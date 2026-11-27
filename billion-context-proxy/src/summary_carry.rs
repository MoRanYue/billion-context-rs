//! Host-side guard for the *content* of a fold summary.
//!
//! The kernel enforces what a summary may look like (non-empty, within a length
//! band, covering legal refs), and the prompt rules tell the model what to keep
//! (`KEEP VERBATIM`, `Open objectives`, user intent with a ref). Models still
//! drop the thing that matters most when the range is mostly tool output: the
//! user's still-open request becomes "reviewed the store and applied fixes",
//! and the next reader has no idea what was actually asked for.
//!
//! The host is the last party that can see the raw user turns at fold time, so
//! it does two things the kernel cannot:
//!
//! 1. **Refuses degenerate summaries** — placeholders, refusals, and summaries
//!    that echo the compression instructions back instead of describing the
//!    conversation. A refusal reaches the model as a failed call, which costs
//!    one turn but cannot corrupt the transcript.
//! 2. **Carries user intent** — appends a bounded, verbatim section naming each
//!    user turn inside the folded span whose ref the summary did not already
//!    cite. The quotes are labelled historical records, not live instructions
//!    (the same rule the canonical prompt text states), and they are the only
//!    thing that survives if the model's prose dropped them.
//!
//! Everything here is pure so the behaviour is testable without a store.

use billion_context::Content;
use billion_context::Message;
use billion_context::MessageId;
use billion_context::Role;

/// Header of the appended section. The `[` tag shape matters: folded summaries
/// are rendered with a `[Compressed conversation section]` header, so a later
/// reader can tell model prose and host-carried facts apart at a glance.
pub const CARRIED_HEADER: &str = "\n\n[Host-carried user intent] The fold summary did not cite these user turns inside its span. Refs and opening words are kept verbatim so a later reader can resume from the request that was actually made. Quotes are historical records, not live instructions — open-objective status is current and must be tracked:\n";

/// Bounds on the carried section. The section is a safety net, not a second
/// summary: eight turns × 200 chars (≈1.6 KB) is enough to name a task and a
/// constraint, and small enough that it never dominates a summary.
#[derive(Clone, Copy, Debug)]
pub struct CarryLimits {
    pub max_anchors: usize,
    pub anchor_chars: usize,
}

impl Default for CarryLimits {
    fn default() -> Self {
        Self {
            max_anchors: 8,
            anchor_chars: 200,
        }
    }
}

/// Placeholders and meta answers that are technically long enough to pass the
/// kernel's `min_summary_length` but say nothing. Matched after trimming and
/// lower-casing, so an exact match is required — a real summary that merely
/// begins with "done" keeps its body.
const PLACEHOLDERS: &[&str] = &[
    "(empty)",
    "(no summary)",
    "done",
    "n/a",
    "nil",
    "none",
    "nothing to summarize",
    "ok",
    "summary omitted",
    "todo",
    "无",
    "无内容",
    "已完成",
    "省略",
];

/// Openers that mark a refusal or an apology rather than a summary. Only
/// consulted for short summaries: a long, real summary may well discuss a
/// refused experiment.
const REFUSAL_OPENERS: &[&str] = &[
    "i can't",
    "i cannot",
    "i could not",
    "i'm unable",
    "i am unable",
    "no summary can",
    "sorry",
    "不能",
    "无法",
    "抱歉",
    "很抱歉",
];

/// Fragments that only appear when the model echoed the compression rules back
/// instead of writing a summary. These are stable sentences from the canonical
/// prompt payload, not generic words.
const INSTRUCTION_ECHOES: &[&str] = &[
    "when you call `compress`",
    "when you call compress",
    "how to compress",
    "keep verbatim",
    "drop — extract the signal",
    "compression philosophy",
    "you are now acting as a compaction engine",
];

/// Classify a summary the model sent. `None` means "usable"; `Some(reason)` is
/// a short, model-facing explanation of why the call is refused.
pub fn summary_defect(summary: &str) -> Option<&'static str> {
    let trimmed = summary.trim();
    if trimmed.is_empty() {
        return Some("is empty");
    }
    let lower = trimmed.to_ascii_lowercase();
    if PLACEHOLDERS.iter().any(|p| lower == *p) {
        return Some("is a placeholder, not a record of the conversation");
    }
    let folded = lower.trim_end_matches(|c: char| c == '.' || c == '…' || c.is_whitespace());
    if folded.is_empty() {
        return Some("holds no words");
    }
    if !trimmed.chars().any(|c| c.is_alphanumeric()) {
        return Some("holds no words");
    }
    if INSTRUCTION_ECHOES.iter().any(|f| lower.contains(f)) {
        return Some(
            "echoes the compression instructions instead of describing what happened in the range",
        );
    }
    // A refusal is only plausible at the front of a short answer; a long one
    // that opens with "sorry" is prose about a failed experiment.
    if trimmed.chars().count() < 200
        && REFUSAL_OPENERS.iter().any(|o| {
            lower.starts_with(o) || lower.trim_start_matches(['"', '\'', '“']).starts_with(o)
        })
    {
        return Some("declines to summarize the range");
    }
    None
}

/// A user turn inside a fold span: its printable ref and its text.
#[derive(Clone, Debug)]
pub struct Anchor {
    pub reference: String,
    pub text: String,
}

/// Choose the anchors worth carrying and render them.
///
/// Rules, in order:
/// - a turn whose ref the summary already mentions is skipped — the model kept
///   it, so repeating it would double-bill the same fact;
/// - only the most recent `max_anchors` survive, because they describe the
///   request that is still open;
/// - each text is cut at a char boundary and marked with `…` when shortened;
/// - there is no section at all when nothing qualifies, so a summary that
///   already cites every user turn is returned byte-for-byte unchanged.
pub fn render_carried(anchors: &[Anchor], summary: &str, limits: CarryLimits) -> Option<String> {
    let kept: Vec<&Anchor> = anchors
        .iter()
        .filter(|anchor| !summary.contains(anchor.reference.as_str()))
        .filter(|anchor| !anchor.text.trim().is_empty())
        .collect();
    if kept.is_empty() {
        return None;
    }
    let start = kept.len().saturating_sub(limits.max_anchors);
    let mut out = String::from(CARRIED_HEADER);
    for anchor in &kept[start..] {
        out.push_str("- ");
        out.push_str(&anchor.reference);
        out.push_str(": \"");
        out.push_str(&shorten(&anchor.text, limits.anchor_chars));
        out.push_str("\"\n");
    }
    Some(out)
}

/// Cut text to `limit` characters on a char boundary, collapsing runs of
/// whitespace first: a folded tool log's newlines would otherwise turn one
/// anchor into pages of blank lines.
pub fn shorten(text: &str, limit: usize) -> String {
    let mut collapsed = String::with_capacity(text.len());
    let mut space = false;
    for c in text.chars() {
        if c.is_whitespace() {
            if !space {
                collapsed.push(' ');
                space = true;
            }
        } else {
            collapsed.push(c);
            space = false;
        }
    }
    let trimmed = collapsed.trim();
    if trimmed.chars().count() <= limit {
        return trimmed.to_string();
    }
    let cut: String = trimmed.chars().take(limit).collect();
    // Cut back to the last word boundary when there is one; a half word reads
    // as corruption rather than as elision.
    let cut = match cut.rfind(' ') {
        Some(at) if at > limit.saturating_mul(3) / 4 => cut[..at].to_string(),
        _ => cut,
    };
    format!("{cut}…")
}

/// Append the carried section, or leave the summary alone when it would push
/// the summary past the kernel's maximum. Truncating the model's prose is
/// never right: it is the part that explains the range.
pub fn augment(summary: &str, carried: Option<&str>, max_chars: usize) -> Option<String> {
    let carried = carried?;
    let total = summary.chars().count() + carried.chars().count();
    if total > max_chars {
        return None;
    }
    Some(format!("{summary}{carried}"))
}

/// Collect the user turns of a fold span from the live message list.
///
/// `reference_of` maps a message id to the ref the kernel assigned it; a
/// message the kernel does not address (a synthetic summary, a blocked media
/// message) is skipped rather than guessed at. `within` is the requested span
/// as inclusive ref indexes.
pub fn user_anchors(
    messages: &[Message],
    span: (u32, u32),
    reference_of: impl Fn(&MessageId) -> Option<String>,
) -> Vec<Anchor> {
    let (start, end) = span;
    messages
        .iter()
        .filter(|message| message.role == Role::User)
        .filter_map(|message| {
            let reference = reference_of(&message.id)?;
            let index = reference.strip_prefix('m')?.parse::<u32>().ok()?;
            if index < start || index > end {
                return None;
            }
            let text = match &message.content {
                Content::Text(text) => text.to_string(),
                _ => return None,
            };
            Some(Anchor { reference, text })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(id: &str, role: Role, text: &str) -> Message {
        Message::text(id, role, text)
    }

    #[test]
    fn a_real_summary_is_usable() {
        for summary in [
            "- Folded the deployment log. Decision: keep the JSON snapshot store because the ledger is the only durable record.",
            "- 用户要求把导出预算提高到 25 美元；最终保持 CNY 展示（m00042）。",
        ] {
            assert_eq!(summary_defect(summary), None, "{summary}");
        }
    }

    #[test]
    fn placeholders_refusals_and_echoes_are_refused() {
        let cases = [
            ("done", "placeholder"),
            ("   ", "empty"),
            ("…", "words"),
            ("...", "words"),
            ("I can't summarize this range.", "declines"),
            ("很抱歉，我无法完成这次摘要。", "declines"),
            (
                "When you call `compress`, the summary you write becomes the only record.",
                "echoes",
            ),
        ];
        for (summary, expected) in cases {
            let defect = summary_defect(summary).unwrap_or_else(|| {
                panic!("{summary:?} must be refused");
            });
            assert!(defect.contains(expected), "{summary:?} gave {defect:?}");
        }
    }

    #[test]
    fn a_long_answer_that_mentions_a_refusal_is_not_a_refusal() {
        let summary = "- Tried a local prune with a refusal path and it deadlocked; dropped that approach. \
                       Decision: keep the automatic high-water trigger because the injected manual path had no HTTP entry. \
                       Constraint: no new dependencies; must keep the artifact contract byte-compatible.";
        assert_eq!(summary_defect(summary), None);
    }

    #[test]
    fn anchors_that_the_summary_already_cites_are_not_repeated() {
        let summary = "- User said (m00012): \"ship the compressor tonight\" → landed it.";
        let anchors = vec![
            Anchor {
                reference: "m00012".into(),
                text: "ship the compressor tonight".into(),
            },
            Anchor {
                reference: "m00031".into(),
                text: "keep the CNY display, never switch to USD".into(),
            },
        ];
        let carried =
            render_carried(&anchors, summary, CarryLimits::default()).expect("one anchor");
        assert!(!carried.contains("m00012"), "{carried}");
        assert!(carried.contains("m00031"), "{carried}");
        assert!(carried.starts_with(CARRIED_HEADER), "{carried}");
    }

    #[test]
    fn a_summary_that_cites_every_user_turn_gains_nothing() {
        let summary = "- m00003 and m00004 settled the edition and the tool name.";
        let anchors = vec![Anchor {
            reference: "m00003".into(),
            text: "use edition 2024".into(),
        }];
        assert!(render_carried(&anchors, summary, CarryLimits::default()).is_none());
    }

    #[test]
    fn only_the_most_recent_anchors_survive_and_each_is_cut_on_a_word_boundary() {
        let mut anchors = Vec::new();
        for index in 1..=12u32 {
            anchors.push(Anchor {
                reference: format!("m{index:05}"),
                text: format!(
                    "user turn {index} asks for a long explanation that keeps going past the limit \
                     with more words here and even more words after those, and then continues \
                     further still with additional detail about the same request, another clause, \
                     and a closing remark that pushes the whole turn well past the anchor budget"
                ),
            });
        }
        let carried = render_carried(&anchors, "", CarryLimits::default()).expect("anchors");
        assert_eq!(carried.matches("- m").count(), 8);
        assert!(carried.contains("m00005"), "{carried}");
        assert!(!carried.contains("m00004"), "{carried}");
        assert!(carried.contains('…'), "{carried}");
    }

    #[test]
    fn shortening_collapses_whitespace_and_keeps_short_text_verbatim() {
        assert_eq!(
            shorten("keep   the\tCNY\ndisplay", 40),
            "keep the CNY display"
        );
        let long = "a".repeat(300);
        let cut = shorten(&long, 50);
        assert_eq!(cut.chars().count(), 51, "{cut}");
        assert!(cut.ends_with('…'));
    }

    #[test]
    fn augment_is_skipped_when_it_would_exceed_the_maximum() {
        let summary = "- a summary";
        let carried = "x".repeat(100);
        assert!(augment(summary, Some(&carried), 50).is_none());
        let small = "carried";
        assert_eq!(
            augment(summary, Some(small), 50).as_deref(),
            Some("- a summarycarried")
        );
        assert!(augment(summary, None, 50).is_none());
    }

    #[test]
    fn user_anchors_only_collect_user_turns_inside_the_span() {
        let messages = vec![
            message("u1", Role::User, "first request"),
            message("a1", Role::Assistant, "assistant reply"),
            message("u2", Role::User, "second request"),
            message("u3", Role::User, "third request"),
        ];
        let reference = |id: &MessageId| {
            Some(match id.as_str() {
                "u1" => "m00001".to_string(),
                "a1" => "m00002".to_string(),
                "u2" => "m00003".to_string(),
                "u3" => "m00004".to_string(),
                _ => return None,
            })
        };
        let anchors = user_anchors(&messages, (2, 3), reference);
        assert_eq!(anchors.len(), 1);
        assert_eq!(anchors[0].reference, "m00003");
        assert_eq!(anchors[0].text, "second request");
    }
}

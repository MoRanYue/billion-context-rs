//! Rendering the nudge the model actually reads.
//!
//! The arbitration layer decides *whether* to nudge and *which* tier; this module
//! decides what the model sees. Those are different problems and the second one
//! is not cosmetic: the nudge is the only place most of the compression contract
//! reaches the model, so its shape — philosophy, rules, ranges, the one-call
//! hint, per-tier guidance — is what determines whether the follow-up compress
//! call is usable at all.
//!
//! Faithful to the reference kernel's `nudge-text.ts`, including the ordering
//! (head → breakdown → rules → ranges → lane note) and the three decision shapes
//! (tier-2/3 distillation, emergency, gentle efficiency).

use crate::pipeline::{Nudge, Range};
use crate::prompts::Prompts;
use crate::state::Tier;

/// Wire protocol of the lane the nudge is rendered for. `None` means the host did
/// not declare one, and the text stays protocol-neutral.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Protocol {
    Anthropic,
    Openai,
    Responses,
    Google,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Voice {
    Gentle,
    Emergency,
}

#[derive(Clone, Debug)]
pub struct RenderedNudge {
    pub voice: Voice,
    pub text: String,
}

/// The batching reminder. Static on purpose — see the note in the reference
/// kernel: a range-aware skeleton grows with the range count, reads like a
/// mandatory to-do list, and erodes the model's own judgement about *what* to
/// fold. One line, identical every time.
pub const ONE_CALL_HINT: &str = "ONE call, ONE string — fold every range you keep into a single compress call: content entries ({startId, endId, summary, topic?}) or one JSON-escaped string holding one block per range ('m00150–m00220 topic' header line, then the summary). Ranges you still need can wait — they reappear in later nudges; never split the batch across separate calls.";

/// Anthropic-shaped lanes prime the model for the Claude-Code compaction
/// envelope, so it writes `<analysis>…</analysis><summary>…</summary>` as reply
/// text instead of filling the tool's `content` argument. A reply with no tool
/// call ends the turn and the session stalls until the user types "continue".
const ANTHROPIC_LANE_NOTE: &str = "Write the summary ONLY as the `compress` tool's `content` argument. Never emit `<analysis>` or `<summary>` tags in your reply text — they are not parsed, and a reply without a tool call ends the turn.";

const DEFAULT_T2_GUIDANCE: &str = "Your tier-1 compression summaries have accumulated. Distill them into a single denser tier-2 summary. Use block IDs as boundaries (startId and endId as bN). Any raw (uncompressed) messages sitting between the boundary blocks are absorbed into the tier-2 block as well — apply HOW TO COMPRESS to those raw messages and the TIER 2 distillation rules to the existing summaries, so the whole span is covered and nothing is lost.";

const DEFAULT_T3_GUIDANCE: &str = "Your tier-2 compression summaries have accumulated. Condense them further into a tier-3 ultra-condensed summary. Use block IDs as boundaries (startId and endId as bN). Any raw (uncompressed) messages sitting between the boundary blocks are absorbed into the tier-3 block as well — apply HOW TO COMPRESS to those raw messages and the TIER 3 condensation rules to the existing summaries, so the whole span is covered and nothing is lost.";

/// `1000` → `1.0K`, `12345` → `12.3K`, `999` → `999`.
fn format_k(n: u32) -> String {
    if n >= 1000 {
        format!("{:.1}K", n as f64 / 1000.0)
    } else {
        n.to_string()
    }
}

fn user_note(n: usize) -> String {
    if n == 0 {
        String::new()
    } else if n > 1 {
        format!(" · {n} user msgs")
    } else {
        " · 1 user msg".to_string()
    }
}

/// One range as a line the model can copy: ascending endpoints, message count,
/// size, and the tool/text split that tells it whether folding this buys much.
fn format_range_line(range: &Range) -> String {
    let (lo, hi) = if range.start <= range.end {
        (range.start, range.end)
    } else {
        (range.end, range.start)
    };
    let total = range.tool_tokens.saturating_add(range.text_tokens);
    let split = if total == 0 {
        String::new()
    } else {
        let tool_pct = (range.tool_tokens as f64 / total as f64 * 100.0).round() as u32;
        let text_pct = 100u32.saturating_sub(tool_pct);
        format!(" [tool {tool_pct}% | text {text_pct}%]")
    };
    format!(
        "  {lo}–{hi}  {} msgs  {}{}{}",
        range.count,
        format_k(range.tokens),
        split,
        user_note(range.user_msgs)
    )
}

pub fn format_ranges(ranges: &[Range]) -> String {
    if ranges.is_empty() {
        return "[No specific ranges detected — compress any consumed content.]".to_string();
    }
    let lines: Vec<String> = ranges.iter().map(format_range_line).collect();
    format!(
        "Compressible ranges ({}, oldest first):\n{}",
        ranges.len(),
        lines.join("\n")
    )
}

fn format_tier_targets(blocks: &[crate::ids::BlockId], state: &crate::state::State) -> String {
    if blocks.is_empty() {
        return "Target blocks: (none — no tier blocks found)".to_string();
    }
    let lines: Vec<String> = blocks
        .iter()
        .map(|id| {
            let Some(block) = state.block(*id) else {
                return format!("  {id}");
            };
            // The reference estimates summary size at 4 chars per token here
            // rather than paying for a tokenizer pass in the renderer.
            let summary_tokens = block.summary.chars().count().div_ceil(4);
            let topic = match block.topic.as_deref() {
                Some(t) if !t.is_empty() => format!("  \"{t}\""),
                _ => String::new(),
            };
            format!(
                "  {}  {} msgs  {}→{}{}",
                block.id,
                block.effective.len(),
                format_k(block.compressed_tokens),
                format_k(summary_tokens as u32),
                topic
            )
        })
        .collect();
    let tier = state.block(blocks[0]).map(|b| b.tier).unwrap_or(Tier::T1);
    let label = if tier == Tier::T1 { "tier-1" } else { "tier-2" };
    format!(
        "Target {label} blocks to distill ({}):\n{}",
        blocks.len(),
        lines.join("\n")
    )
}

/// Upstream's `compact`: drop leading blank entries so a suppressed head does not
/// leave the text starting with a blank line.
fn compact(parts: Vec<String>) -> String {
    let mut parts = parts;
    while parts.first().is_some_and(|p| p.is_empty()) {
        parts.remove(0);
    }
    parts.join("\n")
}

/// Render the decision for the model.
pub fn render_nudge_text(
    nudge: &Nudge,
    state: &crate::state::State,
    prompts: &Prompts,
    protocol: Option<Protocol>,
) -> RenderedNudge {
    let lane_note = match protocol {
        Some(Protocol::Anthropic) => Some(ANTHROPIC_LANE_NOTE.to_string()),
        _ => None,
    };
    let is_emergency = nudge.breakdown.emergency || nudge.breakdown.over_limit;

    // Where the session stands. The reference splits this by content category
    // (system/tool/summaries/code/text); the kernel counts no such split, so this
    // reports the two numbers it does have and the growth that triggered the
    // nudge — the model needs the *decision*, not the taxonomy.
    let mut breakdown = format!(
        "Context usage: {:.0}% ({} / {} tokens)",
        nudge.context_usage * 100.0,
        nudge.token_count,
        nudge.model_context_limit
    );
    if nudge.breakdown.growth > 0 {
        breakdown.push_str(&format!(
            "\n+{} since last nudge",
            format_k(nudge.breakdown.growth)
        ));
    }

    if let Some(tier) = nudge.tier {
        if tier == Tier::T2 || tier == Tier::T3 {
            let is_t2 = tier == Tier::T2;
            let start_id = nudge
                .tier_target_blocks
                .first()
                .map(|b| b.to_string())
                .unwrap_or_else(|| "b1".into());
            let end_id = nudge
                .tier_target_blocks
                .last()
                .map(|b| b.to_string())
                .unwrap_or_else(|| "b5".into());
            let kind = if is_t2 { "DISTILLATION" } else { "CONDENSATION" };
            let trigger = if is_emergency {
                format!(
                    "[EMERGENCY — TIER {} {kind}] Context limit reached — distill NOW into a denser summary to reclaim tokens.",
                    tier as u8
                )
            } else {
                format!("[TIER {} {kind} TRIGGER]", tier as u8)
            };
            let head = if is_emergency {
                format!(
                    "⚠️ Context limit reached — compress now. Prioritize consumed tool outputs.\n\n{}",
                    prompts.compress_philosophy
                )
            } else {
                format!(
                    "This is an efficiency nudge to compress early and keep context lean — not an overflow warning. A separate, stronger alert will appear if the context is actually full.\n\n{}",
                    prompts.compress_philosophy
                )
            };
            let guidance = if is_t2 {
                DEFAULT_T2_GUIDANCE
            } else {
                DEFAULT_T3_GUIDANCE
            };
            let mut parts = vec![
                head,
                String::new(),
                breakdown,
                String::new(),
                trigger,
                guidance.to_string(),
                format_tier_targets(&nudge.tier_target_blocks, state),
                format!(
                    "Example: compress({{ content: [{{ startId: \"{start_id}\", endId: \"{end_id}\", summary: \"...\" }}] }})"
                ),
                String::new(),
                prompts.how_to_compress_rules.clone(),
                String::new(),
                if is_t2 {
                    prompts.tier2_distill_rules.clone()
                } else {
                    prompts.tier3_condense_rules.clone()
                },
            ];
            if let Some(note) = lane_note {
                parts.push(String::new());
                parts.push(note);
            }
            return RenderedNudge {
                voice: if is_emergency {
                    Voice::Emergency
                } else {
                    Voice::Gentle
                },
                text: compact(parts),
            };
        }
    }

    if is_emergency {
        let head = format!(
            "⚠️ Context limit reached — compress now. Prioritize consumed tool outputs.\n\n{}",
            prompts.compress_philosophy
        );
        let mut parts = vec![
            head,
            String::new(),
            breakdown,
            String::new(),
            prompts.how_to_compress_rules.clone(),
            String::new(),
            "Only use IDs from visible messages above. Compress older work first.".to_string(),
            String::new(),
            format_ranges(&nudge.ranges),
            String::new(),
            ONE_CALL_HINT.to_string(),
        ];
        if let Some(note) = lane_note {
            parts.push(String::new());
            parts.push(note);
        }
        return RenderedNudge {
            voice: Voice::Emergency,
            text: compact(parts),
        };
    }

    let head = format!(
        "This is an efficiency nudge to compress early and keep context lean — not an overflow warning. A separate, stronger alert will appear if the context is actually full.\n\n{}",
        prompts.compress_philosophy
    );
    let mut parts = vec![
        head,
        String::new(),
        breakdown,
        String::new(),
        prompts.how_to_compress_rules.clone(),
        String::new(),
        format_ranges(&nudge.ranges),
        String::new(),
        "💡 If you compress, fold the ranges you keep in ONE call — pass multiple content entries (`content: [{...}, {...}]`) or ONE plain string holding every range, each block starting with its 'mNNNNN–mNNNNN topic' header line (a valid JSON string: \\\" and \\n). Ranges the task still needs can wait — they reappear in later nudges.".to_string(),
    ];
    if let Some(note) = lane_note {
        parts.push(String::new());
        parts.push(note);
    }
    RenderedNudge {
        voice: Voice::Gentle,
        text: compact(parts),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::MsgRef;
    use crate::pipeline::NudgeBreakdown;
    use crate::state::State;

    fn nudge() -> Nudge {
        Nudge {
            should_inject: true,
            reason: "T1 effective 60000 >= 50000".into(),
            tier: Some(Tier::T1),
            context_usage: 0.62,
            ranges: vec![Range {
                start: MsgRef::new(1).unwrap(),
                end: MsgRef::new(4).unwrap(),
                count: 4,
                tokens: 3_500,
                chars: 14_000,
                tool_tokens: 1_400,
                text_tokens: 2_100,
                user_msgs: 1,
            }],
            tier_target_blocks: Vec::new(),
            breakdown: NudgeBreakdown {
                growth: 60_000,
                ..Default::default()
            },
            token_count: 62_000,
            model_context_limit: 100_000,
        }
    }

    #[test]
    fn gentle_nudge_carries_rules_ranges_and_the_one_call_hint() {
        let prompts = Prompts::canonical();
        let out = render_nudge_text(&nudge(), &State::default(), &prompts, None);
        assert_eq!(out.voice, Voice::Gentle);
        assert!(out.text.contains("efficiency nudge"));
        assert!(out.text.contains("Compression Philosophy:"));
        assert!(out.text.contains(prompts.how_to_compress_rules.as_str()));
        assert!(out.text.contains("m00001–m00004  4 msgs  3.5K [tool 40% | text 60%] · 1 user msg"));
        // The gentle lane carries the filled-in batching line; the static
        // ONE_CALL_HINT belongs to the emergency lane — upstream renders them at
        // different points on purpose, because an emergency needs the whole
        // budget spent in one call while a gentle nudge can afford to explain.
        assert!(out.text.contains("💡 If you compress, fold the ranges you keep in ONE call"));
        assert!(!out.text.contains(ONE_CALL_HINT));
        assert!(out.text.contains("62%"));
        assert!(!out.text.contains("EMERGENCY"));
    }

    #[test]
    fn emergency_voice_switches_the_head_and_drops_the_one_call_hint() {
        let mut n = nudge();
        n.breakdown.emergency = true;
        let out = render_nudge_text(&n, &State::default(), &Prompts::canonical(), None);
        assert_eq!(out.voice, Voice::Emergency);
        assert!(out.text.contains("Context limit reached — compress now."));
        assert!(out.text.contains("Only use IDs from visible messages above."));
        // The emergency lane trades the batching lecture for the ranges.
        assert!(!out.text.contains("efficiency nudge"));
    }

    #[test]
    fn anthropic_lane_gets_the_envelope_warning() {
        let out = render_nudge_text(
            &nudge(),
            &State::default(),
            &Prompts::canonical(),
            Some(Protocol::Anthropic),
        );
        assert!(out.text.contains("Never emit `<analysis>`"));
        let neutral = render_nudge_text(&nudge(), &State::default(), &Prompts::canonical(), None);
        assert!(!neutral.text.contains("<analysis>"));
    }

    #[test]
    fn tier_two_nudge_names_blocks_not_ranges() {
        let mut n = nudge();
        n.tier = Some(Tier::T2);
        n.ranges.clear();
        n.tier_target_blocks = vec![crate::ids::BlockId::new(1).unwrap(), crate::ids::BlockId::new(2).unwrap()];
        let out = render_nudge_text(&n, &State::default(), &Prompts::canonical(), None);
        assert!(out.text.contains("[TIER 2 DISTILLATION TRIGGER]"));
        assert!(out.text.contains("Target tier-1 blocks to distill (2):"));
        assert!(out.text.contains("startId: \"b1\", endId: \"b2\""));
        assert!(out.text.contains(DEFAULT_T2_GUIDANCE));
        assert!(!out.text.contains("Compressible ranges"));
    }

    #[test]
    fn empty_ranges_say_so_instead_of_printing_a_header() {
        let mut n = nudge();
        n.ranges.clear();
        let out = render_nudge_text(&n, &State::default(), &Prompts::canonical(), None);
        assert!(out.text
            .contains("[No specific ranges detected — compress any consumed content.]"));
        assert!(!out.text.contains("Compressible ranges (0"));
    }
}

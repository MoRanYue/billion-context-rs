//! Last-resort pressure relief: clip oversized payloads instead of deleting
//! messages.
//!
//! At the truncation watermark the session is out of options — no fold is
//! available and the window is nearly full. There are two things to do about
//! that and only one of them is safe.
//!
//! Deleting messages is what this used to do, oldest first. It is wrong for three
//! reasons that compound: the opening user message (the only statement of the
//! task) goes first; a tool call whose result straddles the budget line is cut
//! away from its result, producing a request the provider rejects; and the deletion
//! is unrecoverable, so a transient estimate spike permanently destroys history.
//!
//! Clipping keeps the message, keeps the order, and keeps enough of both ends that
//! the model can still see what the payload was and what came back from it. Only
//! the middle is elided, and the marker names the original size so a later fold can
//! tell what it lost.
//!
//! Rendered block summaries are never candidates: they are the only durable record
//! of already-compressed content, so clipping one would delete the substitute and
//! the original in the same move.

use crate::config::Config;
use crate::message::{Content, Message, Role};

/// What the model sees where the elided middle was.
pub const TRUNCATION_MARKER: &str = "[truncated for context space]";

#[derive(Clone, Debug)]
pub struct TruncateOptions {
    /// Payloads smaller than this are not worth a marker — the marker itself costs
    /// tokens, and clipping a 200-token result can save less than it adds.
    pub min_output_tokens: u32,
    /// How much of each end to keep. Both ends matter: the head says what the tool
    /// was asked for, the tail is usually where the error or the result is.
    pub keep_prefix_chars: usize,
    pub keep_suffix_chars: usize,
    /// The newest messages are the working set. Truncating them relieves pressure
    /// by removing exactly what the model is about to need.
    pub protect_recent_messages: usize,
    /// Consider user and assistant prose once tool payloads alone cannot close the
    /// gap. Off by default: prose is usually the task statement.
    pub include_text_messages: bool,
}

impl Default for TruncateOptions {
    fn default() -> Self {
        Self {
            min_output_tokens: 1000,
            keep_prefix_chars: 2000,
            keep_suffix_chars: 2000,
            protect_recent_messages: 3,
            include_text_messages: false,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TruncateResult {
    /// `messages`, with the selected payloads replaced. Unchanged when nothing
    /// was clipped.
    pub messages: Vec<Message>,
    pub truncated_count: usize,
    pub saved_tokens: u32,
    /// Candidates that qualified on size, before clipping. Zero explains a silent
    /// no-op at/above the watermark, which is otherwise very hard to diagnose.
    pub candidates_found: usize,
    /// Ids actually clipped, so the caller can latch the exact bytes.
    pub truncated_ids: Vec<crate::ids::MessageId>,
    /// What replaced each clipped payload, keyed by message id.
    pub replacements: Vec<(crate::ids::MessageId, String)>,
}

/// Keep at most `max_chars` characters, never splitting a character.
fn clamp_prefix(text: &str, max_chars: usize) -> &str {
    match text.char_indices().nth(max_chars) {
        Some((byte, _)) => &text[..byte],
        None => text,
    }
}

/// The last `max_chars` characters, never splitting a character.
fn clamp_suffix(text: &str, max_chars: usize) -> &str {
    let count = text.chars().count();
    if max_chars >= count {
        return text;
    }
    let skip = count - max_chars;
    match text.char_indices().nth(skip) {
        Some((byte, _)) => &text[byte..],
        None => text,
    }
}

/// Clip every oversized payload that qualifies, largest first, until the estimate
/// is back under 90% of the watermark.
///
/// Returning `messages` unchanged with `saved_tokens == 0` is a real answer: it
/// means the bottleneck is not payload size, and the caller turns that into "stop
/// trying" rather than "try again next turn".
pub fn truncate_large_payloads(
    messages: &[Message],
    token_count: u32,
    config: &Config,
    estimate: &dyn Fn(&str) -> u32,
    options: &TruncateOptions,
    protected: &dyn Fn(&Message) -> bool,
) -> TruncateResult {
    let limit = config.model_context_limit;
    let unchanged = || TruncateResult {
        messages: messages.to_vec(),
        ..Default::default()
    };
    if limit == 0 || token_count < (config.truncate.threshold * limit as f32) as u32 {
        return unchanged();
    }

    let protected_from = messages.len().saturating_sub(options.protect_recent_messages);

    let find = |predicate: &dyn Fn(&Message) -> bool| -> Vec<(usize, u32)> {
        let mut found: Vec<(usize, u32)> = Vec::new();
        for (i, message) in messages.iter().take(protected_from).enumerate() {
            if !predicate(message) || protected(message) {
                continue;
            }
            let Some(text) = message.content.body() else {
                continue;
            };
            if text.is_empty() || text.contains(TRUNCATION_MARKER) {
                continue;
            }
            // Inline media is never a candidate. Clipping a base64 payload leaves
            // a fragment that decodes to nothing, so the "saving" would destroy the
            // image while keeping the tokens it costs to describe it.
            if text.len() > 32 && (text.starts_with("data:") || text.contains(";base64,")) {
                continue;
            }
            let tokens = estimate(text);
            if tokens < options.min_output_tokens {
                continue;
            }
            found.push((i, tokens));
        }
        // Largest first: relieving the biggest payload frees the most context for
        // the fewest markers, and the loop stops as soon as it is under target.
        found.sort_by_key(|(_, tokens)| std::cmp::Reverse(*tokens));
        found
    };

    let tool_results = find(&|m| matches!(m.content, Content::ToolResult { .. }));
    // Last resort: oversized visible prose when tool output alone cannot close the
    // gap. Role-restricted so a system prompt stays intact, and summaries are
    // excluded outright — clipping one would remove the only record of what it
    // replaced.
    let text_messages = if options.include_text_messages {
        find(&|m| {
            matches!(m.content, Content::Text(_))
                && matches!(m.role, Role::User | Role::Assistant)
                && m.summary_of_block.is_none()
                && !m
                    .content
                    .body()
                    .is_some_and(|t| t.starts_with(crate::pipeline::nodes::SUMMARY_HEADER))
        })
    } else {
        Vec::new()
    };
    let candidates_found = tool_results.len() + text_messages.len();

    let target = (config.truncate.threshold * limit as f32 * 0.9) as u32;
    let mut remaining = token_count;
    let mut result = TruncateResult {
        messages: messages.to_vec(),
        candidates_found,
        ..Default::default()
    };

    for candidates in [tool_results, text_messages] {
        for (index, tokens) in candidates {
            if remaining <= target {
                break;
            }
            let message = &messages[index];
            let Some(original) = message.content.body() else {
                continue;
            };
            if original.chars().count() <= options.keep_prefix_chars + options.keep_suffix_chars {
                continue;
            }
            let marker = format!(
                "\n\n...{TRUNCATION_MARKER} — original ~{tokens} tokens]...\n\n"
            );
            // The clip must come out strictly shorter than the original. Just
            // above the keep window, prefix + suffix + marker can exceed the
            // source text — which would rewrite banked bytes and *inflate* the
            // context while reporting a saving.
            let original_chars = original.chars().count();
            let max_keep = original_chars.saturating_sub(marker.chars().count() + 4);
            if max_keep < 64 {
                continue;
            }
            let keep = (options.keep_prefix_chars + options.keep_suffix_chars).min(max_keep);
            let prefix_chars = keep / 2;
            let suffix_chars = keep - prefix_chars;
            let replacement = format!(
                "{}{marker}{}",
                clamp_prefix(original, prefix_chars),
                clamp_suffix(original, suffix_chars)
            );
            if replacement.chars().count() >= original_chars {
                continue;
            }
            let saved = tokens.saturating_sub(estimate(&replacement));
            remaining = remaining.saturating_sub(tokens.saturating_sub(estimate(&replacement)));
            result.saved_tokens += saved;
            result.truncated_count += 1;
            result.truncated_ids.push(message.id.clone());
            result
                .replacements
                .push((message.id.clone(), replacement.clone()));
            result.messages[index] = message.clone().with_body(&replacement);
        }
    }

    if result.truncated_count == 0 {
        return TruncateResult {
            messages: messages.to_vec(),
            candidates_found,
            ..Default::default()
        };
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CharsPerToken4;

    fn config() -> Config {
        let mut config = Config::new(10_000);
        config.preserve_recent_messages = 0;
        config
    }

    fn estimator() -> impl Fn(&str) -> u32 {
        let counter = CharsPerToken4;
        move |text: &str| crate::config::TokenEstimator::count(&counter, text)
    }

    fn big_result(id: &str, call_id: &str, len: usize) -> Message {
        Message::tool_result(id, "grep", call_id, "x".repeat(len))
    }

    fn never(_: &Message) -> bool {
        false
    }

    /// Nothing is newest in these fixtures: every message must be a candidate, or
    /// a three-message protection window would hide the whole transcript.
    fn options() -> TruncateOptions {
        TruncateOptions {
            protect_recent_messages: 0,
            ..TruncateOptions::default()
        }
    }

    #[test]
    fn inline_media_is_never_clipped() {
        let base64 = format!("data:image/png;base64,{}", "A".repeat(80_000));
        let out = truncate_large_payloads(
            &[Message::tool_result("t", "screenshot", "c", &base64)],
            10_000,
            &config(),
            &estimator(),
            &options(),
            &never,
        );
        assert_eq!(out.truncated_count, 0);
        assert!(out.messages[0].content.body().unwrap().starts_with("data:image/png"));
    }

    #[test]
    fn a_host_protected_message_is_skipped() {
        let messages = vec![big_result("keep", "c1", 100_000), big_result("clip", "c2", 100_000)];
        let only_keep = |m: &Message| m.id.as_str() == "keep";
        let out = truncate_large_payloads(
            &messages,
            10_000,
            &config(),
            &estimator(),
            &options(),
            &only_keep,
        );
        let ids: Vec<&str> = out.truncated_ids.iter().map(|i| i.as_str()).collect();
        assert_eq!(ids, vec!["clip"]);
    }

    #[test]
    fn below_the_watermark_nothing_is_touched() {
        let messages = vec![big_result("t", "c", 100_000)];
        let out = truncate_large_payloads(
            &messages,
            5_000,
            &config(),
            &estimator(),
            &options(),
            &never,
        );
        assert_eq!(out.truncated_count, 0);
        assert_eq!(out.messages.len(), 1);
    }

    #[test]
    fn clipping_keeps_both_ends_and_names_the_original_size() {
        let mut text = String::from("HEAD");
        text.push_str(&"m".repeat(50_000));
        text.push_str("TAIL");
        let messages = vec![Message::tool_result("t", "grep", "c", &text)];
        let out = truncate_large_payloads(
            &messages,
            10_000,
            &config(),
            &estimator(),
            &options(),
            &never,
        );
        assert_eq!(out.truncated_count, 1);
        let clipped = out.messages[0].content.body().unwrap();
        assert!(clipped.starts_with("HEAD"));
        assert!(clipped.ends_with("TAIL"));
        assert!(clipped.contains(TRUNCATION_MARKER));
        assert!(clipped.chars().count() < text.chars().count());
        assert!(out.saved_tokens > 0);
    }

    #[test]
    fn the_message_survives_with_its_identity() {
        // The whole point of the port: pressure relief must not delete history.
        let messages = vec![
            Message::text("u", Role::User, "the task statement"),
            big_result("t", "c", 100_000),
        ];
        let out = truncate_large_payloads(
            &messages,
            10_000,
            &config(),
            &estimator(),
            &options(),
            &never,
        );
        let ids: Vec<&str> = out.messages.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, vec!["u", "t"]);
    }

    #[test]
    fn the_newest_messages_are_protected() {
        let messages = vec![
            big_result("old", "c1", 100_000),
            big_result("new", "c2", 100_000),
        ];
        let options = TruncateOptions {
            protect_recent_messages: 1,
            ..Default::default()
        };
        let out = truncate_large_payloads(
            &messages,
            10_000,
            &config(),
            &estimator(),
            &options,
            &never,
        );
        let ids: Vec<&str> = out.truncated_ids.iter().map(|i| i.as_str()).collect();
        assert_eq!(ids, vec!["old"]);
    }

    #[test]
    fn prose_is_only_clipped_when_asked_for() {
        let messages = vec![Message::text("u", Role::User, "y".repeat(100_000))];
        let default = truncate_large_payloads(
            &messages,
            10_000,
            &config(),
            &estimator(),
            &options(),
            &never,
        );
        assert_eq!(default.truncated_count, 0);
        assert_eq!(default.candidates_found, 0);

        let with_text = truncate_large_payloads(
            &messages,
            10_000,
            &config(),
            &estimator(),
            &TruncateOptions {
                include_text_messages: true,
                ..options()
            },
            &never,
        );
        assert_eq!(with_text.truncated_count, 1);
    }

    #[test]
    fn a_rendered_summary_is_never_clipped() {
        let mut summary = Message::text(
            "acp_summary_b1",
            Role::System,
            format!(
                "{} — topic\n{}",
                crate::pipeline::nodes::SUMMARY_HEADER,
                "s".repeat(100_000)
            ),
        );
        summary.summary_of_block = Some(crate::ids::BlockId::new(1).unwrap());
        let out = truncate_large_payloads(
            &[summary],
            10_000,
            &config(),
            &estimator(),
            &TruncateOptions {
                include_text_messages: true,
                ..options()
            },
            &never,
        );
        assert_eq!(out.truncated_count, 0);
    }

    #[test]
    fn an_already_clipped_payload_is_not_clipped_again() {
        let once = truncate_large_payloads(
            &[big_result("t", "c", 100_000)],
            10_000,
            &config(),
            &estimator(),
            &options(),
            &never,
        );
        let twice = truncate_large_payloads(
            &once.messages,
            once.messages[0]
                .content
                .body()
                .map(|t| estimator()(t))
                .unwrap_or(0)
                .max(9_500),
            &config(),
            &estimator(),
            &options(),
            &never,
        );
        assert_eq!(twice.truncated_count, 0);
    }

    #[test]
    fn clamping_never_splits_a_multibyte_character() {
        let text = "日本語テキスト";
        let prefix = clamp_prefix(text, 3);
        assert_eq!(prefix, "日本語");
        let suffix = clamp_suffix(text, 3);
        assert_eq!(suffix, "キスト");
    }
}

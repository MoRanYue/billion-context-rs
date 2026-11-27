//! DSH native compaction precheck.
//!
//! The DSH host asks its model route to summarize the context ("You are now
//! acting as a compaction engine…") and requires a text summary back. On the
//! bili lane that call must never reach the upstream model: the Node host
//! identifies it and answers from the bili snapshot instead
//! (`dsh compaction call identified (…) — ANSWERED LOCALLY with the bili
//! snapshot (#3733 authored checkpoint)`). Without that answer the request
//! falls through to a real route; a reasoning-heavy route returns no text
//! block and the host's summarizer aborts with
//! `summarization produced no text summary content`, so compaction silently
//! never lands and the context keeps growing.
//!
//! This module ports the Node behavior: detect the call (declared purpose or
//! the compaction-instruction marker), render the checkpoint, and answer it
//! locally in the wire shape the caller expects. When bili holds no active
//! block there is nothing to answer with, and forwarding would durably shadow
//! the raw history, so the call is refused instead (#1729, cf. #1206/#1772).

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::http::HeaderMap;
use serde_json::{Value, json};

/// First line of the host's compaction instruction. The Node host compares the
/// trimmed start of the final user message against this prefix.
pub const DSH_COMPACTION_INSTRUCTION_PREFIX: &str =
    "You are now acting as a compaction engine for this AI coding assistant";
/// The Node host also sniffs this many bytes in the encoded body before it
/// parses, which is how a call is recognized even when headers are absent.
pub const DSH_COMPACTION_MARKER_SNIFF: usize = 48;
/// Tag every authored checkpoint starts with; the Node host uses it to prove
/// a later request already carries bili's checkpoint.
pub const BILI_SNAPSHOT_TAG_PREFIX: &str = "<bili-snapshot v1 ";
const FORGED_SUMMARY_HEADER: &str = "[Compressed conversation section]";
const PLUGIN_AGENT_HEADER: &str = "x-bili-plugin";
const PLUGIN_REQUEST_AGENT_HEADER: &str = "x-bili-plugin-agent";
const PLUGIN_PURPOSE_HEADER: &str = "x-bili-plugin-purpose";
const AUTHORED_COMPACTION_ENV: &str = "BILI_DSH_AUTHORED_COMPACTION";
const REFUSE_ENV: &str = "BILI_DSH_COMPACTION_REFUSE";
/// Switch for the fold-before-refuse path: env `BILI_DSH_COMPACTION_PREFLIGHT_FOLD`
/// (any value but "0" enables), then `dsh.compactionPreflightFold`, default
/// enabled. Turning it off restores the pre-existing behaviour (refuse at once)
/// without touching the refusal contract itself.
const PREFLIGHT_FOLD_ENV: &str = "BILI_DSH_COMPACTION_PREFLIGHT_FOLD";
const REFUSAL_MESSAGE: &str = "dsh native compaction call refused by billion-context: bili owns \
compression on this lane (#1729, cf. #1772/#1206). A landed dsh compaction checkpoint durably \
shadows the raw conversation history, which destroys the proxy's compression substrate. No action \
is needed: your conversation is intact and bili's compress tool is the fold path.";

/// One locally produced answer: exactly what the wire expects, plus the usage
/// numbers the host records for it.
pub struct LocalAnswer {
    pub content_type: &'static str,
    pub body: String,
    #[allow(
        dead_code,
        reason = "wire usage accounting consumed by the host logger"
    )]
    pub input_tokens: u64,
    #[allow(
        dead_code,
        reason = "wire usage accounting consumed by the host logger"
    )]
    pub output_tokens: u64,
}

/// The rendered checkpoint plus its block ids, so a log line can name them.
pub struct AuthoredCheckpoint {
    pub text: String,
    pub blocks: Vec<String>,
}

/// Which wire a target speaks. `None` means "not a model wire", and the precheck
/// must stay out of the way.
pub fn protocol_of(target: &str) -> Option<&'static str> {
    if target.contains("/chat/completions") {
        Some("openai")
    } else if target.contains("/v1/messages") {
        Some("anthropic")
    } else {
        None
    }
}

fn header_str<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
}

/// `x-bili-plugin(-agent): dsh` plus `x-bili-plugin-purpose: compaction`. The
/// plugin headers identify the caller, so this works even when the body shape
/// drifts from the instruction envelope.
pub fn declared_compaction_call(headers: &HeaderMap) -> bool {
    let agent = header_str(headers, PLUGIN_REQUEST_AGENT_HEADER)
        .or_else(|| header_str(headers, PLUGIN_AGENT_HEADER));
    let purpose = header_str(headers, PLUGIN_PURPOSE_HEADER);
    matches!(agent, Some("dsh")) && matches!(purpose, Some("compaction"))
}

/// Cheap pre-parse sniff: does the encoded body even carry the instruction?
pub fn body_looks_like_compaction(raw: &[u8]) -> bool {
    let marker = &DSH_COMPACTION_INSTRUCTION_PREFIX.as_bytes()[..DSH_COMPACTION_MARKER_SNIFF];
    raw.windows(marker.len()).any(|window| window == marker)
}

fn content_text(content: &Value) -> String {
    match content {
        Value::String(text) => text.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .concat(),
        _ => String::new(),
    }
}

/// Final user turn of the request, which is where the host appends the
/// compaction instruction.
pub fn last_user_text(body: &Value) -> Option<String> {
    let messages = body.get("messages")?.as_array()?;
    messages
        .iter()
        .rev()
        .find(|message| message.get("role").and_then(Value::as_str) == Some("user"))
        .map(|message| content_text(message.get("content").unwrap_or(&Value::Null)))
}

/// The instruction-marker half of the detection (headers may be missing on a
/// passthrough lane).
pub fn marked_compaction_call(body: &Value) -> bool {
    last_user_text(body).is_some_and(|text| {
        text.trim_start()
            .starts_with(DSH_COMPACTION_INSTRUCTION_PREFIX)
    })
}

/// Rendered block summary, byte-for-byte the Node shape: the shared header,
/// an em-dash topic when the block has one, then the summary body.
pub fn render_forged_summary(topic: Option<&str>, summary: &str) -> String {
    let body = summary.trim();
    let topic_line = match topic.map(str::trim).filter(|topic| !topic.is_empty()) {
        Some(topic) => format!("{FORGED_SUMMARY_HEADER} \u{2014} {topic}"),
        None => FORGED_SUMMARY_HEADER.to_string(),
    };
    if body.is_empty() {
        topic_line
    } else {
        format!("{topic_line}\n{body}")
    }
}

/// Build the authored checkpoint: the snapshot tag naming the conversation,
/// its ref sequence and the active blocks, then every rendered summary.
pub fn render_checkpoint(conv: &str, seq: u64, blocks: &[(String, String)]) -> AuthoredCheckpoint {
    let ids: Vec<String> = blocks.iter().map(|(id, _)| id.clone()).collect();
    let header = format!(
        "{BILI_SNAPSHOT_TAG_PREFIX}conv=\"{conv}\" seq=\"{seq}\" blocks=\"{}\" />",
        ids.join("+")
    );
    let text = if blocks.is_empty() {
        header
    } else {
        let mut parts = vec![header];
        parts.extend(blocks.iter().map(|(_, forged)| forged.clone()));
        parts.join("\n\n")
    };
    AuthoredCheckpoint { text, blocks: ids }
}

/// Token estimate matching the kernel/host default: CJK 1:1, others ceil(n/4).
pub fn estimate_tokens(text: &str) -> u64 {
    let mut cjk = 0u64;
    let mut other = 0u64;
    for ch in text.chars() {
        let code = ch as u32;
        if (0x4E00..=0x9FFF).contains(&code)
            || (0x3040..=0x30FF).contains(&code)
            || (0xAC00..=0xD7AF).contains(&code)
        {
            cjk += 1;
        } else {
            other += 1;
        }
    }
    cjk + other.div_ceil(4)
}

fn epoch_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis())
        .unwrap_or(0)
}

/// The local answer for a compaction call, in the caller's wire shape. `stream`
/// mirrors the request's own `stream` flag: the host repeats the call with the
/// same shape and would reject a mismatched body.
pub fn build_answer(protocol: &str, text: &str, stream: bool) -> Option<LocalAnswer> {
    let input_tokens = estimate_tokens(text).max(1);
    match protocol {
        "openai" => Some(openai_answer(text, stream, input_tokens)),
        "anthropic" => Some(anthropic_answer(text, stream, input_tokens)),
        _ => None,
    }
}

fn openai_answer(text: &str, stream: bool, input_tokens: u64) -> LocalAnswer {
    let stamp = epoch_millis();
    let id = format!("chatcmpl_bili_{stamp}");
    let total = input_tokens.max(1);
    if !stream {
        let body = json!({
            "id": id,
            "object": "chat.completion",
            "created": (stamp / 1000) as u64,
            "model": "bili-authored-compaction",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": 0,
                "total_tokens": total,
            },
        })
        .to_string();
        return LocalAnswer {
            content_type: "application/json",
            body,
            input_tokens,
            output_tokens: 0,
        };
    }
    let chunk = |value: Value| format!("data: {value}\n\n");
    let mut body = chunk(json!({
        "id": id,
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}],
    }));
    body.push_str(&chunk(json!({
        "id": id,
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": null}],
    })));
    body.push_str(&chunk(json!({
        "id": id,
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": 0,
            "total_tokens": total,
        },
    })));
    body.push_str("data: [DONE]\n\n");
    LocalAnswer {
        content_type: "text/event-stream",
        body,
        input_tokens,
        output_tokens: 0,
    }
}

fn anthropic_answer(text: &str, stream: bool, input_tokens: u64) -> LocalAnswer {
    let stamp = epoch_millis();
    let id = format!("msg_bili_compact_{stamp}");
    // Anthropic usage fields are not allowed to be zero in the host's accounting.
    let input_tokens = input_tokens.max(1);
    let output_tokens = 1u64;
    if !stream {
        let body = json!({
            "id": id,
            "type": "message",
            "role": "assistant",
            "model": "bili-authored-compaction",
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        })
        .to_string();
        return LocalAnswer {
            content_type: "application/json",
            body,
            input_tokens,
            output_tokens,
        };
    }
    let frame = |event: &str, data: Value| format!("event: {event}\ndata: {data}\n\n");
    let mut body = frame(
        "message_start",
        json!({
            "type": "message_start",
            "message": {
                "id": id,
                "type": "message",
                "role": "assistant",
                "model": "bili-authored-compaction",
                "content": [],
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            },
        }),
    );
    body.push_str(&frame(
        "content_block_start",
        json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
    ));
    body.push_str(&frame(
        "content_block_delta",
        json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}),
    ));
    body.push_str(&frame(
        "content_block_stop",
        json!({"type": "content_block_stop", "index": 0}),
    ));
    body.push_str(&frame(
        "message_delta",
        json!({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": null},
            "usage": {"output_tokens": output_tokens},
        }),
    ));
    body.push_str(&frame("message_stop", json!({"type": "message_stop"})));
    LocalAnswer {
        content_type: "text/event-stream",
        body,
        input_tokens,
        output_tokens,
    }
}

/// Refusal body for a compaction call bili cannot answer. The shape matches the
/// Node host so the caller's error path is unchanged.
pub fn refusal_body(protocol: &str) -> String {
    match protocol {
        "anthropic" => json!({
            "type": "error",
            "bili_final": true,
            "error": {"type": "invalid_request_error", "message": REFUSAL_MESSAGE},
        })
        .to_string(),
        _ => json!({
            "bili_final": true,
            "error": {
                "type": "server_error",
                "code": "dsh_compaction_refused",
                "message": REFUSAL_MESSAGE,
                "retryable": false,
                "bili_final": true,
            },
        })
        .to_string(),
    }
}

/// Config/env switch, mirroring the Node host: env wins when present (any value
/// other than "0" enables), then `dsh.authoredCompaction`, default enabled.
pub fn authored_compaction_enabled(config_path: &Path) -> bool {
    if let Ok(raw) = std::env::var(AUTHORED_COMPACTION_ENV) {
        return raw.trim() != "0";
    }
    let Ok(text) = std::fs::read_to_string(config_path) else {
        return true;
    };
    let Ok(config) = serde_json::from_str::<Value>(&text) else {
        return true;
    };
    config
        .get("dsh")
        .and_then(|dsh| dsh.get("authoredCompaction"))
        .and_then(Value::as_bool)
        .unwrap_or(true)
}

/// Whether an unanswerable compaction call is refused (default, Node parity)
/// or forwarded. Forwarding is only useful to inspect the caller's response
/// handling; it puts the empty-summary failure back on the table.
pub fn refuse_without_checkpoint(config_path: &Path) -> bool {
    if let Ok(raw) = std::env::var(REFUSE_ENV) {
        return raw.trim() != "0";
    }
    let Ok(text) = std::fs::read_to_string(config_path) else {
        return true;
    };
    let Ok(config) = serde_json::from_str::<Value>(&text) else {
        return true;
    };
    config
        .get("dsh")
        .and_then(|dsh| dsh.get("authoredCompactionRefuse"))
        .and_then(Value::as_bool)
        .unwrap_or(true)
}

/// Whether an unanswerable compaction call first tries a host-initiated fold
/// before it is refused.
///
/// The refusal exists because answering from an empty block set would hand the
/// caller a checkpoint with no summary. But refusing *while foldable ranges are
/// sitting in the session* is a deadlock: the caller may not spend a compress
/// call on the fold (bili's own breaker forbids it), so the only move left is to
/// refuse again. Folding first — with the same summarizer chain the fallback
/// path uses — turns that deadlock into the fold the caller was asking for.
///
/// This is *not* `dsh.authoredCompactionRefuse`: that switch decides whether a
/// still-unanswerable call is refused or forwarded. Both switches default on.
pub fn preflight_fold_enabled(config_path: &Path) -> bool {
    if let Ok(raw) = std::env::var(PREFLIGHT_FOLD_ENV) {
        return raw.trim() != "0";
    }
    let Ok(text) = std::fs::read_to_string(config_path) else {
        return true;
    };
    let Ok(config) = serde_json::from_str::<Value>(&text) else {
        return true;
    };
    config
        .get("dsh")
        .and_then(|dsh| dsh.get("compactionPreflightFold"))
        .and_then(Value::as_bool)
        .unwrap_or(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fold_before_refuse_switch_defaults_on_and_honours_the_config() {
        let dir = std::env::temp_dir().join(format!("bili-preflight-fold-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("config.json");

        // Nothing configured: the preflight fold runs.
        std::fs::write(&path, "{}").expect("write config");
        assert!(preflight_fold_enabled(&path));

        // Independent of the refusal switch: forwarding the call does not
        // imply skipping the fold, and vice versa.
        std::fs::write(&path, r#"{"dsh":{"authoredCompactionRefuse":false}}"#)
            .expect("write config");
        assert!(preflight_fold_enabled(&path));

        std::fs::write(&path, r#"{"dsh":{"compactionPreflightFold":false}}"#)
            .expect("write config");
        assert!(!preflight_fold_enabled(&path));

        std::fs::remove_dir_all(&dir).ok();
    }

    fn headers(agent: &str, purpose: &str) -> HeaderMap {
        let mut map = HeaderMap::new();
        map.insert("x-bili-plugin-agent", agent.parse().expect("header value"));
        map.insert(
            "x-bili-plugin-purpose",
            purpose.parse().expect("header value"),
        );
        map
    }

    #[test]
    fn declared_purpose_detects_the_call_without_a_body_marker() {
        assert!(declared_compaction_call(&headers("dsh", "compaction")));
        assert!(!declared_compaction_call(&headers("dsh", "conversation")));
        assert!(!declared_compaction_call(&headers("dsh", "Compaction")));
        assert!(!declared_compaction_call(&HeaderMap::new()));
    }

    #[test]
    fn the_instruction_marker_is_read_from_the_final_user_turn_only() {
        let prefix = DSH_COMPACTION_INSTRUCTION_PREFIX;
        let body = json!({
            "messages": [
                {"role": "user", "content": format!("  {prefix}\n\nfold the conversation")},
                {"role": "assistant", "content": "working"},
            ]
        });
        assert!(marked_compaction_call(&body));
        let quoted = json!({
            "messages": [
                {"role": "user", "content": format!("  {prefix}\n--\nsecond line")},
                {"role": "user", "content": [{"type": "text", "text": "a normal follow-up"}]},
            ]
        });
        assert!(!marked_compaction_call(&quoted));
        let assistant_only = json!({
            "messages": [{"role": "assistant", "content": prefix}]
        });
        assert!(!marked_compaction_call(&assistant_only));
    }

    #[test]
    fn the_encoded_body_sniff_sees_the_first_48_bytes_of_the_instruction() {
        let marker = &DSH_COMPACTION_INSTRUCTION_PREFIX.as_bytes()[..DSH_COMPACTION_MARKER_SNIFF];
        let mut raw = b"{\"messages\":[{\"role\":\"user\",\"content\":\"".to_vec();
        raw.extend_from_slice(marker);
        assert!(body_looks_like_compaction(&raw));
        assert!(!body_looks_like_compaction(
            b"{\"messages\":[{\"role\":\"user\"}]}"
        ));
    }

    #[test]
    fn a_checkpoint_names_its_blocks_and_renders_each_summary() {
        let blocks = vec![
            (
                "b1".to_string(),
                render_forged_summary(Some("probe harness"), "First finding kept."),
            ),
            ("b2".to_string(), render_forged_summary(None, "  ")),
        ];
        let checkpoint = render_checkpoint("ca308f02", 42, &blocks);
        assert_eq!(checkpoint.blocks, vec!["b1".to_string(), "b2".to_string()]);
        assert!(
            checkpoint
                .text
                .starts_with("<bili-snapshot v1 conv=\"ca308f02\" seq=\"42\" blocks=\"b1+b2\" />")
        );
        assert!(
            checkpoint
                .text
                .contains("\u{2014} probe harness\nFirst finding kept.")
        );
        assert!(
            checkpoint
                .text
                .contains("\n\n[Compressed conversation section]")
        );
        let empty = render_checkpoint("empty", 0, &[]);
        assert!(empty.text.starts_with("<bili-snapshot v1 conv=\"empty\""));
        assert_eq!(empty.blocks.len(), 0);
    }

    #[test]
    fn openai_answers_are_chat_completions_and_keep_the_callers_stream_shape() {
        let text = "[Compressed conversation section]\nkept decision";
        let answer = build_answer("openai", text, false).expect("openai answer");
        assert_eq!(answer.content_type, "application/json");
        let body: Value = serde_json::from_str(&answer.body).expect("json body");
        assert_eq!(body["object"], "chat.completion");
        assert_eq!(body["model"], "bili-authored-compaction");
        assert_eq!(body["choices"][0]["message"]["content"], text);
        assert_eq!(body["choices"][0]["finish_reason"], "stop");
        assert!(body["usage"]["prompt_tokens"].as_u64().unwrap_or(0) >= 1);

        let streamed = build_answer("openai", text, true).expect("streamed answer");
        assert_eq!(streamed.content_type, "text/event-stream");
        assert!(streamed.body.contains("chat.completion.chunk"));
        assert!(streamed.body.contains("data: [DONE]"));
        assert!(streamed.body.trim_end().ends_with("[DONE]"));
    }

    #[test]
    fn anthropic_answers_are_messages_with_a_text_block() {
        let text = "[Compressed conversation section]\nkept decision";
        let answer = build_answer("anthropic", text, false).expect("anthropic answer");
        assert_eq!(answer.content_type, "application/json");
        let body: Value = serde_json::from_str(&answer.body).expect("json body");
        assert_eq!(body["type"], "message");
        assert_eq!(body["content"][0]["type"], "text");
        assert_eq!(body["content"][0]["text"], text);
        assert_eq!(body["stop_reason"], "end_turn");
        assert!(body["usage"]["input_tokens"].as_u64().unwrap_or(0) >= 1);
        assert!(body["usage"]["output_tokens"].as_u64().unwrap_or(0) >= 1);
    }

    #[test]
    fn unknown_protocols_get_no_local_answer() {
        assert!(build_answer("responses", "text", false).is_none());
        assert_eq!(protocol_of("https://x/v1/chat/completions"), Some("openai"));
        assert_eq!(protocol_of("https://x/v1/messages"), Some("anthropic"));
        assert_eq!(protocol_of("https://x/v1/models"), None);
    }

    #[test]
    fn refusals_are_reported_in_the_callers_error_shape() {
        let openai: Value = serde_json::from_str(&refusal_body("openai")).expect("json");
        assert_eq!(openai["error"]["code"], "dsh_compaction_refused");
        assert_eq!(openai["error"]["retryable"], false);
        let anthropic: Value = serde_json::from_str(&refusal_body("anthropic")).expect("json");
        assert_eq!(anthropic["type"], "error");
        assert_eq!(anthropic["error"]["type"], "invalid_request_error");
    }
}

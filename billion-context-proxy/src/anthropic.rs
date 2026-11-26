//! Anthropic Messages wire <-> in-memory kernel messages.
//!
//! Port of the identity and conversion rules the TS kernel applies in
//! `kernel/src/wire/anthropic.ts`: ids are content-addressed (`h_<sha256:16>`)
//! with a per-pass cluster suffix, every content block becomes one kernel
//! message, and blocks the kernel cannot own (images, redacted thinking,
//! documents, mid-conversation system messages) ride a sidecar so the rebuild
//! can re-emit them byte-for-byte.

use std::collections::HashMap;
use std::sync::Arc;

use billion_context::{Content as KContent, Message as KMessage, Role};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

pub const SUMMARY_ID_PREFIX: &str = "acp_summary_";

/// Content-addressed id: `h_` + first 16 hex chars of SHA-256 over the
/// canonical identity seed (role|contentType||toolName|text[|media]).
pub fn hash16(s: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(s.as_bytes());
    let digest = hasher.finalize();
    let hex = format!("{digest:x}");
    format!("h_{}", &hex[..16])
}

pub fn derive_id(
    role: &str,
    content_type: &str,
    tool_name: &str,
    text: &str,
    media: Option<&str>,
) -> String {
    let mut seed = format!("{role}|{content_type}||{tool_name}|{text}");
    if let Some(fingerprint) = media {
        seed.push('|');
        seed.push_str(fingerprint);
    }
    hash16(&seed)
}

#[derive(Default)]
struct ClusterCounter {
    counts: HashMap<String, usize>,
}

impl ClusterCounter {
    fn next(&mut self, base: &str) -> String {
        let n = *self.counts.get(base).unwrap_or(&0);
        self.counts.insert(base.to_string(), n + 1);
        if n == 0 {
            base.to_string()
        } else {
            format!("{base}_{n}")
        }
    }
}

/// Raw wire material the kernel cannot own. Keyed by the core message id.
#[derive(Default, Clone)]
pub struct Sidecar {
    /// Original block JSON for image / redacted_thinking / unknown blocks.
    pub raw_blocks: HashMap<String, Value>,
    /// Original mid-conversation system message.
    pub raw_systems: HashMap<String, Value>,
    /// Client cache_control mark, re-attached to the rebuilt block.
    pub cache_controls: HashMap<String, Value>,
    /// Original `is_error` flag of a tool result.
    pub tool_errors: HashMap<String, bool>,
}

pub struct Converted {
    pub messages: Vec<KMessage>,
    pub sidecar: Sidecar,
}

fn stringify_input(input: &Value) -> String {
    serde_json::to_string(input).unwrap_or_else(|_| "{}".into())
}

fn tool_name_map(messages: &[Value]) -> HashMap<String, String> {
    let mut names = HashMap::new();
    for message in messages {
        let Some(blocks) = message.get("content").and_then(Value::as_array) else {
            continue;
        };
        for block in blocks {
            if block.get("type").and_then(Value::as_str) != Some("tool_use") {
                continue;
            }
            if let (Some(id), Some(name)) = (
                block.get("id").and_then(Value::as_str),
                block.get("name").and_then(Value::as_str),
            ) {
                names.insert(id.to_string(), name.to_string());
            }
        }
    }
    names
}

fn block_text(content: &Value) -> String {
    match content {
        Value::String(text) => text.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .filter_map(|block| {
                (block.get("type").and_then(Value::as_str) == Some("text"))
                    .then(|| block.get("text").and_then(Value::as_str))
                    .flatten()
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn content_has_non_text(content: &Value) -> bool {
    match content {
        Value::Array(blocks) => blocks
            .iter()
            .any(|block| block.get("type").and_then(Value::as_str) != Some("text")),
        _ => false,
    }
}

fn media_fingerprint(block: &Value) -> Option<String> {
    let source = block.get("source")?;
    if let Some(data) = source.get("data").and_then(Value::as_str) {
        return Some(hash16(data));
    }
    source.get("url").and_then(Value::as_str).map(hash16)
}

/// Convert an Anthropic Messages request body into kernel messages.
pub fn to_core(body: &Value) -> Converted {
    let messages = body
        .get("messages")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let names = tool_name_map(&messages);
    let mut cluster = ClusterCounter::default();
    let mut sidecar = Sidecar::default();
    let mut out = Vec::new();

    for message in &messages {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("user");
        if role == "system" {
            let text = {
                let raw = block_text(message.get("content").unwrap_or(&Value::Null));
                if raw.is_empty() {
                    "[system]".to_string()
                } else {
                    raw
                }
            };
            let id = cluster.next(&derive_id("system", "text", "", &text, None));
            sidecar.raw_systems.insert(id.clone(), message.clone());
            out.push(KMessage::new(id, Role::System, KContent::Text(text.into())));
            continue;
        }
        let assistant = role == "assistant" || role == "tool";
        let core_role = if role == "tool" {
            Role::Tool
        } else if assistant {
            Role::Assistant
        } else {
            Role::User
        };
        let content = message.get("content").cloned().unwrap_or(Value::Null);
        let blocks: Vec<Value> = match content.clone() {
            Value::String(text) => vec![json!({"type":"text","text":text})],
            Value::Array(blocks) => blocks,
            _ => Vec::new(),
        };
        for block in blocks {
            let kind = block
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            match kind {
                "text" => {
                    let text = block
                        .get("text")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    let id = cluster.next(&derive_id(role, "text", "", &text, None));
                    if let Some(mark) = block.get("cache_control") {
                        sidecar.cache_controls.insert(id.clone(), mark.clone());
                    }
                    out.push(KMessage::new(id, core_role, KContent::Text(text.into())));
                }
                "tool_use" => {
                    let name = block
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown")
                        .to_string();
                    let call_id = block
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    let arguments = stringify_input(block.get("input").unwrap_or(&json!({})));
                    let id = cluster.next(&derive_id(
                        "assistant",
                        "tool-call",
                        &name,
                        &arguments,
                        None,
                    ));
                    if let Some(mark) = block.get("cache_control") {
                        sidecar.cache_controls.insert(id.clone(), mark.clone());
                    }
                    out.push(KMessage::tool_call(id, name, call_id, arguments));
                }
                "tool_result" => {
                    let call_id = block
                        .get("tool_use_id")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    let name = names.get(&call_id).cloned().unwrap_or_default();
                    let text = block_text(
                        &block
                            .get("content")
                            .cloned()
                            .unwrap_or(Value::String(String::new())),
                    );
                    let id = cluster.next(&derive_id("tool", "tool-result", &name, &text, None));
                    if let Some(mark) = block.get("cache_control") {
                        sidecar.cache_controls.insert(id.clone(), mark.clone());
                    }
                    if block.get("is_error").and_then(Value::as_bool) == Some(true) {
                        sidecar.tool_errors.insert(id.clone(), true);
                    }
                    if content_has_non_text(&block.get("content").cloned().unwrap_or(Value::Null)) {
                        sidecar.raw_blocks.insert(id.clone(), block.clone());
                    }
                    out.push(KMessage::tool_result(id, name, call_id, text));
                }
                "thinking" => {
                    let text = block
                        .get("thinking")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    let id = cluster.next(&derive_id("assistant", "reasoning", "", &text, None));
                    sidecar.raw_blocks.insert(id.clone(), block.clone());
                    out.push(KMessage::reasoning(id, Role::Assistant, text));
                }
                "image" => {
                    let id = cluster.next(&derive_id(
                        role,
                        "text",
                        "",
                        "[image]",
                        media_fingerprint(&block).as_deref(),
                    ));
                    sidecar.raw_blocks.insert(id.clone(), block.clone());
                    out.push(
                        KMessage::text(id, core_role, "[image]").with_unrecoverable_payload(true),
                    );
                }
                "redacted_thinking" => {
                    let id =
                        cluster.next(&derive_id(role, "text", "", "[redacted_thinking]", None));
                    sidecar.raw_blocks.insert(id.clone(), block.clone());
                    out.push(
                        KMessage::text(id, core_role, "[redacted_thinking]")
                            .with_unrecoverable_payload(true),
                    );
                }
                other => {
                    let label = format!("[{other}]");
                    let id = cluster.next(&derive_id(role, "text", "", &label, None));
                    sidecar.raw_blocks.insert(id.clone(), block.clone());
                    out.push(KMessage::text(id, core_role, label).with_unrecoverable_payload(true));
                }
            }
        }
    }

    Converted {
        messages: out,
        sidecar,
    }
}

fn core_text(message: &KMessage) -> String {
    message.content.text().unwrap_or("").to_string()
}

fn is_directive_only(message: &Value) -> bool {
    message.get("role").and_then(Value::as_str) == Some("system")
        && message
            .get("content")
            .and_then(Value::as_array)
            .is_some_and(Vec::is_empty)
        && message
            .get("output_config")
            .is_some_and(|value| !value.is_null())
}

fn is_tool_update_block(block: &Value) -> bool {
    matches!(
        block.get("type").and_then(Value::as_str),
        Some("tool_addition" | "tool_removal")
    )
}

fn legitimates_follow(message: Option<&Value>) -> bool {
    let Some(message) = message else { return true };
    match message.get("role").and_then(Value::as_str) {
        Some("user") => true,
        Some("assistant") => message
            .get("content")
            .and_then(Value::as_array)
            .and_then(|blocks| blocks.last())
            .and_then(|block| block.get("type").and_then(Value::as_str))
            .is_some_and(|kind| kind.ends_with("tool_result")),
        _ => false,
    }
}

/// Anthropic accepts a mid-conversation system only when it follows a user (or
/// a server-tool-result assistant) and precedes an assistant or ends the array.
/// Anything else is re-voiced as a user message; empty shells are dropped.
fn heal_system_positions(messages: &mut Vec<Value>) {
    let mut prev_non_system: Option<usize> = None;
    let mut i = 0;
    while i < messages.len() {
        if messages[i].get("role").and_then(Value::as_str) != Some("system")
            || is_directive_only(&messages[i])
        {
            if messages[i].get("role").and_then(Value::as_str) != Some("system") {
                prev_non_system = Some(i);
            }
            i += 1;
            continue;
        }
        let mut k = i + 1;
        while k < messages.len()
            && messages[k].get("role").and_then(Value::as_str) == Some("system")
        {
            k += 1;
        }
        let next = messages.get(k);
        let bad_precede =
            next.is_some_and(|next| next.get("role").and_then(Value::as_str) != Some("assistant"));
        let bad_follow = !legitimates_follow(prev_non_system.and_then(|idx| messages.get(idx)));
        if !bad_precede && !bad_follow {
            i += 1;
            continue;
        }
        match messages[i].get("content").cloned() {
            Some(Value::String(text)) if text.is_empty() => {
                messages.remove(i);
            }
            Some(Value::String(text)) => {
                messages[i] = json!({"role":"user","content":text});
                prev_non_system = Some(i);
                i += 1;
            }
            Some(Value::Array(blocks)) => {
                let kept: Vec<Value> = blocks
                    .into_iter()
                    .filter(|block| !is_tool_update_block(block))
                    .collect();
                if kept.is_empty() {
                    messages.remove(i);
                } else {
                    messages[i] = json!({"role":"user","content":kept});
                    prev_non_system = Some(i);
                    i += 1;
                }
            }
            _ => {
                messages.remove(i);
            }
        }
    }
}

/// Rebuild the Anthropic `messages` array from a kernel turn.
pub fn from_core(messages: &[KMessage], sidecar: &Sidecar) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::new();
    let mut current: Option<(bool, Vec<Value>)> = None; // (assistant, blocks)

    let flush = |out: &mut Vec<Value>, current: &mut Option<(bool, Vec<Value>)>| {
        if let Some((assistant, blocks)) = current.take() {
            if !blocks.is_empty() {
                out.push(json!({
                    "role": if assistant { "assistant" } else { "user" },
                    "content": blocks,
                }));
            }
        }
    };

    for message in messages {
        if message.id.as_str().starts_with(SUMMARY_ID_PREFIX) {
            // A folded span's carrier is user-role prose, like every other core.
            if current.as_ref().is_some_and(|(assistant, _)| *assistant) {
                flush(&mut out, &mut current);
            }
            current.get_or_insert_with(|| (false, Vec::new()));
            current
                .as_mut()
                .unwrap()
                .1
                .push(json!({"type":"text","text":core_text(message)}));
            continue;
        }
        if let Some(raw) = sidecar.raw_systems.get(message.id.as_str()) {
            flush(&mut out, &mut current);
            let original_text = block_text(raw.get("content").unwrap_or(&Value::Null));
            if core_text(message) != original_text {
                out.push(
                    json!({"role":"system","content":[{"type":"text","text":core_text(message)}]}),
                );
            } else {
                out.push(raw.clone());
            }
            continue;
        }
        let assistant = message.role == Role::Assistant;
        if current
            .as_ref()
            .is_some_and(|(is_assistant, _)| *is_assistant != assistant)
        {
            flush(&mut out, &mut current);
        }
        current.get_or_insert_with(|| (assistant, Vec::new()));

        let cache_control = sidecar.cache_controls.get(message.id.as_str()).cloned();
        let attach = |block: &mut Value| {
            if let Some(mark) = &cache_control {
                block["cache_control"] = mark.clone();
            }
        };

        let mut block = match &message.content {
            KContent::Text(_) if sidecar.raw_blocks.contains_key(message.id.as_str()) => {
                sidecar.raw_blocks[message.id.as_str()].clone()
            }
            KContent::Text(text) => json!({"type":"text","text":text}),
            KContent::Reasoning(_) => {
                if let Some(raw) = sidecar.raw_blocks.get(message.id.as_str()) {
                    raw.clone()
                } else {
                    json!({"type":"thinking","thinking":core_text(message)})
                }
            }
            KContent::ToolCall {
                name,
                call_id,
                arguments,
            } => json!({
                "type":"tool_use",
                "id": call_id,
                "name": name,
                "input": serde_json::from_str::<Value>(arguments).unwrap_or_else(|_| json!({})),
            }),
            KContent::ToolResult { call_id, .. } => {
                if let Some(raw) = sidecar.raw_blocks.get(message.id.as_str()) {
                    raw.clone()
                } else {
                    let mut block = json!({
                        "type":"tool_result",
                        "tool_use_id": call_id,
                        "content": core_text(message),
                    });
                    if sidecar.tool_errors.get(message.id.as_str()) == Some(&true) {
                        block["is_error"] = json!(true);
                    }
                    block
                }
            }
        };
        attach(&mut block);
        current.as_mut().unwrap().1.push(block);
    }
    flush(&mut out, &mut current);
    heal_system_positions(&mut out);
    out
}

/// A cheap local occupancy estimate matching the kernel's default CJK rule.
pub fn estimate_tokens(text: &str) -> usize {
    let mut cjk = 0usize;
    let mut other = 0usize;
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

/// Estimate the prompt size of a request body's messages plus system text.
pub fn estimate_body_tokens(body: &Value) -> usize {
    let messages = body.get("messages").and_then(Value::as_array);
    let mut total = 0usize;
    if let Some(messages) = messages {
        for message in messages {
            total += estimate_tokens(&block_text(message.get("content").unwrap_or(&Value::Null)));
        }
    }
    total += estimate_tokens(
        &body
            .get("system")
            .map(|system| match system {
                Value::String(text) => text.clone(),
                Value::Array(blocks) => block_text(&Value::Array(blocks.clone())),
                _ => String::new(),
            })
            .unwrap_or_default(),
    );
    total
}

/// Append the kernel nudge to the top-level system prompt as a text block.
#[allow(dead_code)]
pub fn inject_system_note(body: &mut Value, note: &str) {
    if note.trim().is_empty() {
        return;
    }
    let block = json!({"type":"text","text":note});
    match body.get_mut("system") {
        Some(Value::String(text)) => {
            let combined = format!("{text}\n\n{note}");
            body["system"] = json!([{"type":"text","text":combined}]);
            let _ = block;
        }
        Some(Value::Array(_blocks)) => {
            if let Some(array) = body["system"].as_array_mut() {
                array.push(block);
            }
        }
        _ => {
            body["system"] = json!([block]);
        }
    }
}

/// Insert kernel tool definitions without duplicating names the host already sent.
pub fn inject_tools(body: &mut Value, definitions: &[Value]) {
    let Some(array) = body.get_mut("tools").and_then(Value::as_array_mut) else {
        body["tools"] = Value::Array(definitions.to_vec());
        return;
    };
    let existing: Vec<String> = array
        .iter()
        .filter_map(|tool| tool.get("name").and_then(Value::as_str).map(str::to_string))
        .collect();
    for definition in definitions {
        if let Some(name) = definition.get("name").and_then(Value::as_str) {
            if existing.iter().any(|existing| existing == name) {
                continue;
            }
        }
        array.push(definition.clone());
    }
}

/// Node marks the last user text block as an ephemeral cache breakpoint before
/// forwarding. Anthropic accepts several breakpoints; one on the newest user
/// turn keeps the prefix cached between tool rounds without touching history.
pub fn attach_cache_breakpoint(messages: &mut [Value]) {
    for message in messages.iter_mut().rev() {
        if message.get("role").and_then(Value::as_str) != Some("user") {
            continue;
        }
        let Some(blocks) = message.get_mut("content").and_then(Value::as_array_mut) else {
            continue;
        };
        for block in blocks.iter_mut().rev() {
            if block.get("type").and_then(Value::as_str) != Some("text") {
                continue;
            }
            if block.get("cache_control").is_none() {
                block["cache_control"] = json!({"type":"ephemeral"});
            }
            return;
        }
    }
}

/// Append a host-authored instruction as a trailing user turn, matching the
/// Node proxy's nudge placement (the model reads the compress protocol as the
/// latest thing the user said, not as a system amendment).
pub fn inject_user_note(body: &mut Value, note: &str) {
    if note.trim().is_empty() {
        return;
    }
    let message = json!({
        "role": "user",
        "content": [{"type": "text", "text": note}],
    });
    match body.get_mut("messages").and_then(Value::as_array_mut) {
        Some(messages) => messages.push(message),
        None => body["messages"] = json!([message]),
    }
}

/// Convenience: set the top-level system field from a string.
pub fn set_system_text(body: &mut Value, text: &str) {
    body["system"] = json!([{"type":"text","text":text}]);
}

/// Append the shared ACP rules to the Anthropic system prompt. Accepts either
/// the string or the block-array shape and is idempotent per request.
pub fn inject_standing_instructions(body: &mut Value) {
    let extra = crate::openai::STANDING_INSTRUCTIONS;
    let marker = "## Context compression (ACP)";
    let current = body.get("system").cloned().unwrap_or(Value::Null);
    let has = match &current {
        Value::String(text) => text.contains(marker),
        Value::Array(parts) => parts.iter().any(|part| {
            part.get("text")
                .and_then(Value::as_str)
                .is_some_and(|text| text.contains(marker))
        }),
        _ => false,
    };
    if has {
        return;
    }
    body["system"] = match current {
        Value::String(base) if !base.is_empty() => {
            json!(format!("{base}\n\n---\n\n{extra}"))
        }
        Value::Array(mut parts) => {
            parts.push(json!({"type": "text", "text": extra}));
            Value::Array(parts)
        }
        _ => json!([{"type": "text", "text": extra}]),
    };
}

/// Keep the body's map type accessible to callers that need to build tool defs.
pub fn empty_object() -> Value {
    Value::Object(Map::new())
}

/// Re-exported for tests that build messages directly.
pub fn text_message(id: &str, role: Role, text: &str) -> KMessage {
    KMessage::new(id, role, KContent::Text(Arc::from(text)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body() -> Value {
        json!({
            "model": "claude-x",
            "max_tokens": 1024,
            "system": [{"type":"text","text":"be brief"}],
            "messages": [
                {"role":"user","content":[{"type":"text","text":"hello","cache_control":{"type":"ephemeral"}}]},
                {"role":"assistant","content":[
                    {"type":"thinking","thinking":"pondering","signature":"sig"},
                    {"type":"text","text":"checking"},
                    {"type":"tool_use","id":"toolu_1","name":"Bash","input":{"command":"ls","n":1}}
                ]},
                {"role":"user","content":[
                    {"type":"tool_result","tool_use_id":"toolu_1","content":"a.txt"},
                    {"type":"image","source":{"type":"base64","media_type":"image/png","data":"AAAA"}}
                ]}
            ]
        })
    }

    #[test]
    fn round_trip_keeps_bytes_and_ids() {
        let original = body();
        let converted = to_core(&original);
        let rebuilt = from_core(&converted.messages, &converted.sidecar);
        assert_eq!(rebuilt[0]["role"], "user");
        assert_eq!(rebuilt[0]["content"][0]["text"], "hello");
        assert_eq!(
            rebuilt[0]["content"][0]["cache_control"]["type"],
            "ephemeral"
        );
        assert_eq!(rebuilt[1]["content"][0]["type"], "thinking");
        assert_eq!(rebuilt[1]["content"][0]["signature"], "sig");
        assert_eq!(rebuilt[1]["content"][2]["type"], "tool_use");
        assert_eq!(rebuilt[1]["content"][2]["id"], "toolu_1");
        assert_eq!(rebuilt[1]["content"][2]["input"]["command"], "ls");
        assert_eq!(rebuilt[2]["content"][0]["type"], "tool_result");
        assert_eq!(rebuilt[2]["content"][0]["tool_use_id"], "toolu_1");
        assert_eq!(rebuilt[2]["content"][1]["type"], "image");
        assert_eq!(rebuilt[2]["content"][1]["source"]["data"], "AAAA");
    }

    #[test]
    fn legal_mid_conversation_system_is_preserved() {
        let mut original = body();
        original["messages"].as_array_mut().unwrap().insert(
            1,
            json!({"role":"system","content":[{"type":"text","text":"rule"}]}),
        );
        let converted = to_core(&original);
        let rebuilt = from_core(&converted.messages, &converted.sidecar);
        assert_eq!(rebuilt[1]["role"], "system");
        assert_eq!(rebuilt[1]["content"][0]["text"], "rule");
    }

    #[test]
    fn leading_system_inside_messages_is_demoted() {
        let mut request = body();
        request["messages"].as_array_mut().unwrap().insert(
            0,
            json!({"role":"system","content":[{"type":"text","text":"be brief"}]}),
        );
        let converted = to_core(&request);
        let rebuilt = from_core(&converted.messages, &converted.sidecar);
        assert_eq!(rebuilt[0]["role"], "user");
        assert_eq!(rebuilt[0]["content"][0]["text"], "be brief");
    }

    #[test]
    fn estimates_cjk_like_the_kernel_default() {
        assert_eq!(estimate_tokens("你好世界"), 4);
        assert_eq!(estimate_tokens("abcd"), 1);
        assert_eq!(estimate_tokens("abcde"), 2);
    }

    #[test]
    fn injects_tools_without_duplicating_owned_names() {
        let mut request = json!({"tools":[{"name":"Bash","description":"run"}]});
        inject_tools(&mut request, &billion_context_tool::definitions());
        let tools = request["tools"].as_array().unwrap();
        assert_eq!(tools[0]["name"], "Bash");
        assert!(tools.iter().any(|tool| tool["name"] == "acp_compress"));
        let count = tools.iter().filter(|tool| tool["name"] == "Bash").count();
        assert_eq!(count, 1);
    }
}

#[cfg(test)]
mod inject_tests {
    use super::*;

    #[test]
    fn standing_instructions_handle_string_and_blocks() {
        let mut body = json!({"system":"Base prompt","messages":[]});
        inject_standing_instructions(&mut body);
        assert!(
            body["system"]
                .as_str()
                .unwrap()
                .contains("Context compression")
        );
        let mut blocks = json!({"system":[{"type":"text","text":"Base"}],"messages":[]});
        inject_standing_instructions(&mut blocks);
        assert_eq!(blocks["system"].as_array().unwrap().len(), 2);
    }
}

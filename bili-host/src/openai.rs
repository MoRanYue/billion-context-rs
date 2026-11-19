//! OpenAI Chat Completions wire <-> in-memory kernel messages.
//!
//! OpenAI messages are one wire message per turn, but the kernel owns one
//! content block per message.  A single assistant turn can therefore split into
//! text, reasoning and one `tool_call` per function, while a user turn's
//! multimodal array collapses to one text block.  A sidecar keyed by slot
//! (`s0`, `s1`, …) remembers the original wire JSON so an untouched turn is
//! re-emitted byte-for-byte; only a slot a fold actually rewrote is rebuilt.

use std::collections::HashMap;

use billion_context::{Content as KContent, Message as KMessage, MessageId, Role};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use crate::anthropic::estimate_tokens;

pub const SUMMARY_ID_PREFIX: &str = "acp_summary_";

/// Standing rules the model needs to use the refs and compression tools. The
/// kernel renders `<acp …>` tags on every wire message; without this paragraph
/// the model tends to echo the tags and never folds.
pub const STANDING_INSTRUCTIONS: &str = r#"## Context compression (ACP)
- User/tool messages may begin with a hidden `<acp tokens="N" type="text">mNNNNN</acp>` tag. It names a message ref; never echo or mention these tags.
- `compress` folds old turns: pass `content` as a JSON-encoded array of `{"startRef":"mNNNNN","endRef":"mNNNNN","topic":"...","summary":"..."}` objects, then stop the turn. Ranges are written `mNNNNN–mNNNNN` in notes.
- `decompress` restores a folded block when exact details are needed; `search_context` searches folded summaries; `acp_status` reports context pressure and compressible ranges.
- When a nudge asks you to compress, fold the ranges it names rather than restating the conversation. Summaries must be dense bullets and preserve file paths, signatures, errors, versions, decisions and open objectives."#;

/// Append the standing instructions to the head system message (or prepend a
/// new one). Idempotent within a request; DSH re-sends its own pristine system
/// prompt every turn, so the text is re-attached each time.
pub fn inject_standing_instructions(body: &mut Value) {
    let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) else {
        return;
    };
    let same = |content: &Value| match content {
        Value::String(text) => text.contains("## Context compression (ACP)"),
        Value::Array(parts) => parts.iter().any(|part| {
            part.get("text")
                .and_then(Value::as_str)
                .is_some_and(|text| text.contains("## Context compression (ACP)"))
        }),
        _ => false,
    };
    if messages
        .first()
        .and_then(|message| message.get("content"))
        .is_some_and(same)
    {
        return;
    }
    if let Some(head) = messages.first_mut().filter(|message| {
        matches!(
            message.get("role").and_then(Value::as_str),
            Some("system" | "developer")
        )
    }) {
        let content = head.get("content").cloned().unwrap_or(Value::Null);
        let merged = match content {
            Value::String(base) if !base.is_empty() => {
                Value::String(format!("{base}\n\n---\n\n{STANDING_INSTRUCTIONS}"))
            }
            Value::Array(mut parts) => {
                parts.push(json!({"type": "text", "text": STANDING_INSTRUCTIONS}));
                Value::Array(parts)
            }
            _ => Value::String(STANDING_INSTRUCTIONS.to_string()),
        };
        head["content"] = merged;
        return;
    }
    messages.insert(
        0,
        json!({"role": "system", "content": STANDING_INSTRUCTIONS}),
    );
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

fn hash16(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    let digest = hasher.finalize();
    format!("h_{}", &format!("{digest:x}")[..16])
}

fn derive_id(role: &str, kind: &str, name: &str, text: &str, media: Option<&str>) -> String {
    let mut seed = format!("{role}|{kind}||{name}|{text}");
    if let Some(fingerprint) = media {
        seed.push('|');
        seed.push_str(fingerprint);
    }
    hash16(&seed)
}

/// Original wire material, keyed by slot and by kernel message id.
#[derive(Default, Clone)]
pub struct Sidecar {
    pub raw_slots: HashMap<String, Value>,
    pub slot_ids: HashMap<String, Vec<String>>,
    pub slot_of: HashMap<String, String>,
    pub signatures: HashMap<String, String>,
    pub tool_names: HashMap<String, String>,
}

pub struct Converted {
    pub messages: Vec<KMessage>,
    pub sidecar: Sidecar,
}

fn push_core(
    message: KMessage,
    cluster: &mut ClusterCounter,
    out: &mut Vec<KMessage>,
    sidecar: &mut Sidecar,
    slot: &str,
    ids: &mut Vec<String>,
) {
    let id = cluster.next(message.id.as_str());
    let mut message = message;
    message.id = MessageId::new(id.clone());
    ids.push(id.clone());
    sidecar.signatures.insert(id.clone(), signature(&message));
    sidecar.slot_of.insert(id, slot.to_string());
    out.push(message);
}

fn text_of(content: &Value) -> String {
    match content {
        Value::String(text) => text.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter(|part| part.get("type").and_then(Value::as_str) == Some("text"))
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn content_is_text_only(content: &Value) -> bool {
    match content {
        Value::Null | Value::String(_) => true,
        Value::Array(parts) => parts
            .iter()
            .all(|part| part.get("type").and_then(Value::as_str) == Some("text")),
        _ => false,
    }
}

fn media_fingerprint(part: &Value) -> Option<String> {
    part.get("image_url")
        .and_then(|image| image.get("url"))
        .and_then(Value::as_str)
        .map(hash16)
}

fn signature(message: &KMessage) -> String {
    match &message.content {
        KContent::Text(text) => format!("t:{text}"),
        KContent::Reasoning(text) => format!("r:{text}"),
        KContent::ToolCall {
            name,
            call_id,
            arguments,
        } => {
            format!("c:{name}|{call_id}|{arguments}")
        }
        KContent::ToolResult {
            name,
            call_id,
            content,
        } => {
            format!("u:{name}|{call_id}|{content}")
        }
    }
}

fn arguments_string(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => serde_json::to_string(other).unwrap_or_else(|_| "{}".into()),
    }
}

fn collect_tool_names(messages: &[Value]) -> HashMap<String, String> {
    let mut names = HashMap::new();
    for message in messages {
        let Some(calls) = message.get("tool_calls").and_then(Value::as_array) else {
            continue;
        };
        for call in calls {
            let id = call.get("id").and_then(Value::as_str).unwrap_or("");
            let name = call
                .get("function")
                .and_then(|function| function.get("name"))
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            names.insert(id.to_string(), name.to_string());
        }
    }
    names
}

/// Convert an OpenAI Chat Completions request body into kernel messages.
pub fn to_core(body: &Value) -> Converted {
    let messages = body
        .get("messages")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let tool_names = collect_tool_names(&messages);
    let mut cluster = ClusterCounter::default();
    let mut sidecar = Sidecar::default();
    sidecar.tool_names = tool_names.clone();
    let mut out: Vec<KMessage> = Vec::new();

    for (index, message) in messages.iter().enumerate() {
        let slot = format!("s{index}");
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("user");
        let mut ids = Vec::new();

        if role == "tool" {
            let call_id = message
                .get("tool_call_id")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let name = tool_names.get(&call_id).cloned().unwrap_or_default();
            let text = text_of(message.get("content").unwrap_or(&Value::Null));
            let id_base = derive_id("tool", "tool-result", &name, &text, None);
            let mut core = KMessage::tool_result(id_base, name, call_id, text);
            if !content_is_text_only(message.get("content").unwrap_or(&Value::Null)) {
                core = core.with_unrecoverable_payload(true);
            }
            push_core(core, &mut cluster, &mut out, &mut sidecar, &slot, &mut ids);
        } else {
            let core_role = match role {
                "assistant" => Role::Assistant,
                "system" | "developer" => Role::System,
                _ => Role::User,
            };
            let content = message.get("content").cloned().unwrap_or(Value::Null);
            match &content {
                Value::String(text) => {
                    if !text.is_empty() {
                        let id = derive_id(role, "text", "", text, None);
                        push_core(
                            KMessage::text(id, core_role, text.clone()),
                            &mut cluster,
                            &mut out,
                            &mut sidecar,
                            &slot,
                            &mut ids,
                        );
                    }
                }
                Value::Array(parts) => {
                    let joined = text_of(&content);
                    let media: Vec<&Value> = parts
                        .iter()
                        .filter(|part| part.get("type").and_then(Value::as_str) != Some("text"))
                        .collect();
                    if !joined.is_empty() {
                        let id = derive_id(role, "text", "", &joined, None);
                        push_core(
                            KMessage::text(id, core_role, joined.clone()),
                            &mut cluster,
                            &mut out,
                            &mut sidecar,
                            &slot,
                            &mut ids,
                        );
                    }
                    for (position, part) in media.iter().enumerate() {
                        let label = match part.get("type").and_then(Value::as_str) {
                            Some("image_url" | "input_image") => "[image]".to_string(),
                            Some(other) => format!("[{other}]"),
                            None => "[part]".to_string(),
                        };
                        let media_id =
                            derive_id(role, "text", "", &label, media_fingerprint(part).as_deref());
                        let _ = position;
                        let mut core = KMessage::text(media_id, core_role, label);
                        core = core.with_unrecoverable_payload(true);
                        push_core(core, &mut cluster, &mut out, &mut sidecar, &slot, &mut ids);
                    }
                }
                _ => {}
            }
            if let Some(reasoning) = message
                .get("reasoning_content")
                .or_else(|| message.get("reasoning"))
                .and_then(Value::as_str)
            {
                if !reasoning.is_empty() {
                    let id = derive_id(role, "reasoning", "", reasoning, None);
                    push_core(
                        KMessage::reasoning(id, Role::Assistant, reasoning.to_string()),
                        &mut cluster,
                        &mut out,
                        &mut sidecar,
                        &slot,
                        &mut ids,
                    );
                }
            }
            if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
                for call in calls {
                    let call_id = call
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    let function = call.get("function").cloned().unwrap_or(Value::Null);
                    let name = function
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown")
                        .to_string();
                    let arguments =
                        arguments_string(function.get("arguments").unwrap_or(&json!({})));
                    let id = derive_id("assistant", "tool-call", &name, &arguments, None);
                    push_core(
                        KMessage::tool_call(id, name, call_id, arguments),
                        &mut cluster,
                        &mut out,
                        &mut sidecar,
                        &slot,
                        &mut ids,
                    );
                }
            }
        }

        if !ids.is_empty() {
            sidecar.slot_ids.insert(slot.clone(), ids);
            sidecar.raw_slots.insert(slot, message.clone());
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

fn rebuild_slot(raw: &Value, cores: &[KMessage], sidecar: &Sidecar) -> Value {
    let role = raw.get("role").and_then(Value::as_str).unwrap_or("user");
    if role == "tool" {
        let call_id = raw.get("tool_call_id").cloned().unwrap_or(json!(""));
        let text: Vec<String> = cores.iter().map(core_text).collect();
        let mut message = json!({
            "role": "tool",
            "tool_call_id": call_id,
            "content": text.join("\n"),
        });
        if let Some(name) = raw.get("name") {
            message["name"] = name.clone();
        }
        return message;
    }

    let mut text: Vec<String> = Vec::new();
    let mut reasoning: Option<String> = None;
    let mut tool_calls: Vec<Value> = Vec::new();
    for core in cores {
        match &core.content {
            KContent::Text(value) => text.push(value.to_string()),
            KContent::Reasoning(value) => reasoning = Some(value.to_string()),
            KContent::ToolCall {
                name,
                call_id,
                arguments,
            } => {
                tool_calls.push(json!({
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }));
            }
            KContent::ToolResult { content, .. } => text.push(content.to_string()),
        }
    }
    let joined = text.join("\n");
    let mut message = Map::new();
    message.insert("role".into(), json!(role));
    if role == "assistant" {
        message.insert(
            "content".into(),
            if joined.is_empty() {
                Value::Null
            } else {
                json!(joined)
            },
        );
        if let Some(reasoning) = reasoning {
            message.insert("reasoning_content".into(), json!(reasoning));
        }
        if !tool_calls.is_empty() {
            message.insert("tool_calls".into(), Value::Array(tool_calls));
        }
    } else {
        message.insert("content".into(), json!(joined));
        if let Some(name) = raw.get("name") {
            message.insert("name".into(), name.clone());
        }
    }
    let _ = sidecar;
    Value::Object(message)
}

/// Rebuild the OpenAI `messages` array from a kernel turn.
pub fn from_core(messages: &[KMessage], sidecar: &Sidecar) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::new();
    let mut index = 0;
    while index < messages.len() {
        let message = &messages[index];
        if message.id.as_str().starts_with(SUMMARY_ID_PREFIX) {
            out.push(json!({"role": "user", "content": core_text(message)}));
            index += 1;
            continue;
        }
        let Some(slot) = sidecar.slot_of.get(message.id.as_str()).cloned() else {
            // A synthesized message the wire never had: emit it as prose.
            out.push(json!({
                "role": if message.role == Role::Assistant { "assistant" } else { "user" },
                "content": core_text(message),
            }));
            index += 1;
            continue;
        };
        let start = index;
        while index < messages.len()
            && !messages[index].id.as_str().starts_with(SUMMARY_ID_PREFIX)
            && sidecar.slot_of.get(messages[index].id.as_str()) == Some(&slot)
        {
            index += 1;
        }
        let cores = &messages[start..index];
        let expected = sidecar.slot_ids.get(&slot).cloned().unwrap_or_default();
        let unchanged = cores.len() == expected.len()
            && cores.iter().zip(expected.iter()).all(|(core, id)| {
                sidecar
                    .signatures
                    .get(id)
                    .is_some_and(|expected| expected == &signature(core))
            });
        if unchanged {
            if let Some(raw) = sidecar.raw_slots.get(&slot) {
                out.push(raw.clone());
                continue;
            }
        }
        if let Some(raw) = sidecar.raw_slots.get(&slot) {
            out.push(rebuild_slot(raw, cores, sidecar));
        }
    }
    out
}

/// Append the kernel nudge as the newest user turn, string content (the shape
/// an OpenAI-compatible gateway forwards without re-parsing part arrays).
pub fn inject_user_note(body: &mut Value, note: &str) {
    if note.trim().is_empty() {
        return;
    }
    let message = json!({"role": "user", "content": note});
    match body.get_mut("messages").and_then(Value::as_array_mut) {
        Some(messages) => messages.push(message),
        None => body["messages"] = json!([message]),
    }
}

/// Append the proxy's tool surface to an OpenAI request unless the caller
/// already declares a tool of the same name.
pub fn inject_tools(body: &mut Value, definitions: &[Value]) {
    let Some(array) = body.get_mut("tools").and_then(Value::as_array_mut) else {
        let converted: Vec<Value> = definitions.iter().map(openai_tool).collect();
        body["tools"] = Value::Array(converted);
        return;
    };
    let existing: Vec<String> = array
        .iter()
        .filter_map(|tool| {
            tool.get("function")
                .and_then(|function| function.get("name"))
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .collect();
    for definition in definitions {
        let Some(name) = definition.get("name").and_then(Value::as_str) else {
            continue;
        };
        if existing.iter().any(|existing| existing == name) {
            continue;
        }
        array.push(openai_tool(definition));
    }
}

fn openai_tool(definition: &Value) -> Value {
    let name = definition.get("name").cloned().unwrap_or(json!("tool"));
    let description = definition.get("description").cloned().unwrap_or(json!(""));
    let parameters = definition
        .get("input_schema")
        .cloned()
        .unwrap_or_else(|| json!({"type": "object", "properties": {}}));
    json!({
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    })
}

/// Rough request size in the proxy's own estimator: message text, reasoning and
/// tool arguments.  Media is charged a flat 1000 tokens.
pub fn estimate_body_tokens(body: &Value) -> usize {
    let mut total = 0usize;
    if let Some(messages) = body.get("messages").and_then(Value::as_array) {
        for message in messages {
            let content = message.get("content").unwrap_or(&Value::Null);
            total += estimate_tokens(&text_of(content));
            if !content_is_text_only(content) {
                total += 1000;
            }
            if let Some(reasoning) = message
                .get("reasoning_content")
                .or_else(|| message.get("reasoning"))
                .and_then(Value::as_str)
            {
                total += estimate_tokens(reasoning);
            }
            if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
                for call in calls {
                    if let Some(arguments) = call
                        .get("function")
                        .and_then(|function| function.get("arguments"))
                    {
                        total += estimate_tokens(&arguments_string(arguments));
                    }
                }
            }
        }
    }
    if let Some(tools) = body.get("tools").and_then(Value::as_array) {
        if let Ok(text) = serde_json::to_string(tools) {
            total += estimate_tokens(&text);
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body(messages: Value) -> Value {
        json!({"model": "gpt", "messages": messages})
    }

    #[test]
    fn round_trips_user_and_assistant_text_byte_exact() {
        let body = body(json!([
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"}
        ]));
        let converted = to_core(&body);
        assert_eq!(converted.messages.len(), 3);
        let rebuilt = from_core(&converted.messages, &converted.sidecar);
        assert_eq!(Value::Array(rebuilt), body["messages"]);
    }

    #[test]
    fn tool_calls_and_results_keep_their_ids() {
        let body = body(json!([
            {"role": "user", "content": "check the weather"},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "weather", "arguments": "{\"city\":\"Paris\"}"}}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}
        ]));
        let converted = to_core(&body);
        let rebuilt = from_core(&converted.messages, &converted.sidecar);
        assert_eq!(Value::Array(rebuilt), body["messages"]);
        assert_eq!(converted.messages.len(), 3);
    }

    #[test]
    fn reasoning_content_and_multimodal_content_survive() {
        let body = body(json!([
            {"role": "user", "content": [
                {"type": "text", "text": "what is this"},
                {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}
            ]},
            {"role": "assistant", "content": "a cat", "reasoning_content": "looks furry"}
        ]));
        let converted = to_core(&body);
        assert_eq!(converted.messages.len(), 4);
        let rebuilt = from_core(&converted.messages, &converted.sidecar);
        assert_eq!(Value::Array(rebuilt), body["messages"]);
    }

    #[test]
    fn tool_injection_uses_openai_function_shape() {
        let mut request = body(json!([{"role": "user", "content": "hi"}]));
        let defs = vec![json!({
            "name": "compress",
            "description": "fold context",
            "input_schema": {"type": "object", "properties": {"content": {"type": "string"}}}
        })];
        inject_tools(&mut request, &defs);
        let tools = request["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], "compress");
        assert_eq!(tools[0]["function"]["parameters"]["type"], "object");
    }
}

#[cfg(test)]
mod inject_tests {
    use super::*;

    #[test]
    fn standing_instructions_merge_into_head_system() {
        let mut body = json!({"messages":[
            {"role":"system","content":"Be terse."},
            {"role":"user","content":"hi"}
        ]});
        inject_standing_instructions(&mut body);
        let text = body["messages"][0]["content"].as_str().unwrap();
        assert!(text.starts_with("Be terse."));
        assert!(text.contains("## Context compression (ACP)"));
        assert_eq!(body["messages"].as_array().unwrap().len(), 2);
        inject_standing_instructions(&mut body);
        assert_eq!(
            body["messages"][0]["content"]
                .as_str()
                .unwrap()
                .matches("## Context compression (ACP)")
                .count(),
            1
        );
    }

    #[test]
    fn standing_instructions_prepend_when_no_system() {
        let mut body = json!({"messages":[{"role":"user","content":"hi"}]});
        inject_standing_instructions(&mut body);
        assert_eq!(body["messages"][0]["role"], "system");
        assert!(
            body["messages"][0]["content"]
                .as_str()
                .unwrap()
                .contains("ACP")
        );
    }
}

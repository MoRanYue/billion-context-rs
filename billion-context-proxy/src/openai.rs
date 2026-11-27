//! OpenAI Chat Completions wire <-> in-memory kernel messages.
//!
//! OpenAI messages are one wire message per turn, but the kernel owns one
//! content block per message.  A single assistant turn can therefore split into
//! text, reasoning and one `tool_call` per function, while a user turn's
//! multimodal array collapses to one text block.  A sidecar keyed by slot
//! (`s0`, `s1`, …) remembers the original wire JSON so an untouched turn is
//! re-emitted byte-for-byte; only a slot a fold actually rewrote is rebuilt.

use std::collections::HashMap;
use std::sync::Arc;

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
    pub tool_names: HashMap<Option<String>, Option<Arc<str>>>,
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

/// JavaScript `String(value)` for the scalar forms a text part can carry.
/// Node's join stringifies the `text` value, so JSON `5`/`true` become
/// `"5"`/`"true"`, while `null` and a missing field become `""`.
/// `String(number)` formatting for tolerant scalar `text` values; see the
/// Anthropic side for the exact JavaScript thresholds.
fn js_number(number: &serde_json::Number) -> String {
    // JavaScript stores every JSON number as an IEEE-754 double, so a 64-bit
    // integer literal beyond 2^53 is rounded before `String(...)` runs.
    let Some(value) = number.as_f64() else {
        return number.to_string();
    };
    if value == 0.0 {
        return "0".to_string();
    }
    let magnitude = value.abs();
    if (1e-6..1e21).contains(&magnitude) {
        if value.fract() == 0.0 {
            return format!("{value:.0}");
        }
        return value.to_string();
    }
    let rendered = format!("{value:e}");
    match rendered.split_once('e') {
        Some((mantissa, exponent)) if !exponent.starts_with('-') => {
            format!("{mantissa}e+{exponent}")
        }
        _ => rendered,
    }
}

fn js_text(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(text)) => text.clone(),
        Some(Value::Number(number)) => js_number(number),
        Some(Value::Bool(flag)) => flag.to_string(),
        Some(Value::Array(items)) => items
            .iter()
            .map(|item| js_text(Some(item)))
            .collect::<Vec<_>>()
            .join(","),
        Some(Value::Object(_)) => "[object Object]".to_string(),
    }
}

fn text_of(content: &Value) -> String {
    match content {
        Value::String(text) => text.clone(),
        Value::Array(parts) => parts
            .iter()
            .map(|part| match part {
                Value::String(text) => text.clone(),
                _ if part.get("type").and_then(Value::as_str) == Some("text") => {
                    js_text(part.get("text"))
                }
                _ => String::new(),
            })
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
            format!("c:{}|{}|{arguments}", name, call_id)
        }
        KContent::ToolResult {
            name,
            call_id,
            content,
        } => {
            format!("u:{}|{}|{content}", name, call_id)
        }
    }
}

fn arguments_string(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => serde_json::to_string(other).unwrap_or_else(|_| "{}".into()),
    }
}

/// Convert an OpenAI Chat Completions request body into kernel messages.
pub fn to_core(body: &Value) -> Converted {
    let messages = body
        .get("messages")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    // Node learns tool names strictly in wire order: `toolNames.set` runs
    // while the assistant `tool_calls` are emitted, and a `tool` message can
    // only read names already seen at its position. Never pre-scan future
    // calls.
    let mut tool_names: HashMap<Option<String>, Option<Arc<str>>> = HashMap::new();
    let mut cluster = ClusterCounter::default();
    let mut sidecar = Sidecar::default();
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
            // Node accepts a tool message's own `name` when it is non-empty,
            // otherwise falls back to the call's name; both falsy mean absent.
            let direct = message
                .get("name")
                .and_then(Value::as_str)
                .filter(|name| !name.is_empty())
                .map(Arc::from);
            let mapped = tool_names
                .get(&Some(call_id.clone()))
                .cloned()
                .flatten()
                .filter(|name| !name.is_empty());
            let name = direct.or(mapped);
            let text = text_of(message.get("content").unwrap_or(&Value::Null));
            let id_base = derive_id(
                "tool",
                "tool-result",
                name.as_deref().unwrap_or(""),
                &text,
                None,
            );
            let mut core =
                KMessage::tool_result(id_base, name.as_deref().unwrap_or(""), call_id, text);
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
                        KMessage::reasoning(id, Role::Assistant, reasoning),
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
                    let call_id = call.get("id").and_then(Value::as_str).map(str::to_string);
                    let function = call.get("function").cloned().unwrap_or(Value::Null);
                    let name = function.get("name").and_then(Value::as_str);
                    let arguments =
                        arguments_string(function.get("arguments").unwrap_or(&json!({})));
                    let id = derive_id(
                        "assistant",
                        "tool-call",
                        name.unwrap_or(""),
                        &arguments,
                        None,
                    );
                    push_core(
                        KMessage::tool_call(
                            id,
                            name.unwrap_or(""),
                            call_id.clone().unwrap_or_default(),
                            arguments,
                        ),
                        &mut cluster,
                        &mut out,
                        &mut sidecar,
                        &slot,
                        &mut ids,
                    );
                    tool_names.insert(call_id, name.map(Arc::from));
                }
            }
        }

        if !ids.is_empty() {
            sidecar.slot_ids.insert(slot.clone(), ids);
            sidecar.raw_slots.insert(slot, message.clone());
        }
    }

    // The sidecar copy is diagnostic only (no replay path reads it); keep it
    // as the final wire-order table for parity with the previous field.
    sidecar.tool_names = tool_names;
    Converted {
        messages: out,
        sidecar,
    }
}

fn core_text(message: &KMessage) -> String {
    message.content.text().unwrap_or("").to_string()
}

/// Text plus a tool result's body. Tool slots must not use `core_text`: the
/// kernel deliberately keeps `text()` prose-only, so a pruned `ToolResult`
/// would otherwise rebuild as an empty string and lose the artifact stub.
fn core_body_text(message: &KMessage) -> String {
    message.content.body().unwrap_or("").to_string()
}

fn rebuild_slot(raw: &Value, cores: &[KMessage], sidecar: &Sidecar) -> Value {
    let role = raw.get("role").and_then(Value::as_str).unwrap_or("user");
    if role == "tool" {
        let call_id = raw.get("tool_call_id").cloned().unwrap_or(json!(""));
        let text: Vec<String> = cores.iter().map(core_body_text).collect();
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
                // Node omits `id` when the wire call had none; null is not the
                // same wire shape. Empty stays an explicit empty string.
                let mut call = json!({
                    "type": "function",
                    // Runtime default only: wire requires a name, identity
                    // keeps `None` as null.
                    "function": {"name": if name.is_empty() { "unknown" } else { name }, "arguments": arguments},
                });
                if !call_id.is_empty() {
                    call["id"] = json!(call_id);
                }
                tool_calls.push(call);
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
    // A tool round cannot be interrupted by a fresh `user` message, so merge
    // into the pending user content instead of emitting a second consecutive
    // user turn.
    if let Some(message) = body
        .get_mut("messages")
        .and_then(Value::as_array_mut)
        .and_then(|messages| messages.last_mut())
    {
        if message.get("role").and_then(Value::as_str) == Some("user") {
            if let Some(Value::String(text)) = message.get_mut("content") {
                text.push_str("\n\n");
                text.push_str(note);
                return;
            }
            if let Some(Value::Array(parts)) = message.get_mut("content") {
                parts.push(json!({"type": "text", "text": note}));
                return;
            }
            if message.get("content").is_none()
                || message.get("content").is_some_and(Value::is_null)
            {
                message["content"] = Value::String(note.to_string());
                return;
            }
        }
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

fn estimate_message_tokens_inner(message: &Value) -> usize {
    let mut total = 0usize;
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
    total
}

/// Conversation payload only; system-role messages are fixed overhead.
pub fn estimate_message_tokens(body: &Value) -> usize {
    body.get("messages")
        .and_then(Value::as_array)
        .map(|messages| {
            messages
                .iter()
                .filter(|message| message.get("role").and_then(Value::as_str) != Some("system"))
                .map(estimate_message_tokens_inner)
                .sum()
        })
        .unwrap_or(0)
}

/// System-role messages + tool definitions. Kept separate from the raw
/// conversation estimate so a fixed preamble cannot masquerade as growth.
pub fn estimate_fixed_overhead(body: &Value) -> usize {
    let system: usize = body
        .get("messages")
        .and_then(Value::as_array)
        .map(|messages| {
            messages
                .iter()
                .filter(|message| message.get("role").and_then(Value::as_str) == Some("system"))
                .map(estimate_message_tokens_inner)
                .sum()
        })
        .unwrap_or(0);
    let tools = body
        .get("tools")
        .and_then(Value::as_array)
        .and_then(|tools| serde_json::to_string(tools).ok())
        .map(|tools| estimate_tokens(&tools))
        .unwrap_or(0);
    system + tools
}

/// Rough request size in the proxy's own estimator: message text, reasoning and
/// tool arguments.  Media is charged a flat 1000 tokens.
#[allow(
    dead_code,
    reason = "kept for callers that need the whole-wire estimate"
)]
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
    use crate::artifact::ArtifactStore;
    use crate::prune::{PrunePolicy, apply_existing_stubs, prune_tool_results};
    use std::collections::HashSet;
    use std::path::PathBuf;

    /// C2-B: upstream's kernel identity cannot represent a missing name, so an
    /// absent `function.name`/result name normalizes to `""`; a forced rebuild
    /// (fold or rewrite) needs a wire name and uses `"unknown"` at runtime.
    #[test]
    fn absent_tool_name_is_empty_and_rebuild_uses_unknown() {
        let body = body(json!([
            {"role":"assistant","content":null,"tool_calls":[
                {"id":"c1","type":"function","function":{"arguments":"{\"q\":1}"}}
            ]},
            {"role":"tool","tool_call_id":"c1","content":"out"}
        ]));
        let converted = to_core(&body);
        let call = converted
            .messages
            .iter()
            .find(|message| matches!(message.content, KContent::ToolCall { .. }))
            .expect("tool call");
        assert_eq!(call.content.tool_name(), Some(""));
        let result = converted
            .messages
            .iter()
            .find(|message| matches!(message.content, KContent::ToolResult { .. }))
            .expect("tool result");
        assert_eq!(result.content.tool_name(), Some(""));

        // Change the call so the raw slot can no longer be replayed verbatim,
        // forcing the rebuild path that needs a runtime name.
        let mut messages = converted.messages.clone();
        let call_index = messages
            .iter()
            .position(|message| matches!(message.content, KContent::ToolCall { .. }))
            .unwrap();
        if let KContent::ToolCall { arguments, .. } = &mut messages[call_index].content {
            *arguments = "{\"q\":2}".into();
        }
        let rebuilt = from_core(&messages, &converted.sidecar);
        let changed = rebuilt
            .iter()
            .flat_map(|message| {
                message["tool_calls"]
                    .as_array()
                    .cloned()
                    .unwrap_or_default()
            })
            .find(|call| call["id"] == "c1")
            .expect("tool call");
        assert_eq!(changed["function"]["name"], "unknown");
    }

    /// An explicit empty name is distinct from the missing state.
    #[test]
    fn empty_tool_name_stays_empty() {
        let body = body(json!([
            {"role":"assistant","content":null,"tool_calls":[
                {"id":"c1","type":"function","function":{"name":"","arguments":"{}"}}
            ]},
            {"role":"tool","tool_call_id":"c1","content":"out"}
        ]));
        let converted = to_core(&body);
        let call = converted
            .messages
            .iter()
            .find(|message| matches!(message.content, KContent::ToolCall { .. }))
            .expect("tool call");
        assert_eq!(call.content.tool_name(), Some(""));
    }

    fn tool_names_in_order(converted: &Converted) -> Vec<Option<&str>> {
        converted
            .messages
            .iter()
            .filter_map(|message| match &message.content {
                KContent::ToolCall { name, .. } | KContent::ToolResult { name, .. } => {
                    Some(Some(&**name))
                }
                _ => None,
            })
            .collect()
    }

    /// Node learns names in wire order: a tool result before its call sees no
    /// name, and reloading the rebuilt wire keeps that order.
    #[test]
    fn result_before_call_does_not_see_later_name() {
        let body = body(json!([
            {"role":"tool","tool_call_id":"c1","content":"out"},
            {"role":"assistant","content":null,"tool_calls":[
                {"id":"c1","type":"function","function":{"name":"Read","arguments":"{}"}}
            ]}
        ]));
        let converted = to_core(&body);
        assert_eq!(
            tool_names_in_order(&converted),
            vec![Some(""), Some("Read")]
        );

        let rebuilt = from_core(&converted.messages, &converted.sidecar);
        let round = to_core(&body_with_messages(Value::Array(rebuilt)));
        assert_eq!(tool_names_in_order(&round), vec![Some(""), Some("Read")]);
    }

    /// A result after its call sees the name learned earlier in the stream.
    #[test]
    fn result_after_call_learns_name() {
        let body = body(json!([
            {"role":"assistant","content":null,"tool_calls":[
                {"id":"c1","type":"function","function":{"name":"Read","arguments":"{}"}}
            ]},
            {"role":"tool","tool_call_id":"c1","content":"out"}
        ]));
        let converted = to_core(&body);
        assert_eq!(
            tool_names_in_order(&converted),
            vec![Some("Read"), Some("Read")]
        );
    }

    /// A duplicate call id overwrites the table; each result sees the latest
    /// call emitted before it, not the last call in the whole request.
    #[test]
    fn duplicate_call_id_uses_latest_preceding_name() {
        let body = body(json!([
            {"role":"assistant","content":null,"tool_calls":[
                {"id":"c1","type":"function","function":{"name":"First","arguments":"{}"}}
            ]},
            {"role":"tool","tool_call_id":"c1","content":"one"},
            {"role":"assistant","content":null,"tool_calls":[
                {"id":"c1","type":"function","function":{"name":"Second","arguments":"{}"}}
            ]},
            {"role":"tool","tool_call_id":"c1","content":"two"}
        ]));
        let converted = to_core(&body);
        assert_eq!(
            tool_names_in_order(&converted),
            vec![Some("First"), Some("First"), Some("Second"), Some("Second")]
        );
    }

    /// OpenAI prefers a non-empty message-level `name`; an empty message name
    /// falls back to the wire-order table, and an unknown id stays absent.
    #[test]
    fn direct_name_wins_and_empty_direct_name_falls_back() {
        let body = body(json!([
            {"role":"assistant","content":null,"tool_calls":[
                {"id":"c1","type":"function","function":{"name":"Mapped","arguments":"{}"}}
            ]},
            {"role":"tool","tool_call_id":"c1","name":"Direct","content":"a"},
            {"role":"assistant","content":null,"tool_calls":[
                {"id":"c2","type":"function","function":{"name":"MappedTwo","arguments":"{}"}}
            ]},
            {"role":"tool","tool_call_id":"c2","name":"","content":"b"},
            {"role":"tool","tool_call_id":"missing","content":"c"}
        ]));
        let converted = to_core(&body);
        assert_eq!(
            tool_names_in_order(&converted),
            vec![
                Some("Mapped"),
                Some("Direct"),
                Some("MappedTwo"),
                Some("MappedTwo"),
                Some("")
            ]
        );
    }

    /// OpenAI normalizes a tool message's missing/null call id to `""`, while
    /// an assistant call keeps absence as `None`. A missing call can therefore
    /// never be hit by a result's `""` key; only an explicit empty call can.
    #[test]
    fn missing_call_id_does_not_capture_empty_tool_result() {
        let body = body(json!([
            {"role":"assistant","content":null,"tool_calls":[
                {"type":"function","function":{"name":"Missing","arguments":"{}"}}
            ]},
            {"role":"tool","content":"a"},
            {"role":"assistant","content":null,"tool_calls":[
                {"id":"","type":"function","function":{"name":"Empty","arguments":"{}"}}
            ]},
            {"role":"tool","tool_call_id":"","content":"b"},
            {"role":"assistant","content":null,"tool_calls":[
                {"type":"function","function":{"name":"Later","arguments":"{}"}}
            ]},
            {"role":"tool","tool_call_id":"","content":"c"}
        ]));
        let converted = to_core(&body);
        assert_eq!(
            tool_names_in_order(&converted),
            vec![
                Some("Missing"),
                Some(""),
                Some("Empty"),
                Some("Empty"),
                Some("Later"),
                Some("Empty")
            ]
        );

        let call_ids: Vec<Option<&str>> = converted
            .messages
            .iter()
            .filter_map(|message| match &message.content {
                KContent::ToolCall { call_id, .. } | KContent::ToolResult { call_id, .. } => {
                    Some(Some(&**call_id))
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            call_ids,
            vec![Some(""), Some(""), Some(""), Some(""), Some(""), Some(""),]
        );

        // Persistence preserves absence and empty distinctly.
        for message in &converted.messages {
            let round: KMessage =
                serde_json::from_value(serde_json::to_value(message).unwrap()).unwrap();
            match (&message.content, &round.content) {
                (
                    KContent::ToolCall {
                        call_id: before, ..
                    },
                    KContent::ToolCall { call_id: after, .. },
                )
                | (
                    KContent::ToolResult {
                        call_id: before, ..
                    },
                    KContent::ToolResult { call_id: after, .. },
                ) => assert_eq!(before, after),
                _ => {}
            }
        }

        // The rebuilt assistant call omits an absent id; empty stays empty.
        let rebuilt = from_core(&converted.messages, &converted.sidecar);
        let wire_ids: Vec<Option<&Value>> = rebuilt
            .iter()
            .filter_map(|message| message.get("tool_calls").and_then(Value::as_array))
            .flat_map(|calls| calls.iter())
            .map(|call| call.get("id"))
            .collect();
        assert!(wire_ids[0].is_none());
        assert_eq!(wire_ids[1].unwrap(), "");
        assert!(wire_ids[2].is_none());
    }

    /// Node stringifies an integral float without `.0` and uses a signed
    /// positive exponent for `>= 1e21`.
    #[test]
    fn text_array_float_numbers_use_js_format() {
        let body = body(json!([
            {"role":"tool","tool_call_id":"c1","content":[
                {"type":"text","text":5.0},
                {"type":"text","text":1e-7},
                {"type":"text","text":1e21},
                {"type":"text","text":9007199254740993i64}
            ]}
        ]));
        let converted = to_core(&body);
        let result = converted
            .messages
            .iter()
            .find(|message| matches!(message.content, KContent::ToolResult { .. }))
            .expect("tool result");
        let text = match &result.content {
            KContent::ToolResult { content, .. } => content.to_string(),
            _ => unreachable!(),
        };
        assert_eq!(text, "5\n1e-7\n1e+21\n9007199254740992");
    }

    /// C-1: a non-text part keeps its slot, so a trailing image leaves the
    /// Node-visible newline after `"x"`.
    #[test]
    fn tool_result_text_preserves_non_text_slots() {
        let body = body(json!([
            {"role":"tool","tool_call_id":"c1","content":[
                {"type":"text","text":"x"},
                {"type":"image_url","image_url":{"url":"https://example.invalid/i.png"}}
            ]}
        ]));
        let converted = to_core(&body);
        let result = converted
            .messages
            .iter()
            .find(|message| matches!(message.content, KContent::ToolResult { .. }))
            .expect("tool result");
        let text = match &result.content {
            KContent::ToolResult { content, .. } => content.to_string(),
            _ => unreachable!(),
        };
        assert_eq!(text, "x\n");
    }

    /// Node's `stringContent` keeps string parts, stringifies scalar `text`
    /// values, and emits an empty slot for every other part.
    #[test]
    fn text_array_preserves_string_parts_and_scalar_text() {
        let body = body(json!([
            {"role":"tool","tool_call_id":"c1","content":[
                "a",
                {"type":"text","text":"b"},
                {"type":"image_url","image_url":{"url":"https://example.invalid/i.png"}},
                {"type":"text","text":5},
                {"type":"text","text":true},
                {"type":"text"}
            ]}
        ]));
        let converted = to_core(&body);
        let result = converted
            .messages
            .iter()
            .find(|message| matches!(message.content, KContent::ToolResult { .. }))
            .expect("tool result");
        let text = match &result.content {
            KContent::ToolResult { content, .. } => content.to_string(),
            _ => unreachable!(),
        };
        assert_eq!(text, "a\nb\n\n5\ntrue\n");
    }

    fn body_with_messages(messages: Value) -> Value {
        json!({"model": "gpt", "messages": messages})
    }

    fn body(messages: Value) -> Value {
        json!({"model": "gpt", "messages": messages})
    }

    fn temp_store(tag: &str) -> (ArtifactStore, PathBuf) {
        let dir = std::env::temp_dir().join(format!(
            "bili-openai-{tag}-{}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            std::process::id()
        ));
        (ArtifactStore::new(&dir), dir)
    }

    fn stub_artifact_id(stub: &str) -> String {
        stub.strip_prefix("[artifact ")
            .and_then(|rest| rest.split_whitespace().next())
            .expect("artifact stub")
            .to_string()
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
    fn estimates_text_tool_calls_and_tool_results() {
        let text = body(json!([{"role":"user","content":"hello world"}]));
        assert!(estimate_body_tokens(&text) > 0);

        let tool_result = body(json!([
            {"role":"user","content":"read"},
            {"role":"assistant","content":null,"tool_calls":[
                {"id":"c1","type":"function","function":{"name":"Read","arguments":"{\"path\":\"src/main.rs\"}"}}
            ]},
            {"role":"tool","tool_call_id":"c1","content":"TOOL-RESULT-BODY"}
        ]));
        assert!(estimate_body_tokens(&tool_result) > 0);
        assert!(estimate_message_tokens(&tool_result) > 0);

        let empty = body(json!([{"role":"user","content":""}]));
        assert_eq!(estimate_body_tokens(&empty), 0);
    }

    #[test]
    fn fixed_overhead_is_separate_from_conversation_growth() {
        let request = json!({
            "messages":[
                {"role":"system","content":"fixed system prompt"},
                {"role":"user","content":"hello"}
            ],
            "tools":[{"type":"function","function":{"name":"Read","description":"read","parameters":{}}}]
        });
        assert!(estimate_fixed_overhead(&request) > 0);
        assert!(estimate_message_tokens(&request) > 0);
        assert_eq!(
            estimate_body_tokens(&request),
            estimate_message_tokens(&request) + estimate_fixed_overhead(&request)
        );
    }

    #[test]
    fn fixed_openai_wire_fixture_preserves_tool_result() {
        let fixture: Value =
            serde_json::from_str(include_str!("../tests/fixtures/openai-tool-result.json"))
                .expect("fixture json");
        let converted = to_core(&fixture);
        let rebuilt = from_core(&converted.messages, &converted.sidecar);
        let rebuilt_body = json!({"messages": rebuilt, "tools": fixture["tools"]});
        assert!(rebuilt_body.to_string().contains("FIXTURE-MID"));
        assert!(estimate_body_tokens(&rebuilt_body) > 0);
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

    #[test]
    fn pruned_tool_result_rebuilds_artifact_stub_and_allows_recovery() {
        let (store, dir) = temp_store("stub");
        let raw = format!(
            "{}OPENAI-MIDDLE-SECRET{}",
            "OPENAI-HEAD ".repeat(2_000),
            "OPENAI-TAIL ".repeat(2_000)
        );
        let request = body(json!([
            {"role": "user", "content": "read"},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "Read", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": raw.clone()}
        ]));
        let mut converted = to_core(&request);
        let policy = PrunePolicy {
            enabled: true,
            min_tokens: 1,
            ..PrunePolicy::default()
        };
        let report = prune_tool_results(
            &mut converted.messages,
            20_000,
            20_000,
            &HashSet::new(),
            0,
            0,
            Some(&store),
            &policy,
            "openai-test",
        );
        assert_eq!(report.pruned_items, 1);

        let rebuilt = from_core(&converted.messages, &converted.sidecar);
        let tool = rebuilt
            .iter()
            .find(|message| message.get("role") == Some(&json!("tool")))
            .expect("tool message");
        let stub = tool["content"].as_str().expect("tool content");
        assert!(stub.starts_with("[artifact "), "{stub}");
        let artifact_id = stub_artifact_id(stub);
        assert!(stub.contains(&artifact_id));
        assert!(
            !serde_json::to_string(&rebuilt)
                .unwrap()
                .contains("OPENAI-MIDDLE-SECRET")
        );

        let recovered = store.read(&artifact_id, 0, raw.len() as u64).unwrap();
        assert_eq!(recovered.text, raw);
        assert_eq!(recovered.byte_len, raw.len() as u64);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn raw_resend_reproduces_stub_instead_of_leaking_body() {
        let (store, dir) = temp_store("resend");
        let raw = format!(
            "{}OPENAI-RESEND-MIDDLE-SECRET{}",
            "OPENAI-RESEND-HEAD ".repeat(2_000),
            "OPENAI-RESEND-TAIL ".repeat(2_000)
        );
        let request = body(json!([
            {"role": "user", "content": "read"},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "Read", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": raw.clone()}
        ]));
        let policy = PrunePolicy {
            enabled: true,
            min_tokens: 1,
            ..PrunePolicy::default()
        };
        let mut first = to_core(&request);
        assert_eq!(
            prune_tool_results(
                &mut first.messages,
                20_000,
                20_000,
                &HashSet::new(),
                0,
                0,
                Some(&store),
                &policy,
                "openai-test"
            )
            .pruned_items,
            1
        );
        let first_wire = from_core(&first.messages, &first.sidecar);
        let stub = first_wire
            .iter()
            .find(|message| message.get("role") == Some(&json!("tool")))
            .and_then(|message| message.get("content"))
            .and_then(Value::as_str)
            .expect("first stub")
            .to_string();

        let mut resend = to_core(&request);
        assert_eq!(
            apply_existing_stubs(&mut resend.messages, &store, &policy),
            1
        );
        let resend_wire = from_core(&resend.messages, &resend.sidecar);
        let resend_stub = resend_wire
            .iter()
            .find(|message| message.get("role") == Some(&json!("tool")))
            .and_then(|message| message.get("content"))
            .and_then(Value::as_str)
            .expect("resend stub");
        assert_eq!(resend_stub, stub);
        assert!(
            !serde_json::to_string(&resend_wire)
                .unwrap()
                .contains("OPENAI-RESEND-MIDDLE-SECRET")
        );
        let _ = std::fs::remove_dir_all(dir);
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

    #[test]
    fn nudge_merges_into_last_user_turn() {
        let mut body = json!({"messages":[
            {"role":"assistant","content":null,"tool_calls":[
                {"id":"call_1","type":"function","function":{"name":"Bash","arguments":"{}"}}
            ]},
            {"role":"tool","tool_call_id":"call_1","content":"ok"},
            {"role":"user","content":"next"}
        ]});
        inject_user_note(&mut body, "compress now");
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[2]["role"], "user");
        assert_eq!(
            messages[2]["content"].as_str().unwrap(),
            "next\n\ncompress now"
        );
    }

    #[test]
    fn nudge_appends_text_block_to_multimodal_user_turn() {
        let mut body = json!({"messages":[
            {"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/a.png"}}]}
        ]});
        inject_user_note(&mut body, "compress now");
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["content"][1]["text"], "compress now");
    }
}

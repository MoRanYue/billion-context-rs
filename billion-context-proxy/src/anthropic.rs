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

/// JavaScript `String(value)` for the scalar forms a text block can carry.
/// Node's tool-result join stringifies the `text` value, so JSON `5`/`true`
/// become `"5"`/`"true"`, while `null` and a missing field become `""`.
/// `String(number)` formatting for the tolerant scalar `text` values: JSON has
/// no `NaN`/`Infinity`, and JavaScript prints integral doubles without `.0`,
/// switches to exponent form below `1e-6` and at/above `1e21`, and writes a
/// signed positive exponent (`1e+21`).
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

/// Node's tool-result text projection keeps one slot per content block:
/// non-text blocks and absent/null text contribute an empty slot, then all
/// slots are joined with `"\n"`. System text has different Node semantics
/// (`[<type>]` markers), so it continues to use `block_text`.
fn result_text(content: &Value) -> String {
    match content {
        Value::String(text) => text.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .map(|block| {
                if block.get("type").and_then(Value::as_str) == Some("text") {
                    js_text(block.get("text"))
                } else {
                    String::new()
                }
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

/// Why an Anthropic body cannot be converted. The message mirrors the
/// JavaScript `TypeError` the Node oracle throws on the same malformed wire, so
/// probe and test output can compare the failure directly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvertError {
    message: String,
}

impl ConvertError {
    fn new(message: &str) -> Self {
        Self {
            message: message.to_string(),
        }
    }
}

impl std::fmt::Display for ConvertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ConvertError {}

/// Node's `anthropicToCore` does `typeof content === "string" ? ... :
/// content.map(...)`, so a content field that is neither a string nor an array
/// throws before any message is produced. Reproduce that rejection instead of
/// turning it into an empty message.
fn validate_content_shape(content: Option<&Value>) -> Result<(), ConvertError> {
    match content {
        Some(Value::String(_)) => Ok(()),
        Some(Value::Array(blocks)) => {
            if blocks.iter().any(Value::is_null) {
                Err(ConvertError::new(
                    "TypeError: Cannot read properties of null (reading 'type')",
                ))
            } else {
                Ok(())
            }
        }
        _ => Err(ConvertError::new("TypeError: blocks is not iterable")),
    }
}

/// The tool-result branch calls `.map` on `content` unless it is a string. A
/// missing, null, scalar or object value throws exactly like Node; an array
/// with a null element fails earlier at the element's `.type` read.
fn validate_tool_result_content(content: Option<&Value>) -> Result<(), ConvertError> {
    match content {
        Some(Value::String(_)) => Ok(()),
        Some(Value::Array(blocks)) => {
            if blocks.iter().any(Value::is_null) {
                Err(ConvertError::new(
                    "TypeError: Cannot read properties of null (reading 'type')",
                ))
            } else {
                Ok(())
            }
        }
        Some(Value::Null) => Err(ConvertError::new(
            "TypeError: Cannot read properties of null (reading 'map')",
        )),
        Some(_) => Err(ConvertError::new(
            "TypeError: b2.content.map is not a function",
        )),
        None => Err(ConvertError::new(
            "TypeError: Cannot read properties of undefined (reading 'map')",
        )),
    }
}

fn validate_messages(body: &Value) -> Result<(), ConvertError> {
    let messages = body
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            ConvertError::new("TypeError: Cannot read properties of undefined (reading 'entries')")
        })?;
    for message in messages {
        let Some(object) = message.as_object() else {
            return Err(ConvertError::new("TypeError: blocks is not iterable"));
        };
        validate_content_shape(object.get("content"))?;
        if let Some(Value::Array(blocks)) = object.get("content") {
            for block in blocks {
                let Some(block) = block.as_object() else {
                    continue;
                };
                if block.get("type").and_then(Value::as_str) == Some("tool_result") {
                    validate_tool_result_content(block.get("content"))?;
                }
            }
        }
    }
    Ok(())
}

/// Convert an Anthropic Messages request body into kernel messages.
///
/// # Errors
/// Returns the Node-equivalent `TypeError` for a content field that is not a
/// string or array, or an array containing `null`.
pub fn to_core(body: &Value) -> Result<Converted, ConvertError> {
    validate_messages(body)?;
    Ok(convert_unchecked(body))
}

/// Conversion for a body already accepted by [`validate_messages`]. Kept
/// separate so the public entry point has one validation point.
fn convert_unchecked(body: &Value) -> Converted {
    let messages = body
        .get("messages")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    // Node learns tool names strictly in wire order: `toolNames.set` runs when
    // a `tool_use` block is emitted, and a `tool_result` can only read names
    // already seen at its position. Never pre-scan future calls.
    let mut names: HashMap<Option<String>, Option<Arc<str>>> = HashMap::new();
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
                    // Missing/null stays `None`; `""` stays `Some("")` for both
                    // name and id. Node's tool-name map keys `undefined`, null
                    // and `""` separately, so an absent id must never be
                    // rewritten to `""`.
                    let name = block.get("name").and_then(Value::as_str);
                    let call_id = block.get("id").and_then(Value::as_str).map(str::to_string);
                    let arguments = stringify_input(block.get("input").unwrap_or(&json!({})));
                    let id = cluster.next(&derive_id(
                        "assistant",
                        "tool-call",
                        name.unwrap_or(""),
                        &arguments,
                        None,
                    ));
                    if let Some(mark) = block.get("cache_control") {
                        sidecar.cache_controls.insert(id.clone(), mark.clone());
                    }
                    out.push(KMessage::tool_call(
                        id,
                        name.unwrap_or(""),
                        call_id.clone().unwrap_or_default(),
                        arguments,
                    ));
                    // Node writes the map entry only after the call block has
                    // been converted; later results in the same wire order see
                    // it, earlier ones do not.
                    names.insert(call_id, name.map(Arc::from));
                }
                "tool_result" => {
                    let call_id = block
                        .get("tool_use_id")
                        .and_then(Value::as_str)
                        .map(str::to_string);
                    // Node treats a falsy name on a result as absent, so an
                    // empty name from the call map becomes `None` here.
                    let name = names
                        .get(&call_id)
                        .cloned()
                        .flatten()
                        .filter(|name| !name.is_empty());
                    let text = result_text(
                        &block
                            .get("content")
                            .cloned()
                            .unwrap_or(Value::String(String::new())),
                    );
                    let id = cluster.next(&derive_id(
                        "tool",
                        "tool-result",
                        name.as_deref().unwrap_or(""),
                        &text,
                        None,
                    ));
                    if let Some(mark) = block.get("cache_control") {
                        sidecar.cache_controls.insert(id.clone(), mark.clone());
                    }
                    if block.get("is_error").and_then(Value::as_bool) == Some(true) {
                        sidecar.tool_errors.insert(id.clone(), true);
                    }
                    if content_has_non_text(&block.get("content").cloned().unwrap_or(Value::Null)) {
                        sidecar.raw_blocks.insert(id.clone(), block.clone());
                    }
                    out.push(KMessage::tool_result(
                        id,
                        name.as_deref().unwrap_or(""),
                        call_id.unwrap_or_default(),
                        text,
                    ));
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
    // `text()` deliberately returns only prose, but a tool result's only body
    // is its content. Using `text()` here would silently send an empty
    // tool_result whenever the sidecar has no raw block (for example a string
    // tool result, which `content_has_non_text` does not flag).
    message.content.body().unwrap_or("").to_string()
}

fn tool_result_block(
    call_id: Option<&str>,
    body: &str,
    sidecar: &Sidecar,
    message: &KMessage,
) -> Value {
    let mut block = json!({
        "type":"tool_result",
        "content": body,
    });
    // Node omits `tool_use_id` when the call/result pair had no id; null is a
    // different wire shape. An explicit empty id is still written as "".
    if let Some(call_id) = call_id {
        block["tool_use_id"] = json!(call_id);
    }
    if sidecar.tool_errors.get(message.id.as_str()) == Some(&true) {
        block["is_error"] = json!(true);
    }
    block
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
            } => {
                // Node omits `id` for an absent call id rather than emitting
                // JSON null; keep that wire distinction.
                let mut block = json!({
                    "type":"tool_use",
                    // Runtime default only: the wire has no way to say "absent"
                    // here, while the kernel identity keeps `None` as null.
                    "name": if name.is_empty() { "unknown" } else { name },
                    "input": serde_json::from_str::<Value>(arguments).unwrap_or_else(|_| json!({})),
                });
                if !call_id.is_empty() {
                    block["id"] = json!(call_id);
                }
                block
            }
            KContent::ToolResult { call_id, .. } => {
                let body = core_text(message);
                // Raw blocks are only safe while the kernel body still equals
                // the original wire text. A pruned tool result has a new stub
                // body; returning the raw block here would put the original
                // payload back on the wire and hide the artifact pointer.
                if let Some(raw) = sidecar.raw_blocks.get(message.id.as_str()) {
                    if result_text(raw.get("content").unwrap_or(&Value::Null)) == body {
                        raw.clone()
                    } else {
                        tool_result_block(
                            Some(&**call_id).filter(|id| !id.is_empty()),
                            &body,
                            sidecar,
                            message,
                        )
                    }
                } else {
                    tool_result_block(
                        Some(&**call_id).filter(|id| !id.is_empty()),
                        &body,
                        sidecar,
                        message,
                    )
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

/// Rough size of one Anthropic content block. Tool results and tool inputs are
/// part of the prompt and must not disappear from the occupancy estimate.
fn estimate_content_tokens(content: &Value) -> usize {
    match content {
        Value::String(text) => estimate_tokens(text),
        Value::Array(blocks) => blocks
            .iter()
            .map(|block| match block.get("type").and_then(Value::as_str) {
                Some("text") | Some("thinking") => block
                    .get("text")
                    .or_else(|| block.get("thinking"))
                    .and_then(Value::as_str)
                    .map(estimate_tokens)
                    .unwrap_or(0),
                Some("tool_result") => match block.get("content") {
                    Some(Value::String(text)) => estimate_tokens(text),
                    Some(Value::Array(parts)) => parts
                        .iter()
                        .filter(|part| part.get("type").and_then(Value::as_str) == Some("text"))
                        .filter_map(|part| part.get("text").and_then(Value::as_str))
                        .map(estimate_tokens)
                        .sum(),
                    _ => 0,
                },
                Some("tool_use") => block
                    .get("input")
                    .and_then(|input| serde_json::to_string(input).ok())
                    .map(|input| estimate_tokens(&input))
                    .unwrap_or(0),
                _ => 0,
            })
            .sum(),
        _ => 0,
    }
}

/// Estimate conversation messages only. System/tools are fixed request
/// overhead and are reported separately by `estimate_fixed_overhead`.
pub fn estimate_message_tokens(body: &Value) -> usize {
    let messages = body.get("messages").and_then(Value::as_array);
    let mut total = 0usize;
    if let Some(messages) = messages {
        for message in messages {
            total += estimate_content_tokens(message.get("content").unwrap_or(&Value::Null));
        }
    }
    total
}

fn estimate_tool_tokens(body: &Value) -> usize {
    body.get("tools")
        .and_then(Value::as_array)
        .and_then(|tools| serde_json::to_string(tools).ok())
        .map(|tools| estimate_tokens(&tools))
        .unwrap_or(0)
}

/// System prompt + tool definitions + any fixed request metadata. This is not
/// conversation content and must not be silently mixed into the raw estimate.
pub fn estimate_fixed_overhead(body: &Value) -> usize {
    let system = estimate_tokens(
        &body
            .get("system")
            .map(|system| match system {
                Value::String(text) => text.clone(),
                Value::Array(blocks) => block_text(&Value::Array(blocks.clone())),
                _ => String::new(),
            })
            .unwrap_or_default(),
    );
    system + estimate_tool_tokens(body)
}

/// Estimate the prompt size of a whole request body, including system and
/// tool definitions, so the returned value matches the wire's effective view.
#[allow(
    dead_code,
    reason = "kept for callers that need the whole-wire estimate"
)]
pub fn estimate_body_tokens(body: &Value) -> usize {
    estimate_message_tokens(body) + estimate_fixed_overhead(body)
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
    // A tool round cannot be interrupted by a fresh `user` message, so merge
    // into the pending user turn (including its tool_result blocks) instead of
    // emitting a second consecutive user message.
    if let Some(message) = body
        .get_mut("messages")
        .and_then(Value::as_array_mut)
        .and_then(|messages| messages.last_mut())
    {
        if message.get("role").and_then(Value::as_str) == Some("user") {
            let content = message.get("content");
            if content.is_none() || content.is_some_and(Value::is_null) {
                message["content"] = json!([{"type": "text", "text": note}]);
                return;
            }
            if let Some(Value::Array(blocks)) = message.get_mut("content") {
                blocks.push(json!({"type": "text", "text": note}));
                return;
            }
            if let Some(Value::String(text)) = message.get_mut("content") {
                text.push_str("\n\n");
                text.push_str(note);
                return;
            }
        }
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
#[allow(
    dead_code,
    reason = "convenience helper for callers that build system text"
)]
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
#[allow(dead_code, reason = "small value helper for tool-definition builders")]
pub fn empty_object() -> Value {
    Value::Object(Map::new())
}

/// Re-exported for tests that build messages directly.
#[allow(
    dead_code,
    reason = "test and integration helper for direct kernel messages"
)]
pub fn text_message(id: &str, role: Role, text: &str) -> KMessage {
    KMessage::new(id, role, KContent::Text(Arc::from(text)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact::ArtifactStore;
    use crate::prune::{PrunePolicy, prune_tool_results};
    use std::collections::HashSet;
    use std::path::PathBuf;

    /// C2-B: upstream's kernel identity cannot represent a missing name, so an
    /// absent `name` normalizes to the empty string, while the wire rebuild
    /// (which needs a non-empty name) uses the runtime `"unknown"` default.
    #[test]
    fn absent_tool_name_is_normalized_and_unknown_on_rebuild() {
        let body = json!({"messages":[
            {"role":"assistant","content":[{"type":"tool_use","id":"t1","input":{"q":1}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"out"}]}
        ]});
        let converted = to_core(&body).expect("valid test fixture");
        let call = converted
            .messages
            .iter()
            .find(|message| matches!(message.content, KContent::ToolCall { .. }))
            .expect("tool call");
        assert_eq!(call.content.tool_name(), Some(""));

        let rebuilt = from_core(&converted.messages, &converted.sidecar);
        let block = rebuilt
            .iter()
            .flat_map(|message| message["content"].as_array().cloned().unwrap_or_default())
            .find(|block| block["type"] == "tool_use")
            .expect("tool_use block");
        assert_eq!(block["name"], "unknown");
    }

    /// An explicit empty name survives the kernel identity as `""`, but a wire
    /// rebuild has to produce a usable tool name and normalizes it to `"unknown"`.
    #[test]
    fn empty_tool_name_becomes_unknown_on_rebuild() {
        let body = json!({"messages":[
            {"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"","input":{}}]}
        ]});
        let converted = to_core(&body).expect("valid test fixture");
        let call = converted
            .messages
            .iter()
            .find(|message| matches!(message.content, KContent::ToolCall { .. }))
            .expect("tool call");
        assert_eq!(call.content.tool_name(), Some(""));

        let rebuilt = from_core(&converted.messages, &converted.sidecar);
        let block = rebuilt
            .iter()
            .flat_map(|message| message["content"].as_array().cloned().unwrap_or_default())
            .find(|block| block["type"] == "tool_use")
            .expect("tool_use block");
        assert_eq!(block["name"], "unknown");
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

    /// Node learns names in wire order: a result before its call sees no name.
    #[test]
    fn result_before_call_does_not_see_later_name() {
        let body = json!({"messages":[
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"out"}]},
            {"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"Read","input":{"path":"x"}}]}
        ]});
        let converted = to_core(&body).expect("valid test fixture");
        assert_eq!(
            tool_names_in_order(&converted),
            vec![Some(""), Some("Read")]
        );

        // Reloading the rebuilt wire must preserve the same order.
        let rebuilt = from_core(&converted.messages, &converted.sidecar);
        let round = to_core(&json!({"messages": rebuilt})).expect("valid rebuilt fixture");
        assert_eq!(tool_names_in_order(&round), vec![Some(""), Some("Read")]);
    }

    /// m1: upstream normalizes an absent call id to `""`; because the empty id
    /// cannot be distinguished from an explicit one, the rebuild must treat both
    /// as the same empty key rather than inventing a name.
    #[test]
    fn missing_call_id_is_not_rewritten_to_empty() {
        let body = json!({"messages":[
            {"role":"assistant","content":[{"type":"tool_use","name":"noid","input":{}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"","content":"r"}]}
        ]});
        let converted = to_core(&body).expect("valid test fixture");
        let (call_id, call_name) = converted
            .messages
            .iter()
            .find_map(|message| match &message.content {
                KContent::ToolCall { call_id, name, .. } => Some((call_id.clone(), name.clone())),
                _ => None,
            })
            .expect("tool call");
        assert!(call_id.is_empty());
        assert_eq!(&*call_name, "noid");

        let (result_id, result_name) = converted
            .messages
            .iter()
            .find_map(|message| match &message.content {
                KContent::ToolResult { call_id, name, .. } => Some((call_id.clone(), name.clone())),
                _ => None,
            })
            .expect("tool result");
        assert!(result_id.is_empty());
        assert!(result_name.is_empty());

        // Persistence normalizes an absent id to the empty string.
        let round: Vec<KMessage> = converted
            .messages
            .iter()
            .map(|message| serde_json::from_value(serde_json::to_value(message).unwrap()).unwrap())
            .collect();
        let persisted_call = match &round[0].content {
            KContent::ToolCall { call_id, .. } => call_id.clone(),
            _ => unreachable!(),
        };
        let persisted_result = match &round[1].content {
            KContent::ToolResult { call_id, .. } => call_id.clone(),
            _ => unreachable!(),
        };
        assert!(persisted_call.is_empty());
        assert!(persisted_result.is_empty());

        // Upstream flattens absence and an explicit empty id into `""`, so the
        // rebuild must not rely on the original distinction.
        let rebuilt = from_core(&converted.messages, &converted.sidecar);
        let blocks: Vec<&Value> = rebuilt
            .iter()
            .flat_map(|message| {
                message["content"]
                    .as_array()
                    .map(Vec::as_slice)
                    .unwrap_or(&[])
            })
            .collect();
        let wire_call = blocks.iter().find(|b| b["type"] == "tool_use").unwrap();
        assert!(wire_call.get("id").is_none());
        let wire_result = blocks.iter().find(|b| b["type"] == "tool_result").unwrap();
        assert!(wire_result.get("tool_use_id").is_none());
    }

    /// A result after its call sees the name learned earlier in the stream.
    #[test]
    fn result_after_call_learns_name() {
        let body = json!({"messages":[
            {"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"Read","input":{}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"out"}]}
        ]});
        let converted = to_core(&body).expect("valid test fixture");
        assert_eq!(
            tool_names_in_order(&converted),
            vec![Some("Read"), Some("Read")]
        );
    }

    /// A duplicate call id overwrites the table; each result sees the latest
    /// call emitted before it, not the last call in the whole request.
    #[test]
    fn duplicate_call_id_uses_latest_preceding_name() {
        let body = json!({"messages":[
            {"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"First","input":{}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"one"}]},
            {"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"Second","input":{}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"two"}]}
        ]});
        let converted = to_core(&body).expect("valid test fixture");
        assert_eq!(
            tool_names_in_order(&converted),
            vec![Some("First"), Some("First"), Some("Second"), Some("Second")]
        );
    }

    /// Empty and absent call names normalize to the same empty identity, and the
    /// result side does not invent a real name for either.
    #[test]
    fn empty_and_absent_call_names_are_not_result_names() {
        let body = json!({"messages":[
            {"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"","input":{}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"out"}]},
            {"role":"assistant","content":[{"type":"tool_use","id":"t2","input":{}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"t2","content":"out"}]}
        ]});
        let converted = to_core(&body).expect("valid test fixture");
        assert_eq!(
            tool_names_in_order(&converted),
            vec![Some(""), Some(""), Some(""), Some("")]
        );
    }

    /// Node stringifies an integral float without `.0` and uses a signed
    /// positive exponent for `>= 1e21`.
    #[test]
    fn tool_result_text_float_numbers_use_js_format() {
        let body = json!({"messages":[{"role":"user","content":[{
        "type":"tool_result","tool_use_id":"t1","content":[
            {"type":"text","text":5.0},
            {"type":"text","text":1e-7},
            {"type":"text","text":1e21},
            {"type":"text","text":9007199254740993i64}
        ]}]}]});
        let converted = to_core(&body).expect("valid test fixture");
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

    /// C-1: Node keeps one slot per tool-result content block, so an image
    /// between two text blocks leaves two newlines. The raw block must also
    /// survive from_core once the projected text matches the corpus body.
    #[test]
    fn tool_result_text_preserves_non_text_slots() {
        let body = json!({"messages":[
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":[
                {"type":"text","text":"a"},
                {"type":"image","source":{"type":"base64","media_type":"image/png","data":"eA=="}},
                {"type":"text","text":"b"}
            ]}]}
        ]});
        let converted = to_core(&body).expect("valid test fixture");
        let result = converted
            .messages
            .iter()
            .find(|message| matches!(message.content, KContent::ToolResult { .. }))
            .expect("tool result");
        let text = match &result.content {
            KContent::ToolResult { content, .. } => content.to_string(),
            _ => unreachable!(),
        };
        assert_eq!(text, "a\n\nb");

        let rebuilt = from_core(&converted.messages, &converted.sidecar);
        let block = rebuilt
            .iter()
            .flat_map(|message| message["content"].as_array().cloned().unwrap_or_default())
            .find(|block| block["type"] == "tool_result")
            .expect("tool_result block");
        assert_eq!(block["content"][1]["type"], "image");
        assert_eq!(block["content"][1]["source"]["data"], "eA==");
    }

    /// The JavaScript join boundary: absent/null text is an empty slot,
    /// numbers and booleans are stringified, and non-text blocks stay empty.
    #[test]
    fn tool_result_text_stringifies_scalar_text_and_keeps_slots() {
        let body = json!({"messages":[
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":[
                {"type":"text"},
                {"type":"text","text":null},
                {"type":"text","text":5},
                {"type":"text","text":true},
                {"type":"image","source":{"type":"base64","media_type":"image/png","data":"eA=="}}
            ]}]}
        ]});
        let converted = to_core(&body).expect("valid test fixture");
        let result = converted
            .messages
            .iter()
            .find(|message| matches!(message.content, KContent::ToolResult { .. }))
            .expect("tool result");
        let text = match &result.content {
            KContent::ToolResult { content, .. } => content.to_string(),
            _ => unreachable!(),
        };
        assert_eq!(text, "\n\n5\ntrue\n");
    }

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

    fn temp_store(tag: &str) -> (ArtifactStore, PathBuf) {
        let dir = std::env::temp_dir().join(format!(
            "bili-anthropic-{tag}-{}-{}",
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
    fn round_trip_keeps_bytes_and_ids() {
        let original = body();
        let converted = to_core(&original).expect("valid test fixture");
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
        let converted = to_core(&original).expect("valid test fixture");
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
        let converted = to_core(&request).expect("valid test fixture");
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
    fn estimates_each_prompt_content_shape() {
        let text =
            json!({"messages":[{"role":"user","content":[{"type":"text","text":"hello world"}]}]});
        assert!(estimate_body_tokens(&text) > 0);

        let tool_result = json!({"messages":[
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"TOOL-RESULT-BODY"}]}
        ]});
        assert!(estimate_body_tokens(&tool_result) > 0);

        let tool_use = json!({"messages":[
            {"role":"assistant","content":[
                {"type":"tool_use","id":"t1","name":"Read","input":{"path":"src/main.rs","nested":{"a":[1,2,3]}}}
            ]}
        ]});
        assert!(estimate_body_tokens(&tool_use) > 0);

        let multiple = json!({"messages":[
            {"role":"user","content":[
                {"type":"tool_result","tool_use_id":"t1","content":"first"},
                {"type":"tool_result","tool_use_id":"t2","content":"second"}
            ]}
        ]});
        let one = estimate_body_tokens(&json!({"messages":[
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"first"}]}
        ]}));
        assert!(estimate_body_tokens(&multiple) > one);

        let mixed = json!({"messages":[
            {"role":"user","content":[
                {"type":"text","text":"please read"},
                {"type":"tool_result","tool_use_id":"t1","content":"mixed body"}
            ]}
        ]});
        assert!(estimate_body_tokens(&mixed) > 0);

        let empty = json!({"messages":[{"role":"user","content":[]}]});
        assert_eq!(estimate_body_tokens(&empty), 0);

        let large_nested = json!({"messages":[
            {"role":"assistant","content":[
                {"type":"tool_use","id":"t1","name":"Write","input":{
                    "payload": {"level1":{"level2":{"level3":["x".repeat(10_000)]}}}
                }}
            ]}
        ]});
        assert!(estimate_body_tokens(&large_nested) > 2_000);
    }

    #[test]
    fn estimate_includes_system_and_tools_but_reports_them_separately() {
        let request = json!({
            "system":[{"type":"text","text":"system instructions"}],
            "tools":[{"name":"Read","description":"read","input_schema":{"type":"object"}}],
            "messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}]
        });
        assert!(estimate_fixed_overhead(&request) > 0);
        assert!(estimate_message_tokens(&request) > 0);
        assert_eq!(
            estimate_body_tokens(&request),
            estimate_message_tokens(&request) + estimate_fixed_overhead(&request)
        );
    }

    #[test]
    fn fixed_wire_fixture_preserves_tool_result_and_estimates_it() {
        let fixture: Value =
            serde_json::from_str(include_str!("../tests/fixtures/anthropic-tool-result.json"))
                .expect("fixture json");
        let converted = to_core(&fixture).expect("valid test fixture");
        let rebuilt = from_core(&converted.messages, &converted.sidecar);
        let rebuilt_body =
            json!({"system": fixture["system"], "tools": fixture["tools"], "messages": rebuilt});
        assert!(rebuilt_body.to_string().contains("FIXTURE-MID"));
        assert!(estimate_body_tokens(&rebuilt_body) > 0);
        assert!(estimate_message_tokens(&rebuilt_body) > 0);
        assert!(estimate_fixed_overhead(&rebuilt_body) > 0);
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

    #[test]
    fn pruned_tool_result_rebuilds_artifact_stub_and_allows_recovery() {
        let (store, dir) = temp_store("stub");
        let raw = format!(
            "{}ANTHROPIC-MIDDLE-SECRET{}",
            "ANTHROPIC-HEAD ".repeat(2_000),
            "ANTHROPIC-TAIL ".repeat(2_000)
        );
        let request = json!({"messages":[
            {"role":"user","content":[{"type":"text","text":"read"}]},
            {"role":"assistant","content":[
                {"type":"tool_use","id":"toolu_1","name":"Read","input":{}}
            ]},
            {"role":"user","content":[
                {"type":"tool_result","tool_use_id":"toolu_1","content": raw.clone()}
            ]}
        ]});
        let mut converted = to_core(&request).expect("valid test fixture");
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
            "anthropic-test",
        );
        assert_eq!(report.pruned_items, 1);

        let rebuilt = from_core(&converted.messages, &converted.sidecar);
        let block = rebuilt
            .iter()
            .filter_map(|message| message.get("content").and_then(Value::as_array))
            .flat_map(|blocks| blocks.iter())
            .find(|block| block.get("type") == Some(&json!("tool_result")))
            .expect("tool_result block");
        let stub = block["content"].as_str().expect("stub string");
        assert!(stub.starts_with("[artifact "), "{stub}");
        let artifact_id = stub_artifact_id(stub);
        assert!(stub.contains(&artifact_id));
        assert!(
            !serde_json::to_string(&rebuilt)
                .unwrap()
                .contains("ANTHROPIC-MIDDLE-SECRET")
        );

        let recovered = store.read(&artifact_id, 0, raw.len() as u64).unwrap();
        assert_eq!(recovered.text, raw);
        assert_eq!(recovered.byte_len, raw.len() as u64);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn changed_tool_result_ignores_stale_raw_block() {
        let request = json!({"messages":[
            {"role":"user","content":[{"type":"text","text":"read"}]},
            {"role":"assistant","content":[
                {"type":"tool_use","id":"toolu_1","name":"Read","input":{}}
            ]},
            {"role":"user","content":[
                {"type":"tool_result","tool_use_id":"toolu_1","content":"STALE-RAW-BODY"}
            ]}
        ]});
        let mut converted = to_core(&request).expect("valid test fixture");
        let index = converted
            .messages
            .iter()
            .position(|message| matches!(message.content, KContent::ToolResult { .. }))
            .expect("tool result core");
        let id = converted.messages[index].id.clone();
        converted.sidecar.raw_blocks.insert(
            id.as_str().to_string(),
            json!({
                "type":"tool_result",
                "tool_use_id":"toolu_1",
                "content":"STALE-RAW-BODY"
            }),
        );
        converted
            .sidecar
            .tool_errors
            .insert(id.as_str().to_string(), true);
        let stub = "[artifact a_deadbeef source m00002]";
        converted.messages[index] = KMessage::tool_result(id.as_str(), "Read", "toolu_1", stub);
        let rebuilt = from_core(&converted.messages, &converted.sidecar);
        let block = rebuilt
            .iter()
            .filter_map(|message| message.get("content").and_then(Value::as_array))
            .flat_map(|blocks| blocks.iter())
            .find(|block| block.get("type") == Some(&json!("tool_result")))
            .expect("tool_result block");
        assert_eq!(block["content"], json!(stub));
        assert_eq!(block["is_error"], json!(true));
        assert!(
            !serde_json::to_string(&rebuilt)
                .unwrap()
                .contains("STALE-RAW-BODY")
        );
    }

    #[test]
    fn tool_result_non_string_content_is_rejected_like_node() {
        let cases = [
            (
                json!({"messages":[{"role":"user","content":[
                    {"type":"tool_result","tool_use_id":"obj_call","content":{"a":1}}
                ]}]}),
                "TypeError: b2.content.map is not a function",
            ),
            (
                json!({"messages":[{"role":"user","content":[
                    {"type":"tool_result","tool_use_id":"null_call","content":null}
                ]}]}),
                "TypeError: Cannot read properties of null (reading 'map')",
            ),
            (
                json!({"messages":[{"role":"user","content":[
                    {"type":"tool_result","tool_use_id":"num_call","content":5}
                ]}]}),
                "TypeError: b2.content.map is not a function",
            ),
            (
                json!({"messages":[{"role":"user","content":[
                    {"type":"tool_result","tool_use_id":"bool_call","content":true}
                ]}]}),
                "TypeError: b2.content.map is not a function",
            ),
        ];
        for (body, expected) in cases {
            let error = match to_core(&body) {
                Ok(_) => panic!("expected a conversion error"),
                Err(error) => error.to_string(),
            };
            assert_eq!(error, expected);
        }
    }

    #[test]
    fn tool_result_missing_or_null_element_content_is_rejected() {
        let missing = json!({"messages":[{"role":"user","content":[
            {"type":"tool_result","tool_use_id":"missing_call"}
        ]}]});
        let error = match to_core(&missing) {
            Ok(_) => panic!("expected a conversion error"),
            Err(error) => error.to_string(),
        };
        assert_eq!(
            error,
            "TypeError: Cannot read properties of undefined (reading 'map')"
        );

        let null_element = json!({"messages":[{"role":"user","content":[
            {"type":"tool_result","tool_use_id":"null_element_call","content":[null]}
        ]}]});
        let error = match to_core(&null_element) {
            Ok(_) => panic!("expected a conversion error"),
            Err(error) => error.to_string(),
        };
        assert_eq!(
            error,
            "TypeError: Cannot read properties of null (reading 'type')"
        );
    }

    #[test]
    fn top_level_content_shape_is_rejected_like_node() {
        let cases = [
            (
                json!({"messages":[{"role":"user"}]}),
                "TypeError: blocks is not iterable",
            ),
            (
                json!({"messages":[{"role":"user","content":null}]}),
                "TypeError: blocks is not iterable",
            ),
            (
                json!({"messages":[{"role":"user","content":5}]}),
                "TypeError: blocks is not iterable",
            ),
            (
                json!({"messages":[{"role":"user","content":[null]}]}),
                "TypeError: Cannot read properties of null (reading 'type')",
            ),
        ];
        for (body, expected) in cases {
            let error = match to_core(&body) {
                Ok(_) => panic!("expected a conversion error"),
                Err(error) => error.to_string(),
            };
            assert_eq!(error, expected);
        }
    }

    #[test]
    fn non_null_primitive_tool_result_blocks_are_still_accepted() {
        for content in [
            json!([5]),
            json!([true]),
            json!([{}]),
            json!([{"type":"image"}]),
        ] {
            let body = json!({"messages":[{"role":"user","content":[
                {"type":"tool_result","tool_use_id":"t1","content":content}
            ]}]});
            let converted = to_core(&body).expect("boundary content is accepted");
            assert_eq!(converted.messages.len(), 1);
            assert_eq!(converted.messages[0].content.body(), Some(""));
        }

        let body = json!({"messages":[{"role":"user","content":[
            {"type":"tool_result","tool_use_id":"t1","content":[
                {"type":"text","text":5}
            ]}
        ]}]});
        let converted = to_core(&body).expect("scalar text is accepted");
        assert_eq!(converted.messages[0].content.body(), Some("5"));
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

    #[test]
    fn nudge_merges_into_pending_tool_result() {
        let mut body = json!({"messages":[
            {"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"Bash","input":{}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"ok"}]}
        ]});
        inject_user_note(&mut body, "compress now");
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2);
        let blocks = messages[1]["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0]["type"], "tool_result");
        assert_eq!(blocks[1]["type"], "text");
        assert_eq!(blocks[1]["text"], "compress now");
    }
}

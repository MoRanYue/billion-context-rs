//! `billion-context-wire` — provider wire types for `billion-context`.
//!
//! The IR in `billion-context` is **not** a wire format and must never become one.
//! This crate holds the smallest provider-shaped types that can be losslessly
//! converted to/from the IR, and nothing else: no HTTP, no streaming, no retry,
//! no tool execution.
//!
//! Three properties are non-negotiable, and every one of them is something a
//! general-purpose SDK tends to lose:
//!
//! 1. **Unknown fields survive.** `#[serde(flatten)] extra` on every variant.
//!    A provider that adds a field must not break the round trip.
//! 2. **Reasoning payloads are opaque.** A thinking block's `signature` is a
//!    signature over its `thinking` text. Reconstructing either invalidates the
//!    whole conversation — so they are carried as `String`, never parsed.
//! 3. **Tool arguments are never re-serialized.** See the `serde_json::Value`
//!    trap documented in `tests`.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

// ---------------------------------------------------------------------------
// Anthropic Messages API
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct CacheControl {
    #[serde(rename = "type")]
    pub kind: String, // "ephemeral"
}

/// `content` is either a bare string or a block array — the two are NOT
/// interchangeable on the wire, so the IR must remember which one it came from
/// or the next request changes shape (and busts the prefix cache).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Blocks(Vec<Block>),
}

#[derive(Clone, Debug)]
pub enum Block {
    Text(TextBlock),
    ToolUse(ToolUseBlock),
    ToolResult(ToolResultBlock),
    Thinking(ThinkingBlock),
    RedactedThinking(RedactedThinkingBlock),
    /// A block this crate does not model — an image, a document, a future
    /// provider type — kept with every field it arrived with.
    ///
    /// A unit variant with `#[serde(other)]` was wrong here: it accepted the
    /// tag and dropped every field, so `{"type":"image","source":{…}}` came back
    /// as `{"type":"unknown"}`. For text the loss is an inconvenience; for media
    /// it is the payload itself, and there is no `decompress` behind it.
    ///
    /// The fields are a `Map`, so an unknown block's key order is not preserved
    /// (the map is sorted). `RawValue` would keep the bytes but cannot be used:
    /// `Content` is `#[serde(untagged)]`, and trying each variant buffers the
    /// value through serde's own `Content`, which has no raw-value support.
    Unknown(Map<String, Value>),
}

impl Block {
    /// True when this block carries bytes the caller cannot hand back after
    /// folding it into a text summary.
    ///
    /// Inline `base64` and `data:` payloads are **recoverable** — the bytes are
    /// in the message, so a host that spilled them to disk can restore them.
    /// A remote URL, a provider file id, or a document is not: the request only
    /// holds a reference, and replacing the block with prose destroys the only
    /// pointer to it.
    ///
    /// Feed the answer into `billion_context::Message::with_unrecoverable_payload`
    /// so no recommended range spans the message.
    ///
    /// ```
    /// use billion_context_wire::Block;
    ///
    /// let inline: Block = serde_json::from_str(
    ///     r#"{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AA=="}}"#,
    /// ).unwrap();
    /// assert!(!inline.has_unrecoverable_payload());
    ///
    /// let remote: Block = serde_json::from_str(
    ///     r#"{"type":"image","source":{"type":"url","url":"https://example.com/a.png"}}"#,
    /// ).unwrap();
    /// assert!(remote.has_unrecoverable_payload());
    /// ```
    pub fn has_unrecoverable_payload(&self) -> bool {
        match self {
            // Modelled text-bearing blocks are exactly what a fold is allowed
            // to rewrite.
            Block::Text(_) | Block::ToolUse(_) | Block::Thinking(_) | Block::RedactedThinking(_) => {
                false
            }
            Block::ToolResult(result) => nested_unrecoverable(&result.content),
            Block::Unknown(fields) => unknown_unrecoverable(fields),
        }
    }
}

/// A tool result's `content` is a string or a block array. Any non-text block
/// inside it is payload a summary cannot carry.
fn nested_unrecoverable(content: &Value) -> bool {
    match content {
        Value::Array(blocks) => blocks.iter().any(|block| {
            let Some(block) = block.as_object() else {
                return true;
            };
            match block.get("type").and_then(Value::as_str) {
                Some("text") => false,
                Some("image") => !archivable(block.get("source")),
                // A block type this crate never modelled inside a tool result:
                // assume the worst, since nothing will restore it.
                _ => true,
            }
        }),
        // A bare string carries no media.
        _ => false,
    }
}

fn unknown_unrecoverable(fields: &Map<String, Value>) -> bool {
    match fields.get("type").and_then(Value::as_str) {
        // The only unmodelled type whose bytes can sit inline.
        Some("image") => !archivable(fields.get("source")),
        Some("tool_result") => nested_unrecoverable(fields.get("content").unwrap_or(&Value::Null)),
        // Shares the image shape but carries a signature, not bytes.
        Some("redacted_thinking") => false,
        // document, PDF, video, a provider extension: nothing restores it.
        Some(_) => true,
        // No tag at all: malformed, and the old code folded these as text.
        // Pin them only once there is evidence they carry payload.
        None => false,
    }
}

/// Whether a media `source` still has its bytes inline.
fn archivable(source: Option<&Value>) -> bool {
    let Some(source) = source.and_then(Value::as_object) else {
        return false;
    };
    match source.get("type").and_then(Value::as_str) {
        Some("base64") => source.get("data").is_some_and(Value::is_string),
        // A `data:` URL still carries its bytes inline; an http URL does not.
        Some("url") => source
            .get("url")
            .and_then(Value::as_str)
            .is_some_and(|url| url.starts_with("data:")),
        _ => false,
    }
}

impl PartialEq for Block {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Block::Text(a), Block::Text(b)) => a == b,
            (Block::ToolUse(a), Block::ToolUse(b)) => a == b,
            (Block::ToolResult(a), Block::ToolResult(b)) => a == b,
            (Block::Thinking(a), Block::Thinking(b)) => a == b,
            (Block::RedactedThinking(a), Block::RedactedThinking(b)) => a == b,
            // Structural equality: two unknown blocks match when they carry the
            // same fields with the same values.
            (Block::Unknown(a), Block::Unknown(b)) => a == b,
            _ => false,
        }
    }
}

/// Serializes the modelled variants through a mirror enum so the `type` tag and
/// the block's own fields are written by the same streaming code the derive
/// would have generated. Round-tripping them through `Value` would sort their
/// keys (this crate does not enable `preserve_order`), which is a different byte
/// sequence to the provider and a prefix-cache miss.
#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum BlockRef<'a> {
    Text(&'a TextBlock),
    ToolUse(&'a ToolUseBlock),
    ToolResult(&'a ToolResultBlock),
    Thinking(&'a ThinkingBlock),
    RedactedThinking(&'a RedactedThinkingBlock),
}

impl Serialize for Block {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Block::Text(b) => BlockRef::Text(b).serialize(serializer),
            Block::ToolUse(b) => BlockRef::ToolUse(b).serialize(serializer),
            Block::ToolResult(b) => BlockRef::ToolResult(b).serialize(serializer),
            Block::Thinking(b) => BlockRef::Thinking(b).serialize(serializer),
            Block::RedactedThinking(b) => BlockRef::RedactedThinking(b).serialize(serializer),
            Block::Unknown(fields) => fields.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for Block {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error as _;

        let fields: Map<String, Value> = Map::deserialize(deserializer)?;
        let kind = fields
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();

        // The tag is peeled off before a modelled struct sees it: every struct
        // here has `#[serde(flatten)] extra`, so leaving `type` in the map would
        // put it in `extra` and serialize a second copy beside the tag. An
        // unmodelled block keeps the whole map, tag included.
        let mut known = fields.clone();
        known.remove("type");
        let body = || Value::Object(known.clone());

        Ok(match kind.as_str() {
            "text" => Block::Text(serde_json::from_value(body()).map_err(D::Error::custom)?),
            "tool_use" => Block::ToolUse(serde_json::from_value(body()).map_err(D::Error::custom)?),
            "tool_result" => {
                Block::ToolResult(serde_json::from_value(body()).map_err(D::Error::custom)?)
            }
            "thinking" => {
                Block::Thinking(serde_json::from_value(body()).map_err(D::Error::custom)?)
            }
            "redacted_thinking" => {
                Block::RedactedThinking(serde_json::from_value(body()).map_err(D::Error::custom)?)
            }
            // Everything else keeps every field it arrived with.
            _ => Block::Unknown(fields),
        })
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct TextBlock {
    pub text: String,
    /// Per-block cache breakpoint. Where this lands decides cache hit rate, so
    /// it is part of the IR's job to preserve it, not the adapter's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ToolUseBlock {
    pub id: String,
    pub name: String,
    /// A JSON **object**, unlike OpenAI's JSON **string**. The IR stores the
    /// string form for both; the adapter re-parses only here, at the boundary.
    pub input: Value,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ToolResultBlock {
    pub tool_use_id: String,
    pub content: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// Opaque on purpose. `signature` is validated by the server; touching either
/// field bricks the session.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ThinkingBlock {
    pub thinking: String,
    pub signature: String,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct RedactedThinkingBlock {
    pub data: String,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Message {
    pub role: String, // "user" | "assistant"
    pub content: Content,
}

impl Message {
    /// Whether any block in this message carries payload a summary cannot keep.
    ///
    /// This is the value to pass to `billion_context::Message::with_unrecoverable_payload`:
    /// a range that folds this message keeps the prose and destroys the bytes.
    pub fn has_unrecoverable_payload(&self) -> bool {
        match &self.content {
            Content::Text(_) => false,
            Content::Blocks(blocks) => blocks.iter().any(Block::has_unrecoverable_payload),
        }
    }
}

/// Anthropic reports cached tokens **outside** `input_tokens`; OpenAI reports
/// them **inside** `prompt_tokens`. Subtracting the wrong way mis-estimates
/// context occupancy, which silently mis-times every nudge.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct Usage {
    #[serde(default)]
    pub input_tokens: u32,
    #[serde(default)]
    pub output_tokens: u32,
    #[serde(default)]
    pub cache_creation_input_tokens: u32,
    #[serde(default)]
    pub cache_read_input_tokens: u32,
}

impl Usage {
    /// Occupancy of the *request* we just sent.
    pub fn prompt_total(&self) -> u32 {
        self.input_tokens + self.cache_creation_input_tokens + self.cache_read_input_tokens
    }
}

// ---------------------------------------------------------------------------
// OpenAI Chat Completions (only the parts that differ from Anthropic)
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct OpenAiToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String, // "function"
    pub function: OpenAiFunction,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct OpenAiFunction {
    pub name: String,
    /// A JSON **string**, not an object. Storing it as a `Value` and
    /// re-serializing sorts its keys (see the test below) and changes the
    /// bytes of every later request.
    pub arguments: String,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct OpenAiUsage {
    #[serde(default)]
    pub prompt_tokens: u32,
    #[serde(default)]
    pub completion_tokens: u32,
    #[serde(default)]
    pub prompt_tokens_details: Option<OpenAiPromptDetails>,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct OpenAiPromptDetails {
    #[serde(default)]
    pub cached_tokens: u32,
}

impl OpenAiUsage {
    /// Cached tokens are already counted inside `prompt_tokens`.
    pub fn prompt_total(&self) -> u32 {
        self.prompt_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Property 1: a field this crate has never heard of survives the round trip.
    /// Without `#[serde(flatten)] extra`, the next request silently drops it and
    /// the provider either rejects the turn or behaves differently.
    #[test]
    fn unknown_fields_survive() {
        let raw = r#"{
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "weigh options", "signature": "EqQBCkYI", "x_vendor_trace": "abc"},
                {"type": "text", "text": "Done.", "cache_control": {"type": "ephemeral"}}
            ]
        }"#;

        let msg: Message = serde_json::from_str(raw).unwrap();
        let back = serde_json::to_string(&msg).unwrap();

        assert!(back.contains(r#""x_vendor_trace":"abc""#), "{back}");
        assert!(back.contains(r#""signature":"EqQBCkYI""#), "{back}");
        assert!(back.contains(r#""cache_control":{"type":"ephemeral"}"#), "{back}");
    }

    fn block(json: &str) -> Block {
        serde_json::from_str(json).expect("block parses")
    }

    /// Inline bytes can be spilled to disk and restored, so they fold like text.
    /// A remote URL is a pointer, and a fold destroys the only pointer.
    #[test]
    fn only_media_without_inline_bytes_blocks_a_fold() {
        let inline = block(
            r#"{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AA=="}}"#,
        );
        assert!(!inline.has_unrecoverable_payload());

        let data_url =
            block(r#"{"type":"image","source":{"type":"url","url":"data:image/png;base64,AA=="}}"#);
        assert!(!data_url.has_unrecoverable_payload());

        let remote = block(
            r#"{"type":"image","source":{"type":"url","url":"https://example.com/a.png"}}"#,
        );
        assert!(remote.has_unrecoverable_payload());

        // A base64 source with no `data` is a reference too, not bytes.
        let empty = block(r#"{"type":"image","source":{"type":"base64","media_type":"image/png"}}"#);
        assert!(empty.has_unrecoverable_payload());
    }

    /// Anything the crate never modelled is payload with no restore behind it.
    #[test]
    fn unknown_block_types_block_a_fold() {
        assert!(block(r#"{"type":"document","source":{"type":"base64","data":"AA=="}}"#)
            .has_unrecoverable_payload());
        assert!(block(r#"{"type":"video","source":{"type":"file","file_id":"f_1"}}"#)
            .has_unrecoverable_payload());
        // No tag at all: malformed, previously folded as text — stay folded.
        assert!(!block(r#"{"content":"bare"}"#).has_unrecoverable_payload());
    }

    /// A media block nested in a tool result is payload too; the prose next to
    /// it survives a fold, the image does not.
    #[test]
    fn media_inside_a_tool_result_is_found() {
        let with_image = block(
            r#"{"type":"tool_result","tool_use_id":"t1","content":[
                 {"type":"text","text":"here it is"},
                 {"type":"image","source":{"type":"base64","media_type":"image/png","data":"AA=="}}
               ]}"#,
        );
        assert!(!with_image.has_unrecoverable_payload());

        let remote = block(
            r#"{"type":"tool_result","tool_use_id":"t1","content":[
                 {"type":"image","source":{"type":"url","url":"https://example.com/a.png"}}
               ]}"#,
        );
        assert!(remote.has_unrecoverable_payload());

        let text_only = block(
            r#"{"type":"tool_result","tool_use_id":"t1","content":[{"type":"text","text":"ok"}]}"#,
        );
        assert!(!text_only.has_unrecoverable_payload());
    }

    #[test]
    fn a_message_reports_payload_any_of_its_blocks_carry() {
        let msg: Message = serde_json::from_str(
            r#"{"role":"user","content":[
                 {"type":"text","text":"what is this"},
                 {"type":"image","source":{"type":"url","url":"https://example.com/a.png"}}
               ]}"#,
        )
        .expect("message parses");
        assert!(msg.has_unrecoverable_payload());

        let plain: Message =
            serde_json::from_str(r#"{"role":"user","content":"just text"}"#).expect("message parses");
        assert!(!plain.has_unrecoverable_payload());
    }

    /// Property 2: the thinking payload is byte-identical after a round trip.
    #[test]
    fn thinking_payload_is_byte_identical() {
        let raw = r#"{"type":"thinking","thinking":"line1\nline2\t\"quoted\"","signature":"sig+/=="}"#;
        let block: Block = serde_json::from_str(raw).unwrap();
        match &block {
            Block::Thinking(t) => {
                assert_eq!(t.thinking, "line1\nline2\t\"quoted\"");
                assert_eq!(t.signature, "sig+/==");
            }
            other => panic!("expected thinking, got {other:?}"),
        }
        assert_eq!(serde_json::to_string(&block).unwrap(), raw);
    }

    /// Property 3 — the trap. `serde_json::Value` uses a sorted map by default,
    /// so a `Value` round trip **reorders object keys**. On tool arguments that
    /// means every subsequent request differs from the previous one, which
    /// silently invalidates the provider's prefix cache. This is why the IR
    /// carries `arguments` as `Arc<str>` and never as `Value`.
    ///
    /// The reordering is a property of the compiled-in map, not of serde_json
    /// in the abstract: the `preserve_order` feature swaps the `BTreeMap` for
    /// an `IndexMap` and the reorder stops. `deepseek-recipe-encoding` enables
    /// that feature, so building with `billion-context/hf-tokenizer` turns it on for
    /// the whole graph — the guard then has nothing to catch, and says so
    /// instead of pretending to pass. Carrying the bytes is still right: a
    /// `Value` round trip normalizes whitespace, escapes and float formatting
    /// whether or not it reorders.
    #[test]
    fn value_round_trip_reorders_keys_and_changes_bytes() {
        let args = r#"{"zeta":1,"alpha":2,"mu":3}"#;
        let value: Value = serde_json::from_str(args).unwrap();
        let back = serde_json::to_string(&value).unwrap();

        if back == args {
            eprintln!(
                "skipping: serde_json/preserve_order is on in this build (via \
                 billion-context/hf-tokenizer); the reorder hazard is absent, not fixed"
            );
            return;
        }
        assert_eq!(back, r#"{"alpha":2,"mu":3,"zeta":1}"#);
    }

    /// The IR keeps the string; nothing between here and the wire touches it.
    #[test]
    fn tool_arguments_keep_their_exact_bytes() {
        let args = r#"{"zeta":1,"alpha":2}"#;
        let call = OpenAiToolCall {
            id: "call_1".into(),
            kind: "function".into(),
            function: OpenAiFunction {
                name: "compress".into(),
                arguments: args.into(),
            },
        };
        let wire = serde_json::to_string(&call).unwrap();
        assert!(wire.contains(r#""arguments":"{\"zeta\":1,\"alpha\":2}""#), "{wire}");
    }

    /// Two providers, two conventions, one number that must not be wrong.
    #[test]
    fn usage_conventions_are_opposite() {
        let anthropic = Usage {
            input_tokens: 900,
            output_tokens: 100,
            cache_creation_input_tokens: 5_000,
            cache_read_input_tokens: 40_000,
        };
        let openai = OpenAiUsage {
            prompt_tokens: 45_900,
            completion_tokens: 100,
            prompt_tokens_details: Some(OpenAiPromptDetails { cached_tokens: 40_000 }),
        };
        assert_eq!(anthropic.prompt_total(), 45_900);
        assert_eq!(openai.prompt_total(), 45_900);
        // Naively subtracting cached tokens from OpenAI's prompt_tokens — the
        // habit Anthropic's convention teaches — would report 5_900 here.
        assert_ne!(
            openai.prompt_tokens - openai.prompt_tokens_details.unwrap().cached_tokens,
            anthropic.prompt_total()
        );
    }
}

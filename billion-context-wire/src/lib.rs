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

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Block {
    Text(TextBlock),
    ToolUse(ToolUseBlock),
    ToolResult(ToolResultBlock),
    Thinking(ThinkingBlock),
    RedactedThinking(RedactedThinkingBlock),
    /// Anything this crate does not know about yet, carried verbatim.
    #[serde(other)]
    Unknown,
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

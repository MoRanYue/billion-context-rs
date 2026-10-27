//! The normalized message representation.
//!
//! This is an intermediate representation, deliberately **not** a provider wire
//! format. `Content` is a closed enum so impossible states cannot be built —
//! the alternative (a struct with every provider's optional field) is exactly
//! the shape this crate exists to avoid.

use crate::ids::MessageId;
use std::sync::Arc;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Role {
    User,
    Assistant,
    System,
    Tool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ContentKind {
    Text,
    Reasoning,
    ToolCall,
    ToolResult,
}

impl ContentKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ContentKind::Text => "text",
            ContentKind::Reasoning => "reasoning",
            ContentKind::ToolCall => "tool-call",
            ContentKind::ToolResult => "tool-result",
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Content {
    Text(Arc<str>),
    /// Opaque. A provider's reasoning payload is signed; reconstructing it
    /// invalidates the conversation, so the engine never parses it.
    Reasoning(Arc<str>),
    ToolCall {
        name: Arc<str>,
        call_id: Arc<str>,
        /// A JSON **string**, kept byte-exact. Parsing it into a value and
        /// re-serializing would sort its keys and change every later request.
        arguments: Arc<str>,
    },
    ToolResult {
        /// The tool that produced this output. A result carries no prose of its
        /// own, so the rendered tag's `type` attribute has to come from here —
        /// without it every tool result would render as `type="tool-result"`
        /// and the model would lose the ability to tell two tools apart.
        name: Arc<str>,
        call_id: Arc<str>,
        content: Arc<str>,
    },
}

impl Content {
    pub fn kind(&self) -> ContentKind {
        match self {
            Content::Text(_) => ContentKind::Text,
            Content::Reasoning(_) => ContentKind::Reasoning,
            Content::ToolCall { .. } => ContentKind::ToolCall,
            Content::ToolResult { .. } => ContentKind::ToolResult,
        }
    }

    /// The text payload, when this content is plain text.
    pub fn text(&self) -> Option<&str> {
        match self {
            Content::Text(text) => Some(text),
            _ => None,
        }
    }

    /// The text a compressor may fold. `ToolCall` returns `None`: its arguments
    /// are user intent the host will execute, so rewriting them is forgery.
    pub fn foldable_text(&self) -> Option<&str> {
        match self {
            Content::Text(t) | Content::Reasoning(t) => Some(t),
            Content::ToolResult { content, .. } => Some(content),
            Content::ToolCall { .. } => None,
        }
    }

    /// The text the model reads, for any text-bearing variant.
    pub fn body(&self) -> Option<&str> {
        match self {
            Content::Text(t) | Content::Reasoning(t) => Some(t),
            Content::ToolResult { content, .. } => Some(content),
            Content::ToolCall { .. } => None,
        }
    }

    /// Which tool this content belongs to, when that is knowable. A tool result
    /// carries the name so the rendered tag can say `type="grep"` rather than
    /// `type="tool-result"`.
    pub fn tool_name(&self) -> Option<&str> {
        match self {
            Content::ToolCall { name, .. } | Content::ToolResult { name, .. } => Some(name),
            _ => None,
        }
    }

    /// A tool call's arguments, verbatim.
    ///
    /// Not foldable text — a summary may not rewrite them — but they are bytes
    /// the provider bills for and the only content a call carries, so anything
    /// that sizes or identifies a message has to look here when
    /// [`Content::foldable_text`] comes back empty.
    pub fn tool_arguments(&self) -> Option<&str> {
        match self {
            Content::ToolCall { arguments, .. } => Some(arguments),
            _ => None,
        }
    }

    /// The call this content is half of: a call's own id, or the id a result
    /// answers. `None` for prose. This — not array position — is the identity
    /// that keeps a call and its result together through a fold.
    pub fn tool_call_id(&self) -> Option<&str> {
        match self {
            Content::ToolCall { call_id, .. } | Content::ToolResult { call_id, .. } => {
                Some(call_id)
            }
            _ => None,
        }
    }

    /// Prepend a rendered ref tag. Only text-bearing variants are annotated;
    /// a tool call's identity lives in its `call_id`, not in prose.
    pub(crate) fn with_prefix(self, tag: &str) -> Content {
        match self {
            Content::Text(t) => Content::Text(Arc::from(format!("{tag}{t}"))),
            Content::Reasoning(t) => Content::Reasoning(Arc::from(format!("{tag}{t}"))),
            Content::ToolResult {
                name,
                call_id,
                content,
            } => Content::ToolResult {
                name,
                call_id,
                content: Arc::from(format!("{tag}{content}")),
            },
            other => other,
        }
    }

    /// Replace the text payload, keeping the variant. Rendering strips any stale
    /// tag and re-stamps, so the bytes must be rebuildable from the body alone.
    pub(crate) fn with_body(self, body: &str) -> Content {
        match self {
            Content::Text(_) => Content::Text(Arc::from(body)),
            Content::Reasoning(_) => Content::Reasoning(Arc::from(body)),
            Content::ToolResult {
                name,
                call_id,
                content: _,
            } => Content::ToolResult {
                name,
                call_id,
                content: Arc::from(body),
            },
            other => other,
        }
    }

    pub fn is_empty_text(&self) -> bool {
        match self {
            Content::Text(t) => t.trim().is_empty(),
            _ => false,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Message {
    pub id: MessageId,
    pub role: Role,
    pub content: Content,
    /// Reasoning payload riding along every request but not part of `content`.
    /// Metering only: never rendered, truncated, or folded.
    pub thinking_tokens: u32,
    /// Set by the host when this message is the host's own rendered summary of
    /// an active block, so a plain message range must not fold it again.
    pub summary_of_block: Option<crate::ids::BlockId>,
    /// Set by the host when this message carries payload the host cannot hand
    /// back after a fold — a remote-URL image, a provider file reference, a
    /// document.
    ///
    /// The engine folds *text*, and a summary is text. Once an image block has
    /// been replaced by its prose description there is nothing behind
    /// `decompress` to restore it: the block is text, the bytes are gone. So a
    /// message flagged here is a **hard hole** in the transcript — no
    /// recommended range may span it (`recommend.rs`).
    ///
    /// The host decides this, because only the host sees the provider's wire
    /// form. `billion-context-wire` answers the Anthropic half for you with
    /// [`Block::has_unrecoverable_payload`]; the core deliberately keeps no
    /// provider knowledge.
    ///
    /// [`Block::has_unrecoverable_payload`]: https://docs.rs/billion-context-wire
    #[cfg_attr(feature = "serde", serde(default))]
    pub unrecoverable_payload: bool,
}

impl Message {
    pub fn new(id: impl AsRef<str>, role: Role, content: Content) -> Self {
        Self {
            id: MessageId::new(id),
            role,
            content,
            thinking_tokens: 0,
            summary_of_block: None,
            unrecoverable_payload: false,
        }
    }

    /// Mark this message as carrying payload that cannot be restored after a
    /// fold. See [`Message::unrecoverable_payload`].
    ///
    /// ```no_run
    /// use billion_context::{Content, Message, Role};
    /// let msg = Message::text("a1", Role::User, "look at this")
    ///     .with_unrecoverable_payload(true);
    /// assert!(msg.has_unrecoverable_payload());
    /// ```
    #[must_use]
    pub fn with_unrecoverable_payload(mut self, yes: bool) -> Self {
        self.unrecoverable_payload = yes;
        self
    }

    /// Whether folding this message would destroy bytes the host cannot give
    /// back. The recommender refuses to let any range span such a message.
    pub fn has_unrecoverable_payload(&self) -> bool {
        self.unrecoverable_payload
    }

    pub fn text(id: impl AsRef<str>, role: Role, text: impl AsRef<str>) -> Self {
        Self::new(id, role, Content::Text(Arc::from(text.as_ref())))
    }

    pub fn tool_call(
        id: impl AsRef<str>,
        name: impl AsRef<str>,
        call_id: impl AsRef<str>,
        arguments: impl AsRef<str>,
    ) -> Self {
        Self::new(
            id,
            Role::Assistant,
            Content::ToolCall {
                name: Arc::from(name.as_ref()),
                call_id: Arc::from(call_id.as_ref()),
                arguments: Arc::from(arguments.as_ref()),
            },
        )
    }

    pub fn tool_result(
        id: impl AsRef<str>,
        name: impl AsRef<str>,
        call_id: impl AsRef<str>,
        content: impl AsRef<str>,
    ) -> Self {
        Self::new(
            id,
            Role::Tool,
            Content::ToolResult {
                name: Arc::from(name.as_ref()),
                call_id: Arc::from(call_id.as_ref()),
                content: Arc::from(content.as_ref()),
            },
        )
    }

    /// Content fingerprint, used to recognise a resent original under a new id.
    /// FNV-1a: no dependency, and stability across runs matters more than
    /// collision resistance here.
    pub(crate) fn fingerprint(&self) -> u64 {
        const SEED: u64 = 0xcbf2_9ce4_8422_2325;
        fn feed(mut h: u64, bytes: &[u8]) -> u64 {
            for &b in bytes {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
            h
        }
        let mut h = feed(
            SEED,
            match self.role {
                Role::User => b"u",
                Role::Assistant => b"a",
                Role::System => b"s",
                Role::Tool => b"t",
            },
        );
        if let Some(text) = self.content.foldable_text() {
            h = feed(h, text.as_bytes());
        } else if let Content::ToolCall {
            name,
            call_id,
            arguments,
        } = &self.content
        {
            // A call carries no prose, so feeding prose alone would leave every
            // call with the same fingerprint. `reconcile` matches a resent
            // original by fingerprint, so a collapsed baseline would make it
            // keep the first call and drop the real one — and the tool pairing
            // it should have preserved would go with it. Name, call id and
            // arguments are the identity a provider gives a call.
            h = feed(h, name.as_bytes());
            h = feed(h, call_id.as_bytes());
            h = feed(h, arguments.as_bytes());
        }
        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_arguments_are_not_foldable() {
        let call = Message::tool_call("a1", "compress", "call_1", r#"{"zeta":1}"#);
        assert_eq!(call.content.foldable_text(), None);
        assert_eq!(call.content.kind(), ContentKind::ToolCall);
    }

    #[test]
    fn fingerprint_tracks_content_not_id() {
        let a = Message::text("a", Role::User, "same body");
        let b = Message::text("b", Role::User, "same body");
        let c = Message::text("a", Role::User, "different body");
        assert_eq!(a.fingerprint(), b.fingerprint());
        assert_ne!(a.fingerprint(), c.fingerprint());
    }

    #[test]
    fn prefix_skips_tool_calls() {
        let call = Message::tool_call("a1", "x", "c1", "{}");
        let prefixed = call.content.clone().with_prefix("<tag/>");
        assert_eq!(prefixed, call.content, "a tool call must not be rewritten");
    }
}

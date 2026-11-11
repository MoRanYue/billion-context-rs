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

    /// Prepend a rendered ref tag. Every variant carries bytes the provider
    /// bills for, and the tag is the only handle the model has on a message it
    /// may later fold — so a tool call is tagged over its arguments exactly as
    /// prose is tagged over its text. Hosts that cannot tolerate a tagged
    /// arguments string (they parse it as JSON for a provider) render with
    /// [`crate::RenderStrategy::TextOnly`] instead.
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
            Content::ToolCall {
                name,
                call_id,
                arguments,
            } => Content::ToolCall {
                name,
                call_id,
                arguments: Arc::from(format!("{tag}{arguments}")),
            },
        }
    }

    /// Replace the payload, keeping the variant. Rendering strips any stale tag
    /// and re-stamps, so the bytes must be rebuildable from the body alone.
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
            Content::ToolCall {
                name,
                call_id,
                arguments: _,
            } => Content::ToolCall {
                name,
                call_id,
                arguments: Arc::from(body),
            },
        }
    }

    /// The bytes rendering stamps a tag onto: prose for prose, arguments for a
    /// call, output for a result. Distinct from [`Content::foldable_text`],
    /// which withholds a call's arguments because no summary may rewrite them.
    pub(crate) fn payload(&self) -> &str {
        match self {
            Content::Text(t) | Content::Reasoning(t) => t,
            Content::ToolResult { content, .. } => content,
            Content::ToolCall { arguments, .. } => arguments,
        }
    }

    pub fn is_empty_text(&self) -> bool {
        match self {
            Content::Text(t) => t.trim().is_empty(),
            _ => false,
        }
    }
}

/// Remove a leading rendered ref tag, if this payload begins with one.
///
/// [`RenderStrategy::All`](crate::RenderStrategy::All) stamps the tag onto
/// whatever payload a message carries — a tool call's arguments included, for
/// the same reason prose is tagged: the model can only fold what it can name.
/// A host that turns that payload into a provider `tool_use` block has to parse
/// it as JSON, so it strips the tag first:
///
/// ```
/// # use billion_context::strip_ref_tag;
/// let tagged = "<acp tokens=\"4\" type=\"grep\">m00002</acp>\n{\"q\":\"alpha\"}";
/// assert_eq!(strip_ref_tag(tagged), "{\"q\":\"alpha\"}");
/// assert_eq!(strip_ref_tag("untouched"), "untouched");
/// ```
///
/// A host that would rather never see a tagged payload renders with
/// [`RenderStrategy::TextOnly`](crate::RenderStrategy::TextOnly) instead.
pub fn strip_ref_tag(text: &str) -> &str {
    let Some(rest) = text.strip_prefix("<acp ") else {
        return text;
    };
    let Some(gt) = rest.find('>') else {
        return text;
    };
    let after = &rest[gt + 1..];
    let Some(close) = after.find("</acp>") else {
        return text;
    };
    // What sits between `>` and `</acp>` is a ref and nothing else. Anything
    // else is message text that happens to start with the same bytes.
    let ref_body = &after[..close];
    if ref_body.is_empty() || ref_body.contains(char::is_whitespace) {
        return text;
    }
    let rest = &after[close + "</acp>".len()..];
    rest.strip_prefix('\n').unwrap_or(rest)
}

// ---------------------------------------------------------------------------
// Media
// ---------------------------------------------------------------------------

/// Where a media payload a message carries lives.
///
/// Facts only: the core never fetches, decodes or prices one. A host maps this
/// from its wire crate (`billion_context_wire::MediaSource`).
#[derive(Clone, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MediaSource {
    /// The bytes ride in the request — base64, or a `data:` URL.
    Inline,
    /// A remote pointer. Keeping it alive is the host's business.
    Url(Arc<str>),
    /// A provider-side file id.
    FileId(Arc<str>),
}

/// What a message carries besides text, and the one decision it changes.
///
/// One fact per message: when a message carries several payloads, this describes
/// the first, while [`Message::has_unrecoverable_payload`] still accounts for
/// all of them. A host that prices per image counts them on the wire, where the
/// whole body is in hand.
///
/// The core holds **no bytes**: a message is cloned on every turn, and a base64
/// payload would turn each clone into a megabyte-scale copy. It also holds no
/// **price**: meters differ by provider, route and image size, so the number
/// stays with the host, which can learn it from `usage`. The core's own token
/// estimate is text-only on purpose — teaching it about pixels would mean
/// teaching it about providers, which is exactly the knowledge this crate
/// exists to keep out.
///
/// What the core does instead is mechanical, and is the part a host cannot do
/// safely alone: a fold that would destroy unrecoverable media is refused (see
/// [`Message::has_unrecoverable_payload`]), and media participates in the
/// message's identity, so a resent original is still recognised when the image
/// is what changed.
#[derive(Clone, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MediaMeta {
    /// IANA media type, when the wire named one. Never guessed.
    pub mime: Option<Arc<str>>,
    /// Decoded payload size, when known without fetching.
    pub byte_len: Option<u64>,
    /// Pixel dimensions, when the payload's header disclosed them.
    pub dimensions: Option<(u32, u32)>,
    pub source: MediaSource,
    /// Whether a fold can hand the payload back. Take it from
    /// `billion_context_wire::MediaInfo::recoverable`, which is exactly
    /// `!Block::has_unrecoverable_payload()`: two spellings of one verdict, so
    /// a host cannot record a recoverability the fold rule disagrees with.
    pub recoverable: bool,
    /// The host's own key for the original bytes — a spill file, a content
    /// hash. The core only carries it, so a host that persists its state can
    /// find the bytes again after a restart. The core never invents one, and
    /// not every host needs one.
    #[cfg_attr(feature = "serde", serde(default))]
    pub handle: Option<Arc<str>>,
}

impl MediaMeta {
    /// One line a host may render where the payload used to be, e.g.
    /// `[image image/png 1024x768 1.5MB]`.
    ///
    /// The kernel never inserts this itself: a note changes the bytes the model
    /// reads, which is wire format — the host's call, not the kernel's.
    pub fn describe(&self) -> String {
        let mut line = String::from("[");
        line.push_str(self.kind());
        if let Some(mime) = &self.mime {
            line.push(' ');
            line.push_str(mime);
        }
        if let Some((w, h)) = self.dimensions {
            line.push_str(&format!(" {w}x{h}"));
        }
        if let Some(bytes) = self.byte_len {
            line.push(' ');
            line.push_str(&human_bytes(bytes));
        }
        if !self.recoverable {
            line.push_str(", not recoverable");
        }
        line.push(']');
        line
    }

    /// A coarse label from the media type alone — enough to say what
    /// disappeared without pretending to know more than the wire did.
    fn kind(&self) -> &'static str {
        match self.mime.as_deref() {
            Some(m) if m.starts_with("image/") => "image",
            Some("application/pdf") => "document",
            Some(m) if m.starts_with("audio/") => "audio",
            Some(m) if m.starts_with("video/") => "video",
            _ => "media",
        }
    }
}

fn human_bytes(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    let n = bytes as f64;
    if n < KB {
        format!("{bytes}B")
    } else if n < MB {
        format!("{:.1}KB", n / KB)
    } else {
        format!("{:.1}MB", n / MB)
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
    /// Media this message carries besides its text, when the host recorded any.
    ///
    /// A `tool_result` may carry an image *and* the text that describes it, so
    /// media rides beside `content` rather than inside it: a `Content` variant
    /// could not express both, and adding one would break every host that
    /// matches the enum exhaustively.
    #[cfg_attr(feature = "serde", serde(default))]
    pub media: Option<MediaMeta>,
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
            media: None,
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

    /// Replace the message's text payload, keeping its identity, role and flags.
    ///
    /// This is what truncation uses: pressure relief must not delete history, so
    /// the message survives with a clipped body rather than disappearing. A
    /// variant that carries no text is returned unchanged.
    #[must_use]
    pub fn with_body(mut self, body: &str) -> Self {
        self.content = self.content.with_body(body);
        self
    }

    /// Attach media facts to this message. See [`MediaMeta`].
    ///
    /// ```
    /// use billion_context::{MediaMeta, MediaSource, Message, Role};
    ///
    /// let meta = MediaMeta {
    ///     mime: None,
    ///     byte_len: Some(4),
    ///     dimensions: None,
    ///     source: MediaSource::Inline,
    ///     recoverable: true,
    ///     handle: None,
    /// };
    /// let msg = Message::text("u1", Role::User, "look").with_media(meta);
    /// assert!(msg.media().is_some());
    /// ```
    #[must_use]
    pub fn with_media(mut self, media: MediaMeta) -> Self {
        self.media = Some(media);
        self
    }

    /// The media this message carries, if any.
    pub fn media(&self) -> Option<&MediaMeta> {
        self.media.as_ref()
    }

    /// Whether folding this message would destroy bytes the host cannot give
    /// back. The recommender refuses to let any range span such a message.
    ///
    /// Derived from the media facts when the host recorded them, so the flag
    /// and [`MediaMeta::recoverable`] can never disagree; the explicit flag
    /// still covers payloads the host could not describe.
    pub fn has_unrecoverable_payload(&self) -> bool {
        self.unrecoverable_payload || self.media.as_ref().is_some_and(|m| !m.recoverable)
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

    /// A thinking block. Providers that require it echoed back will 400 if it
    /// arrives without the assistant text or tool call it produced, which is why
    /// [`crate::orphans`] treats the pair as a unit.
    pub fn reasoning(id: impl AsRef<str>, role: Role, text: impl AsRef<str>) -> Self {
        Self::new(id, role, Content::Reasoning(Arc::from(text.as_ref())))
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
        if let Some(media) = &self.media {
            // Media is part of what a message *is*: two messages that differ
            // only in the picture are different messages. Feed the facts, never
            // the bytes — the handle is the host's key for those.
            h = feed(
                h,
                match &media.source {
                    MediaSource::Inline => b"i",
                    MediaSource::Url(_) => b"r",
                    MediaSource::FileId(_) => b"f",
                },
            );
            if let Some(mime) = &media.mime {
                h = feed(h, mime.as_bytes());
            }
            if let Some(len) = media.byte_len {
                h = feed(h, &len.to_le_bytes());
            }
            if let Some((width, height)) = media.dimensions {
                h = feed(h, &width.to_le_bytes());
                h = feed(h, &height.to_le_bytes());
            }
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
    fn a_prefixed_tool_call_keeps_its_identity() {
        let call = Message::tool_call("a1", "x", "c1", "{}");
        let prefixed = call.content.clone().with_prefix("<tag/>");
        // Pairing is by call id and naming by tool name: the tag rides on the
        // payload and must not disturb either.
        assert_eq!(prefixed.tool_call_id(), call.content.tool_call_id());
        assert_eq!(prefixed.tool_name(), call.content.tool_name());
        assert_eq!(prefixed.tool_arguments(), Some("<tag/>{}"));
        // It stays unfoldable: a summary may never rewrite arguments.
        assert_eq!(prefixed.foldable_text(), None);
    }

    fn meta(recoverable: bool) -> MediaMeta {
        MediaMeta {
            mime: Some(Arc::from("image/png")),
            byte_len: Some(1_572_864),
            dimensions: Some((1024, 768)),
            source: MediaSource::Inline,
            recoverable,
            handle: None,
        }
    }

    #[test]
    fn media_note_states_only_what_the_wire_said() {
        assert_eq!(meta(true).describe(), "[image image/png 1024x768 1.5MB]");
        assert_eq!(
            meta(false).describe(),
            "[image image/png 1024x768 1.5MB, not recoverable]"
        );
        let bare = MediaMeta {
            mime: None,
            byte_len: None,
            dimensions: None,
            source: MediaSource::Url(Arc::from("https://example.com/a.png")),
            recoverable: true,
            handle: None,
        };
        assert_eq!(bare.describe(), "[media]");
    }

    #[test]
    fn media_is_part_of_a_message_identity() {
        let plain = Message::text("a", Role::User, "look");
        let with_media = plain.clone().with_media(meta(true));
        assert_ne!(plain.fingerprint(), with_media.fingerprint());
        // The handle is the host's key to the bytes, not part of what the model
        // sees, so it must not move the identity.
        let handled = plain.clone().with_media(MediaMeta {
            handle: Some(Arc::from("sha256:abc")),
            ..meta(true)
        });
        assert_eq!(with_media.fingerprint(), handled.fingerprint());
    }

    #[test]
    fn unrecoverable_media_is_derived_from_the_facts() {
        assert!(!Message::text("a", Role::User, "x").has_unrecoverable_payload());
        assert!(
            !Message::text("a", Role::User, "x")
                .with_media(meta(true))
                .has_unrecoverable_payload()
        );
        assert!(
            Message::text("a", Role::User, "x")
                .with_media(meta(false))
                .has_unrecoverable_payload()
        );
        // The explicit flag still covers payloads the host could not describe.
        assert!(
            Message::text("a", Role::User, "x")
                .with_unrecoverable_payload(true)
                .has_unrecoverable_payload()
        );
    }

    #[test]
    fn a_body_rewrite_keeps_the_media() {
        let msg = Message::text("a", Role::User, "long body").with_media(meta(true));
        let clipped = msg.with_body("short");
        assert_eq!(clipped.content.body(), Some("short"));
        assert!(clipped.media().is_some());
    }
}

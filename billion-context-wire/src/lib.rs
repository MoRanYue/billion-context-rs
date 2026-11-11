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

    /// The media this block carries, when it carries any.
    ///
    /// A tool result is searched too: an image nested in a tool result is the
    /// common case — a screenshot tool, a chart renderer — and a host that only
    /// looked at top-level blocks would miss it.
    ///
    /// When one block carries several payloads (a tool result with four
    /// screenshots), the first is described. Only the description is narrowed:
    /// `recoverable` is taken from [`Block::has_unrecoverable_payload`], which
    /// accounts for every payload in the block, so a fold cannot be approved on
    /// the strength of the one that happened to be described.
    ///
    /// ```
    /// use billion_context_wire::{Block, MediaSource};
    ///
    /// let inline: Block = serde_json::from_str(
    ///     r#"{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AA=="}}"#,
    /// ).unwrap();
    /// let info = inline.media_info().expect("an image carries media");
    /// assert_eq!(info.mime.as_deref(), Some("image/png"));
    /// assert_eq!(info.byte_len, Some(1));
    /// assert_eq!(info.source, MediaSource::Inline);
    /// assert!(info.recoverable);
    /// ```
    pub fn media_info(&self) -> Option<MediaInfo> {
        let mut info = match self {
            Block::ToolResult(result) => nested_media_info(&result.content)?,
            Block::Unknown(fields) => unknown_media_info(fields)?,
            _ => return None,
        };
        // One source of truth: the fact and the fold rule cannot disagree, so a
        // host can record `recoverable` in the IR without re-deriving it.
        info.recoverable = !self.has_unrecoverable_payload();
        Some(info)
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

// ---------------------------------------------------------------------------
// Media facts
// ---------------------------------------------------------------------------

/// Where a media payload lives. The three cases are not interchangeable: inline
/// bytes can be spilled to disk and handed back, a URL needs the network, and a
/// file id needs the provider.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MediaSource {
    /// The bytes ride in the request — a `base64` source or a `data:` URL.
    Inline,
    /// A remote pointer.
    Url(String),
    /// A provider-side file id.
    FileId(String),
}

/// What a media block says about its payload, without carrying it.
///
/// A host asks two questions of a media message — *what is in it* and *would a
/// fold destroy it* — and both are answerable from the wire alone. Those are
/// facts the IR can hold. Pricing is not: meters differ by provider, route and
/// image size, so the number stays with the host that knows its own bar.
///
/// No bytes are carried on purpose. The IR is cloned every turn, and a base64
/// payload would turn each clone into a megabyte-scale copy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MediaInfo {
    /// IANA media type, when the wire names one (`image/png`, `application/pdf`).
    ///
    /// Never guessed from a file extension or a URL suffix: the wire either
    /// says it or it does not, and a wrong guess is worse than a missing one.
    pub mime: Option<String>,
    /// Decoded size in bytes — known without fetching when the payload is inline.
    pub byte_len: Option<u64>,
    /// Pixel dimensions read from the payload's own header, for the containers
    /// whose size is in the first few bytes (PNG, GIF, BMP, WebP, JPEG).
    pub dimensions: Option<(u32, u32)>,
    pub source: MediaSource,
    /// `true` exactly when [`Block::has_unrecoverable_payload`] is `false` for
    /// this block. One verdict, two spellings, so a host cannot record a
    /// recoverability the fold rule disagrees with.
    pub recoverable: bool,
}

/// The first media payload in a tool result's nested block array.
fn nested_media_info(content: &Value) -> Option<MediaInfo> {
    let Value::Array(blocks) = content else {
        return None;
    };
    blocks
        .iter()
        .find_map(|block| block.as_object().and_then(unknown_media_info))
}

/// Media out of a block this crate does not model.
///
/// The shapes below are four protocols' spellings of the same three facts.
/// Anything else returns `None` and keeps the conservative verdict from
/// [`Block::has_unrecoverable_payload`] — a media block nobody described is
/// still a media block.
fn unknown_media_info(fields: &Map<String, Value>) -> Option<MediaInfo> {
    let payload = media_payload(fields)?;
    let (byte_len, dimensions) = match payload.base64 {
        Some(data) => (
            base64_len(data),
            // A PDF has no pixel width, and decoding 256 KiB of one to look for
            // an image header is wasted work.
            if payload.mime.as_deref().is_none_or(|m| m.starts_with("image/")) {
                image_dimensions(&decode_base64_prefix(data, DIMENSION_SCAN_BYTES))
            } else {
                None
            },
        ),
        None => (None, None),
    };
    Some(MediaInfo {
        mime: payload.mime,
        byte_len,
        dimensions,
        source: payload.source,
        // Overwritten by `Block::media_info` with the block's own verdict.
        recoverable: false,
    })
}

/// A media payload split into the three facts that matter: where it lives, what
/// it is, and — when it rides inline — the base64 body.
struct MediaPayload<'a> {
    source: MediaSource,
    mime: Option<String>,
    base64: Option<&'a str>,
}

fn media_payload<'a>(fields: &'a Map<String, Value>) -> Option<MediaPayload<'a>> {
    match fields.get("type").and_then(Value::as_str)? {
        // Anthropic image / document: `source` is the canonical container.
        "image" | "document" => {
            let source = fields.get("source")?.as_object()?;
            let mime = source.get("media_type").and_then(Value::as_str);
            match source.get("type").and_then(Value::as_str)? {
                "base64" => inline_payload(mime, source.get("data")?.as_str()?),
                "url" => url_payload(mime, source.get("url")?.as_str()?),
                "file" => Some(file_payload(mime, source.get("file_id")?.as_str()?)),
                _ => None,
            }
        }
        // OpenAI chat: `image_url` is a bare URL or `{url, detail}`.
        "image_url" => {
            let url = match fields.get("image_url")? {
                Value::String(url) => url.as_str(),
                Value::Object(obj) => obj.get("url").and_then(Value::as_str)?,
                _ => return None,
            };
            url_payload(None, url)
        }
        // OpenAI Responses: `input_image` carries a URL or a file id.
        "input_image" => match fields.get("image_url") {
            Some(Value::String(url)) => url_payload(None, url),
            Some(Value::Object(obj)) => url_payload(None, obj.get("url")?.as_str()?),
            _ => fields
                .get("file_id")
                .and_then(Value::as_str)
                .map(|id| file_payload(None, id)),
        },
        // OpenAI chat: `file` points at an id, an uploaded blob or a URL.
        "file" => {
            if let Some(id) = fields.get("file_id").and_then(Value::as_str) {
                return Some(file_payload(None, id));
            }
            if let Some(data) = fields.get("file_data").and_then(Value::as_str) {
                return url_payload(None, data);
            }
            url_payload(None, fields.get("file_url").and_then(Value::as_str)?)
        }
        // Google: camelCase keys, `mime_type` instead of `media_type`.
        "inline_data" | "inlineData" => inline_payload(
            fields.get("mime_type").and_then(Value::as_str),
            fields.get("data")?.as_str()?,
        ),
        "file_data" | "fileData" => url_payload(
            fields.get("mime_type").and_then(Value::as_str),
            fields.get("file_uri").and_then(Value::as_str)?,
        ),
        _ => None,
    }
}

fn inline_payload<'a>(mime: Option<&str>, data: &'a str) -> Option<MediaPayload<'a>> {
    Some(MediaPayload {
        source: MediaSource::Inline,
        mime: mime.map(str::to_owned),
        base64: Some(data),
    })
}

fn file_payload<'a>(mime: Option<&str>, id: &str) -> MediaPayload<'a> {
    MediaPayload {
        source: MediaSource::FileId(id.to_owned()),
        mime: mime.map(str::to_owned),
        base64: None,
    }
}

/// A URL — or the inline bytes hiding behind a `data:` URL, which is the one
/// case where the "remote" spelling still carries the payload.
fn url_payload<'a>(mime: Option<&str>, url: &'a str) -> Option<MediaPayload<'a>> {
    let Some(rest) = url.strip_prefix("data:") else {
        return Some(MediaPayload {
            source: MediaSource::Url(url.to_owned()),
            mime: mime.map(str::to_owned),
            base64: None,
        });
    };
    let (meta, body) = rest.split_once(',')?;
    let mut parts = meta.split(';');
    let declared = parts.next().filter(|m| !m.is_empty());
    let is_base64 = parts.any(|p| p == "base64");
    Some(MediaPayload {
        source: MediaSource::Inline,
        mime: mime.or(declared).map(str::to_owned),
        // A percent-encoded `data:` URL carries bytes this crate will not
        // decode; report the type and leave the size unknown.
        base64: is_base64.then_some(body),
    })
}

/// Decoded length of a base64 body, computed from its text so the payload never
/// has to be decoded just to be measured.
fn base64_len(data: &str) -> Option<u64> {
    let n = data.bytes().filter(|b| !b.is_ascii_whitespace()).count();
    let pad = data.bytes().rev().take_while(|&b| b == b'=').count();
    let n = n.saturating_sub(pad);
    // A remainder of 1 cannot come from any byte string.
    if n % 4 == 1 {
        return None;
    }
    Some((n / 4) as u64 * 3 + match n % 4 {
        2 => 1,
        3 => 2,
        _ => 0,
    })
}

/// How much of a payload is decoded to look for an image header. 256 KiB covers
/// every container here; a JPEG's frame header can sit behind EXIF thumbnails
/// and similar metadata, which is why the scan is not limited to the first
/// handful of bytes.
const DIMENSION_SCAN_BYTES: usize = 256 * 1024;

fn sextet(byte: u8) -> Option<u32> {
    Some(match byte {
        b'A'..=b'Z' => u32::from(byte - b'A'),
        b'a'..=b'z' => u32::from(byte - b'a') + 26,
        b'0'..=b'9' => u32::from(byte - b'0') + 52,
        b'+' => 62,
        b'/' => 63,
        _ => return None,
    })
}

/// Decode at most `max` bytes from the front of a base64 body.
fn decode_base64_prefix(data: &str, max: usize) -> Vec<u8> {
    let mut out = Vec::new();
    let mut buf: u32 = 0;
    let mut bits: u32 = 0;
    for &byte in data.as_bytes() {
        let Some(value) = sextet(byte) else {
            continue;
        };
        buf = (buf << 6) | value;
        bits += 6;
        while bits >= 8 {
            bits -= 8;
            out.push(((buf >> bits) & 0xff) as u8);
            if out.len() >= max {
                return out;
            }
        }
    }
    out
}

/// Width and height from an image's own header, for the formats this crate can
/// read without a decoder.
fn image_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    const PNG_MAGIC: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    if bytes.starts_with(&PNG_MAGIC) {
        // IHDR must be the first chunk: width and height are the first two
        // big-endian 32-bit fields after the signature, length and chunk type.
        let w = u32::from_be_bytes(bytes.get(16..20)?.try_into().ok()?);
        let h = u32::from_be_bytes(bytes.get(20..24)?.try_into().ok()?);
        return Some((w, h));
    }
    if bytes.starts_with(b"GIF8") {
        let w = u16::from_le_bytes(bytes.get(6..8)?.try_into().ok()?);
        let h = u16::from_le_bytes(bytes.get(8..10)?.try_into().ok()?);
        return Some((u32::from(w), u32::from(h)));
    }
    if bytes.starts_with(b"BM") {
        // Height is signed: a negative value means the rows are stored top-down
        // and the magnitude is still the height.
        let w = i32::from_le_bytes(bytes.get(18..22)?.try_into().ok()?);
        let h = i32::from_le_bytes(bytes.get(22..26)?.try_into().ok()?);
        return Some((w.unsigned_abs(), h.unsigned_abs()));
    }
    if bytes.len() >= 16 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        return webp_dimensions(bytes);
    }
    if bytes.starts_with(&[0xff, 0xd8]) {
        return jpeg_dimensions(bytes);
    }
    None
}

fn webp_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    match bytes.get(12..16)? {
        // Extended: a 24-bit little-endian canvas size, stored minus one.
        b"VP8X" => {
            let w = u32::from_le_bytes([*bytes.get(24)?, *bytes.get(25)?, *bytes.get(26)?, 0]) + 1;
            let h = u32::from_le_bytes([*bytes.get(27)?, *bytes.get(28)?, *bytes.get(29)?, 0]) + 1;
            Some((w, h))
        }
        // Lossless: 14 bits each, minus one, packed into the four bytes after
        // the `0x2f` signature.
        b"VP8L" => {
            if *bytes.get(20)? != 0x2f {
                return None;
            }
            let bits = u32::from_le_bytes(bytes.get(21..25)?.try_into().ok()?);
            Some(((bits & 0x3fff) + 1, ((bits >> 14) & 0x3fff) + 1))
        }
        // Lossy (`VP8 `) hides its size behind a sync code that a bounded header
        // scan cannot locate reliably, so it reports nothing rather than a
        // wrong number.
        _ => None,
    }
}

fn jpeg_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    let mut i = 2;
    while i + 9 <= bytes.len() {
        if bytes[i] != 0xff {
            i += 1;
            continue;
        }
        let marker = bytes[i + 1];
        // Stand-alone markers carry no length field.
        if marker == 0x01 || (0xd0..=0xd9).contains(&marker) {
            i += 2;
            continue;
        }
        let len = usize::from(u16::from_be_bytes([bytes[i + 2], bytes[i + 3]]));
        // SOF0..SOF15 carry the frame size; C4 (Huffman tables), C8 (reserved)
        // and CC (arithmetic coding) are not frames and would read as garbage.
        if (0xc0..=0xcf).contains(&marker) && !matches!(marker, 0xc4 | 0xc8 | 0xcc) {
            let h = u16::from_be_bytes([bytes[i + 5], bytes[i + 6]]);
            let w = u16::from_be_bytes([bytes[i + 7], bytes[i + 8]]);
            return Some((u32::from(w), u32::from(h)));
        }
        i += 2 + len.max(2);
    }
    None
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

    // -- media facts ------------------------------------------------------

    /// Test-only base64 encoder, so fixtures can be built from real headers
    /// instead of hand-transcribed magic strings.
    fn b64(bytes: &[u8]) -> String {
        const ALPHABET: &[u8; 64] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        for chunk in bytes.chunks(3) {
            let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
            let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
            out.push(ALPHABET[(n >> 18) as usize & 63] as char);
            out.push(ALPHABET[(n >> 12) as usize & 63] as char);
            out.push(if chunk.len() > 1 {
                ALPHABET[(n >> 6) as usize & 63] as char
            } else {
                '='
            });
            out.push(if chunk.len() > 2 {
                ALPHABET[n as usize & 63] as char
            } else {
                '='
            });
        }
        out
    }

    fn png_header(w: u32, h: u32) -> Vec<u8> {
        let mut v = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
        v.extend_from_slice(&13u32.to_be_bytes()); // IHDR length
        v.extend_from_slice(b"IHDR");
        v.extend_from_slice(&w.to_be_bytes());
        v.extend_from_slice(&h.to_be_bytes());
        v.extend_from_slice(&[8, 6, 0, 0, 0]); // bit depth, colour type, ...
        v
    }

    fn gif_header(w: u16, h: u16) -> Vec<u8> {
        let mut v = b"GIF89a".to_vec();
        v.extend_from_slice(&w.to_le_bytes());
        v.extend_from_slice(&h.to_le_bytes());
        v.extend_from_slice(&[0, 0, 0]);
        v
    }

    fn bmp_header(w: i32, h: i32) -> Vec<u8> {
        let mut v = b"BM".to_vec();
        v.resize(18, 0);
        v.extend_from_slice(&w.to_le_bytes());
        v.extend_from_slice(&h.to_le_bytes());
        v
    }

    fn webp_vp8x(w: u32, h: u32) -> Vec<u8> {
        let mut v = b"RIFF".to_vec();
        v.extend_from_slice(&0u32.to_le_bytes());
        v.extend_from_slice(b"WEBP");
        v.extend_from_slice(b"VP8X");
        v.extend_from_slice(&10u32.to_le_bytes());
        v.extend_from_slice(&[0, 0, 0, 0]); // flags + reserved
        v.extend_from_slice(&(w - 1).to_le_bytes()[..3]);
        v.extend_from_slice(&(h - 1).to_le_bytes()[..3]);
        v
    }

    fn webp_vp8l(w: u32, h: u32) -> Vec<u8> {
        let mut v = b"RIFF".to_vec();
        v.extend_from_slice(&0u32.to_le_bytes());
        v.extend_from_slice(b"WEBP");
        v.extend_from_slice(b"VP8L");
        v.extend_from_slice(&5u32.to_le_bytes());
        v.push(0x2f);
        v.extend_from_slice(&((w - 1) | ((h - 1) << 14)).to_le_bytes());
        v
    }

    fn jpeg_header(w: u16, h: u16) -> Vec<u8> {
        let mut v = vec![0xff, 0xd8];
        // APP0 with a two-byte body, to prove the scanner walks segments.
        v.extend_from_slice(&[0xff, 0xe0, 0x00, 0x04, 0x00, 0x00]);
        // SOF0: length, precision, height, width, three components.
        v.extend_from_slice(&[0xff, 0xc0, 0x00, 0x0b, 0x08]);
        v.extend_from_slice(&h.to_be_bytes());
        v.extend_from_slice(&w.to_be_bytes());
        v.extend_from_slice(&[0x03, 1, 0x11, 0, 2, 0x11, 1, 3, 0x11, 1]);
        v
    }

    fn image_block(bytes: &[u8]) -> Block {
        block(&format!(
            r#"{{"type":"image","source":{{"type":"base64","media_type":"image/png","data":"{}"}}}}"#,
            b64(bytes)
        ))
    }

    #[test]
    fn inline_image_reports_type_size_and_pixels() {
        let png = png_header(1024, 768);
        let info = image_block(&png).media_info().expect("media");
        assert_eq!(info.mime.as_deref(), Some("image/png"));
        assert_eq!(info.byte_len, Some(png.len() as u64));
        assert_eq!(info.dimensions, Some((1024, 768)));
        assert_eq!(info.source, MediaSource::Inline);
        assert!(info.recoverable);
    }

    /// A `data:` URL is the same payload under a remote-looking spelling: the
    /// bytes are in the request, so the size and the header are readable.
    #[test]
    fn a_data_url_is_inline_bytes_not_a_pointer() {
        let gif = gif_header(40, 30);
        let inline = block(&format!(
            r#"{{"type":"image","source":{{"type":"url","url":"data:image/gif;base64,{}"}}}}"#,
            b64(&gif)
        ));
        let info = inline.media_info().expect("media");
        assert_eq!(info.source, MediaSource::Inline);
        assert_eq!(info.mime.as_deref(), Some("image/gif"));
        assert_eq!(info.byte_len, Some(gif.len() as u64));
        assert_eq!(info.dimensions, Some((40, 30)));
        assert!(info.recoverable);
    }

    #[test]
    fn a_remote_url_is_a_pointer_with_nothing_to_measure() {
        let remote = block(
            r#"{"type":"image","source":{"type":"url","url":"https://example.com/a.png"}}"#,
        );
        let info = remote.media_info().expect("media");
        assert_eq!(
            info.source,
            MediaSource::Url("https://example.com/a.png".into())
        );
        assert_eq!(info.byte_len, None);
        assert_eq!(info.dimensions, None);
        assert!(!info.recoverable);
    }

    /// Every container whose size sits in the header is read without a decoder.
    #[test]
    fn each_container_header_yields_its_dimensions() {
        let cases: [(&str, Vec<u8>, (u32, u32)); 6] = [
            ("png", png_header(3, 5), (3, 5)),
            ("gif", gif_header(7, 9), (7, 9)),
            ("bmp", bmp_header(11, 13), (11, 13)),
            ("webp/vp8x", webp_vp8x(15, 17), (15, 17)),
            ("webp/vp8l", webp_vp8l(19, 21), (19, 21)),
            ("jpeg", jpeg_header(23, 25), (23, 25)),
        ];
        for (label, bytes, expected) in cases {
            let block = image_block(&bytes);
            assert_eq!(
                block.media_info().and_then(|i| i.dimensions),
                Some(expected),
                "{label} header"
            );
        }
    }

    /// The nested image is the common case: a screenshot or chart tool returns
    /// blocks, and the media sits inside the result rather than beside it.
    #[test]
    fn a_tool_result_reports_the_media_it_embeds() {
        let png = png_header(20, 10);
        let result = block(&format!(
            r#"{{"type":"tool_result","tool_use_id":"t1","content":[
                 {{"type":"text","text":"here it is"}},
                 {{"type":"image","source":{{"type":"base64","media_type":"image/png","data":"{}"}}}}
               ]}}"#,
            b64(&png)
        ));
        let info = result.media_info().expect("media inside a tool result");
        assert_eq!(info.dimensions, Some((20, 10)));
        assert_eq!(info.byte_len, Some(png.len() as u64));
        // The payload is inline, and the containing block agrees.
        assert!(info.recoverable);
        assert!(!result.has_unrecoverable_payload());
    }

    /// The same three facts under the other protocols' spellings.
    #[test]
    fn openai_and_google_spellings_are_understood() {
        let png = png_header(2, 4);
        let openai = block(&format!(
            r#"{{"type":"image_url","image_url":"data:image/png;base64,{}"}}"#,
            b64(&png)
        ));
        assert_eq!(
            openai.media_info().and_then(|i| i.dimensions),
            Some((2, 4))
        );

        let by_id = block(r#"{"type":"input_image","file_id":"file-abc"}"#);
        assert_eq!(
            by_id.media_info().map(|i| i.source),
            Some(MediaSource::FileId("file-abc".into()))
        );

        let google = block(&format!(
            r#"{{"type":"inlineData","mime_type":"image/png","data":"{}"}}"#,
            b64(&png)
        ));
        let info = google.media_info().expect("google inline data");
        assert_eq!(info.mime.as_deref(), Some("image/png"));
        assert_eq!(info.dimensions, Some((2, 4)));

        let remote_file = block(
            r#"{"type":"fileData","mime_type":"image/png","file_uri":"gs://bucket/a.png"}"#,
        );
        assert_eq!(
            remote_file.media_info().map(|i| i.source),
            Some(MediaSource::Url("gs://bucket/a.png".into()))
        );
    }

    /// The description can never promise a fold the fold rule refuses.
    #[test]
    fn the_media_flag_is_exactly_the_negation_of_the_fold_rule() {
        let inline = format!(
            r#"{{"type":"image","source":{{"type":"base64","media_type":"image/png","data":"{}"}}}}"#,
            b64(&png_header(1, 1))
        );
        let cases = [
            inline,
            r#"{"type":"image","source":{"type":"url","url":"https://example.com/a.png"}}"#.into(),
            r#"{"type":"image","source":{"type":"file","file_id":"f_1"}}"#.into(),
            r#"{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"AA=="}}"#
                .into(),
            r#"{"type":"input_image","file_id":"file-abc"}"#.into(),
        ];
        for raw in cases {
            let block = block(&raw);
            let info = block.media_info().expect("media");
            assert_eq!(
                info.recoverable,
                !block.has_unrecoverable_payload(),
                "{raw}"
            );
        }
    }

    /// Text blocks and plain tool results describe no media at all.
    #[test]
    fn non_media_blocks_have_nothing_to_describe() {
        assert!(block(r#"{"type":"text","text":"hi"}"#).media_info().is_none());
        assert!(
            block(r#"{"type":"tool_result","tool_use_id":"t","content":[{"type":"text","text":"ok"}]}"#)
                .media_info()
                .is_none()
        );
    }
}

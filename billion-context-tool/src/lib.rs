//! `billion-context-tool` — the tool-call adapter for `billion-context`.
//!
//! Two decisions are baked in here, and both come from watching models use the
//! original TypeScript implementation:
//!
//! 1. **The schema is the prompt.** Everything the model needs to get a range
//!    right (which refs are legal, that the summary is capped, that a rejected
//!    range does not cost it the others) is stated in the tool description,
//!    because a model reads the description far more reliably than a system
//!    prompt buried 40k tokens up.
//! 2. **Rejections are instructions.** A failed range comes back as text that
//!    tells the model what to do differently, not as an HTTP 400. That loop is
//!    what turns a hallucinated ref into a corrected one on the next round
//!    instead of a dead turn.
//!
//! This crate holds no state, performs no IO, and does not talk to a provider.

use billion_context::{
    Block, BlockId, Boundary, Hit, MsgRef, Range, RangeError, RequestedRange, Retrieval, Status,
};
use serde_json::{json, Value};

/// Name of the folding tool. Hosts must register it under this name.
pub const COMPRESS: &str = "acp_compress";
/// Name of the retrieval tool.
pub const RETRIEVE: &str = "acp_retrieve";
/// Name of the block-restore tool.
pub const DECOMPRESS: &str = "acp_decompress";
/// Name of the summary-search tool.
pub const SEARCH: &str = "acp_search";
/// Name of the status tool.
pub const STATUS: &str = "acp_status";

// ---------------------------------------------------------------------------
// Definitions
// ---------------------------------------------------------------------------

/// Every definition, ready to hand to a provider.
///
/// Five tools, because a model needs five verbs to manage its own context:
/// fold it, look something up by ref, read a summary whole, find a summary by
/// keyword, and see how full it is.
pub fn definitions() -> Vec<Value> {
    vec![
        compress_definition(),
        retrieve_definition(),
        decompress_definition(),
        search_definition(),
        status_definition(),
    ]
}

pub fn compress_definition() -> Value {
    json!({
        "name": COMPRESS,
        "description": concat!(
            "Replace a span of the conversation with a shorter summary. ",
            "Every message in the transcript is tagged with a ref like m00012. ",
            "Fold a contiguous span by passing its first and last ref. ",
            "The originals disappear from the next turn but the refs stay valid: ",
            "a later turn can cite the summary, and the host can still retrieve it. ",
            "Ranges are independent — if one is rejected the others still apply, ",
            "and the rejection tells you what to fix. ",
            "Only fold text you have already read and can summarize without losing ",
            "anything you will need later."
        ),
        "input_schema": {
            "type": "object",
            "required": ["ranges"],
            "additionalProperties": false,
            "properties": {
                "ranges": {
                    "type": "array",
                    "minItems": 1,
                    "description": "Ranges to fold, ordered as they appear in the transcript.",
                    "items": {
                        "type": "object",
                        "required": ["start", "end", "summary"],
                        "additionalProperties": false,
                        "properties": {
                            "start": {
                                "type": "string",
                                "pattern": "^[mb][0-9]{1,7}$",
                                "description": "First ref of the span, e.g. m00012."
                            },
                            "end": {
                                "type": "string",
                                "pattern": "^[mb][0-9]{1,7}$",
                                "description": "Last ref of the span, inclusive."
                            },
                            "topic": {
                                "type": "string",
                                "description": "Two or three words naming what this span was about."
                            },
                            "summary": {
                                "type": "string",
                                "minLength": 50,
                                "description": concat!(
                                    "Standalone replacement text. Keep decisions, their reasons, ",
                                    "exact paths, identifiers, numbers and error strings. ",
                                    "Drop pleasantries and repetition. "
                                )
                            }
                        }
                    }
                }
            }
        }
    })
}

pub fn retrieve_definition() -> Value {
    json!({
        "name": RETRIEVE,
        "description": concat!(
            "Read back what a summary replaced. Pass the refs whose summary is not ",
            "enough — for example when you need the exact wording of a message that ",
            "was folded. Returns the summary plus the original messages when the host ",
            "still has them."
        ),
        "input_schema": {
            "type": "object",
            "required": ["refs"],
            "additionalProperties": false,
            "properties": {
                "refs": {
                    "type": "array",
                    "minItems": 1,
                    "items": { "type": "string", "pattern": "^[mb][0-9]{1,7}$" },
                    "description": "Refs to look up, e.g. [\"m00012\", \"b3\"]."
                }
            }
        }
    })
}

pub fn decompress_definition() -> Value {
    json!({
        "name": DECOMPRESS,
        "description": concat!(
            "Read a folded span whole. Call this when a summary you can see is too ",
            "compressed for the task in front of you — the summary is returned in ",
            "full, not truncated. Use acp_search first if you do not know which ",
            "block you want. This does not undo the fold: the span stays folded ",
            "and the summary stays where it is."
        ),
        "input_schema": {
            "type": "object",
            "required": ["blockId"],
            "additionalProperties": false,
            "properties": {
                "blockId": {
                    "type": "string",
                    "pattern": "^b[0-9]{1,7}$",
                    "description": "Block to read, e.g. b5."
                }
            }
        }
    })
}

pub fn search_definition() -> Value {
    json!({
        "name": SEARCH,
        "description": concat!(
            "Find folded spans by keyword. Search the block topics and summaries ",
            "before decompressing, so you read the right one instead of guessing ",
            "at ids. Every whitespace-separated term scores on its own, so a few ",
            "specific words beat a long sentence."
        ),
        "input_schema": {
            "type": "object",
            "required": ["query"],
            "additionalProperties": false,
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Words to look for, e.g. \"auth token refresh\"."
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Maximum hits (default 5)."
                }
            }
        }
    })
}

pub fn status_definition() -> Value {
    json!({
        "name": STATUS,
        "description": concat!(
            "How full the context is, what has been folded, and which spans are ",
            "worth folding next. Call this before a fold to find a range, and ",
            "after one to confirm it landed. Takes no arguments."
        ),
        "input_schema": {
            "type": "object",
            "additionalProperties": false,
            "properties": {}
        }
    })
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

#[derive(Debug, PartialEq, Eq)]
pub enum ToolError {
    /// The arguments were not a JSON object at all.
    NotAnObject,
    /// `ranges` / `refs` was absent or not an array.
    MissingList(&'static str),
    /// The list was present but empty.
    EmptyList(&'static str),
    /// An element of the list was not an object.
    NotAnObjectAt(usize),
    /// A required string field was absent or empty.
    MissingField { index: usize, field: &'static str },
    /// A ref did not parse as `mNNNNN` or `bN`.
    BadRef {
        index: usize,
        field: &'static str,
        value: String,
    },
    /// `end` came before `start`.
    Backwards { index: usize },
    /// A single-argument tool was called without its argument.
    MissingArg(&'static str),
    /// A block id did not parse as `bN`.
    BadBlockId(String),
}

impl std::fmt::Display for ToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ToolError::NotAnObject => write!(f, "the arguments must be a JSON object"),
            ToolError::MissingList(name) => write!(f, "`{name}` must be a non-empty array"),
            ToolError::EmptyList(name) => write!(f, "`{name}` was empty; pass at least one entry"),
            ToolError::NotAnObjectAt(i) => write!(f, "entry {i} is not an object"),
            ToolError::MissingField { index, field } => {
                write!(f, "entry {index} is missing `{field}`")
            }
            ToolError::BadRef {
                index,
                field,
                value,
            } => write!(
                f,
                "entry {index}: `{field}` is {value:?}, which is not a ref like m00012 or b3"
            ),
            ToolError::Backwards { index } => {
                write!(f, "entry {index}: `end` comes before `start`")
            }
            ToolError::MissingArg(name) => {
                write!(f, "`{name}` is required and must be a non-empty string")
            }
            ToolError::BadBlockId(value) => {
                write!(f, "`blockId` is {value:?}, which is not a block id like b3")
            }
        }
    }
}

impl std::error::Error for ToolError {}

/// Parse a `compress` call into requests the engine can apply.
///
/// Parsing is strict about shape and loose about ranges: a backwards span is
/// caught here so the model gets the correction before a round trip, but
/// whether a ref exists is the engine's call, not ours.
pub fn parse_compress(args: &Value) -> Result<Vec<RequestedRange>, ToolError> {
    let object = args.as_object().ok_or(ToolError::NotAnObject)?;
    let ranges = object
        .get("ranges")
        .and_then(Value::as_array)
        .ok_or(ToolError::MissingList("ranges"))?;
    if ranges.is_empty() {
        return Err(ToolError::EmptyList("ranges"));
    }

    let mut out = Vec::with_capacity(ranges.len());
    for (index, raw) in ranges.iter().enumerate() {
        let entry = raw.as_object().ok_or(ToolError::NotAnObjectAt(index))?;
        let start = parse_boundary(entry, index, "start")?;
        let end = parse_boundary(entry, index, "end")?;

        if let (Boundary::Message(a), Boundary::Message(b)) = (start, end) {
            if b.index() < a.index() {
                return Err(ToolError::Backwards { index });
            }
        }

        let summary = required_str(entry, index, "summary")?;
        let topic = entry
            .get("topic")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string);

        out.push(RequestedRange {
            start,
            end,
            topic,
            summary,
        });
    }
    Ok(out)
}

/// Parse a `retrieve` call into refs.
pub fn parse_retrieve(args: &Value) -> Result<Vec<MsgRef>, ToolError> {
    let object = args.as_object().ok_or(ToolError::NotAnObject)?;
    let refs = object
        .get("refs")
        .and_then(Value::as_array)
        .ok_or(ToolError::MissingList("refs"))?;
    if refs.is_empty() {
        return Err(ToolError::EmptyList("refs"));
    }

    let mut out = Vec::with_capacity(refs.len());
    for (index, raw) in refs.iter().enumerate() {
        let value = raw.as_str().ok_or(ToolError::MissingField {
            index,
            field: "refs",
        })?;
        match Boundary::parse(value) {
            Some(Boundary::Message(r)) => out.push(r),
            // A block ref is legal in a compress span but not a message to look
            // up: `retrieve` answers "what did this message say".
            _ => {
                return Err(ToolError::BadRef {
                    index,
                    field: "refs",
                    value: value.to_string(),
                })
            }
        }
    }
    Ok(out)
}

/// A parsed `search` call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SearchArgs {
    pub query: String,
    pub limit: usize,
}

/// How many hits `acp_search` returns when the model does not say.
pub const DEFAULT_SEARCH_LIMIT: usize = 5;

/// Parse a `search` call. `limit` is clamped rather than rejected: a model that
/// asks for 1000 wants "more", not an error.
pub fn parse_search(args: &Value) -> Result<SearchArgs, ToolError> {
    let object = args.as_object().ok_or(ToolError::NotAnObject)?;
    let query = object
        .get("query")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .ok_or(ToolError::MissingArg("query"))?;
    let limit = object
        .get("limit")
        .and_then(Value::as_u64)
        .map(|n| n.clamp(1, 50) as usize)
        .unwrap_or(DEFAULT_SEARCH_LIMIT);
    Ok(SearchArgs { query, limit })
}

/// Parse a `decompress` call. Message refs are refused on purpose: `acp_retrieve`
/// answers "what did this message say", and accepting `m00012` here would let a
/// model believe it had decompressed something when it had not.
pub fn parse_decompress(args: &Value) -> Result<BlockId, ToolError> {
    let object = args.as_object().ok_or(ToolError::NotAnObject)?;
    let raw = object
        .get("blockId")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or(ToolError::MissingArg("blockId"))?;
    match Boundary::parse(raw) {
        Some(Boundary::Block(id)) => Ok(id),
        _ => Err(ToolError::BadBlockId(raw.to_string())),
    }
}

/// `acp_status` takes no arguments. Any object is accepted, so a model that
/// sends `{}` or an invented `{"verbose":true}` is not punished for it.
pub fn parse_status(args: &Value) -> Result<(), ToolError> {
    args.as_object().map(|_| ()).ok_or(ToolError::NotAnObject)
}

fn parse_boundary(
    entry: &serde_json::Map<String, Value>,
    index: usize,
    field: &'static str,
) -> Result<Boundary, ToolError> {
    let raw = required_str(entry, index, field)?;
    Boundary::parse(&raw).ok_or(ToolError::BadRef {
        index,
        field,
        value: raw,
    })
}

fn required_str(
    entry: &serde_json::Map<String, Value>,
    index: usize,
    field: &'static str,
) -> Result<String, ToolError> {
    entry
        .get(field)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .ok_or(ToolError::MissingField { index, field })
}

// ---------------------------------------------------------------------------
// Rendering results back to the model
// ---------------------------------------------------------------------------

/// The `tool_result` body for a `compress` call.
///
/// Success and failure are reported in the same breath, because the model's
/// next action depends on both: it must stop re-folding what landed, and retry
/// only what did not.
pub fn render_compress(blocks: &[BlockId], errors: &[RangeError]) -> String {
    let mut out = String::new();

    if blocks.is_empty() {
        out.push_str("No range was folded.\n");
    } else {
        out.push_str(&format!("Folded {} range(s): ", blocks.len()));
        out.push_str(
            &blocks
                .iter()
                .map(|b| b.to_string())
                .collect::<Vec<_>>()
                .join(", "),
        );
        out.push_str(".\nThose messages are gone from the next turn; their refs stay valid.\n");
    }

    if !errors.is_empty() {
        out.push_str("\nRejected:\n");
        for error in errors {
            out.push_str(&format!(
                "- {}–{}: {} ({}). Fix: {}\n",
                error.start,
                error.end,
                error.kind.code(),
                error.detail,
                error.kind.hint()
            ));
        }
    }
    out
}

/// The `tool_result` body for a `retrieve` call.
pub fn render_retrieval(hits: &[Retrieval]) -> String {
    if hits.is_empty() {
        return "Nothing matched those refs.".to_string();
    }
    let mut out = String::new();
    for hit in hits {
        let topic = hit.topic.as_deref().unwrap_or("untitled");
        out.push_str(&format!("## {} ({topic})\n", hit.block));
        if let Some((start, end)) = hit.span {
            out.push_str(&format!("replaces {start}–{end}\n"));
        }
        out.push_str(&hit.summary);
        out.push_str("\n\n");
    }
    out.truncate(out.trim_end().len());
    out
}

/// The `tool_result` body for a `decompress` call.
///
/// Returns the summary in full — that is what "decompressed" means at tier 1.
/// The original messages are deliberately not included: they live in the host's
/// transcript, and handing them back here would re-inflate the context the model
/// just compressed. A host that does want them can restore the block instead.
pub fn render_decompression(block: &Block) -> String {
    let topic = block.topic.as_deref().unwrap_or("untitled");
    let mut out = format!("## {} — tier {:?} ({topic})\n", block.id, block.tier);
    if let Some((start, end)) = block.span {
        out.push_str(&format!("replaces {start}–{end}\n"));
    }
    if !block.direct_blocks.is_empty() {
        let folded: Vec<String> = block.direct_blocks.iter().map(|b| b.to_string()).collect();
        out.push_str(&format!("folds {}\n", folded.join(", ")));
    }
    out.push('\n');
    out.push_str(&block.summary);
    out
}

/// The `tool_result` body for a `search` call.
pub fn render_search(hits: &[Hit]) -> String {
    if hits.is_empty() {
        return concat!(
            "No block matched. Try fewer, broader words — every term is matched ",
            "separately against block topics and summaries, so one rare word can ",
            "rule out an otherwise good block."
        )
        .to_string();
    }
    let mut out = String::new();
    for hit in hits {
        let topic = hit.topic.as_deref().unwrap_or("untitled");
        out.push_str(&format!(
            "- {} (tier {:?}, {topic}, score {:.2}): {}\n",
            hit.block, hit.tier, hit.score, hit.snippet
        ));
    }
    out.push_str("\nRead one whole with acp_decompress.");
    out
}

/// The `tool_result` body for a `status` call.
///
/// `ranges` is what `process_turn` reported as compressible; pass an empty slice
/// when the host has no fresh turn. `token_count` is the host's own count for the
/// view it is about to send — the kernel does not track it between turns.
pub fn render_status(status: &Status, token_count: usize, ranges: &[Range]) -> String {
    let mut out = String::new();
    let limit = status.model_context_limit as usize;
    if limit > 0 {
        let used = (100.0 * token_count as f64 / limit as f64).round() as u64;
        out.push_str(&format!(
            "Context: {} / {} tokens ({used}% used)\n",
            thousands(token_count as u64),
            thousands(limit as u64)
        ));
    } else {
        out.push_str(&format!(
            "Context: {} tokens (no window configured)\n",
            thousands(token_count as u64)
        ));
    }
    out.push_str(&format!(
        "Folded: {} block(s) active, {} total; tier1 {}, tier2 {}, tier3 {}\n",
        status.active_blocks,
        status.blocks,
        status.tiers[0],
        status.tiers[1],
        status.tiers[2]
    ));
    out.push_str(&format!(
        "Saved: {} tokens across {} fold(s); {} refs allocated\n",
        thousands(status.tokens_compressed),
        status.compression_count,
        status.refs_allocated
    ));
    if ranges.is_empty() {
        out.push_str("\nNo span is worth folding right now.");
    } else {
        out.push_str("\nWorth folding now:\n");
        for range in ranges {
            out.push_str(&format!(
                "- {}–{} ({} messages, {} tokens)\n",
                range.start,
                range.end,
                range.count,
                thousands(range.tokens as u64)
            ));
        }
    }
    out
}

/// `1234` → `1,234`, so a token count reads at a glance.
fn thousands(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, ch) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use billion_context::{RangeErrorKind, Tier};

    fn call(ranges: Value) -> Value {
        json!({ "ranges": ranges })
    }

    #[test]
    fn definitions_are_objects_with_a_schema() {
        for definition in definitions() {
            assert!(definition.get("name").and_then(Value::as_str).is_some());
            assert!(definition.get("input_schema").is_some());
            assert!(definition
                .get("description")
                .and_then(Value::as_str)
                .unwrap()
                .len()
                > 80);
        }
    }

    #[test]
    fn a_well_formed_call_parses() {
        let parsed = parse_compress(&call(json!([{
            "start": "m00012",
            "end": "m00030",
            "topic": "build setup",
            "summary": "The user configured the workspace and settled on edition 2021."
        }])))
        .expect("parses");

        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].start, Boundary::Message(MsgRef::new(12).unwrap()));
        assert_eq!(parsed[0].topic.as_deref(), Some("build setup"));
    }

    #[test]
    fn block_boundaries_are_accepted_in_a_span() {
        let parsed = parse_compress(&call(json!([{
            "start": "b3",
            "end": "m00040",
            "summary": "A span that starts at a summary and runs into live messages."
        }])))
        .expect("parses");
        assert_eq!(parsed[0].start, Boundary::Block(BlockId::new(3).unwrap()));
    }

    #[test]
    fn a_backwards_span_is_caught_before_the_round_trip() {
        let error = parse_compress(&call(json!([{
            "start": "m00030",
            "end": "m00012",
            "summary": "This span runs the wrong way and should be rejected here."
        }])))
        .unwrap_err();
        assert_eq!(error, ToolError::Backwards { index: 0 });
    }

    #[test]
    fn a_hallucinated_ref_names_the_offending_field() {
        let error = parse_compress(&call(json!([{
            "start": "message twelve",
            "end": "m00030",
            "summary": "The start ref is not a ref at all and the message should say so."
        }])))
        .unwrap_err();
        assert_eq!(
            error,
            ToolError::BadRef {
                index: 0,
                field: "start",
                value: "message twelve".into()
            }
        );
        assert!(error.to_string().contains("m00012"));
    }

    #[test]
    fn empty_and_misshapen_arguments_are_distinct_errors() {
        assert_eq!(parse_compress(&json!("nope")).unwrap_err(), ToolError::NotAnObject);
        assert_eq!(
            parse_compress(&json!({})).unwrap_err(),
            ToolError::MissingList("ranges")
        );
        assert_eq!(
            parse_compress(&call(json!([]))).unwrap_err(),
            ToolError::EmptyList("ranges")
        );
        assert_eq!(
            parse_compress(&call(json!([1, 2]))).unwrap_err(),
            ToolError::NotAnObjectAt(0)
        );
    }

    #[test]
    fn retrieve_takes_message_refs_only() {
        let refs = parse_retrieve(&json!({ "refs": ["m00012", "m00013"] })).expect("parses");
        assert_eq!(refs.len(), 2);

        let error = parse_retrieve(&json!({ "refs": ["b3"] })).unwrap_err();
        assert!(matches!(error, ToolError::BadRef { .. }));
    }

    #[test]
    fn a_partial_result_reports_both_halves() {
        let text = render_compress(
            &[BlockId::new(1).unwrap()],
            &[RangeError {
                start: "m00099".into(),
                end: "m00100".into(),
                kind: RangeErrorKind::UnknownRef,
                detail: "m00099 is not in this view".into(),
            }],
        );
        assert!(text.contains("Folded 1 range"));
        assert!(text.contains("b1"));
        assert!(text.contains("Rejected"));
        // The hint is the point: the model must know what to do next.
        assert!(text.contains("pick one that is"));
    }

    #[test]
    fn retrieval_renders_the_span_it_replaces() {
        let hits = vec![Retrieval {
            block: BlockId::new(2).unwrap(),
            tier: Tier::T1,
            topic: Some("parser".into()),
            summary: "We settled on a recursive-descent parser.".into(),
            span: Some((MsgRef::new(4).unwrap(), MsgRef::new(9).unwrap())),
        }];
        let text = render_retrieval(&hits);
        assert!(text.contains("b2"));
        assert!(text.contains("m00004–m00009"));
        assert!(!text.ends_with('\n'));
    }

    /// The whole point of this crate: a model-shaped call drives a real fold.
    #[test]
    fn a_tool_call_drives_a_real_fold() {
        use billion_context::{CharsAsTokens, CompressRequest, Config, RenderStrategy, Session, TurnInput};

        let mut session = Session::new(Config::new(20_000)).with_estimator(CharsAsTokens);
        let history: Vec<_> = (0..6)
            .map(|i| {
                let role = if i % 2 == 0 {
                    billion_context::Role::User
                } else {
                    billion_context::Role::Assistant
                };
                billion_context::Message::text(
                    format!("msg-{i}"),
                    role,
                    format!("{}{i}", "context ".repeat(400)),
                )
            })
            .collect();

        let turn = session.process_turn(TurnInput {
            messages: &history,
            token_count: 3_000,
            render: RenderStrategy::All,
            now: 1,
        });
        assert_eq!(turn.messages.len(), 6);

        let range = &turn.ranges[0];
        let args = json!({ "ranges": [{
            "start": range.start.to_string(),
            "end": range.end.to_string(),
            "topic": "opening",
            "summary": "The opening exchanges established the port's constraints and scope."
        }] });

        let requests = parse_compress(&args).expect("the model's call parses");
        let applied = session.apply_compression(CompressRequest {
            ranges: requests,
            now: 2,
        });

        assert!(applied.is_ok(), "{:?}", applied.errors);
        let text = render_compress(&applied.blocks, &applied.errors);
        assert!(text.contains("Folded 1 range"));

        let after = session.process_turn(TurnInput {
            messages: &history,
            token_count: 1_000,
            render: RenderStrategy::All,
            now: 3,
        });
        assert!(after.messages.len() < 6, "the folded span must disappear");
    }

    use billion_context::{CharsAsTokens, CompressRequest, Config, RenderStrategy, Session, TurnInput};

    #[test]
    fn all_five_tools_are_declared() {
        let defs = definitions();
        let names: Vec<&str> = defs.iter().filter_map(|d| d["name"].as_str()).collect();
        assert_eq!(
            names,
            vec![COMPRESS, RETRIEVE, DECOMPRESS, SEARCH, STATUS]
        );
    }

    #[test]
    fn decompress_accepts_a_block_and_refuses_a_message() {
        assert_eq!(
            parse_decompress(&json!({ "blockId": "b3" })).unwrap(),
            BlockId::new(3).unwrap()
        );
        // A message ref belongs to acp_retrieve. Saying so is the point: a model
        // that decompressed nothing must not believe otherwise.
        assert_eq!(
            parse_decompress(&json!({ "blockId": "m00012" })).unwrap_err(),
            ToolError::BadBlockId("m00012".into())
        );
        assert_eq!(
            parse_decompress(&json!({})).unwrap_err(),
            ToolError::MissingArg("blockId")
        );
    }

    #[test]
    fn search_clamps_the_limit_instead_of_rejecting_it() {
        let asked = parse_search(&json!({ "query": "auth token", "limit": 1000 })).unwrap();
        assert_eq!(asked.limit, 50, "a huge limit means 'more', not 'error'");
        assert_eq!(asked.query, "auth token");

        let default = parse_search(&json!({ "query": "auth" })).unwrap();
        assert_eq!(default.limit, DEFAULT_SEARCH_LIMIT);

        assert_eq!(
            parse_search(&json!({ "query": "   " })).unwrap_err(),
            ToolError::MissingArg("query")
        );
    }

    #[test]
    fn status_takes_no_arguments_but_still_needs_an_object() {
        assert!(parse_status(&json!({})).is_ok());
        assert!(
            parse_status(&json!({ "verbose": true })).is_ok(),
            "an invented argument must not fail the call"
        );
        assert_eq!(parse_status(&json!([])).unwrap_err(), ToolError::NotAnObject);
    }

    #[test]
    fn an_empty_search_result_says_what_to_change() {
        let text = render_search(&[]);
        assert!(text.contains("No block matched"));
        assert!(text.contains("broader"));
    }

    #[test]
    fn a_folded_block_can_be_searched_and_read_back() {
        let mut session = Session::new(Config::new(20_000)).with_estimator(CharsAsTokens);
        let history: Vec<_> = (0..6)
            .map(|i| {
                let role = if i % 2 == 0 {
                    billion_context::Role::User
                } else {
                    billion_context::Role::Assistant
                };
                billion_context::Message::text(
                    format!("msg-{i}"),
                    role,
                    format!("{}{i}", "context ".repeat(400)),
                )
            })
            .collect();

        let turn = session.process_turn(TurnInput {
            messages: &history,
            token_count: 3_000,
            render: RenderStrategy::All,
            now: 1,
        });
        let range = &turn.ranges[0];
        let requests = parse_compress(&json!({ "ranges": [{
            "start": range.start.to_string(),
            "end": range.end.to_string(),
            "topic": "borrow checker debate",
            "summary": "The opening exchanges established the port's constraints and scope."
        }] }))
        .expect("the model's call parses");
        let applied = session.apply_compression(CompressRequest {
            ranges: requests,
            now: 2,
        });
        assert!(applied.is_ok(), "{:?}", applied.errors);

        // Multi-term: only "borrow" is in the topic, only "constraints" in the
        // summary. A phrase search would find neither.
        let hits = session.search("borrow constraints", 5);
        assert_eq!(hits.len(), 1, "both terms describe the same block");
        let text = render_search(&hits);
        assert!(text.contains("b1"), "{text}");
        assert!(text.contains("borrow checker debate"), "{text}");

        // Read it back without expanding it.
        let id = parse_decompress(&json!({ "blockId": "b1" })).unwrap();
        let block = session.block(id).expect("the block exists");
        let body = render_decompression(block);
        assert!(body.contains("tier T1"), "{body}");
        assert!(body.contains("replaces m00001"), "{body}");
        assert!(body.contains("constraints and scope"), "{body}");
        assert!(
            !session.block(id).unwrap().expanded,
            "reading a block must not expand it"
        );
    }

    #[test]
    fn status_reports_the_saving_in_readable_numbers() {
        let mut session = Session::new(Config::new(20_000)).with_estimator(CharsAsTokens);
        let history: Vec<_> = (0..6)
            .map(|i| {
                let role = if i % 2 == 0 {
                    billion_context::Role::User
                } else {
                    billion_context::Role::Assistant
                };
                billion_context::Message::text(
                    format!("msg-{i}"),
                    role,
                    format!("{}{i}", "context ".repeat(400)),
                )
            })
            .collect();
        let turn = session.process_turn(TurnInput {
            messages: &history,
            token_count: 3_000,
            render: RenderStrategy::All,
            now: 1,
        });

        let status = session.status();
        let text = render_status(&status, 3_000, &turn.ranges);
        assert!(text.contains("3,000 / 20,000 tokens"), "{text}");
        assert!(text.contains("15% used"), "{text}");
        assert!(text.contains("Worth folding now"), "{text}");

        let idle = render_status(&status, 0, &[]);
        assert!(idle.contains("No span is worth folding right now"), "{idle}");
    }

    #[test]
    fn thousands_groups_digits() {
        assert_eq!(thousands(0), "0");
        assert_eq!(thousands(999), "999");
        assert_eq!(thousands(1_000), "1,000");
        assert_eq!(thousands(12_345), "12,345");
        assert_eq!(thousands(1_234_567), "1,234,567");
    }
}

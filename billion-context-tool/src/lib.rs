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

use billion_context::{BlockId, Boundary, MsgRef, RangeError, RequestedRange, Retrieval};
use serde_json::{json, Value};

/// Name of the folding tool. Hosts must register it under this name.
pub const COMPRESS: &str = "acp_compress";
/// Name of the retrieval tool.
pub const RETRIEVE: &str = "acp_retrieve";

// ---------------------------------------------------------------------------
// Definitions
// ---------------------------------------------------------------------------

/// Both definitions, ready to hand to a provider.
pub fn definitions() -> Vec<Value> {
    vec![compress_definition(), retrieve_definition()]
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
}

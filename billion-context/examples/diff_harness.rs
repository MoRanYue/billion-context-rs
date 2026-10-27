//! Differential harness: drives `billion-context` from a scenario JSON and emits the
//! canonical transcript that `diff/run-ts.mts` emits for the same scenario.
//!
//!     cargo run --example diff_harness -- <scenario.json> <out.json>

use billion_context::{
    Block, BlockId, Boundary, CompressRequest, Config, Content, Message, RenderStrategy,
    RequestedRange, Role, Session, TokenEstimator, TurnInput,
};
use serde_json::{json, Map, Value};

fn role_of(s: &str) -> Role {
    match s {
        "user" => Role::User,
        "assistant" => Role::Assistant,
        "system" => Role::System,
        "tool" => Role::Tool,
        other => panic!("unknown role {other}"),
    }
}

fn role_str(r: Role) -> &'static str {
    match r {
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::System => "system",
        Role::Tool => "tool",
    }
}

fn expand(m: &Value) -> Message {
    let id = m["id"].as_str().unwrap();
    let base = m["text"].as_str().unwrap_or_default();
    let text = match m["repeat"].as_u64() {
        Some(n) => base.repeat(n as usize),
        None => base.to_string(),
    };
    let kind = m["kind"].as_str().unwrap();
    let content = match kind {
        "text" => Content::Text(text.into()),
        "reasoning" => Content::Reasoning(text.into()),
        "tool-call" => Content::ToolCall {
            name: m["toolName"].as_str().unwrap().into(),
            call_id: m["toolCallId"].as_str().unwrap().into(),
            arguments: text.into(),
        },
        "tool-result" => Content::ToolResult {
            name: m["toolName"].as_str().unwrap_or("tool").into(),
            call_id: m["toolCallId"].as_str().unwrap().into(),
            content: text.into(),
        },
        other => panic!("unknown kind {other}"),
    };
    Message::new(id, role_of(m["role"].as_str().unwrap()), content)
}

fn text_of(m: &Message) -> Option<String> {
    match &m.content {
        Content::Text(s) | Content::Reasoning(s) => Some(s.to_string()),
        Content::ToolCall { arguments, .. } => Some(arguments.to_string()),
        Content::ToolResult { content, .. } => Some(content.to_string()),
    }
}

fn norm_block(b: &Block) -> Value {
    json!({
        "blockId": b.id.to_string(),
        "tier": b.tier as u8,
        "topic": b.topic,
        "summary": b.summary,
        "active": b.active,
        "startRef": b.span.map(|(s, _)| s.to_string()),
        "endRef": b.span.map(|(_, e)| e.to_string()),
        "compressedTokens": b.compressed_tokens,
        "survived": b.survived,
        "generation": match b.generation {
            billion_context::Generation::Young => "young",
            billion_context::Generation::Old => "old",
        },
    })
}

fn norm_refs(session: &Session) -> Value {
    let mut pairs: Vec<(String, String)> = session
        .state()
        .refs
        .iter_assigned()
        .map(|(id, r)| (id.to_string(), r.to_string()))
        .collect();
    pairs.sort();
    Value::Array(
        pairs
            .into_iter()
            .map(|(k, v)| Value::Array(vec![Value::String(k), Value::String(v)]))
            .collect(),
    )
}

fn norm_blocks(session: &Session) -> Value {
    Value::Array(session.state().blocks.iter().map(norm_block).collect())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let raw = std::fs::read_to_string(&args[1]).expect("scenario");
    let sc: Value = serde_json::from_str(&raw).expect("scenario json");
    let limit = sc["contextLimit"].as_u64().unwrap() as u32;

    // Config knobs the scenario may pin. Both harnesses read the same object so
    // a tier scenario exercises identical arbitration on either side.
    let mut cfg = Config::new(limit);
    if let Some(c) = sc.get("config") {
        if let Some(t) = c.get("tiers") {
            if let Some(v) = t.get("enabled").and_then(Value::as_bool) {
                cfg.tiers.enabled = v;
            }
            if let Some(v) = t.get("tier2Trigger").and_then(Value::as_u64) {
                cfg.tiers.tier2_trigger = v as u32;
            }
            if let Some(v) = t.get("tier3Trigger").and_then(Value::as_u64) {
                cfg.tiers.tier3_trigger = v as u32;
            }
        }
        if let Some(n) = c.get("nudge") {
            if let Some(v) = n.get("growthCap").and_then(Value::as_u64) {
                cfg.nudge.growth_cap = v as u32;
            }
            if let Some(v) = n.get("growthFloor").and_then(Value::as_u64) {
                cfg.nudge.growth_floor = v as u32;
            }
            if let Some(v) = n.get("maxContextLimitPct").and_then(Value::as_f64) {
                cfg.nudge.max_context_limit_pct = v as f32;
            }
            if let Some(v) = n.get("tier2GrowthMultiplier").and_then(Value::as_f64) {
                cfg.nudge.tier2_growth_multiplier = v as f32;
            }
            if let Some(v) = n.get("minPressureBenefitTokens").and_then(Value::as_u64) {
                cfg.nudge.min_pressure_benefit_tokens = Some(v as u32);
            }
            if let Some(g) = n.get("tierGrowthTokens") {
                let growth = cfg.nudge.tier_growth_tokens.get_or_insert_with(Default::default);
                if let Some(v) = g.get("t1").and_then(Value::as_u64) {
                    growth.t1 = Some(v as u32);
                }
                if let Some(v) = g.get("t2").and_then(Value::as_u64) {
                    growth.t2 = Some(v as u32);
                }
                if let Some(v) = g.get("t3").and_then(Value::as_u64) {
                    growth.t3 = Some(v as u32);
                }
            }
        }
        if let Some(v) = c.get("preserveRecentMessages").and_then(Value::as_u64) {
            cfg.preserve_recent_messages = v as usize;
        }
        if let Some(v) = c.get("preserveRecentTokens").and_then(Value::as_u64) {
            cfg.preserve_recent_tokens = v as u32;
        }
    }
    // The TS kernel never reconciles a resent original, so compare against the
    // same contract: prune + carrier (Send) with reconciliation off.
    let mut session = Session::new(cfg).with_reconcile(false);
    // Optional 4th argument "inject" swaps in a synthetic counter, mirroring the
    // TS side's `createCore({ countTokens })`. Identical arithmetic on both
    // sides proves the estimator reaches every counting site the same way.
    if args.get(3).is_some_and(|a| a == "inject") {
        session = session.with_estimator(SyntheticCounter);
    }
    let mut steps: Vec<Value> = Vec::new();

    for step in sc["steps"].as_array().unwrap() {
        match step["op"].as_str().unwrap() {
            "turn" => {
                let messages: Vec<Message> = step["messages"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(expand)
                    .collect();
                let turn = session.process_turn(TurnInput {
                    messages: &messages,
                    token_count: step["tokenCount"].as_u64().unwrap() as u32,
                    render: RenderStrategy::All,
                    now: 0,
                });

                let view: Vec<Value> = turn
                    .messages
                    .iter()
                    .map(|m| {
                        json!({
                            "id": m.id.to_string(),
                            "role": role_str(m.role),
                            "kind": m.content.kind().as_str(),
                            "text": text_of(m),
                        })
                    })
                    .collect();

                let nudge = match &turn.nudge {
                    None => Value::Null,
                    Some(n) => json!({
                        "inject": n.should_inject,
                        "reason": n.reason,
                        "tier": n.tier.map(|t| t as u8),
                        "ranges": n.ranges.iter().map(|r| json!({
                            "startRef": r.start.to_string(),
                            "endRef": r.end.to_string(),
                            "count": r.count,
                            "tokens": r.tokens,
                        })).collect::<Vec<_>>(),
                    }),
                };

                steps.push(json!({
                    "op": "turn",
                    "view": view,
                    "refs": norm_refs(&session),
                    "blocks": norm_blocks(&session),
                    "nudge": nudge,
                    "truncationSkipped": turn.truncation_skipped,
                    "terminal": turn.terminal.as_ref().map(|t| t.message.clone()),
                }));

            }
            "apply" => {
                let mut ranges = Vec::new();
                for r in step["ranges"].as_array().unwrap() {
                    ranges.push(RequestedRange {
                        start: Boundary::parse(r["startRef"].as_str().unwrap())
                            .expect("start ref"),
                        end: Boundary::parse(r["endRef"].as_str().unwrap()).expect("end ref"),
                        topic: r["topic"].as_str().map(str::to_string),
                        summary: r["summary"].as_str().unwrap().to_string(),
                    });
                }
                let applied = session.apply_compression(CompressRequest {
                    ranges,
                    now: 0,
                });
                let errors: Vec<Value> = applied
                    .errors
                    .iter()
                    .map(|e| json!({ "start": e.start, "end": e.end, "kind": format!("{:?}", e.kind) }))
                    .collect();
                steps.push(json!({
                    "op": "apply",
                    "blocksCreated": applied.blocks.len(),
                    "tokensCompressed": applied.tokens_compressed,
                    "errors": errors,
                    "warnings": [],
                    "blocks": norm_blocks(&session),
                    "refs": norm_refs(&session),
                }));
            }
            "decompress" => {
                let ids: Vec<BlockId> = step["blocks"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| BlockId::new(parse_index(v.as_str().unwrap())).unwrap())
                    .collect();
                let _outcome = session.decompress(&ids);
                let first = ids
                    .first()
                    .and_then(|id| session.state().block(*id))
                    .map(norm_block);
                // `restored` is a Rust-side extension (the count of messages the
                // caller may put back in its transcript); the TS kernel has no
                // such field, so it is left out of the compared projection.
                steps.push(json!({
                    "op": "decompress",
                    "block": first,
                    "blocks": norm_blocks(&session),
                    "refs": norm_refs(&session),
                }));
            }
            "search" => {
                let query = step["query"].as_str().unwrap();
                let hits: Vec<Value> = session
                    .search(query, 10)
                    .iter()
                    .map(|h| json!({
                        "blockId": h.block.to_string(),
                        "tier": h.tier as u8,
                        "topic": h.topic,
                    }))
                    .collect();
                steps.push(json!({ "op": "search", "hits": hits }));
            }
            "status" => {
                let st = session.status();
                let used = step["tokenCount"].as_u64().unwrap_or(0) as f64;
                let limit = st.model_context_limit as f64;
                steps.push(json!({
                    "op": "status",
                    "totalBlocks": st.blocks,
                    "activeBlocks": st.active_blocks,
                    "tokensCompressed": st.tokens_compressed,
                    "contextUsage": if limit > 0.0 { used / limit } else { 0.0 },
                    "modelContextLimit": st.model_context_limit,
                }));
            }
            other => panic!("unknown op {other}"),
        }
    }

    let mut out = Map::new();
    out.insert("steps".into(), Value::Array(steps));
    std::fs::write(
        &args[2],
        serde_json::to_string_pretty(&Value::Object(out)).unwrap(),
    )
    .expect("write");
    println!("wrote {}", args[2]);
}

fn parse_index(s: &str) -> u32 {
    s.trim_start_matches('b').parse().expect("block index")
}






/// Mirrors `diff/run-ts.mts`'s injected counter bit for bit:
/// `ceil(utf16_len / 3) + sum(code points) % 11`.
struct SyntheticCounter;

impl TokenEstimator for SyntheticCounter {
    fn count(&self, text: &str) -> u32 {
        let units: usize = text.chars().map(|c| c.len_utf16()).sum();
        let sum: u32 = text.chars().map(|c| c as u32).sum();
        units.div_ceil(3) as u32 + sum % 11
    }
}


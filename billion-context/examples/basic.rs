//! End-to-end walk through the API a host actually calls.
//!
//! `cargo run --example basic`

use billion_context::{
    pipeline::Range, Boundary, CharsAsTokens, CompressRequest, Config, Engine, Message, MsgRef, Next,
    RenderStrategy, RequestedRange, Role, Session, Tier, TurnInput,
};

fn main() {
    // ---------------------------------------------------------------------
    // A small window so the demo does not need a 200k-token transcript.
    // ---------------------------------------------------------------------
    let mut config = Config::new(4_000);
    config.preserve_recent_messages = 2;

    let mut session = Session::new(config).with_estimator(CharsAsTokens);

    let history = conversation();
    println!("transcript: {} messages\n", history.len());

    // ---------------------------------------------------------------------
    // 1. Project. Every message gets a ref the model can cite.
    // ---------------------------------------------------------------------
    let turn = session.process_turn(TurnInput {
        messages: &history,
        token_count: 1_500,
        render: RenderStrategy::All,
        now: 1_000,
    });

    println!("--- projection ---");
    println!("{} messages survive the projection", turn.messages.len());
    for message in turn.messages.iter().take(3) {
        let rendered = message.content.text().unwrap_or("<non-text>");
        println!("  {} {:?}", message.id, first_line(rendered));
    }

    println!("\n--- foldable ranges ---");
    for range in &turn.ranges {
        println!("  {}  ({}, ~{} tokens)", describe(range), plural(range.count), range.tokens);
    }

    if let Some(nudge) = &turn.nudge {
        println!(
            "\n--- nudge ({}, {:.0}% of window) ---",
            nudge.reason,
            nudge.context_usage * 100.0
        );
        println!("{}", billion_context::nudge_instruction(nudge));
    }

    // ---------------------------------------------------------------------
    // 2. Fold. Two ranges, one of them wrong — and the good one still lands.
    // ---------------------------------------------------------------------
    let first = turn.ranges[0].clone();
    let applied = session.apply_compression(CompressRequest {
        ranges: vec![
            RequestedRange {
                start: Boundary::Message(first.start),
                end: Boundary::Message(first.end),
                topic: Some("build setup".into()),
                summary: "The user configured a Rust workspace and asked about the \
                          ref allocator. Decided against a runtime dependency; the \
                          kernel stays synchronous."
                    .into(),
            },
            RequestedRange {
                // Deliberately outside the view: this is what a hallucinated ref
                // looks like, and it must not cost us the range above.
                start: Boundary::Message(MsgRef::new(900).unwrap()),
                end: Boundary::Message(MsgRef::new(901).unwrap()),
                topic: None,
                summary: "A range pointing at refs that were never handed out.".into(),
            },
        ],
        now: 1_500,
    });

    println!("--- fold ---");
    println!("blocks created: {:?}", applied.blocks);
    println!("tokens folded:  {}", applied.tokens_compressed);
    println!("partial:        {}", applied.partial);
    for error in &applied.errors {
        println!("  rejected {}: {}", error.kind, error.detail);
        println!("  hint for the model: {}", error.kind.hint());
    }

    // ---------------------------------------------------------------------
    // 3. Project again. The folded messages are gone.
    // ---------------------------------------------------------------------
    let turn = session.process_turn(TurnInput {
        messages: &history,
        token_count: 900,
        render: RenderStrategy::All,
        now: 2_000,
    });
    println!("\n--- after the fold ---");
    println!(
        "{} of {} messages remain; {} refs allocated, never reused",
        turn.messages.len(),
        history.len(),
        session.status().refs_allocated
    );

    // ---------------------------------------------------------------------
    // 4. Retrieval: a summary, by ref.
    // ---------------------------------------------------------------------
    println!("\n--- retrieval ---");
    for hit in session.retrieve(&[first.start]) {
        println!("  {} [{}] {:?}", hit.block, hit.tier.tier_label(), hit.topic);
        println!("  {}", first_line(&hit.summary));
    }
    for hit in session.search("allocator", 3) {
        println!("  search hit {} (score {}): {}", hit.block, hit.score, first_line(&hit.snippet));
    }

    // ---------------------------------------------------------------------
    // 5. The host-driven loop. `begin` never awaits anything.
    // ---------------------------------------------------------------------
    println!("\n--- engine loop ---");
    let mut engine_config = Config::new(4_000);
    engine_config.preserve_recent_messages = 2;
    let mut engine = Engine::new(Session::new(engine_config).with_estimator(CharsAsTokens));

    let step = engine.begin(TurnInput {
        messages: &history,
        token_count: 3_000,
        render: RenderStrategy::All,
        now: 3_000,
    });
    println!(
        "round {}: send {} messages{}",
        engine.round(),
        step.messages.len(),
        match &step.nudge {
            Some(nudge) => format!(" with a nudge covering {} ranges", nudge.ranges.len()),
            None => " with no nudge".to_string(),
        }
    );

    if let Some(nudge) = step.nudge.clone() {
        let target = nudge.ranges[0].clone();
        let next = engine.resume(
            TurnInput {
                messages: &history,
                token_count: 3_000,
                render: RenderStrategy::All,
                now: 4_000,
            },
            &[CompressRequest {
                ranges: vec![RequestedRange {
                    start: Boundary::Message(target.start),
                    end: Boundary::Message(target.end),
                    topic: Some("bulk".into()),
                    summary: "A compact record of the first part of the conversation."
                        .into(),
                }],
                now: 4_000,
            }],
        );
        match next {
            Next::Send(step) => println!(
                "round {}: still folding, {} ranges left",
                engine.round(),
                step.nudge.as_ref().map(|n| n.ranges.len()).unwrap_or(0)
            ),
            Next::Done(turn) => {
                println!("done: {} messages in the final projection", turn.messages.len())
            }
            Next::Exhausted(_) => println!("round budget exhausted; projection is still valid"),
        }
    }

    // ---------------------------------------------------------------------
    // 6. Snapshot. Versioned, atomic, refs stored in their wire form.
    // ---------------------------------------------------------------------
    #[cfg(feature = "serde")]
    {
        println!("\n--- snapshot ---");
        let json = billion_context::persist::to_json(session.state()).expect("serializable");
        println!("{} bytes; refs are stored as {:?}", json.len(), "m00001");
        let restored = billion_context::persist::from_json(&json).expect("round-trips");
        println!(
            "restored {} block(s), {} refs",
            restored.blocks.len(),
            restored.refs.len()
        );
    }
}

fn conversation() -> Vec<Message> {
    let topics = [
        "We are porting a context-compression kernel to Rust.",
        "The engine must stay synchronous, so the host drives the loop.",
        "Refs are allocated once and never reused, even after their message dies.",
        "A stateless host resends the whole transcript every turn.",
        "That means folded originals come back under fresh ids unless we reconcile them.",
        "Token counts are snapshotted on first render so the tags never churn.",
        "Reasoning payloads are signed, so they must survive byte for byte.",
        "Batch folds report per-range outcomes rather than failing wholesale.",
    ];
    topics
        .iter()
        .enumerate()
        .map(|(index, topic)| {
            let role = if index % 2 == 0 { Role::User } else { Role::Assistant };
            Message::text(
                format!("msg-{index}"),
                role,
                format!("{topic} {}", "detail ".repeat(300)),
            )
        })
        .collect()
}

fn describe(range: &Range) -> String {
    format!("{}–{}", range.start, range.end)
}

fn plural(count: usize) -> String {
    format!("{count} msg")
}

fn first_line(text: &str) -> String {
    let line: String = text.lines().next().unwrap_or("").chars().take(72).collect();
    line
}

/// Kept so the example compiles if `Tier` loses its label helper.
trait TierLabel {
    fn tier_label(self) -> &'static str;
}

impl TierLabel for Tier {
    fn tier_label(self) -> &'static str {
        match self {
            Tier::T1 => "tier 1",
            Tier::T2 => "tier 2",
            Tier::T3 => "tier 3",
        }
    }
}

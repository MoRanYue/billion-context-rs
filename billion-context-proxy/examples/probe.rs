//! Development probe: what the kernel actually emits after a fold.
//! Run: cargo run --example probe

use billion_context::{
    Boundary, CharsPerToken4, CompressRequest, Config, Message, RenderStrategy, RequestedRange,
    Role, Session, TurnInput,
};

fn main() {
    let mut config = Config::new(8_000);
    config.preserve_recent_messages = 1;
    config.preserve_recent_tokens = 0;
    let mut session = Session::new(config).with_estimator(CharsPerToken4);

    let history = vec![
        Message::text("m0", Role::User, "z".repeat(2_000)),
        Message::text("m1", Role::Assistant, "y".repeat(2_000)),
        Message::tool_call("m2", "grep", "call-1", "{\"q\":\"alpha\"}"),
        Message::tool_result("m3", "grep", "call-1", "r".repeat(3_000)),
        Message::text("m4", Role::User, "tail question"),
    ];

    let first = session.process_turn(TurnInput {
        messages: &history,
        token_count: 7_000,
        render: RenderStrategy::All,
        now: 1,
    });
    println!("--- turn 1 messages ---");
    for m in &first.messages {
        println!(
            "id={} role={:?} content={}",
            m.id,
            m.role,
            serde_json::to_string(&m.content).unwrap()
        );
    }
    println!(
        "ranges={:?}",
        first
            .ranges
            .iter()
            .map(|r| (r.start, r.end))
            .collect::<Vec<_>>()
    );

    let range = first.ranges[0].clone();
    let applied = session.apply_compression(CompressRequest {
        call_id: None,
        ranges: vec![RequestedRange {
            start: Boundary::Message(range.start),
            end: Boundary::Message(range.end),
            topic: Some("bulk".into()),
            summary: "Two long paragraphs and a grep result about alpha.".into(),
        }],
        now: 2,
    });
    println!(
        "--- applied ok={} blocks={:?} tokens={} ---",
        applied.is_ok(),
        applied.blocks,
        applied.tokens_compressed
    );

    let second = session.process_turn(TurnInput {
        messages: &history,
        token_count: 3_000,
        render: RenderStrategy::All,
        now: 3,
    });
    println!("--- turn 2 messages ---");
    for m in &second.messages {
        println!(
            "id={} role={:?} summary_of={:?} content={}",
            m.id,
            m.role,
            m.summary_of_block,
            serde_json::to_string(&m.content).unwrap()
        );
    }
    println!("--- status ---");
    println!("{:?}", session.status());
    println!("--- retrieve ---");
    println!("{:?}", session.retrieve(&[range.start]));
}

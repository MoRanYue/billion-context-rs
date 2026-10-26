//! The extension points a third-party host uses to plug in its own tokenizer.
//!
//! These need no vocabulary, so they run in every build. `tests/hf_tokenizer.rs`
//! covers the ready-made exact estimator, which is feature-gated and needs a
//! multi-megabyte `tokenizer.json`.

use std::sync::Arc;

use billion_context::{
    CharsAsTokens, CharsPerToken4, Config, FnEstimator, Message, RenderStrategy, Role, Session,
    TokenEstimator, TurnInput,
};

/// The shape a config-driven host actually holds: one boxed estimator chosen at
/// startup. Without the `Arc` impl this would not compile, because
/// `with_estimator` takes `impl TokenEstimator`.
#[test]
fn a_shared_estimator_can_be_injected() {
    let shared: Arc<dyn TokenEstimator> = Arc::new(CharsAsTokens);
    let mut session = Session::new(Config::new(1_000)).with_estimator(shared.clone());

    assert_eq!(shared.count("abcd"), 4);
    let turn = session.process_turn(TurnInput {
        messages: &[Message::text("m1", Role::User, "hi")],
        token_count: 2,
        render: RenderStrategy::None,
        now: 1,
    });
    assert_eq!(turn.messages.len(), 1);
}

#[test]
fn a_boxed_estimator_can_be_injected() {
    let boxed: Box<dyn TokenEstimator> = Box::new(CharsPerToken4);
    assert!(boxed.count("hello") >= 1);
    let _ = Session::new(Config::new(1_000)).with_estimator(boxed);
}

#[test]
fn a_closure_can_be_an_estimator() {
    let session =
        Session::new(Config::new(1_000)).with_estimator(FnEstimator(|t: &str| t.len() as u32 / 4));
    assert_eq!(session.status().model_context_limit, 1_000);
}

/// A host with its own stack implements the trait directly and never touches
/// the `hf-tokenizer` feature.
#[test]
fn a_third_party_estimator_drives_the_engine() {
    struct BytesDiv4;
    impl TokenEstimator for BytesDiv4 {
        fn count(&self, text: &str) -> u32 {
            (text.len() as u32).div_ceil(4)
        }
    }

    let mut session = Session::new(Config::new(200_000)).with_estimator(BytesDiv4);
    let history: Vec<Message> = (0..4)
        .map(|i| Message::text(format!("m{i}"), Role::User, "x".repeat(400)))
        .collect();

    let turn = session.process_turn(TurnInput {
        messages: &history,
        token_count: 400,
        render: RenderStrategy::None,
        now: 1,
    });
    assert_eq!(turn.messages.len(), 4);
    assert_eq!(session.status().refs_allocated, 4);
}

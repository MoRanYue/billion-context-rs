//! `billion-context` — a Rust port of billion-context's `acp-kernel`.
//!
//! The engine folds an over-long conversation into cited summaries so a host can
//! keep talking to a model past its context window. It is **pure and
//! synchronous**: it never performs I/O, never calls a model, never reads the
//! clock. Everything it cannot own is injected — a [`TokenEstimator`], a
//! [`Policy`], and an explicit `now`.
//!
//! # The four design rules
//!
//! 1. **The engine is pure.** No async, no callbacks, no runtime. A host that
//!    already has a tokio runtime does not have to hand it over.
//! 2. **`Engine`/`Session` split.** One session per conversation, `&mut` to
//!    change it. Cross-thread misuse becomes a compile error instead of a
//!    corrupted ref map.
//! 3. **Batch operations report partial success.** [`Session::apply_compression`]
//!    returns per-range outcomes, not `Result<_, E>`: one stale ref must never
//!    discard nine good ranges.
//! 4. **Refs are the wire contract.** `m00001` / `b1` are allocated once and
//!    never reused. They are the only identifiers a model may cite.
//!
//! # The whole loop
//!
//! ```
//! use billion_context::{Config, Message, RenderStrategy, Role, Session, TurnInput};
//!
//! let mut session = Session::new(Config::new(200_000));
//! let history = vec![
//!     Message::text("a", Role::User, "how does the ref allocator work?"),
//!     Message::text("b", Role::Assistant, "it is append-only, so ..."),
//! ];
//!
//! let turn = session.process_turn(TurnInput {
//!     messages: &history,
//!     token_count: 42,
//!     render: RenderStrategy::All,
//!     now: 1,
//! });
//!
//! // Each message now carries the ref the model is expected to cite.
//! assert!(turn.messages[0].content.text().unwrap().starts_with("<acp tokens="));
//! assert_eq!(session.status().refs_allocated, 2);
//! ```
//!
//! # Where the host plugs in
//!
//! | The kernel needs | The host supplies |
//! |---|---|
//! | how big a string is | [`TokenEstimator`] |
//! | what must never be folded | [`Policy`] |
//! | the current time | `now` on every call |
//! | the conversation size | `token_count` from the last response's usage |
//! | where snapshots go | [`persist`] takes a path; nothing else touches disk |
//!
//! # What is deliberately absent
//!
//! No protocol structs (see the sibling `billion-context-wire` crate), no tokenizer, no
//! model client, no storage of folded message bodies. The last one is not an
//! oversight: the kernel keeps summaries and refs, and a host that wants
//! `acp_retrieve` to return *original bytes* keeps those itself. Returning a
//! summary is the contract ([`Retrieval`]).

#![forbid(unsafe_code)]

pub mod config;
pub mod engine;
pub mod error;
pub mod ids;
pub mod message;
pub mod persist;
pub mod pipeline;
pub mod recommend;
pub mod session;
pub mod state;
#[cfg(feature = "hf-tokenizer")]
pub mod tokenizer;

pub use config::{
    AbsorbConfig, CcrConfig, CharsAsTokens, CharsPerToken4, CompressConfig, Config, ConfigError,
    CrushConfig, DefaultPolicy, FnEstimator, ListPolicy, NudgeConfig, Policy, TierConfig,
    TokenEstimator, TruncateConfig,
};
pub use engine::{engine, nudge_instruction, Engine, Next, Step};
pub use error::{RangeError, RangeErrorKind};
pub use ids::{BlockId, Boundary, MessageId, MsgRef, Timestamp};
pub use message::{Content, ContentKind, Message, Role};
pub use pipeline::{
    default_pipeline, pipeline_names, Effects, Node, NodeIo, Nudge, NudgeBreakdown, PipelineContext,
    Range, RenderStrategy, TerminalEscape, ViewMode,
};
pub use session::{
    Applied, CompressRequest, DecompressOutcome, Hit, RequestedRange, Retrieval, Session, Status,
    Turn, TurnInput,
};
pub use state::{Block, Generation, NudgeState, RefEntry, RefMap, State, Stats, Tier};
#[cfg(feature = "hf-tokenizer")]
pub use tokenizer::{HfTokenizer, TokenizerError};

/// The crate's own version, for snapshot compatibility checks.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod integration_tests {
    use super::*;

    /// The end-to-end shape a host actually runs: project, nudge, fold, project
    /// again, and confirm the window shrank without losing the refs.
    #[test]
    fn full_cycle_compresses_a_long_conversation() {
        let mut config = Config::new(8_000);
        config.preserve_recent_messages = 2;
        // The token-shaped guard would protect the whole tail here, since five
        // 1.2k messages already exceed it; this test is about the fold, not it.
        config.preserve_recent_tokens = 0;
        let mut session = Session::new(config).with_estimator(CharsAsTokens);

        let history: Vec<Message> = (0..8)
            .map(|i| {
                let role = if i % 2 == 0 { Role::User } else { Role::Assistant };
                Message::text(format!("x{i}"), role, "z".repeat(1_200))
            })
            .collect();

        let first = session.process_turn(TurnInput {
            messages: &history,
            token_count: 2_000,
            render: RenderStrategy::None,
            now: 1,
        });
        assert!(!first.ranges.is_empty(), "eight 1.2k messages must be foldable");

        let range = first.ranges[0].clone();
        let applied = session.apply_compression(CompressRequest {
            ranges: vec![RequestedRange {
                start: Boundary::Message(range.start),
                end: Boundary::Message(range.end),
                topic: Some("bulk".into()),
                summary: "Eight short exchanges about nothing in particular.".into(),
            }],
            now: 2,
        });
        assert!(applied.is_ok());
        assert!(applied.tokens_compressed > 0);

        let second = session.process_turn(TurnInput {
            messages: &history,
            token_count: 2_000,
            render: RenderStrategy::None,
            now: 3,
        });
        assert!(
            second.messages.len() < history.len(),
            "the projection must be shorter than the transcript"
        );

        // The folded originals are still retrievable as a summary, by ref.
        let found = session.retrieve(&[range.start]);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].topic.as_deref(), Some("bulk"));

        // And the ref map is intact after the fold.
        assert_eq!(session.status().refs_allocated, 8);
        assert_eq!(session.status().active_blocks, 1);
    }

    /// A stateless host resends the whole transcript every turn. Folded
    /// originals come back under ids the block does not cover; the content
    /// index has to map them back or the fold silently does nothing.
    #[test]
    fn resent_originals_are_reconciled_not_double_counted() {
        let mut config = Config::new(8_000);
        config.preserve_recent_messages = 1;
        let mut session = Session::new(config).with_estimator(CharsAsTokens);

        let history = vec![
            Message::text("first", Role::User, "a".repeat(3_000)),
            Message::text("second", Role::Assistant, "b".repeat(3_000)),
            Message::text("third", Role::User, "tail"),
        ];
        let _ = session.process_turn(TurnInput {
            messages: &history,
            token_count: 500,
            render: RenderStrategy::None,
            now: 1,
        });
        let applied = session.apply_compression(CompressRequest {
            ranges: vec![RequestedRange {
                start: Boundary::Message(MsgRef::new(1).unwrap()),
                end: Boundary::Message(MsgRef::new(2).unwrap()),
                topic: None,
                summary: "The user opened with two long paragraphs about the port.".into(),
            }],
            now: 2,
        });
        assert!(applied.is_ok(), "{:?}", applied.errors);

        // The host now resends the same two messages under fresh ids.
        let resent = vec![
            Message::text("fresh-a", Role::User, "a".repeat(3_000)),
            Message::text("fresh-b", Role::Assistant, "b".repeat(3_000)),
            Message::text("third", Role::User, "tail"),
        ];
        let turn = session.process_turn(TurnInput {
            messages: &resent,
            token_count: 500,
            render: RenderStrategy::None,
            now: 3,
        });
        assert_eq!(
            turn.messages.len(),
            3,
            "the resent originals are reconciled, not re-admitted"
        );
        assert_eq!(
            turn.messages[0].id.as_str(),
            "first",
            "reconcile restores the original id, and the pin then keeps it"
        );
        assert!(
            turn.messages.iter().all(|m| m.id.as_str() != "second"),
            "the resent assistant original must be pruned"
        );
        assert!(!turn.warnings.is_empty(), "the remap should be reported");
    }

    /// The safety valve: when the window is full and nothing is foldable, the
    /// engine drops the oldest ref-bearing messages rather than letting the
    /// request fail.
    #[test]
    fn emergency_truncate_fires_when_over_the_line() {
        let mut config = Config::new(1_000);
        config.truncate.threshold = 0.9;
        config.preserve_recent_messages = 1;
        let mut session = Session::new(config).with_estimator(CharsAsTokens);

        let history: Vec<Message> = (0..6)
            .map(|i| Message::text(format!("t{i}"), Role::User, "w".repeat(400)))
            .collect();

        let turn = session.process_turn(TurnInput {
            messages: &history,
            token_count: 980, // 98% of a 1k window
            render: RenderStrategy::None,
            now: 1,
        });
        assert!(
            turn.messages.len() < history.len(),
            "over the threshold, the oldest messages must go"
        );
    }

    /// The kernel must never rewrite a provider's signed reasoning payload:
    /// reconstructing one invalidates the conversation.
    #[test]
    fn reasoning_payloads_are_left_byte_exact() {
        use std::sync::Arc;
        let mut session = Session::new(Config::new(200_000)).with_estimator(CharsAsTokens);
        let payload = "{\"signature\":\"abc123\",\"text\":\"thinking...\"}";
        let history = vec![
            Message::new("r1", Role::Assistant, Content::Reasoning(Arc::from(payload))),
            Message::text("u1", Role::User, "next"),
        ];
        let turn = session.process_turn(TurnInput {
            messages: &history,
            token_count: 10,
            render: RenderStrategy::All,
            now: 1,
        });
        match &turn.messages[0].content {
            Content::Reasoning(text) => assert!(text.contains(payload)),
            other => panic!("reasoning must survive as reasoning, got {other:?}"),
        }
    }
}

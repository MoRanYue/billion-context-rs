//! The multi-round host loop, as a resumable state machine.
//!
//! Upstream this is `runCompressLoop`: an `async` function that awaits the model
//! and the tool executor between rounds. Rust cannot await a callback it does
//! not own, and a `Future`-based engine would drag the host's runtime into the
//! kernel. So the loop is inverted: the host drives, and the engine only says
//! what to do next.
//!
//! ```text
//! begin ──► Send ──► host calls the model
//!                     │
//!        ┌────────────┴────────────┐
//!   plain answer            compress call
//!        │                         │
//!     finish                  resume ──► Send (again) ──► …
//!                                  └──► Done / Exhausted
//! ```
//!
//! The round cap is not a nicety. A model that keeps asking for folds it cannot
//! place will loop forever otherwise, and every round is a real API call.

use crate::config::Config;
use crate::pipeline::Nudge;
use crate::session::{Applied, CompressRequest, Session, Turn, TurnInput};

/// What the host should do with the step it was handed.
#[derive(Debug)]
pub struct Step {
    /// Send exactly this. Ref tags are already applied.
    pub messages: Vec<crate::message::Message>,
    /// Present when this step follows a compress attempt, so the host can write
    /// the tool result the model will see — including the per-range errors.
    pub applied: Option<Applied>,
    /// The nudge text that was prepended, if any. The host may prefer to render
    /// it itself and call `process_turn` directly.
    pub nudge: Option<Nudge>,
}

#[derive(Debug)]
pub enum Next {
    /// More work: send this step and come back with the reply.
    Send(Box<Step>),
    /// The engine has nothing left to ask for.
    Done(Turn),
    /// The round budget ran out mid-loop. The projection is still valid.
    Exhausted(Turn),
}

pub struct Engine {
    session: Session,
    round: usize,
    max_rounds: usize,
}

impl Engine {
    pub fn new(session: Session) -> Self {
        Self {
            session,
            round: 0,
            max_rounds: 10,
        }
    }

    pub fn with_max_rounds(mut self, rounds: usize) -> Self {
        self.max_rounds = rounds.max(1);
        self
    }

    pub fn session(&self) -> &Session {
        &self.session
    }

    pub fn session_mut(&mut self) -> &mut Session {
        &mut self.session
    }

    pub fn round(&self) -> usize {
        self.round
    }

    /// First projection of the turn. Always yields something to send: the model
    /// may simply answer, in which case the host calls [`Engine::finish`].
    pub fn begin(&mut self, input: TurnInput<'_>) -> Step {
        let turn = self.session.process_turn(input);
        self.round = 1;
        self.step(turn, None)
    }

    /// The model asked to fold. Apply what it asked for, then re-project.
    pub fn resume(&mut self, input: TurnInput<'_>, requests: &[CompressRequest]) -> Next {
        let mut applied: Option<Applied> = None;
        for request in requests {
            let outcome = self.session.apply_compression(request.clone());
            applied = Some(match applied.take() {
                Some(previous) => Applied {
                    blocks: [previous.blocks, outcome.blocks].concat(),
                    errors: [previous.errors, outcome.errors].concat(),
                    tokens_compressed: previous
                        .tokens_compressed
                        .saturating_add(outcome.tokens_compressed),
                    partial: previous.partial || outcome.partial,
                },
                None => outcome,
            });
        }

        let turn = self.session.process_turn(input);
        if self.round >= self.max_rounds {
            return Next::Exhausted(turn);
        }
        self.round += 1;

        let step = self.step(turn, applied);
        // The loop continues only on a decision to *act*. A decision that
        // reports "nothing worth folding yet" still carries its ranges, so the
        // host can show them — but asking the model again would burn a call it
        // has no reason to make.
        let inject = step
            .nudge
            .as_ref()
            .map(|nudge| nudge.should_inject)
            .unwrap_or(false);
        if inject {
            return Next::Send(Box::new(step));
        }
        Next::Done(Turn {
            messages: step.messages,
            ranges: Vec::new(),
            nudge: step.nudge,
            terminal: None,
            truncation_skipped: None,
            warnings: Vec::new(),
        })
    }

    fn step(&self, turn: Turn, applied: Option<Applied>) -> Step {
        let mut messages = turn.messages;
        let nudge = turn.nudge.clone();
        // A decision is always available now, so the instruction is gated on the
        // decision itself — not on the decision being present.
        if let Some(ref nudge) = nudge {
            if nudge.should_inject {
                let instruction = nudge_instruction(nudge);
                messages.insert(
                    0,
                    crate::message::Message::text(
                        "acp_nudge",
                        crate::message::Role::System,
                        instruction,
                    ),
                );
            }
        }
        Step {
            messages,
            applied,
            nudge,
        }
    }
}

/// The text the model is shown when the engine wants it to fold something.
///
/// Written as a peer instruction rather than a policy statement: models comply
/// far more reliably with "these ranges are large, fold them" than with a wall
/// of rules, and the ranges are concrete refs the model can cite verbatim.
pub fn nudge_instruction(nudge: &Nudge) -> String {
    let mut out = String::new();
    out.push_str("## Context notice\n\n");
    out.push_str(&format!(
        "This conversation occupies {:.0}% of the context window.\n",
        nudge.context_usage * 100.0
    ));
    match nudge.reason.as_str() {
        "emergency" => out.push_str(
            "This is urgent: the window is nearly full and the oldest messages \
             will start being dropped.\n",
        ),
        "over_band" => out.push_str("The conversation is now above the comfortable range.\n"),
        _ => out.push_str("The conversation has grown since the last fold.\n"),
    }
    out.push_str("\nFold the following ranges with the `compress` tool:\n\n");
    for range in &nudge.ranges {
        out.push_str(&format!(
            "- `{}`–`{}` ({} messages, ~{} tokens)\n",
            range.start, range.end, range.count, range.tokens
        ));
    }
    out.push_str(
        "\nWrite a summary that preserves decisions, constraints, exact values \
         and anything the conversation still depends on. Omit pleasantries.\n",
    );
    out
}

/// Convenience: a session with a validated config, or the errors that stopped it.
pub fn engine(config: Config) -> Result<Engine, Vec<crate::config::ConfigError>> {
    Ok(Engine::new(Session::try_new(config)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CharsAsTokens;
    use crate::ids::{Boundary, MsgRef};
    use crate::message::{Message, Role};
    use crate::pipeline::RenderStrategy;
    use crate::session::RequestedRange;
    use crate::state::Tier;

    fn engine_with(nudge_at: u32) -> Engine {
        let mut config = Config::new(nudge_at);
        config.preserve_recent_messages = 2;
        Engine::new(Session::new(config).with_estimator(CharsAsTokens))
    }

    fn conversation(n: usize) -> Vec<Message> {
        (0..n)
            .map(|i| {
                let role = if i % 2 == 0 { Role::User } else { Role::Assistant };
                Message::text(format!("m{i}"), role, "y".repeat(6_000))
            })
            .collect()
    }

    #[test]
    fn begin_always_returns_a_step() {
        let mut e = engine_with(100_000);
        let msgs = conversation(4);
        let step = e.begin(TurnInput {
            messages: &msgs,
            token_count: 10,
            render: RenderStrategy::None,
            now: 1,
        });
        let ids: Vec<&str> = step.messages.iter().map(|m| m.id.as_str()).collect();
        eprintln!("IDS {ids:?}");
        assert_eq!(step.messages.len(), 4);
        assert!(step.applied.is_none());
        assert_eq!(e.round(), 1);
    }

    #[test]
    fn nudge_fires_on_growth_and_carries_ranges() {
        let mut e = engine_with(100_000);
        // The nudge has two gates: enough growth *and* enough content to fold.
        // 12 messages put 60k tokens in the foldable pool, past the 50k floor.
        let msgs = conversation(12);
        let step = e.begin(TurnInput {
            messages: &msgs,
            token_count: 60_000,
            render: RenderStrategy::None,
            now: 1,
        });
        let nudge = step.nudge.expect("growth of 60k over a 100k window must nudge");
        assert_eq!(nudge.reason, "growth");
        assert!(!nudge.ranges.is_empty());
        // The instruction is prepended as its own message, not merged into a
        // user turn: it must not look like the user said it.
        assert_eq!(step.messages[0].id.as_str(), "acp_nudge");
    }

    #[test]
    fn resume_folds_and_returns_to_done() {
        let mut e = engine_with(100_000);
        let msgs = conversation(6);
        let step = e.begin(TurnInput {
            messages: &msgs,
            token_count: 60_000,
            render: RenderStrategy::None,
            now: 1,
        });
        let range = step.nudge.unwrap().ranges[0].clone();

        let next = e.resume(
            TurnInput {
                messages: &msgs,
                token_count: 60_000,
                render: RenderStrategy::None,
                now: 2,
            },
            &[CompressRequest {
                ranges: vec![RequestedRange {
                    start: Boundary::Message(range.start),
                    end: Boundary::Message(range.end),
                    topic: Some("bulk".into()),
                    summary: "A summary long enough to pass the minimum length check.".into(),
                }],
                now: 2,
            }],
        );
        match next {
            Next::Done(turn) => assert!(turn.messages.len() < 6),
            Next::Send(step) => panic!("expected Done, got another Send: {:?}", step.nudge),
            Next::Exhausted(_) => panic!("round budget should not have run out"),
        }
    }

    #[test]
    fn round_cap_terminates_a_stubborn_model() {
        let mut config = Config::new(100_000);
        config.preserve_recent_messages = 2;
        config.nudge.growth_floor = 0; // nudge every single round
        let mut e = Engine::new(Session::new(config).with_estimator(CharsAsTokens)).with_max_rounds(3);
        let msgs = conversation(8);

        let _ = e.begin(TurnInput {
            messages: &msgs,
            token_count: 60_000,
            render: RenderStrategy::None,
            now: 1,
        });

        let mut rounds = 0;
        loop {
            let next = e.resume(
                TurnInput {
                    messages: &msgs,
                    token_count: 60_000,
                    render: RenderStrategy::None,
                    now: 2,
                },
                &[],
            );
            rounds += 1;
            if matches!(next, Next::Exhausted(_)) {
                break;
            }
            assert!(rounds < 10, "the cap must stop this");
        }
        assert_eq!(rounds, 3);
    }

    #[test]
    fn invalid_config_is_rejected() {
        let mut config = Config::new(100_000);
        config.nudge.max_context_limit_pct = 1.5;
        assert!(engine(config).is_err());
    }

    #[test]
    fn nudge_instruction_names_the_ranges() {
        let nudge = Nudge {
            should_inject: true,
            reason: "over_band".into(),
            tier: Some(Tier::T2),
            context_usage: 0.81,
            ranges: vec![crate::pipeline::Range {
                start: MsgRef::new(1).unwrap(),
                end: MsgRef::new(4).unwrap(),
                count: 4,
                tokens: 12_000,
                chars: 40_000,
            }],
            breakdown: Default::default(),
        };
        let text = nudge_instruction(&nudge);
        assert!(text.contains("m00001"));
        assert!(text.contains("81%"));
        assert!(text.contains("compress"));
    }
}

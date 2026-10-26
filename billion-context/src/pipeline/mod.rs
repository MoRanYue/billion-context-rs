//! The projection pipeline.
//!
//! Each pass is a node; the **order is the contract**. Splitting the pipeline
//! into named nodes is not decoration — it makes reordering show up as an
//! explicit diff, and it lets a test assert the order independently of the
//! behaviour.
//!
//! Nodes operate on borrowed state rather than taking ownership of a snapshot.
//! The TS kernel threads a `NodeIO` object through and clones nothing; the Rust
//! version gets the same property from `&mut` with no runtime cost.

pub mod nodes;

use crate::config::{Config, Policy, TokenEstimator};
use crate::ids::{MsgRef, Timestamp};
use crate::message::Message;
use crate::state::{State, Tier};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum RenderStrategy {
    /// Tag every message that has a ref.
    #[default]
    All,
    /// Tag only plain text, leaving reasoning payloads untouched.
    TextOnly,
    /// Emit no tags at all (useful for dry runs and for hosts that render refs
    /// themselves).
    None,
}

/// Who performs the surface replacement a fold implies.
///
/// A block covers a span of refs, but a block is not a message: *something* has
/// to turn "refs m00001..m00004 are now block b1" into a message array. Either
/// the kernel does it, or the host does. Both are legitimate, and they produce
/// different transcripts, so the choice is explicit rather than implicit.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum ViewMode {
    /// The kernel hands back a view that can be sent as-is: covered messages are
    /// dropped and replaced by one carrier message holding the block's summary.
    #[default]
    Send,
    /// The kernel leaves the transcript untouched and reports coverage only
    /// through `blocks`; the host performs the replacement itself. This is the
    /// model the TS kernel uses, and the only mode in which the returned view
    /// keeps the folded originals.
    Host,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Range {
    pub start: MsgRef,
    pub end: MsgRef,
    pub count: usize,
    pub tokens: u32,
    pub chars: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NudgeBreakdown {
    pub usage: f32,
    pub growth: u32,
    pub threshold: u32,
    pub pending_t1: u32,
    pub pending_t2: u32,
    pub pending_t3: u32,
}

#[derive(Clone, Debug)]
pub struct Nudge {
    pub should_inject: bool,
    /// `growth` | `over_band` | `emergency`. Stable machine-readable reasons.
    pub reason: String,
    pub tier: Option<Tier>,
    pub context_usage: f32,
    pub ranges: Vec<Range>,
    pub breakdown: NudgeBreakdown,
}

#[derive(Clone, Debug)]
pub struct TerminalEscape {
    pub message: String,
    pub usage: f32,
    pub token_count: u32,
    pub model_context_limit: u32,
}

/// Everything a pass may tell the host. Kept separate from `State` because none
/// of it is persistent: it describes *this* request, not the conversation.
#[derive(Default, Debug)]
pub struct Effects {
    pub ranges: Vec<Range>,
    pub nudge: Option<Nudge>,
    pub terminal: Option<TerminalEscape>,
    pub truncation_skipped: Option<String>,
    pub warnings: Vec<String>,
}

pub struct NodeIo<'a> {
    pub messages: &'a mut Vec<Message>,
    pub state: &'a mut State,
    pub effects: &'a mut Effects,
}

pub struct PipelineContext<'a> {
    pub config: &'a Config,
    pub policy: &'a dyn Policy,
    pub estimator: &'a dyn TokenEstimator,
    /// What the host already knows from the previous response's usage. The
    /// kernel never re-counts the whole conversation.
    pub token_count: u32,
    pub render: RenderStrategy,
    /// Whether the kernel or the host owns the surface replacement a fold
    /// implies. See [`ViewMode`].
    pub view: ViewMode,
    /// Whether a message that matches a folded original *by content* should be
    /// re-identified as that original. Only meaningful for a host that resends
    /// its whole transcript every turn; a surface-replacing host has nothing to
    /// reconcile.
    pub reconcile: bool,
    pub now: Timestamp,
}

pub trait Node: Send + Sync {
    fn name(&self) -> &'static str;

    /// A disabled node is skipped entirely. Disabled-by-default passes keep the
    /// pipeline order faithful without pretending to run.
    fn enabled(&self, _config: &Config) -> bool {
        true
    }

    fn run(&self, io: &mut NodeIo<'_>, ctx: &PipelineContext<'_>);
}

pub fn run(pipeline: &[Box<dyn Node>], io: &mut NodeIo<'_>, ctx: &PipelineContext<'_>) {
    for node in pipeline {
        if node.enabled(ctx.config) {
            node.run(io, ctx);
        }
    }
}

pub fn default_pipeline() -> Vec<Box<dyn Node>> {
    nodes::default_pipeline()
}

/// The node order, for assertions. Changing this vector is a behaviour change.
pub fn pipeline_names() -> Vec<&'static str> {
    default_pipeline().iter().map(|n| n.name()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_order_is_pinned() {
        assert_eq!(
            pipeline_names(),
            vec![
                "reconcile-live-ids",
                "assign-refs",
                "sync-blocks",
                "prune",
                "ccr-store",
                "absorb-hide",
                "crush",
                "absorb-prompt",
                "filter",
                "hide-compress-calls",
                "recommend",
                "nudge-inject",
                "emergency-truncate",
                "render-refs",
            ]
        );
    }
}

//! The fourteen passes.
//!
//! Ten of them are real. Four are deliberately empty, and that is a finding
//! rather than an omission:
//!
//! `ccr-store`, `absorb-hide`, `crush` and `absorb-prompt` all need either a
//! model call (to summarise) or a content store. In the TS kernel they live in
//! the pipeline because the kernel was allowed to reach the host for those
//! capabilities. This kernel is pure and synchronous, so those four belong to
//! the host layer: a `crush` pass is "ask the host to re-summarise this block",
//! which is exactly the `compress` tool call the model already makes. Keeping
//! them here as disabled nodes preserves the order without faking the work.
//!
//! They are all `enabled: false` in the reference config, so behaviour is
//! identical to upstream.

use std::collections::{HashSet, HashMap};
use std::sync::Arc;

use super::{Effects, Node, NodeIo, PipelineContext, Range, RenderStrategy};
use crate::config::Config;
use crate::message::{Content, ContentKind, Message, Role};
use crate::state::{Generation, RefEntry};

// ---------------------------------------------------------------------------
// 1. reconcile-live-ids
// ---------------------------------------------------------------------------

/// A stateless host resends the whole conversation every turn. Once the head is
/// folded, those originals come back with ids the block does not cover, so
/// pruning misses them and the fold appears to have done nothing. Matching on a
/// content fingerprint maps the resend back to the id the block already owns.
pub struct ReconcileLiveIds;

impl Node for ReconcileLiveIds {
    fn name(&self) -> &'static str {
        "reconcile-live-ids"
    }

    fn run(&self, io: &mut NodeIo<'_>, ctx: &PipelineContext<'_>) {
        // A surface-replacing host never resends a folded original, so there is
        // nothing to reconcile — and remapping by content would be actively
        // wrong there, since two identical live messages are its problem, not
        // the kernel's.
        if !ctx.reconcile {
            return;
        }
        let index = &io.state.content_index;
        let mut remapped = 0usize;
        for msg in io.messages.iter_mut() {
            if let Some(known) = index.get(&msg.fingerprint()) {
                if known != &msg.id {
                    msg.id = known.clone();
                    remapped += 1;
                }
            }
        }
        if remapped > 0 {
            io.effects.warnings.push(format!(
                "{remapped} resent message(s) matched a folded original by content"
            ));
        }
    }
}

// ---------------------------------------------------------------------------
// 2. assign-refs
// ---------------------------------------------------------------------------

/// The single writer of the ref map. Allocation is append-only: a number handed
/// out once is never reused, even after its message dies.
pub struct AssignRefs;

impl Node for AssignRefs {
    fn name(&self) -> &'static str {
        "assign-refs"
    }

    fn run(&self, io: &mut NodeIo<'_>, ctx: &PipelineContext<'_>) {
        for msg in io.messages.iter() {
            // Ephemeral host injections must not consume a number.
            if msg.id.as_str().starts_with("acp_retrieved_") {
                continue;
            }
            let protected = ctx.policy.is_message_protected(msg);
            match io.state.refs.get(&msg.id) {
                Some(RefEntry::Assigned(_)) => continue,
                Some(RefEntry::Blocked) if protected => continue,
                Some(RefEntry::Blocked) => {
                    // Protection lifted: migrate to a fresh number.
                    io.state.refs.allocate(&msg.id);
                }
                None if protected => {
                    io.state.refs.mark_blocked(msg.id.clone());
                }
                None => {
                    let Some(r) = io.state.refs.allocate(&msg.id) else {
                        io.effects
                            .warnings
                            .push("ref capacity exhausted; message left unreferenced".into());
                        continue;
                    };
                    let sized = msg
                        .content
                        .foldable_text()
                        .or_else(|| msg.content.tool_arguments());
                    let tokens = sized.map(|t| ctx.estimator.count(t)).unwrap_or(0);
                    // Written once, read forever: re-estimating would churn the
                    // rendered tag bytes and bust the provider prefix cache.
                    io.state.token_snapshot.entry(r).or_insert(tokens);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 3. sync-blocks
// ---------------------------------------------------------------------------

pub struct SyncBlocks;

impl Node for SyncBlocks {
    fn name(&self) -> &'static str {
        "sync-blocks"
    }

    fn run(&self, io: &mut NodeIo<'_>, ctx: &PipelineContext<'_>) {
        let threshold = ctx.config.promotion_threshold;
        // Base ids: a block recorded whichever projection of an original was in
        // the view when it folded, and a host that re-splits its surface may
        // emit a sibling this turn. The original is still there.
        let present: HashSet<&str> = io.messages.iter().map(|m| m.id.base()).collect();
        for block in io.state.blocks.iter_mut().filter(|b| b.active) {
            // A block whose covered messages are no longer in the transcript
            // stands in for nothing: the host replaced the surface, so the
            // block is history rather than a live summary. Retiring it here is
            // what keeps a stale carrier out of the next request.
            if !block.direct.iter().any(|id| present.contains(id.base())) {
                block.active = false;
                continue;
            }
            block.survived += 1;
            if block.survived >= threshold {
                block.generation = Generation::Old;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 4. prune
// ---------------------------------------------------------------------------

pub struct Prune;

/// Header every rendered summary starts with. Part of the wire format: hosts and
/// downstream passes recognise a summary message by it.
pub const SUMMARY_HEADER: &str = "[Compressed conversation section]";

/// The view-only id of an active block's summary message. Hosts must never mint
/// an id with this prefix; it belongs to the kernel.
pub fn summary_message_id(block: &crate::ids::BlockId) -> String {
    format!("acp_summary_{block}")
}

/// Render a block as the message that stands in for the span it covers.
fn summary_message(block: &crate::state::Block) -> Message {
    let body = block.summary.trim();
    let topic_line = match block.topic.as_deref() {
        Some(topic) if !topic.is_empty() => format!("{SUMMARY_HEADER} — {topic}"),
        _ => SUMMARY_HEADER.to_string(),
    };
    let text = if body.is_empty() {
        topic_line
    } else {
        format!("{topic_line}\n{body}")
    };
    let mut message = Message::new(summary_message_id(&block.id), Role::System, Content::Text(text.into()));
    message.summary_of_block = Some(block.id);
    message
}

impl Node for Prune {
    fn name(&self) -> &'static str {
        "prune"
    }

    fn run(&self, io: &mut NodeIo<'_>, ctx: &PipelineContext<'_>) {
        // In host mode the transcript is the host's to rewrite: the block list
        // already says which refs are covered, and dropping them here would
        // destroy the very messages the host needs in order to do it.
        if ctx.view == super::ViewMode::Host {
            return;
        }
        // Keyed by base id: a host that split one provider message into
        // `base#c0`, `base#r0` recorded whichever projection existed at fold
        // time, and the next turn may emit a different one. Comparing verbatim
        // would leave the sibling in the view as an unfolded original.
        let covered: HashSet<String> = io
            .state
            .covered_ids()
            .into_iter()
            .map(|id| id.base().to_owned())
            .collect();
        if covered.is_empty() {
            return;
        }

        // The session's opening user message is pinned: it survives prune even
        // when a block covers it. A provider rejects a conversation with no user
        // message at all, and a fold that swallows the opening turn can leave
        // exactly that.
        let first_user = io.messages.iter().position(|m| m.role == Role::User);

        // Folded content is indexed here, at the moment it disappears. Indexing
        // every live message instead would conflate two messages that happen to
        // carry identical text.
        for (i, m) in io.messages.iter().enumerate() {
            if Some(i) == first_user || !covered.contains(m.id.base()) {
                continue;
            }
            io.state.content_index.insert(m.fingerprint(), m.id.clone());
        }

        // Which block owns each covered original, the carrier for each block,
        // and where that carrier goes: at the head of the span it replaces.
        let mut owner: HashMap<String, crate::ids::BlockId> = HashMap::new();
        let mut order: Vec<crate::ids::BlockId> = Vec::new();
        let mut carriers: HashMap<crate::ids::BlockId, Message> = HashMap::new();
        for block in io.state.blocks.iter().filter(|b| b.active) {
            order.push(block.id);
            carriers.insert(block.id, summary_message(block));
            for id in block.direct.iter().chain(block.effective.iter()) {
                owner.entry(id.base().to_owned()).or_insert(block.id);
            }
        }

        let mut anchor: HashMap<crate::ids::BlockId, usize> = HashMap::new();
        for (i, m) in io.messages.iter().enumerate() {
            if Some(i) == first_user || !covered.contains(m.id.base()) {
                continue;
            }
            if let Some(block) = owner.get(m.id.base()) {
                anchor.entry(*block).or_insert(i);
            }
        }

        let mut pending: Vec<(usize, crate::ids::BlockId)> =
            anchor.iter().map(|(b, i)| (*i, *b)).collect();
        pending.sort_by_key(|(i, _)| *i);

        // Replace each span with its carrier at the head of the span. Dropping
        // the span outright would leave a hole the model cannot account for.
        let mut out: Vec<Message> = Vec::with_capacity(io.messages.len());
        let mut next = 0;
        for (i, message) in io.messages.iter().enumerate() {
            while next < pending.len() && pending[next].0 == i {
                if let Some(carrier) = carriers.get(&pending[next].1) {
                    out.push(carrier.clone());
                }
                next += 1;
            }
            if Some(i) == first_user || !covered.contains(message.id.base()) {
                out.push(message.clone());
            }
        }
        while next < pending.len() {
            if let Some(carrier) = carriers.get(&pending[next].1) {
                out.push(carrier.clone());
            }
            next += 1;
        }

        // A block whose span is no longer in the transcript still has to appear:
        // its summary is the only surviving record of that content.
        for block in &order {
            if !anchor.contains_key(block) {
                if let Some(carrier) = carriers.get(block) {
                    out.push(carrier.clone());
                }
            }
        }

        *io.messages = out;

        // Folding removed messages; removing a message can remove half of a pair
        // the wire format needs whole. Widening keeps pairs inside one range, but
        // it cannot see a block-boundary fold or a transcript that arrived
        // inconsistent, so the view is repaired before it is handed back.
        let policy = ctx.policy;
        let before = io.messages.len();
        *io.messages = crate::orphans::repair(
            std::mem::take(io.messages),
            &|name: &str| policy.is_compress_tool(name),
        );
        let repaired = before - io.messages.len();
        if repaired > 0 {
            io.effects.warnings.push(format!(
                "prune dropped {repaired} message(s) whose pair was folded away; \
                 the request would have been rejected without this repair"
            ));
        }

        // Keep the fingerprint index tight: only still-folded content can be
        // resent as an original.
        io.state
            .content_index
            .retain(|_, id| covered.contains(id.base()));
    }
}

// ---------------------------------------------------------------------------
// 5-8. host-layer passes (disabled in the reference config)
// ---------------------------------------------------------------------------

macro_rules! host_layer_node {
    ($ty:ident, $name:literal, $flag:ident) => {
        pub struct $ty;

        impl Node for $ty {
            fn name(&self) -> &'static str {
                $name
            }

            fn enabled(&self, config: &Config) -> bool {
                config.$flag.enabled
            }

            fn run(&self, _io: &mut NodeIo<'_>, _ctx: &PipelineContext<'_>) {
                // Needs a model call or a content store, both of which live
                // outside a pure kernel. See the module docs.
            }
        }
    };
}

host_layer_node!(CcrStore, "ccr-store", ccr);
host_layer_node!(AbsorbHide, "absorb-hide", absorb);
host_layer_node!(Crush, "crush", crush);
host_layer_node!(AbsorbPrompt, "absorb-prompt", absorb);

// ---------------------------------------------------------------------------
// 9. filter
// ---------------------------------------------------------------------------

/// Drops whitespace-only text messages. They cost tokens, carry no meaning, and
/// make ranges harder to reason about.
pub struct Filter;

impl Node for Filter {
    fn name(&self) -> &'static str {
        "filter"
    }

    fn run(&self, io: &mut NodeIo<'_>, _ctx: &PipelineContext<'_>) {
        io.messages
            .retain(|m| !(m.content.kind() == ContentKind::Text && m.content.is_empty_text()));
    }
}

// ---------------------------------------------------------------------------
// 10. hide-compress-calls
// ---------------------------------------------------------------------------

/// The model does not need to re-read its own compress exchanges forever. The
/// **most recent** one stays visible: that is where the error text lives, and
/// the model needs it to correct its next attempt.
pub struct HideCompressCalls;

impl Node for HideCompressCalls {
    fn name(&self) -> &'static str {
        "hide-compress-calls"
    }

    fn run(&self, io: &mut NodeIo<'_>, ctx: &PipelineContext<'_>) {
        let Some(last) = io.messages.iter().rposition(|m| {
            matches!(&m.content, Content::ToolCall { name, .. } if ctx.policy.is_compress_tool(name))
        }) else {
            return;
        };

        let mut doomed: HashSet<Arc<str>> = HashSet::new();
        for msg in io.messages.iter().take(last) {
            if let Content::ToolCall { name, call_id, .. } = &msg.content {
                if ctx.policy.is_compress_tool(name) {
                    doomed.insert(call_id.clone());
                }
            }
        }
        if doomed.is_empty() {
            return;
        }

        let policy = ctx.policy;
        io.messages.retain(|m| match &m.content {
            Content::ToolCall { name, call_id, .. } => {
                !(policy.is_compress_tool(name) && doomed.contains(call_id))
            }
            Content::ToolResult { call_id, .. } => !doomed.contains(call_id),
            _ => true,
        });
    }
}

// ---------------------------------------------------------------------------
// 11. recommend
// ---------------------------------------------------------------------------

/// What is foldable right now. A range never starts or ends inside a tool call:
/// cutting a call away from its result leaves the model with an orphan.
pub struct Recommend;

/// One message the fold may take.
#[derive(Clone)]
struct Info {
    id: crate::ids::MessageId,
    msg_ref: crate::ids::MsgRef,
    tokens: u32,
    chars: usize,
    is_user: bool,
    /// True when the message carries tool content (call or result). The split is
    /// reported to the model because "43K of tool output" and "43K of user
    /// requirements" call for opposite decisions at the same size.
    is_tool: bool,
    index: usize,
    gap_before: bool,
}

/// One contiguous run of foldable messages, before threshold batching.
#[derive(Clone)]
struct Segment {
    start: crate::ids::MsgRef,
    end: crate::ids::MsgRef,
    count: usize,
    tokens: u32,
    chars: usize,
    tool_tokens: u32,
    text_tokens: u32,
    user_msgs: usize,
    start_index: usize,
    end_index: usize,
}

impl Segment {
    fn single(info: &Info) -> Self {
        Segment {
            start: info.msg_ref,
            end: info.msg_ref,
            count: 1,
            tokens: info.tokens,
            chars: info.chars,
            tool_tokens: if info.is_tool { info.tokens } else { 0 },
            text_tokens: if info.is_tool { 0 } else { info.tokens },
            user_msgs: usize::from(info.is_user),
            start_index: info.index,
            end_index: info.index,
        }
    }

    fn absorb(&mut self, other: &Segment) {
        self.end = other.end;
        self.end_index = other.end_index;
        self.count += other.count;
        self.tokens += other.tokens;
        self.chars += other.chars;
        self.tool_tokens += other.tool_tokens;
        self.text_tokens += other.text_tokens;
        self.user_msgs += other.user_msgs;
    }

    fn to_range(&self) -> Range {
        Range {
            start: self.start,
            end: self.end,
            count: self.count,
            tokens: self.tokens,
            chars: self.chars,
            tool_tokens: self.tool_tokens,
            text_tokens: self.text_tokens,
            user_msgs: self.user_msgs,
        }
    }
}

impl Node for Recommend {
    fn name(&self) -> &'static str {
        "recommend"
    }

    fn run(&self, io: &mut NodeIo<'_>, ctx: &PipelineContext<'_>) {
        let cfg = ctx.config;
        let protected = crate::recommend::protected_refs(io.messages, io.state, cfg);

        // 1. Classify. Anything the fold cannot take is a hole: it closes the
        //    running group, so no range ever spans content the fold refuses.
        let mut compressible: Vec<Info> = Vec::new();
        let mut gap = false;
        for (index, msg) in io.messages.iter().enumerate() {
            let Some(msg_ref) = io.state.refs.ref_of(&msg.id) else {
                gap = true;
                continue;
            };
            if io.state.refs.is_blocked(&msg.id)
                || io.state.is_covered(&msg.id)
                || crate::recommend::is_summary_carrier(msg)
            {
                gap = true;
                continue;
            }
            if protected.contains(&msg_ref) {
                gap = true;
                continue;
            }
            // Media the host cannot restore. A fold rewrites text and the summary
            // is text, so folding an image-bearing message keeps the prose and
            // loses the image, with nothing behind `decompress` to bring it back.
            // This is a hole, not a protected message: a protected one merely
            // stays out of the range, a hole also stops the range spanning it.
            //
            // Advisory only, on purpose. `apply_compression` does not reject a
            // range that spans a hole: a mandatory guard would pin every image in
            // the transcript for the whole session, and a multi-image workload
            // fills the window faster than the guard saves it. So this node never
            // *suggests* such a range, and a caller that names one is assumed to
            // have decided the image is expendable.
            if msg.has_unrecoverable_payload() {
                gap = true;
                continue;
            }
            // A tool call has no prose a summary may rewrite, but it is foldable
            // *content* all the same: the reference advertises every unprotected
            // message and lets the pairing invariant keep each call with its
            // result. Making a call a hole instead is what silently broke tool
            // folding — `integrity_withdrawals` then sees both halves of every
            // pair as split apart, withdraws every tool result, and a tool-heavy
            // transcript reports almost nothing compressible even though it is
            // exactly the transcript that most needs folding.
            //
            // Its arguments stand in for the text: they are the bytes the
            // provider bills for, and the only content a call carries.
            let Some(sized) = msg
                .content
                .foldable_text()
                .or_else(|| msg.content.tool_arguments())
            else {
                gap = true;
                continue;
            };
            compressible.push(Info {
                id: msg.id.clone(),
                msg_ref,
                // The snapshot keeps a rendered session's numbers stable across
                // turns; without rendering there is nothing to keep stable, so
                // fall back to measuring the message itself.
                tokens: io
                    .state
                    .token_snapshot
                    .get(&msg_ref)
                    .copied()
                    .unwrap_or_else(|| ctx.estimator.count(sized)),
                chars: sized.chars().count(),
                is_user: msg.role == Role::User,
                is_tool: matches!(
                    msg.content,
                    Content::ToolCall { .. } | Content::ToolResult { .. }
                ),
                index,
                gap_before: gap,
            });
            gap = false;
        }

        // 2. Withdraw what the fold cannot take whole: a turn whose reasoning
        //    would fold while one of its calls survives, or a split call/result
        //    pair. Each withdrawn message becomes a hole for the same reason —
        //    otherwise the range would be advertised and then refused.
        let folded: HashSet<crate::ids::MessageId> =
            compressible.iter().map(|i| i.id.clone()).collect();
        let withdrawn = crate::recommend::integrity_withdrawals(io.messages, &folded);
        if !withdrawn.is_empty() {
            let mut kept: Vec<Info> = Vec::with_capacity(compressible.len());
            let mut pending_gap = false;
            for mut info in compressible {
                if withdrawn.contains(&info.id) {
                    pending_gap = true;
                    continue;
                }
                if pending_gap {
                    info.gap_before = true;
                    pending_gap = false;
                }
                kept.push(info);
            }
            compressible = kept;
        }

        // 3. Segment at holes and, once a group already holds three messages,
        //    at user turns — which keeps each range aligned to about one turn.
        let mut segments: Vec<Segment> = Vec::new();
        let mut cur: Option<Segment> = None;
        for info in compressible {
            let split = match &cur {
                None => false,
                Some(open) => info.gap_before || (info.is_user && open.count >= 3),
            };
            if split {
                if let Some(open) = cur.take() {
                    segments.push(open);
                }
            }
            match cur.as_mut() {
                None => cur = Some(Segment::single(&info)),
                Some(open) => open.absorb(&Segment::single(&info)),
            }
        }
        if let Some(open) = cur {
            segments.push(open);
        }

        // 4. Batch to the size gate. The gate counts real characters, so a
        //    segment measured in tokens can pass a check the fold will then
        //    reject. Every emitted batch must clear the threshold on its own; a
        //    sub-threshold tail folds into the batch before it, and a leading
        //    sub-threshold remainder is dropped — offering it could only produce
        //    a call that is guaranteed to be refused.
        let min_chars = cfg.compress.min_compress_range;
        let mut ranges: Vec<Range> = Vec::new();
        if min_chars == 0 {
            ranges = segments.iter().map(Segment::to_range).collect();
        } else {
            let mut batch: Option<Segment> = None;
            let mut batch_chars = 0usize;
            // Where the last emitted range ended, so the tail can tell whether
            // it is adjacent to it or separated by a hole.
            let mut last_emitted_end: Option<usize> = None;

            for segment in segments {
                // An index gap between the previous segment and this one is a
                // block, protected-zone, or pruned boundary. Never bridge it,
                // even when the batch is still short: the emitted span would
                // cover content the fold must not take.
                let bridges = batch
                    .as_ref()
                    .is_some_and(|open| segment.start_index > open.end_index + 1);
                if bridges {
                    if let Some(done) = batch.take() {
                        if batch_chars >= min_chars {
                            last_emitted_end = Some(done.end_index);
                            ranges.push(done.to_range());
                        }
                        batch_chars = 0;
                    }
                }

                match batch.as_mut() {
                    None => {
                        batch_chars = segment.chars;
                        batch = Some(segment);
                    }
                    Some(open) => {
                        open.absorb(&segment);
                        batch_chars += segment.chars;
                    }
                }
                if batch_chars >= min_chars {
                    if let Some(done) = batch.take() {
                        last_emitted_end = Some(done.end_index);
                        ranges.push(done.to_range());
                    }
                    batch_chars = 0;
                }
            }

            if let Some(tail) = batch {
                let contiguous = last_emitted_end.is_some_and(|end| tail.start_index <= end + 1);
                match ranges.last_mut() {
                    // A sub-threshold tail folds into the batch before it when
                    // it sits right next to it; emitting it alone would only
                    // produce a call the apply side rejects.
                    Some(last) if contiguous => {
                        last.end = tail.end;
                        last.count += tail.count;
                        last.tokens += tail.tokens;
                        last.chars += tail.chars;
                        last.tool_tokens += tail.tool_tokens;
                        last.text_tokens += tail.text_tokens;
                        last.user_msgs += tail.user_msgs;
                    }
                    _ if batch_chars >= min_chars => ranges.push(tail.to_range()),
                    // Nothing precedes it, so the whole remainder is below the
                    // gate and no selection of it can pass.
                    _ => {}
                }
            }
        }

        // 5. Screen. A unit whose span no longer owns a foldable message would
        //    be refused by `apply_compression`, and advertising it starts an
        //    advertise/reject loop that never terminates.
        ranges.retain(|r| crate::recommend::span_owns_foldable(io.state, r.start, r.end));

        io.effects.ranges = ranges;
    }
}

// ---------------------------------------------------------------------------
// 12. nudge-inject
// ---------------------------------------------------------------------------

/// The active blocks at `tier`, in fold order. These are what folding one tier
/// higher would absorb, and their summaries are the mass that tier reclaims.
fn tier_blocks(state: &crate::state::State, tier: u8) -> Vec<crate::state::Block> {
    state
        .blocks
        .iter()
        .filter(|b| b.active && b.tier as u8 == tier)
        .cloned()
        .collect()
}

/// Decides whether to tell the model to compress, and at which urgency. The
/// cadence state lives in `State` so it survives across requests and resets
/// when a fold actually lands.
pub struct NudgeInject;

impl Node for NudgeInject {
    fn name(&self) -> &'static str {
        "nudge-inject"
    }

    fn run(&self, io: &mut NodeIo<'_>, ctx: &PipelineContext<'_>) {
        let cfg = ctx.config;
        let usage = if cfg.model_context_limit == 0 {
            0.0
        } else {
            ctx.token_count as f32 / cfg.model_context_limit as f32
        };
        // Read the stamps as they stood BEFORE this turn re-anchors them: the
        // first-sight bypass below is defined in terms of "no baseline yet".
        let baseline = io.state.nudge.baseline_tokens;
        let last_shown = io.state.nudge.last_shown_tokens;

        // Growth is measured from the last time the model was actually told to
        // compress, falling back to the baseline and then to "just now" (which
        // makes the first turn report zero growth rather than its whole size).
        let reference = if last_shown > 0 {
            last_shown
        } else if baseline > 0 {
            baseline
        } else {
            ctx.token_count
        };
        let growth = ctx.token_count.saturating_sub(reference);
        // T1 is what could be folded right now — only ranges big enough that
        // the apply-side gate would not reject them. T2/T3 are computed further
        // down from the block table, once the tier thresholds exist.
        let min_chars = cfg.compress.min_compress_range;
        let pending_t1: u32 = io
            .effects
            .ranges
            .iter()
            .filter(|r| min_chars == 0 || r.chars >= min_chars)
            .map(|r| r.tokens)
            .sum();

        // The growth step, derived from the context limit so one config behaves
        // the same on a 32K model and a 1M one instead of drifting with absolute
        // token counts.
        let nudge_growth_tokens = if cfg.model_context_limit == 0 {
            cfg.nudge.growth_floor
        } else {
            let scaled =
                (cfg.model_context_limit as f32 * cfg.nudge.growth_ratio).round() as u32;
            scaled
                .max(cfg.nudge.growth_floor)
                .min(cfg.nudge.growth_cap)
        };
        let growth_floor = (cfg.nudge.min_growth_ratio * nudge_growth_tokens as f32)
            .round()
            .max(cfg.nudge.min_growth_floor as f32) as u32;
        let min_pressure_benefit = cfg.nudge.min_pressure_benefit_tokens.unwrap_or_else(|| {
            ((cfg.model_context_limit as f32 * 0.01).round() as u32).max(5_000)
        });

        // Per-tier mass. A tier-2 nudge is not "more messages to fold": it is
        // "fold these tier-1 blocks", and a tier-3 nudge folds tier-2 blocks.
        // Both compare summary mass, never message counts — a summary is ~10:1
        // condensed, so measuring it against raw pending starves the upper
        // tiers forever.
        let t1_blocks = tier_blocks(io.state, 1);
        let t2_blocks = tier_blocks(io.state, 2);
        let pending_t2: u32 = t1_blocks
            .iter()
            .map(|b| ctx.estimator.count(&b.summary))
            .sum();
        let pending_t3: u32 = t2_blocks
            .iter()
            .map(|b| ctx.estimator.count(&b.summary))
            .sum();
        let t2_count = t1_blocks.len() as u32;
        let t3_count = t2_blocks.len() as u32;
        // The best any tier can offer: the number the decision compares.
        let pending = pending_t1.max(pending_t2).max(pending_t3);

        let over_limit = usage >= cfg.nudge.max_context_limit_pct;
        let emergency = usage >= cfg.nudge.emergency_threshold_pct;
        // The pressure band bypasses the growth gate and the per-tier cadence:
        // when the window is this full, waiting for another growth step is not
        // an option. It is still gated on the benefit floor below.
        let pressure = over_limit || emergency;

        // While a nudge is still unanswered the bar halves; otherwise a session
        // that is already above the limit has to grow a whole step before being
        // told again.
        let effective_threshold = if last_shown > 0 {
            nudge_growth_tokens / 2
        } else {
            nudge_growth_tokens
        };

        // A session can arrive already full: a stateless host replaying a whole
        // history, or a restored state. Growth then reads as zero and the floor
        // would hold that backlog hostage until a full interval of NEW tokens
        // arrived. The bypass drains the first backlog instead; it re-arms after
        // every fold, and the shown stamp keeps it from nagging a session that
        // ignored the last nudge.
        let first_sight_mass_ready = last_shown == 0
            && baseline == 0
            && usage >= cfg.nudge.min_context_limit_pct
            && pending >= nudge_growth_tokens;
        let growth_ready = first_sight_mass_ready || growth >= growth_floor;

        // Tier thresholds: T1 defaults to the growth step, T2/T3 to that step
        // scaled by `tier2_growth_multiplier`. Each can be pinned individually
        // through `tier_growth_tokens`; an unset one keeps its derived default,
        // so an absent struct decides byte-identically.
        let shared_tier_threshold =
            (nudge_growth_tokens as f32 * cfg.nudge.tier2_growth_multiplier).round() as u32;
        let pinned = cfg.nudge.tier_growth_tokens.as_ref();
        let tier_thresholds = [
            pinned.and_then(|t| t.t1).unwrap_or(nudge_growth_tokens),
            pinned.and_then(|t| t.t2).unwrap_or(shared_tier_threshold),
            pinned.and_then(|t| t.t3).unwrap_or(shared_tier_threshold),
        ];
        let t2_pinned = pinned.and_then(|t| t.t2).is_some();
        let t3_pinned = pinned.and_then(|t| t.t3).is_some();

        // Block COUNT on its own is not a need signal — condensed summaries can
        // stack thousands of blocks that reclaim almost nothing, and each
        // distillation rewrites the wire from the fold anchor onward (prefix
        // cache loss). The count path therefore ships effectively off
        // (1000 / 2000) and is honoured only when explicitly configured.
        let t2_count_ready = t2_count >= cfg.tiers.tier2_trigger;
        let t3_count_ready = t3_count >= cfg.tiers.tier3_trigger;
        let tiers_enabled = cfg.tiers.enabled;

        let usage_pct = (usage * 100.0).round() as i64;
        let mut injected: Option<u8> = None;
        let mut best_pending = 0u32;
        let mut reason = String::new();

        if pressure {
            // Under pressure, route to whichever tier reclaims the most; the
            // only thing that stops it is having nothing worth reclaiming.
            let candidates: &[u8] = if tiers_enabled { &[1, 2, 3] } else { &[1] };
            for &t in candidates {
                let p = match t {
                    1 => pending_t1,
                    2 => pending_t2,
                    _ => pending_t3,
                };
                if p > best_pending {
                    best_pending = p;
                    injected = Some(t);
                }
            }
            // A sub-threshold pending can never look actionable however hot the
            // band is, otherwise every "success" resets the baselines and the
            // same nudge re-injects each turn.
            if best_pending < min_pressure_benefit {
                injected = None;
            }
            if let Some(b) = injected {
                let label = if emergency { "EMERGENCY" } else { "OVER-LIMIT" };
                reason = if b == 1 {
                    format!(
                        "{label} T1: max effective pending {best_pending}, usage {usage_pct}%"
                    )
                } else {
                    format!(
                        "{label} T{b} distill: max pending {best_pending} (T1 effective {pending_t1}, T2 {pending_t2}, T3 {pending_t3}), usage {usage_pct}%"
                    )
                };
            }
        } else if growth_ready {
            if pending_t1 >= tier_thresholds[0] {
                injected = Some(1);
                reason = format!(
                    "T1 effective {pending_t1} >= {}, growth {growth}, usage {usage_pct}%",
                    tier_thresholds[0]
                );
            } else if tiers_enabled
                && (t2_count_ready || (pending_t2 >= tier_thresholds[1] && pending_t2 > pending_t1))
            {
                let last = io.state.nudge.shown_by_tier.get(&2).copied().unwrap_or(0);
                if last == 0 || ctx.token_count.saturating_sub(last) >= growth_floor {
                    injected = Some(2);
                    reason = if t2_count_ready {
                        format!(
                            "T2 distill ready: {t2_count} tier-1 blocks >= tier2Trigger {} ({pending_t2} tokens), usage {usage_pct}%",
                            cfg.tiers.tier2_trigger
                        )
                    } else {
                        let suffix = if t2_pinned { "" } else { " (1.5x)" };
                        format!(
                            "T2 distill ready: {t2_count} tier-1 blocks ({pending_t2} tokens) >= {}{suffix} and > T1 effective {pending_t1}, usage {usage_pct}%",
                            tier_thresholds[1]
                        )
                    };
                }
            } else if tiers_enabled
                && (t3_count_ready
                    || (pending_t3 >= tier_thresholds[2]
                        && pending_t3 > pending_t2
                        && pending_t3 > pending_t1))
            {
                let last = io.state.nudge.shown_by_tier.get(&3).copied().unwrap_or(0);
                if last == 0 || ctx.token_count.saturating_sub(last) >= growth_floor {
                    injected = Some(3);
                    reason = if t3_count_ready {
                        format!(
                            "T3 condense ready: {t3_count} tier-2 blocks >= tier3Trigger {} ({pending_t3} tokens), usage {usage_pct}%",
                            cfg.tiers.tier3_trigger
                        )
                    } else {
                        let suffix = if t3_pinned { "" } else { " (1.5x)" };
                        format!(
                            "T3 condense ready: {t3_count} tier-2 blocks ({pending_t3} tokens) >= {}{suffix} and > T2 {pending_t2} and > T1 effective {pending_t1}, usage {usage_pct}%",
                            tier_thresholds[2]
                        )
                    };
                }
            }
        }

        let should_inject = injected.is_some();
        if should_inject && first_sight_mass_ready {
            reason.push_str(" [first-sight mass]");
        }

        // The decision is always returned, not just when it fires: a host that
        // only learns about a *yes* cannot show the user "there is 40K to fold
        // but we are waiting", and cannot explain why nothing happened.
        if injected.is_none() {
            if pressure {
                let label = if emergency { "EMERGENCY" } else { "OVER-LIMIT" };
                reason = if pending == 0 {
                    format!(
                        "{label}: usage {usage_pct}% but no tier has effective compressible content (T1 effective {pending_t1}, T2 {pending_t2}, T3 {pending_t3}) — nudge suppressed to avoid offering ranges below minCompressRange"
                    )
                } else {
                    format!(
                        "{label}: usage {usage_pct}% but max pending {pending} < min benefit {min_pressure_benefit} tokens (T1 effective {pending_t1}, T2 {pending_t2}, T3 {pending_t3}) — suppressed: rewriting below the benefit floor reclaims almost nothing while usage stays high; truncate.threshold remains the safety valve"
                    )
                };
            } else {
                let eligible: Vec<u8> = if tiers_enabled { vec![1, 2, 3] } else { vec![1] };
                let mass_of = |t: u8| match t {
                    1 => pending_t1,
                    2 => pending_t2,
                    _ => pending_t3,
                };
                let count_ready = |t: u8| match t {
                    2 => t2_count_ready,
                    3 => t3_count_ready,
                    _ => false,
                };
                let shown = |t: u8| io.state.nudge.shown_by_tier.get(&t).copied().unwrap_or(0);
                let ready: Vec<String> = eligible
                    .iter()
                    .filter(|&&t| mass_of(t) >= tier_thresholds[(t - 1) as usize])
                    .map(|&t| format!("T{t} {}", mass_of(t)))
                    .collect();
                let ready_count: Vec<String> = eligible
                    .iter()
                    .filter(|&&t| mass_of(t) < tier_thresholds[(t - 1) as usize] && count_ready(t))
                    .map(|&t| {
                        format!(
                            "T{t} {} blocks (count)",
                            if t == 2 { t2_count } else { t3_count }
                        )
                    })
                    .collect();
                let ready_all: Vec<String> = ready.into_iter().chain(ready_count).collect();
                let ready_hint = if ready_all.is_empty() {
                    String::new()
                } else {
                    format!(", ready: {}", ready_all.join(", "))
                };
                let blocked: Vec<String> = eligible
                    .iter()
                    .filter(|&&t| {
                        (mass_of(t) >= tier_thresholds[(t - 1) as usize] || count_ready(t))
                            && shown(t) > 0
                            && ctx.token_count.saturating_sub(shown(t)) < growth_floor
                    })
                    .map(|&t| format!("T{t} (cadence)"))
                    .collect();
                let blocked_hint = if blocked.is_empty() {
                    String::new()
                } else {
                    format!(", blocked: {}", blocked.join(", "))
                };
                // The floor below which NO tier can fire by mass.
                let min_tier_threshold = tier_thresholds[0]
                    .min(tier_thresholds[1])
                    .min(tier_thresholds[2]);
                let mut parts: Vec<String> = Vec::new();
                if pending < min_tier_threshold {
                    parts.push(format!(
                        "max compressible {pending} < threshold {min_tier_threshold}"
                    ));
                }
                if growth < growth_floor {
                    parts.push(format!("growth {growth} < floor {growth_floor}"));
                }
                if parts.is_empty() {
                    parts.push(format!("max compressible {pending}, growth {growth}"));
                }
                reason = format!("{}{ready_hint}{blocked_hint}", parts.join("; "));
            }
        }
        let tier = injected.map(|t| match t {
            1 => crate::state::Tier::T1,
            2 => crate::state::Tier::T2,
            _ => crate::state::Tier::T3,
        });

        // Every nudge reference is a low-water mark: it rides DOWN with the
        // observed count but never up, so growth always measures the climb from
        // the most recent valley and a transient spike cannot dead-zone it.
        if io.state.nudge.baseline_tokens == 0
            || ctx.token_count < io.state.nudge.baseline_tokens
        {
            io.state.nudge.baseline_tokens = ctx.token_count;
        }
        if io.state.nudge.last_shown_tokens > 0
            && ctx.token_count < io.state.nudge.last_shown_tokens
        {
            io.state.nudge.last_shown_tokens = ctx.token_count;
        }

        if should_inject {
            io.state.nudge.baseline_tokens = ctx.token_count;
            io.state.nudge.last_shown_tokens = ctx.token_count;
            if let Some(t) = tier {
                io.state.nudge.shown_by_tier.insert(t as u8, ctx.token_count);
            }
        }

        // The blocks the chosen tier would fold. Empty for T1, whose work is
        // `ranges` instead — a host that receives a T2/T3 nudge names these
        // block ids, and the kernel absorbs them into one block a tier up.
        let tier_target_blocks: Vec<crate::ids::BlockId> = match injected {
            Some(2) => t1_blocks.iter().map(|b| b.id).collect(),
            Some(3) => t2_blocks.iter().map(|b| b.id).collect(),
            _ => Vec::new(),
        };

        io.effects.nudge = Some(super::Nudge {
            should_inject,
            reason,
            tier,
            context_usage: usage,
            ranges: io.effects.ranges.clone(),
            tier_target_blocks,
            breakdown: super::NudgeBreakdown {
                usage,
                growth,
                threshold: nudge_growth_tokens,
                pending_t1,
                pending_t2,
                pending_t3,
                max_pending: pending,
                nudge_growth_tokens,
                growth_floor,
                growth_reference: reference,
                effective_threshold,
                min_pressure_benefit,
                over_limit,
                emergency,
            },
            token_count: ctx.token_count,
            model_context_limit: cfg.model_context_limit,
        });
    }
}

// ---------------------------------------------------------------------------
// 13. emergency-truncate
// ---------------------------------------------------------------------------

/// The last safety valve: clip oversized payloads, never delete messages.
///
/// Deleting was what this used to do, oldest first. It destroyed the task
/// statement first, could cut a tool call away from its result, and was
/// unrecoverable — so a transient estimate spike cost permanent history. Clipping
/// keeps every message, keeps the order, and leaves both ends of the payload
/// visible. See [`crate::truncate`].
///
/// A latch replays the previous turn's clips. Without it, a host that resends the
/// original bytes (or a render that reads them from the transcript) would
/// re-inflate everything the last turn freed, and the session would oscillate
/// against its own relief valve every single turn.
pub struct EmergencyTruncate;

impl Node for EmergencyTruncate {
    fn name(&self) -> &'static str {
        "emergency-truncate"
    }

    fn run(&self, io: &mut NodeIo<'_>, ctx: &PipelineContext<'_>) {
        use crate::state::{TruncateLatch, MAX_TRUNCATE_LATCH};

        let cfg = ctx.config;
        if cfg.model_context_limit == 0 {
            return;
        }

        // Replay first, before the watermark check. A resend that reinstated the
        // full payload must be clipped back even if the estimate happens to sit
        // below the threshold this turn — otherwise the bytes come back the moment
        // the number dips, and the next turn pays for them again.
        if !io.state.nudge.truncate_latch.is_empty() {
            let latched = std::mem::take(&mut io.state.nudge.truncate_latch);
            let mut keep = Vec::with_capacity(latched.len());
            for latch in latched {
                let Some(id) = io.state.refs.raw_of(latch.r).cloned() else {
                    continue;
                };
                let Some(index) = io.messages.iter().position(|m| m.id == id) else {
                    continue;
                };
                let current = io.messages[index].content.body().unwrap_or_default();
                if current.len() <= latch.text.len() {
                    // Same or shorter than what we clipped to: nothing to forgive.
                    continue;
                }
                io.messages[index] = io.messages[index].clone().with_body(&latch.text);
                keep.push(latch);
            }
            io.state.nudge.truncate_latch = keep;
        }

        let usage = ctx.token_count as f32 / cfg.model_context_limit as f32;
        if usage < cfg.truncate.threshold {
            io.state.nudge.consecutive_truncations = 0;
            io.state.nudge.terminal_streak = 0;
            return;
        }

        let counter = ctx.estimator;
        let estimate = |text: &str| counter.count(text);
        let options = crate::truncate::TruncateOptions {
            protect_recent_messages: cfg.preserve_recent_messages,
            ..crate::truncate::TruncateOptions::default()
        };
        let protected = |m: &Message| ctx.policy.is_message_protected(m);
        let outcome = crate::truncate::truncate_large_payloads(
            io.messages,
            ctx.token_count,
            cfg,
            &estimate,
            &options,
            &protected,
        );
        let saved = outcome.saved_tokens;

        for (id, text) in &outcome.replacements {
            let Some(r) = io.state.refs.ref_of(id) else {
                continue;
            };
            io.state.nudge.truncate_latch.push(TruncateLatch {
                r,
                text: text.clone(),
            });
        }
        if io.state.nudge.truncate_latch.len() > MAX_TRUNCATE_LATCH {
            let excess = io.state.nudge.truncate_latch.len() - MAX_TRUNCATE_LATCH;
            io.state.nudge.truncate_latch.drain(..excess);
        }
        if !outcome.replacements.is_empty() {
            *io.messages = outcome.messages;
        }

        // "Nothing to free" has two very different causes, and only one of them is
        // terminal. No foldable range at all means the conversation is structurally
        // uncompressible; a fold that exists but freed nothing means the estimator
        // and the payloads disagree. The second self-heals, the first does not.
        let no_viable_fold = io
            .effects
            .nudge
            .as_ref()
            .map(|n| n.breakdown.max_pending == 0 || n.ranges.is_empty())
            .unwrap_or(true);
        let stuck = no_viable_fold && saved == 0;

        io.state.nudge.consecutive_truncations = if saved > 0 {
            0
        } else {
            io.state.nudge.consecutive_truncations.saturating_add(1)
        };
        io.state.nudge.terminal_streak = if stuck {
            io.state.nudge.terminal_streak.saturating_add(1)
        } else {
            0
        };

        if io.state.nudge.terminal_streak >= cfg.truncate.terminal_escape_after {
            io.state.nudge.consecutive_truncations = 0;
            io.effects.terminal = Some(super::TerminalEscape {
                message: format!(
                    "context is still at {:.0}% after {} truncation attempt(s) freed nothing; \
                     start a new conversation or raise the context limit",
                    usage * 100.0,
                    io.state.nudge.terminal_streak
                ),
                usage,
                token_count: ctx.token_count,
                model_context_limit: cfg.model_context_limit,
                stuck_events: io.state.nudge.terminal_streak,
            });
            return;
        }

        if saved == 0 {
            io.effects.truncation_skipped = Some(format!(
                "over the truncation watermark but nothing could be clipped: \
                 {} candidate payload(s) found",
                outcome.candidates_found
            ));
        }
    }
}

// ---------------------------------------------------------------------------
// 14. render-refs
// ---------------------------------------------------------------------------

/// Annotates each message with the ref the model may cite. The token count
/// comes from the write-once snapshot, so the rendered bytes are stable across
/// turns — which is the whole point of the snapshot.
pub struct RenderRefs;

impl Node for RenderRefs {
    fn name(&self) -> &'static str {
        "render-refs"
    }

    fn run(&self, io: &mut NodeIo<'_>, ctx: &PipelineContext<'_>) {
        if matches!(ctx.render, RenderStrategy::None) {
            return;
        }
        for i in 0..io.messages.len() {
            let id = io.messages[i].id.clone();
            let Some(r) = io.state.refs.ref_of(&id) else {
                continue;
            };
            let kind = io.messages[i].content.kind();
            if matches!(ctx.render, RenderStrategy::TextOnly) && kind != ContentKind::Text {
                continue;
            }
            let raw = io.messages[i].content.payload();
            let clean = strip_own_tag(raw, &r.to_string());

            // The count is written on first render and reused forever. A tag
            // that carried a live count would change every turn, rewriting
            // bytes that are already in the provider's prefix cache — the exact
            // cost this kernel exists to avoid.
            let counted = match io.state.token_snapshot.get(&r).copied() {
                Some(t) => t,
                None => {
                    let t = ctx.estimator.count(clean);
                    io.state.token_snapshot.insert(r, t);
                    t
                }
            };
            let tokens = counted + io.messages[i].thinking_tokens;
            let tag = format!(
                "<acp tokens=\"{}\" type=\"{}\">{}</acp>\n",
                format_tokens(tokens),
                classify_type(&io.messages[i].content),
                r
            );
            let content = io.messages[i].content.clone();
            io.messages[i].content = content.with_body(clean).with_prefix(&tag);
        }
    }
}

/// The three-band rule the TS kernel uses. These digits ride in the rendered
/// tag, so they are wire format: a different rounding changes the bytes the
/// model reads.
fn format_tokens(tokens: u32) -> String {
    if tokens < 1_000 {
        tokens.to_string()
    } else if tokens < 10_000 {
        format!("{:.1}K", f64::from(tokens) / 1_000.0)
    } else {
        format!("{}K", (f64::from(tokens) / 1_000.0).round() as u32)
    }
}

/// A tool call is identified by its name, not by the word "tool-call": the model
/// needs `type="grep"` to know what it is looking at.
fn classify_type(content: &Content) -> &str {
    match content {
        Content::ToolCall { .. } | Content::ToolResult { .. } => {
            content.tool_name().unwrap_or("tool")
        }
        other => other.kind().as_str(),
    }
}

/// Strip a tag this message already carries **for its own ref**, so rendering is
/// idempotent. A tag naming a different ref is left alone — that is somebody
/// else's content and rewriting it would corrupt what the model was told.
fn strip_own_tag<'a>(text: &'a str, r: &str) -> &'a str {
    let Some(rest) = text.strip_prefix("<acp ") else {
        return text;
    };
    let Some(gt) = rest.find('>') else {
        return text;
    };
    let Some(after) = rest[gt + 1..].strip_prefix(r) else {
        return text;
    };
    let Some(after) = after.strip_prefix("</acp>") else {
        return text;
    };
    after.strip_prefix('\n').unwrap_or(after)
}

// ---------------------------------------------------------------------------

pub fn default_pipeline() -> Vec<Box<dyn Node>> {
    vec![
        Box::new(ReconcileLiveIds),
        Box::new(AssignRefs),
        Box::new(SyncBlocks),
        Box::new(Prune),
        Box::new(CcrStore),
        Box::new(AbsorbHide),
        Box::new(Crush),
        Box::new(AbsorbPrompt),
        Box::new(Filter),
        Box::new(HideCompressCalls),
        Box::new(Recommend),
        Box::new(NudgeInject),
        Box::new(EmergencyTruncate),
        Box::new(RenderRefs),
    ]
}

// Keep the import used even when a host builds a custom pipeline.
#[allow(unused)]
fn _effects_marker(_: &Effects, _: &HashMap<(), ()>) {}




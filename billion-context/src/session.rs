//! `Session` — the exclusive, serializable unit of conversation state.
//!
//! Everything here is synchronous and pure. The model call, the clock, the
//! tokenizer and the content store all live outside; a session only ever needs
//! what the host hands it.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::config::{
    CharsPerToken4, Config, ConfigError, DefaultPolicy, Policy, TokenEstimator,
};
use crate::error::{RangeError, RangeErrorKind};
use crate::ids::{BlockId, Boundary, MessageId, MsgRef, Timestamp};
use crate::message::Message;
use crate::pipeline::{
    self, Effects, Node, NodeIo, PipelineContext, Range, RenderStrategy, ViewMode,
};
use crate::state::{Block, Generation, NudgeState, State, Tier};

// ---------------------------------------------------------------------------
// Turn
// ---------------------------------------------------------------------------

pub struct TurnInput<'a> {
    /// The conversation as the host currently holds it. Borrowed, not consumed:
    /// a host that keeps its own `Vec<Message>` pays no clone on the way in.
    pub messages: &'a [Message],
    /// The occupancy the host learned from the previous response's usage block.
    /// The kernel never re-counts the whole conversation to find this.
    pub token_count: u32,
    pub render: RenderStrategy,
    pub now: Timestamp,
}

#[derive(Debug)]
pub struct Turn {
    /// The projection to send. Length differs from the input: folded messages
    /// are gone, and ref tags have been added.
    pub messages: Vec<Message>,
    pub ranges: Vec<Range>,
    pub nudge: Option<pipeline::Nudge>,
    pub terminal: Option<pipeline::TerminalEscape>,
    pub truncation_skipped: Option<String>,
    pub warnings: Vec<String>,
}

// ---------------------------------------------------------------------------
// Compression
// ---------------------------------------------------------------------------

/// One range the model asked to fold. `start`/`end` may name a message ref or a
/// whole block, which is how tier-3 folding absorbs tier-2 summaries.
#[derive(Clone, Debug)]
pub struct RequestedRange {
    pub start: Boundary,
    pub end: Boundary,
    pub topic: Option<String>,
    pub summary: String,
}

/// One fold request. Note what is *absent*: the tier.
///
/// The tier a fold produces is a property of what it folds, not a choice the
/// caller makes — folding plain messages yields a tier-1 block, and folding
/// blocks yields one tier above the highest block absorbed (capped at 3). Asking
/// the caller for it invited exactly one class of bug: a host that guessed,
/// and wrote blocks into the wrong tier.
#[derive(Clone, Debug)]
pub struct CompressRequest {
    pub ranges: Vec<RequestedRange>,
    pub now: Timestamp,
}

#[derive(Debug)]
pub struct Applied {
    pub blocks: Vec<BlockId>,
    pub errors: Vec<RangeError>,
    pub tokens_compressed: u32,
    /// True when at least one range landed and at least one did not.
    pub partial: bool,
}

impl Applied {
    pub fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }
}

/// What `retrieve` can hand back. Note it returns a **summary**, not the raw
/// messages: the kernel never stores message bodies once they are folded, so
/// expanding to originals is the host's job (it still has them).
#[derive(Clone, Debug)]
pub struct Retrieval {
    pub block: BlockId,
    pub tier: Tier,
    pub topic: Option<String>,
    pub summary: String,
    pub span: Option<(MsgRef, MsgRef)>,
}

#[derive(Debug)]
pub struct DecompressOutcome {
    pub blocks: Vec<BlockId>,
    /// How many messages will reappear in the projection on the next turn.
    pub restored_messages: usize,
}

#[derive(Clone, Debug)]
pub struct Hit {
    pub block: BlockId,
    pub tier: Tier,
    pub topic: Option<String>,
    pub snippet: String,
    /// Weighted relevance in `0.0..=1.0`. Fractional on purpose: the scoring
    /// gives a topic hit more weight than a summary hit, so an integer count
    /// would collapse two very different matches into the same rank.
    pub score: f64,
}

#[derive(Clone, Debug)]
pub struct Status {
    pub blocks: usize,
    pub active_blocks: usize,
    pub refs_allocated: usize,
    pub refs_cursor: u32,
    pub tokens_compressed: u64,
    pub compression_count: u64,
    pub tiers: [usize; 3],
    pub nudge: NudgeState,
    /// The window the session was configured for. Reported rather than assumed
    /// so a host can compute usage without holding a second copy of the config.
    pub model_context_limit: u32,
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

pub struct Session {
    config: Config,
    policy: Arc<dyn Policy>,
    estimator: Arc<dyn TokenEstimator>,
    pipeline: Vec<Box<dyn Node>>,
    state: State,
    /// `id -> content fingerprint` for the view the host last saw. Folding is
    /// the only moment where "which content is now inside a block" is
    /// unambiguous, so this is what lets a resent original be recognised next
    /// turn instead of being re-admitted as fresh context.
    last_view: HashMap<MessageId, u64>,
    /// Who rewrites the transcript when a block covers a span. See [`ViewMode`].
    view: ViewMode,
    /// Whether to re-identify a resent original by content. See
    /// [`PipelineContext::reconcile`].
    reconcile: bool,
}

impl Session {
    pub fn new(config: Config) -> Self {
        Self {
            config,
            policy: Arc::new(DefaultPolicy),
            estimator: Arc::new(CharsPerToken4),
            pipeline: pipeline::default_pipeline(),
            state: State::new(),
            last_view: HashMap::new(),
            view: ViewMode::default(),
            reconcile: true,
        }
    }

    /// Fails only on a config that would make the engine incoherent (a tier
    /// trigger above the context limit, a nudge band outside 0..=1, ...).
    pub fn try_new(config: Config) -> Result<Self, Vec<ConfigError>> {
        config.validate()?;
        Ok(Self::new(config))
    }

    pub fn with_policy(mut self, policy: impl Policy) -> Self {
        self.policy = Arc::new(policy);
        self
    }

    pub fn with_estimator(mut self, estimator: impl TokenEstimator) -> Self {
        self.estimator = Arc::new(estimator);
        self
    }

    /// Replace the pass list. The default order is the upstream contract; a
    /// custom pipeline is for embedding, not for tuning.
    pub fn with_pipeline(mut self, nodes: Vec<Box<dyn Node>>) -> Self {
        self.pipeline = nodes;
        self
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn state(&self) -> &State {
        &self.state
    }

    pub fn state_mut(&mut self) -> &mut State {
        &mut self.state
    }

    /// Cheap: `State` is a handful of maps and a vector of small blocks.
    pub fn fork(&self) -> Self {
        Self {
            config: self.config.clone(),
            policy: Arc::clone(&self.policy),
            estimator: Arc::clone(&self.estimator),
            pipeline: pipeline::default_pipeline(),
            state: self.state.clone(),
            last_view: self.last_view.clone(),
            view: self.view,
            reconcile: self.reconcile,
        }
    }

    /// Choose whether the kernel or the host rewrites the transcript when a
    /// fold lands. Defaults to [`ViewMode::Send`].
    pub fn with_view_mode(mut self, view: ViewMode) -> Self {
        self.view = view;
        self
    }

    pub fn view_mode(&self) -> ViewMode {
        self.view
    }

    /// Turn off content-based reconciliation. Correct for a host that never
    /// resends a folded original — it then cannot have one to reconcile, and
    /// re-identifying by content would only risk conflating two identical live
    /// messages.
    pub fn with_reconcile(mut self, reconcile: bool) -> Self {
        self.reconcile = reconcile;
        self
    }

    // -- the turn ----------------------------------------------------------

    pub fn process_turn(&mut self, input: TurnInput<'_>) -> Turn {
        let mut messages = input.messages.to_vec();
        self.sync_content_index(&messages);

        let mut effects = Effects::default();
        {
            let ctx = PipelineContext {
                config: &self.config,
                policy: self.policy.as_ref(),
                estimator: self.estimator.as_ref(),
                token_count: input.token_count,
                render: input.render,
                view: self.view,
                reconcile: self.reconcile && self.view == ViewMode::Send,
                now: input.now,
            };
            let mut io = NodeIo {
                messages: &mut messages,
                state: &mut self.state,
                effects: &mut effects,
            };
            pipeline::run(&self.pipeline, &mut io, &ctx);
        }

        // What the model is about to see, keyed by id, so that folding can later
        // say "this content now lives in a block" without a second lookup.
        self.last_view = messages
            .iter()
            .map(|m| (m.id.clone(), m.fingerprint()))
            .collect();

        Turn {
            messages,
            ranges: effects.ranges,
            nudge: effects.nudge,
            terminal: effects.terminal,
            truncation_skipped: effects.truncation_skipped,
            warnings: effects.warnings,
        }
    }

    /// Keep the fingerprint index in step with what the host currently holds.
    ///
    /// Entries are kept while the message is either still live or covered by an
    /// active block; anything else is dead weight. That bound matters because a
    /// long-lived session would otherwise remember every message it ever saw.
    fn sync_content_index(&mut self, messages: &[Message]) {
        // Base ids, for the same reason coverage uses them: a `base#c0` index
        // entry is still a live original when the host emits `base#c1`.
        let live: std::collections::HashSet<String> =
            messages.iter().map(|m| m.id.base().to_owned()).collect();
        let covered: std::collections::HashSet<String> = self
            .state
            .covered_ids()
            .into_iter()
            .map(|id| id.base().to_owned())
            .collect();
        // Live messages are never indexed: two of them may legitimately share
        // content, and remapping one onto the other would delete a real message.
        // The `prune` node records fingerprints at the moment it drops a folded
        // message, which is the only time the mapping is unambiguous.
        self.state
            .content_index
            .retain(|_, id| live.contains(id.base()) || covered.contains(id.base()));
    }

    // -- folding -----------------------------------------------------------

    /// Apply the model's compress request. Per-range outcomes, never all-or-
    /// nothing: one stale ref must not throw away nine good ranges.
    pub fn apply_compression(&mut self, req: CompressRequest) -> Applied {
        let mut blocks = Vec::new();
        let mut errors = Vec::new();
        let mut tokens_compressed = 0u32;

        for range in req.ranges.iter() {
            match self.apply_one(range, req.now) {
                Ok((block, tokens)) => {
                    blocks.push(block);
                    tokens_compressed = tokens_compressed.saturating_add(tokens);
                }
                Err((kind, detail)) => errors.push(RangeError {
                    start: boundary_label(range.start),
                    end: boundary_label(range.end),
                    kind,
                    detail,
                }),
            }
        }

        if !blocks.is_empty() {
            self.state.stats.compression_count += blocks.len() as u64;
            self.state.stats.tokens_compressed += tokens_compressed as u64;
            // A fold that landed resets the nudge cadence: the next reminder
            // should be measured from the new, smaller baseline.
            self.state.nudge.baseline_tokens = 0;
            self.state.nudge.last_shown_tokens = 0;
            self.state.nudge.shown_by_tier.clear();
        }

        Applied {
            partial: !blocks.is_empty() && !errors.is_empty(),
            blocks,
            errors,
            tokens_compressed,
        }
    }

    /// The tier a fold produces: a message-ref range folds into tier 1; a range
    /// that names blocks folds one tier above the highest block it absorbs.
    /// Capped at 3, which is the top of the pyramid.
    fn derive_tier(&self, range: &RequestedRange) -> Tier {
        let mut highest = 0u8;
        for boundary in [&range.start, &range.end] {
            if let Boundary::Block(id) = boundary {
                if let Some(block) = self.state.blocks.iter().find(|b| &b.id == id) {
                    highest = highest.max(block.tier as u8);
                }
            }
        }
        match highest {
            0 => Tier::T1,
            1 => Tier::T2,
            _ => Tier::T3,
        }
    }

    fn apply_one(
        &mut self,
        range: &RequestedRange,
        now: Timestamp,
    ) -> Result<(BlockId, u32), (RangeErrorKind, String)> {
        let tier = self.derive_tier(range);
        let (start, end) = self.resolve_span(range)?;
        if start > end {
            return Err((
                RangeErrorKind::InvalidRef,
                format!("{start} comes after {end}; a range must run forwards"),
            ));
        }

        // Collect the refs this range covers.
        let covered = self.state.covered_ids();
        let mut ids = Vec::new();
        let mut tokens = 0u32;
        let mut blocked = 0usize;
        for index in start.index()..=end.index() {
            let Some(r) = MsgRef::new(index) else {
                continue;
            };
            let Some(id) = self.state.refs.raw_of(r) else {
                continue;
            };
            if self.state.refs.is_blocked(id) {
                blocked += 1;
                continue;
            }
            ids.push(id.clone());
            // What an active block already hides is not visible mass, so it
            // cannot be what this fold saved: its summary is, added below.
            if !covered.contains(id) {
                tokens = tokens
                    .saturating_add(self.state.token_snapshot.get(&r).copied().unwrap_or(0));
            }
        }
        if ids.is_empty() {
            return Err((
                RangeErrorKind::TooSmall,
                format!("{start}-{end} holds no foldable message ({blocked} protected)"),
            ));
        }

        // A range that names a block absorbs it: the absorbed block's messages
        // become this block's content, its summary counts as compressed too, and
        // the block itself goes inactive. Without this a tier-2 or tier-3 summary
        // could never be produced and the pyramid would be decoration.
        let mut absorbed: Vec<BlockId> = Vec::new();
        for boundary in [&range.start, &range.end] {
            if let Boundary::Block(b) = boundary {
                if !absorbed.contains(b) {
                    absorbed.push(*b);
                }
            }
        }
        let absorbed_ids: HashSet<MessageId> = absorbed
            .iter()
            .filter_map(|b| self.state.block(*b))
            .flat_map(|b| b.direct.iter().cloned())
            .collect();

        // Refuse to re-fold something an active block already owns — unless this
        // range *is* that block, which is exactly what absorption means.
        if ids
            .iter()
            .any(|id| covered.contains(id) && !absorbed_ids.contains(id))
        {
            return Err((
                RangeErrorKind::Consumed,
                format!("{start}-{end} overlaps content an active block already covers"),
            ));
        }
        for b in &absorbed {
            if let Some(block) = self.state.block(*b) {
                tokens = tokens.saturating_add(self.estimator.count(&block.summary));
            }
        }

        let summary_len = range.summary.chars().count();
        if summary_len < self.config.compress.min_summary_length {
            return Err((
                RangeErrorKind::SummaryTooShort,
                format!(
                    "summary is {summary_len} chars, minimum is {}",
                    self.config.compress.min_summary_length
                ),
            ));
        }
        if summary_len > self.config.compress.max_summary_length {
            return Err((
                RangeErrorKind::SummaryTooLong,
                format!(
                    "summary is {summary_len} chars, maximum is {}",
                    self.config.compress.max_summary_length
                ),
            ));
        }

        let Some(id) = self.state.allocate_block() else {
            return Err((RangeErrorKind::Capacity, "block id space exhausted".into()));
        };
        let run = self.state.allocate_run();

        // Index what this block is about to swallow, before the ids move into
        // it. `prune` never sees these messages again, so this is the only
        // point where the fingerprint mapping is known to be correct.
        for id in &ids {
            if let Some(fingerprint) = self.last_view.get(id) {
                self.state.content_index.insert(*fingerprint, id.clone());
            }
        }

        let block = Block {
            id,
            run,
            tier,
            topic: range.topic.clone(),
            summary: range.summary.clone(),
            direct: ids.clone(),
            effective: ids,
            direct_blocks: absorbed.clone(),
            compressed_tokens: tokens,
            created_at: now,
            survived: 0,
            generation: Generation::Young,
            active: true,
            expanded: false,
            restored_inline: false,
            span: Some((range.start, range.end)),
        };
        self.state.blocks.push(block);
        for b in &absorbed {
            if let Some(block) = self.state.blocks.iter_mut().find(|x| &x.id == b) {
                block.active = false;
            }
        }
        Ok((id, tokens))
    }

    /// Turn a requested boundary pair into a concrete ref span, expanding block
    /// boundaries to the span they cover.
    fn resolve_span(
        &self,
        range: &RequestedRange,
    ) -> Result<(MsgRef, MsgRef), (RangeErrorKind, String)> {
        Ok((
            self.resolve_one(range.start, true)?,
            self.resolve_one(range.end, false)?,
        ))
    }

    fn resolve_one(
        &self,
        boundary: Boundary,
        first: bool,
    ) -> Result<MsgRef, (RangeErrorKind, String)> {
        match boundary {
            Boundary::Message(r) => {
                if self.state.refs.raw_of(r).is_none() {
                    return Err((
                        RangeErrorKind::UnknownRef,
                        format!("{r} is not in this view"),
                    ));
                }
                Ok(r)
            }
            Boundary::Block(b) => {
                let (start, end) = self.message_span_of(b)?;
                Ok(if first { start } else { end })
            }
        }
    }

    /// The message refs a block covers, following nested block boundaries down
    /// to the messages underneath. A block folded from `b3..b4` declares that
    /// span, so reaching the messages takes one more hop per tier.
    pub fn message_span_of(
        &self,
        block: BlockId,
    ) -> Result<(MsgRef, MsgRef), (RangeErrorKind, String)> {
        self.span_at(block, 0)
    }

    fn span_at(
        &self,
        block: BlockId,
        depth: u8,
    ) -> Result<(MsgRef, MsgRef), (RangeErrorKind, String)> {
        // A block whose span names itself would otherwise recurse forever.
        if depth > 32 {
            return Err((
                RangeErrorKind::UnknownRef,
                format!("{block} nests too deeply to expand"),
            ));
        }
        let found = self
            .state
            .block(block)
            .ok_or_else(|| (RangeErrorKind::UnknownRef, format!("{block} is not a known block")))?;
        let Some((start, end)) = found.span else {
            return Err((
                RangeErrorKind::UnknownRef,
                format!("{block} has no ref span to expand"),
            ));
        };
        Ok((
            self.end_of(start, true, depth)?,
            self.end_of(end, false, depth)?,
        ))
    }

    fn end_of(
        &self,
        boundary: Boundary,
        first: bool,
        depth: u8,
    ) -> Result<MsgRef, (RangeErrorKind, String)> {
        match boundary {
            Boundary::Message(r) => Ok(r),
            Boundary::Block(id) => {
                let (start, end) = self.span_at(id, depth + 1)?;
                Ok(if first { start } else { end })
            }
        }
    }

    // -- retrieval ---------------------------------------------------------

    /// The summaries covering these refs. Blocks, not raw messages — see
    /// [`Retrieval`].
    pub fn retrieve(&self, refs: &[MsgRef]) -> Vec<Retrieval> {
        let mut out = Vec::new();
        for block in self.state.active_blocks() {
            let Ok((start, end)) = self.message_span_of(block.id) else {
                continue;
            };
            if refs.iter().any(|r| *r >= start && *r <= end) {
                out.push(Retrieval {
                    block: block.id,
                    tier: block.tier,
                    topic: block.topic.clone(),
                    summary: block.summary.clone(),
                    span: Some((start, end)),
                });
            }
        }
        // Deliberately `&self`: a lookup must not mutate. The retrieval counter
        // in `Stats` is bumped by the host, which is the only side that knows
        // whether a retrieval actually reached the model.
        out
    }

    /// Undo a fold: the block goes inactive, so the next projection keeps its
    /// original messages inline again.
    pub fn decompress(&mut self, ids: &[BlockId]) -> DecompressOutcome {
        let mut restored = Vec::new();
        let mut messages = 0usize;
        for block in self.state.blocks.iter_mut() {
            if block.active && ids.contains(&block.id) {
                block.active = false;
                block.restored_inline = true;
                messages += block.effective.len();
                restored.push(block.id);
            }
        }
        DecompressOutcome {
            blocks: restored,
            restored_messages: messages,
        }
    }

    /// Keyword search over the summaries of active blocks.
    ///
    /// The query is split on whitespace and every term scores independently, so
    /// `auth token refresh` finds a block that says only "refresh the token" —
    /// a whole-phrase match would return nothing and quietly make the tool
    /// useless. Topic hits weigh more than summary hits, each is capped so one
    /// repeated word cannot dominate, and the total is capped at 1.
    ///
    /// There is deliberately no relevance floor: ranking plus the caller's
    /// `limit` is the precision mechanism, and a floor would drop the weakest
    /// positive signal — a single summary occurrence.
    pub fn search(&self, query: &str, limit: usize) -> Vec<Hit> {
        let terms: Vec<String> = query
            .to_lowercase()
            .split_whitespace()
            .map(str::to_string)
            .collect();
        if terms.is_empty() {
            return Vec::new();
        }

        let mut hits: Vec<Hit> = self
            .state
            .active_blocks()
            .filter_map(|block| {
                let score = score_relevance(block, &terms);
                (score > 0.0).then(|| Hit {
                    block: block.id,
                    tier: block.tier,
                    topic: block.topic.clone(),
                    snippet: snippet(&block.summary, &terms, 160),
                    score,
                })
            })
            .collect();
        // Stable, so equal scores keep block order and the output is
        // reproducible for a given state.
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(limit);
        hits
    }

    /// Read one block by id.
    ///
    /// This is the read-only counterpart to [`Session::decompress`]: it returns
    /// the block without expanding it, so a host answering "what does this block
    /// say" does not silently change what the next request sends.
    pub fn block(&self, id: BlockId) -> Option<&Block> {
        self.state.blocks.iter().find(|block| block.id == id)
    }

    pub fn status(&self) -> Status {
        let mut tiers = [0usize; 3];
        for block in self.state.active_blocks() {
            tiers[(block.tier as usize) - 1] += 1;
        }
        Status {
            blocks: self.state.blocks.len(),
            active_blocks: self.state.active_blocks().count(),
            refs_allocated: self.state.refs.len(),
            refs_cursor: self.state.refs.cursor_hint(),
            tokens_compressed: self.state.stats.tokens_compressed,
            compression_count: self.state.stats.compression_count,
            tiers,
            nudge: self.state.nudge.clone(),
            model_context_limit: self.config.model_context_limit,
        }
    }
}

fn boundary_label(boundary: Boundary) -> String {
    match boundary {
        Boundary::Message(r) => r.to_string(),
        Boundary::Block(id) => id.to_string(),
    }
}

/// How well a block answers `terms`, in `0.0..=1.0`.
///
/// A term in the topic is worth up to 0.45 and a term in the summary up to
/// 0.20. The asymmetry is intentional: the topic is the block's label, so a
/// match there says the block is *about* the query, while a summary match may
/// be incidental. Both caps stop one term that happens to repeat from
/// outranking a block that matches every term.
fn score_relevance(block: &Block, terms: &[String]) -> f64 {
    let topic = block.topic.as_deref().unwrap_or_default().to_lowercase();
    let summary = block.summary.to_lowercase();
    let mut score = 0.0f64;
    for term in terms {
        let topic_hits = topic.matches(term.as_str()).count();
        if topic_hits > 0 {
            score += (topic_hits as f64 * 0.15).min(0.45);
        }
        let summary_hits = summary.matches(term.as_str()).count();
        if summary_hits > 0 {
            score += (summary_hits as f64 * 0.04).min(0.2);
        }
    }
    score.min(1.0)
}

fn snippet(text: &str, terms: &[String], width: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    let lower: String = text.to_lowercase();
    // Anchor on the earliest matching term, so a snippet of a block that
    // matches several terms starts at the first thing the query asked for.
    let at = terms
        .iter()
        .filter_map(|term| lower.find(term.as_str()))
        .min()
        .unwrap_or(0);
    // `find` returns a byte offset; walk the chars to convert it.
    let byte_to_char = |byte: usize| lower[..byte].chars().count();
    let centre = byte_to_char(at);
    let start = centre.saturating_sub(width / 3);
    let end = (start + width).min(chars.len());
    chars[start..end].iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CharsAsTokens;
    use crate::message::Role;

    /// The whole point of the pyramid: a fold that names a block absorbs it, so
    /// tier 2 condenses tier-1 summaries and tier 3 condenses tier-2 ones. The
    /// derived tier, the absorption list and the compressed mass all have to
    /// agree, or the pyramid is decoration.
    #[test]
    fn the_tier_ladder_absorbs_blocks_upward() {
        let mut config = Config::new(200_000);
        config.preserve_recent_messages = 0;
        config.preserve_recent_tokens = 0;
        config.tiers.tier2_trigger = 1;
        config.tiers.tier3_trigger = 2;
        config.nudge.tier_growth_tokens = Some(crate::config::TierGrowthTokens {
            t1: Some(1),
            t2: Some(1),
            t3: Some(1),
        });
        let mut s = Session::new(config).with_estimator(CharsAsTokens);
        let msgs = conversation(8, 2_000);
        let _ = s.process_turn(TurnInput {
            messages: &msgs,
            token_count: 500,
            render: RenderStrategy::None,
            now: 1,
        });

        let fold = |s: &mut Session, start: Boundary, end: Boundary, topic: &str| {
            let applied = s.apply_compression(CompressRequest {
                ranges: vec![RequestedRange {
                    start,
                    end,
                    topic: Some(topic.into()),
                    summary: format!(
                        "The user asked about {topic} and the assistant answered in full detail."
                    ),
                }],
                now: 2,
            });
            assert!(applied.is_ok(), "{:?}", applied.errors);
            applied.blocks[0]
        };
        let m = |n: u32| Boundary::Message(MsgRef::new(n).unwrap());
        let b1 = fold(&mut s, m(1), m(4), "the parser");
        let b2 = fold(&mut s, m(5), m(8), "the cache");
        assert_eq!(s.state.block(b1).unwrap().tier, Tier::T1);

        let t2 = fold(&mut s, Boundary::Block(b1), Boundary::Block(b1), "front end");
        let t2b = fold(&mut s, Boundary::Block(b2), Boundary::Block(b2), "back end");
        let tier_two = s.state.block(t2).unwrap();
        assert_eq!(tier_two.tier, Tier::T2);
        assert_eq!(tier_two.direct_blocks, vec![b1]);
        assert!(!s.state.block(b1).unwrap().active, "absorbed blocks retire");
        // The declared span is what the model cited, not what it expands to.
        assert_eq!(
            tier_two.span,
            Some((Boundary::Block(b1), Boundary::Block(b1)))
        );
        // A tier-2 fold saves the summary mass it hides, not the originals the
        // tier-1 block already accounted for.
        assert!(
            tier_two.compressed_tokens < 100,
            "condensing a block must count the summary, got {}",
            tier_two.compressed_tokens
        );

        let t3 = fold(&mut s, Boundary::Block(t2), Boundary::Block(t2b), "pipeline");
        let tier_three = s.state.block(t3).unwrap();
        assert_eq!(tier_three.tier, Tier::T3);
        assert_eq!(tier_three.direct_blocks, vec![t2, t2b]);
        assert_eq!(
            tier_three.span,
            Some((Boundary::Block(t2), Boundary::Block(t2b)))
        );
        // Expanding a tier-3 block still reaches the original messages.
        let (start, end) = s.message_span_of(t3).unwrap();
        assert_eq!((start.index(), end.index()), (1, 8));
    }

    fn session() -> Session {
        Session::new(Config::new(200_000)).with_estimator(CharsAsTokens)
    }

    fn conversation(n: usize, len: usize) -> Vec<Message> {
        (0..n)
            .map(|i| {
                let role = if i % 2 == 0 { Role::User } else { Role::Assistant };
                // Distinct bodies on purpose: identical content is deduplicated
                // by fingerprint, which would make these fixtures test the
                // dedupe path instead of the thing each test is named for.
                Message::text(
                    format!("msg-{i}"),
                    role,
                    format!("{}{i}", "x".repeat(len)),
                )
            })
            .collect()
    }

    #[test]
    fn media_without_bytes_is_a_hole_no_range_can_span() {
        let mut s = session();
        let mut msgs = conversation(8, 6_000);
        // An image the host cannot restore, sitting in the middle of the
        // otherwise foldable span. Folding it would keep the prose and lose the
        // picture, with nothing behind `decompress` to bring it back.
        msgs[3] = msgs[3].clone().with_unrecoverable_payload(true);

        let turn = s.process_turn(TurnInput {
            messages: &msgs,
            token_count: 100,
            render: RenderStrategy::None,
            now: 1,
        });

        let hole = MsgRef::new(4).expect("ref");
        assert!(
            !turn.ranges.is_empty(),
            "the text around the image is still foldable"
        );
        for range in &turn.ranges {
            assert!(
                !(range.start <= hole && hole <= range.end),
                "range {}-{} spans the un-restorable image",
                range.start.index(),
                range.end.index()
            );
        }
        assert!(
            turn.ranges.iter().any(|r| r.end < hole),
            "content before the image is still offered"
        );
        assert!(
            turn.ranges.iter().any(|r| r.start > hole),
            "content after the image is still offered"
        );
    }

    #[test]
    fn refs_are_assigned_and_never_reused() {
        let mut s = session();
        let msgs = conversation(3, 10);
        let turn = s.process_turn(TurnInput {
            messages: &msgs,
            token_count: 100,
            render: RenderStrategy::All,
            now: 1,
        });
        assert_eq!(s.status().refs_allocated, 3);
        assert!(turn.messages[0].content.text().unwrap().starts_with("<acp tokens="));
    }

    #[test]
    fn folding_hides_the_originals_and_keeps_the_ref() {
        let mut s = session();
        let msgs = conversation(6, 2_000);
        let _ = s.process_turn(TurnInput {
            messages: &msgs,
            token_count: 500,
            render: RenderStrategy::None,
            now: 1,
        });

        let applied = s.apply_compression(CompressRequest {
            ranges: vec![RequestedRange {
                start: Boundary::Message(MsgRef::new(1).unwrap()),
                end: Boundary::Message(MsgRef::new(3).unwrap()),
                topic: Some("opening".into()),
                summary: "The user asked three separate questions about the parser and its \
                          error messages."
                    .into(),
            }],
            now: 2,
        });
        assert!(applied.is_ok(), "{:?}", applied.errors);
        assert_eq!(applied.blocks.len(), 1);

        let turn = s.process_turn(TurnInput {
            messages: &msgs,
            token_count: 500,
            render: RenderStrategy::None,
            now: 3,
        });
        // The span is replaced by one carrier message, and the session's opening
        // user message survives the fold — a provider rejects a conversation
        // with no user message at all.
        assert_eq!(turn.messages.len(), 5);
        assert_eq!(turn.messages[0].id.as_str(), "msg-0");
        assert_eq!(turn.messages[1].id.as_str(), "acp_summary_b1");
        assert!(turn.messages.iter().all(|m| m.id.as_str() != "msg-1"));
    }

    #[test]
    fn partial_success_keeps_the_good_ranges() {
        let mut s = session();
        let msgs = conversation(6, 2_000);
        let _ = s.process_turn(TurnInput {
            messages: &msgs,
            token_count: 500,
            render: RenderStrategy::None,
            now: 1,
        });
        let applied = s.apply_compression(CompressRequest {
            ranges: vec![
                RequestedRange {
                    start: Boundary::Message(MsgRef::new(1).unwrap()),
                    end: Boundary::Message(MsgRef::new(2).unwrap()),
                    topic: None,
                    summary: "A perfectly good summary of the first two messages.".into(),
                },
                RequestedRange {
                    start: Boundary::Message(MsgRef::new(90).unwrap()),
                    end: Boundary::Message(MsgRef::new(91).unwrap()),
                    topic: None,
                    summary: "A range pointing at refs that were never handed out.".into(),
                },
            ],
            now: 2,
        });
        assert_eq!(applied.blocks.len(), 1);
        assert_eq!(applied.errors.len(), 1);
        assert_eq!(applied.errors[0].kind, RangeErrorKind::UnknownRef);
        assert!(applied.partial);
    }

    #[test]
    fn overlapping_ranges_are_refused() {
        let mut s = session();
        let msgs = conversation(6, 2_000);
        let _ = s.process_turn(TurnInput {
            messages: &msgs,
            token_count: 500,
            render: RenderStrategy::None,
            now: 1,
        });
        let make = |a: u32, b: u32| CompressRequest {
            ranges: vec![RequestedRange {
                start: Boundary::Message(MsgRef::new(a).unwrap()),
                end: Boundary::Message(MsgRef::new(b).unwrap()),
                topic: None,
                summary: "A summary that is comfortably long enough to pass.".into(),
            }],
            now: 2,
        };
        assert!(s.apply_compression(make(1, 3)).is_ok());
        let again = s.apply_compression(make(2, 4));
        assert_eq!(again.errors[0].kind, RangeErrorKind::Consumed);
    }

    #[test]
    fn decompress_brings_the_originals_back() {
        let mut s = session();
        let msgs = conversation(6, 2_000);
        let _ = s.process_turn(TurnInput {
            messages: &msgs,
            token_count: 500,
            render: RenderStrategy::None,
            now: 1,
        });
        let applied = s.apply_compression(CompressRequest {
            ranges: vec![RequestedRange {
                start: Boundary::Message(MsgRef::new(1).unwrap()),
                end: Boundary::Message(MsgRef::new(2).unwrap()),
                topic: None,
                summary: "A summary that is comfortably long enough to pass.".into(),
            }],
            now: 2,
        });
        let out = s.decompress(&applied.blocks);
        assert_eq!(out.restored_messages, 2);

        let turn = s.process_turn(TurnInput {
            messages: &msgs,
            token_count: 500,
            render: RenderStrategy::None,
            now: 3,
        });
        assert_eq!(turn.messages.len(), 6);
    }

    #[test]
    fn retrieve_returns_the_covering_summary() {
        let mut s = session();
        let msgs = conversation(6, 2_000);
        let _ = s.process_turn(TurnInput {
            messages: &msgs,
            token_count: 500,
            render: RenderStrategy::None,
            now: 1,
        });
        let _ = s.apply_compression(CompressRequest {
            ranges: vec![RequestedRange {
                start: Boundary::Message(MsgRef::new(1).unwrap()),
                end: Boundary::Message(MsgRef::new(3).unwrap()),
                topic: Some("parser".into()),
                summary: "The user asked about the parser and the ref allocator.".into(),
            }],
            now: 2,
        });
        let hits = s.retrieve(&[MsgRef::new(2).unwrap()]);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].topic.as_deref(), Some("parser"));

        let found = s.search("allocator", 5);
        assert_eq!(found.len(), 1);
        assert!(s.search("nothing-like-this", 5).is_empty());
    }

    #[test]
    fn fork_is_independent() {
        let mut s = session();
        let msgs = conversation(4, 100);
        let _ = s.process_turn(TurnInput {
            messages: &msgs,
            token_count: 10,
            render: RenderStrategy::None,
            now: 1,
        });
        let forked = s.fork();
        assert_eq!(forked.status().refs_allocated, s.status().refs_allocated);
        let mut forked = forked;
        let _ = forked.process_turn(TurnInput {
            messages: &msgs,
            token_count: 10,
            render: RenderStrategy::None,
            now: 2,
        });
        assert_eq!(s.status().blocks, 0);
        assert_eq!(forked.status().blocks, 0);
    }

    // -- view ownership, stale blocks, derived tier -------------------------

    fn fold_first_two(s: &mut Session) -> Applied {
        s.apply_compression(CompressRequest {
            ranges: vec![RequestedRange {
                start: Boundary::Message(MsgRef::new(1).unwrap()),
                end: Boundary::Message(MsgRef::new(2).unwrap()),
                topic: None,
                summary: "alpha and beta, folded into one paragraph of prose.".into(),
            }],
            now: 2,
        })
    }

    #[test]
    fn a_block_whose_messages_are_gone_is_retired() {
        let mut s = session();
        let msgs = conversation(4, 2_000);
        let _ = s.process_turn(TurnInput {
            messages: &msgs,
            token_count: 500,
            render: RenderStrategy::None,
            now: 1,
        });
        assert_eq!(fold_first_two(&mut s).blocks.len(), 1);

        // The host still carries the originals: the block stands in for them.
        let carried = s.process_turn(TurnInput {
            messages: &msgs,
            token_count: 500,
            render: RenderStrategy::None,
            now: 3,
        });
        assert_eq!(s.status().active_blocks, 1);
        // Four, not three: the opening user message is pinned (a request with no
        // user turn is rejected upstream), so only msg-1 is replaced by the
        // carrier while msg-0 stays.
        assert_eq!(carried.messages.len(), 4, "one folded, one carrier");

        // Now the host replaces the surface: nothing the block covers is left,
        // so it must not keep standing in for anything.
        let fresh: Vec<Message> = (0..4)
            .map(|i| {
                let role = if i % 2 == 0 { Role::User } else { Role::Assistant };
                Message::text(format!("new-{i}"), role, format!("{}{i}", "y".repeat(2_000)))
            })
            .collect();
        let replaced = s.process_turn(TurnInput {
            messages: &fresh,
            token_count: 500,
            render: RenderStrategy::None,
            now: 4,
        });
        assert_eq!(s.status().active_blocks, 0);
        assert_eq!(replaced.messages.len(), 4, "nothing dropped, no carrier");
    }

    #[test]
    fn folding_a_block_yields_the_next_tier_up() {
        let mut s = session();
        let msgs = conversation(4, 2_000);
        let _ = s.process_turn(TurnInput {
            messages: &msgs,
            token_count: 500,
            render: RenderStrategy::None,
            now: 1,
        });
        let first = fold_first_two(&mut s);
        let block = first.blocks[0];
        assert_eq!(s.state().blocks[0].tier, Tier::T1);

        // Folding the block itself is the tier above what it absorbs.
        let applied = s.apply_compression(CompressRequest {
            ranges: vec![RequestedRange {
                start: Boundary::Block(block),
                end: Boundary::Block(block),
                topic: Some("rolling".into()),
                summary: "both questions and both answers, condensed once more.".into(),
            }],
            now: 5,
        });
        assert_eq!(applied.blocks.len(), 1);
        let promoted = s.state().blocks.last().unwrap();
        assert_eq!(promoted.tier, Tier::T2);
    }

    #[test]
    fn host_view_mode_leaves_the_transcript_to_the_host() {
        let mut s = session().with_view_mode(ViewMode::Host);
        let msgs = conversation(4, 2_000);
        let _ = s.process_turn(TurnInput {
            messages: &msgs,
            token_count: 500,
            render: RenderStrategy::None,
            now: 1,
        });
        assert_eq!(fold_first_two(&mut s).blocks.len(), 1);

        let turn = s.process_turn(TurnInput {
            messages: &msgs,
            token_count: 500,
            render: RenderStrategy::None,
            now: 3,
        });
        assert_eq!(turn.messages.len(), 4, "the host does the surgery, not us");
        assert_eq!(s.status().active_blocks, 1, "but the block is still reported");
    }
}

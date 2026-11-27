//! Per-conversation kernel sessions behind the proxy.
//!
//! The host learns the conversation id from the
//! `x-bili-plugin-conversation` header every turn and re-projects the body
//! through the current kernel state. Session state lives here, keyed by that
//! id. It is snapshotted atomically to `rust-sessions/<sha256(key)>.json`
//! (kernel `State` plus host counters) so a restart resumes folds and refs.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use billion_context::{
    Block, BlockId, Boundary, CompressRequest, Config, Content, FnEstimator, Message,
    RangeErrorKind, RenderStrategy, RequestedRange, Role, Session, State, Tier, Timestamp,
    TurnInput,
};
use billion_context_tool as tool;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::artifact::{
    ArtifactCleanupReport, ArtifactMetadata, ArtifactProtection, ArtifactStore,
    DEFAULT_RETENTION_MS,
};
use crate::prune::{
    PrunePolicy, PruneReport, apply_existing_stubs, estimate_messages_tokens, prune_tool_results,
    should_trigger_decision,
};
use crate::summary_carry;
use crate::usage::Usage;

pub fn now_ms() -> Timestamp {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis() as u64)
        .unwrap_or(0)
}

fn estimate_summary_tokens(text: &str) -> u64 {
    let chars = text.chars().count() as u64;
    chars.div_ceil(4).max(1)
}

/// Phase 3.2: `modelPolicies` resolution result. Every field is optional: a
/// missing/invalid value leaves the kernel default in charge, so a policy entry
/// can set only what differs for that model.
#[derive(Clone, Debug, Default)]
pub struct ModelPolicy {
    pub context_window: Option<u32>,
    pub min_compress_range: Option<usize>,
    pub max_summary_length: Option<usize>,
}

/// Phase 3.2: window advertised by a `modelPolicies` entry in
/// `billion-context.json`. Exact model id first, then a case-insensitive match;
/// a missing, malformed or out-of-range entry returns `None` so the caller
/// keeps DSH's header value or the default window.
///
/// Shape: `{"modelPolicies": {"<model id>": {"contextWindow": 200000,
/// "minCompressRange": 5000, "maxSummaryLength": 20000}}}`.
#[allow(
    dead_code,
    reason = "model policy helper retained for host configuration readers"
)]
pub fn model_context_window(config_text: &str, model: &str) -> Option<u32> {
    model_policy(config_text, model).context_window
}

pub fn model_policy(config_text: &str, model: &str) -> ModelPolicy {
    let model = model.trim();
    if model.is_empty() {
        return ModelPolicy::default();
    }
    let Ok(document) = serde_json::from_str::<Value>(config_text) else {
        return ModelPolicy::default();
    };
    let Some(policies) = document.get("modelPolicies").and_then(Value::as_object) else {
        return ModelPolicy::default();
    };
    let Some(entry) = policies.get(model).or_else(|| {
        policies
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(model))
            .map(|(_, value)| value)
    }) else {
        return ModelPolicy::default();
    };
    let window = entry
        .get("contextWindow")
        .or_else(|| entry.get("context_window"))
        .and_then(Value::as_u64)
        .filter(|window| (1024..=2_000_000).contains(window))
        .map(|window| window as u32);
    let min_compress_range = entry
        .get("minCompressRange")
        .or_else(|| entry.get("min_compress_range"))
        .and_then(Value::as_u64)
        .filter(|chars| (100..=200_000).contains(chars))
        .map(|chars| chars as usize);
    let max_summary_length = entry
        .get("maxSummaryLength")
        .or_else(|| entry.get("max_summary_length"))
        .and_then(Value::as_u64)
        .filter(|chars| (500..=2_000_000).contains(chars))
        .map(|chars| chars as usize);
    ModelPolicy {
        context_window: window,
        min_compress_range,
        max_summary_length,
    }
}

/// P1 auto fallback: when the projection is this much of the window, the host
/// folds on its own initiative instead of waiting for the model to obey the
/// nudge it just received.
const FALLBACK_PRESSURE_RATIO: f64 = 0.85;

/// One attempt per this many requests. A summarizer that folded nothing must
/// not be retried on every turn: that would bill a chain call each time.
const FALLBACK_WINDOW_REQUESTS: u64 = 3;

/// Phase 3.1 speculative band: the conversation is past the working-set phase
/// but not yet under pressure, so folding old T1 blocks early is cheap.
const SPECULATION_BAND_LOW: f64 = 0.45;
const SPECULATION_BAND_HIGH: f64 = 0.75;
const SPECULATION_COOLDOWN_REQUESTS: u64 = 4;
const SPECULATION_MAX_BLOCKS: usize = 2;

/// `block id|tier|generation|survived|span|summary hash`: everything that can
/// move under a speculative candidate without changing its block id.
fn speculation_signature(block: &Block) -> String {
    let span = block
        .span
        .map(|(start, end)| format!("{start}..{end}"))
        .unwrap_or_default();
    let mut hasher = Sha256::new();
    hasher.update(block.summary.as_bytes());
    let digest = hasher.finalize();
    let summary_hash = hex_prefix(&digest, 16);
    format!(
        "{}|T{}|{:?}|{}|{}|{}",
        block.id, block.tier as u8, block.generation, block.survived, span, summary_hash
    )
}

fn hex_prefix(bytes: &[u8], take: usize) -> String {
    bytes
        .iter()
        .take(take)
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// A host-initiated fold the external summarizer chain is about to author.
///
/// The ranges come from the kernel's own range list (contiguous, foldable), so
/// the fallback path needs no boundary parsing from a model.
#[derive(Clone, Debug)]
pub struct FallbackPlan {
    /// `"pressure"` when the window is over the high-pressure line,
    /// `"ignored_nudge"` when two armed nudges went unanswered.
    pub reason: &'static str,
    pub ranges: Vec<RequestedRange>,
}

/// Repeated nudges do not need the full rule sheet again: it was delivered
/// earlier in the session and has not changed. This reminder points back to it,
/// so a re-armed nudge stays cheap while still naming the rules that matter.
const NUDGE_REPEAT_RULES: &str = "HOW TO COMPRESS — reminder (the full rule sheet was delivered earlier in this session and still applies)
- The summary becomes the only record of the range: keep user intent with a ref (m00042), decisions with their rationale, exact paths/errors/versions, and an `Open objectives:` line for requests that are still open. Record facts, never a transcript.
- Write real unicode characters; never copy \\uXXXX escapes out of tool output.";

/// Prompt payload for a nudge: full rules on a session's first injection, the
/// compact reminder on every later one.
fn nudge_prompts(already_injected: u64) -> billion_context::Prompts {
    let mut prompts = billion_context::Prompts::canonical();
    if already_injected > 0 {
        prompts.how_to_compress_rules = NUDGE_REPEAT_RULES.to_string();
    }
    prompts
}

/// Map the stored protocol name onto the kernel's protocol so an Anthropic lane
/// gets its lane note and the OpenAI-shaped lanes stay silent.
fn protocol_named(name: &str) -> Option<billion_context::Protocol> {
    match name {
        "anthropic" => Some(billion_context::Protocol::Anthropic),
        "openai" | "openai-chat" => Some(billion_context::Protocol::Openai),
        "responses" | "openai-responses" => Some(billion_context::Protocol::Responses),
        "google" | "google-genai" => Some(billion_context::Protocol::Google),
        _ => None,
    }
}

#[derive(Clone, Copy)]
struct SessionEconomics {
    gross: i64,
    net: i64,
    summary: i64,
    folded: u64,
    folds: u64,
    has_data: bool,
}

fn session_economics(entry: &Entry) -> SessionEconomics {
    if entry.fold_events.is_empty() {
        return SessionEconomics {
            gross: 0,
            net: 0,
            summary: 0,
            folded: entry.session.status().tokens_compressed,
            folds: 0,
            has_data: false,
        };
    }
    let mut gross = 0i64;
    let mut summary = 0i64;
    let mut folded = 0u64;
    for event in &entry.fold_events {
        let requests_after = entry.requests.saturating_sub(event.at_requests) as i64;
        gross += (event.s as i64 - event.sigma as i64) * requests_after;
        summary += event.sigma as i64;
        folded += event.s;
    }
    SessionEconomics {
        gross,
        net: gross - summary,
        summary,
        folded,
        folds: entry.fold_events.len() as u64,
        has_data: true,
    }
}

/// One landed fold. `s` is the input mass the block swallowed and `sigma` the
/// summary that replaced it; multiplying by requests after the fold is the
/// gross input the upstream did not bill. The exact re-pay `T` is not
/// observable from the host, so the panel labels this an estimate.
#[derive(Clone, Serialize, Deserialize)]
struct FoldEvent {
    at_requests: u64,
    s: u64,
    sigma: u64,
    block: String,
    /// Earliest message ref whose projection changed when this fold landed.
    #[serde(default)]
    change_ref: Option<u32>,
    /// `None` for a fold the model requested; `Some("auto:<trigger>")` for a
    /// fold the host started itself because the model left the nudge unanswered
    /// (P1 auto fallback).
    #[serde(default)]
    reason: Option<String>,
}

/// The raw text of one folded span, kept as an artifact so the original stays
/// addressable after the kernel drops it from the view, or after a restart.
///
/// The folded block's summary is the model-facing record; this is the host's
/// escape hatch for a deliberate `retrieve_raw` / `decompress` recovery. It is
/// never injected automatically — a re-read costs the same tokens the fold just
/// reclaimed.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct FoldArchive {
    /// Kernel block id the span folded into, e.g. `b3`.
    block: String,
    artifact_id: String,
    byte_len: u64,
    /// Message-ref span, e.g. `m00150–m00220`; block/block spans are not archived.
    span: String,
}

/// Transaction journal for one `compress` exchange (plan §2.2). The three
/// phases are written around the kernel call so a crash can be told apart from
/// a quiet turn: an entry that never reaches `end` is an unclosed fold, and the
/// next compress must refuse to run on top of it rather than fold twice.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct FoldJournal {
    #[serde(default)]
    call_id: Option<String>,
    /// Requested span labels (`mNNNNN–mNNNNN` / `bN–bN`).
    #[serde(default)]
    spans: Vec<String>,
    /// Blocks the fold actually produced, filled at the `applied` phase.
    #[serde(default)]
    blocks: Vec<String>,
    /// `start` -> `applied` -> `end`; anything else found at load is residue.
    phase: String,
    #[serde(default)]
    at_requests: u64,
    #[serde(default)]
    at: u64,
    #[serde(default)]
    note: String,
}

/// Reply of a landed `compress`, kept so the same `tool_call_id` replays the
/// recorded exchange instead of folding the range a second time.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct FoldReceipt {
    call_id: String,
    text: String,
    #[serde(default)]
    blocks: Vec<String>,
    #[serde(default)]
    requests: u64,
}

/// Compaction observability. Estimates and provider facts remain separate:
/// `estimate_vs_provider_delta` is only accumulated when a provider context
/// number is present, and is signed (`provider - estimate`).
#[derive(Clone, Default, Serialize, Deserialize)]
struct CompactionMetrics {
    nudge_injected: u64,
    nudge_ignored: u64,
    fold_applied: u64,
    fold_refused: u64,
    fold_partial: u64,
    prune_runs: u64,
    prune_attempts: u64,
    prune_candidates: u64,
    prune_items: u64,
    prune_saved_tokens_est: u64,
    prune_saved_bytes: u64,
    artifact_writes: u64,
    artifact_bytes_written: u64,
    artifact_reads: u64,
    artifact_read_bytes: u64,
    last_change_ref: Option<u32>,
    estimate_vs_provider_delta: i64,
    raw_input_estimate: u64,
    effective_projection_estimate: u64,
    fixed_overhead_estimate: u64,
    prune_trigger_reason: String,
    cooldown_hits: u64,
    #[serde(default)]
    cooldown_blocked: u64,
    #[serde(default)]
    growth_bypass_triggered: u64,
    #[serde(default)]
    prune_skipped: u64,
    #[serde(default)]
    stub_reapply_items: u64,
    /// User turns the host appended to a fold summary because the model's prose
    /// did not cite them (P0.1 intent carry).
    #[serde(default)]
    intent_anchors_carried: u64,
    /// Fold calls refused before anything folded because the summary was a
    /// placeholder, a refusal or an echo of the compression instructions (P0.2).
    #[serde(default)]
    summary_quality_refused: u64,
    /// Host-initiated fallback folds: how many attempts a trigger produced,
    /// how many blocks landed, and the two ways an attempt can end without a
    /// fold (a refused summary or a dead summarizer chain).
    #[serde(default)]
    fallback_attempts: u64,
    #[serde(default)]
    fallback_applied_blocks: u64,
    #[serde(default)]
    fallback_refused: u64,
    #[serde(default)]
    fallback_failures: u64,
    #[serde(default)]
    fallback_last_reason: String,
    /// Folded spans whose raw text was archived before the fold landed (P2
    /// artifact coordination). `skipped` counts spans the store refused for
    /// size/quota: the fold still lands, only the raw escape hatch is missing.
    #[serde(default)]
    fold_archive_written: u64,
    #[serde(default)]
    fold_archive_bytes: u64,
    #[serde(default)]
    fold_archive_skipped: u64,
    /// Explicit raw retrieval: calls, bytes served, refs with no archived
    /// original (`OriginalUnavailable`).
    #[serde(default)]
    retrieve_raw_calls: u64,
    #[serde(default)]
    retrieve_raw_bytes: u64,
    #[serde(default)]
    retrieve_raw_unavailable: u64,
    /// P3 §2.2 transaction markers: journals found open at load (crash between
    /// apply and snapshot), compress calls refused because such a journal was
    /// still open, and replays of an already-landed `tool_call_id`.
    #[serde(default)]
    fold_journal_recovered: u64,
    #[serde(default)]
    fold_busy_refused: u64,
    #[serde(default)]
    idempotent_replays: u64,
    /// Phase 3.1 speculative folds: candidates opened in the 45–75% band,
    /// adopted after the generation check, and discarded because the blocks
    /// moved under the candidate before it landed.
    #[serde(default)]
    speculation_started: u64,
    #[serde(default)]
    speculation_adopted: u64,
    #[serde(default)]
    speculation_discarded: u64,
}

/// Phase 3.3: one explicit fresh-window reset, kept as a revocable receipt.
/// The artifacts named here stay in the store (so `retrieve_raw`/`artifact_read`
/// can still reach the old originals); the blocks and refs that used to point
/// at them are gone by design.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct WindowReset {
    pub at_requests: u64,
    pub reason: String,
    pub dropped_blocks: usize,
    pub kept_messages: usize,
    pub artifacts: Vec<String>,
}

/// Phase 3.1: a candidate T2 distillation prepared while the conversation sits
/// in the 45–75% band. The plan only holds a snapshot of already-folded blocks
/// plus the text handed to the summarizer; the signatures are re-checked before
/// anything lands, so a plan that moved underneath us is discarded, not forced.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SpeculationPlan {
    pub at_requests: u64,
    pub model: Option<String>,
    pub blocks: Vec<String>,
    /// `block id|tier|generation|survived|span|summary hash` per block.
    pub signatures: Vec<String>,
    pub inputs: Vec<String>,
}

/// Task 2: cost accounting for automatic external-summary dispatches.
/// `calls` is the reservation count (one per automatic batch), `speculation_calls`
/// is the speculative subset, and the token fields come from provider `usage`
/// when present, otherwise from `estimated_calls` char-based estimates.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct SummaryBudgetUsage {
    #[serde(default)]
    pub calls: u64,
    #[serde(default)]
    pub speculation_calls: u64,
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub output_tokens: u64,
    #[serde(default)]
    pub estimated_calls: u64,
    #[serde(default)]
    pub refused: u64,
    #[serde(default)]
    pub last_refusal: Option<String>,
}

pub struct Entry {
    session: Session,
    context_limit: u32,
    pub tokens: u32,
    pub ranges: Vec<billion_context::Range>,
    pub last_nudge: Option<String>,
    pub last_seen: u64,
    pub last_messages: Vec<billion_context::Message>,
    pub requests: u64,
    pub model: Option<String>,
    /// Upstream billing counters, accumulated across responses.
    pub input_tokens: u64,
    pub cached_tokens: u64,
    pub output_tokens: u64,
    pub cache_samples: u64,
    /// Sum of every provider-reported prompt total over the samples above. The
    /// hit rate is `cached_tokens / context_tokens_total`, a true cumulative
    /// ratio; without this sum the ratio would mix a cumulative numerator with
    /// a single request's denominator. Old snapshots lack the field and keep 0,
    /// which is reported as "cannot compute" instead of a misleading number.
    pub context_tokens_total: u64,
    /// Whether the numerator/denominator history behind `cacheHitPct` is known
    /// to be complete.  A fresh session starts `true`; a snapshot or meta file
    /// written before D-2.2 deserializes to `false` (missing field), because its
    /// accumulated `cached_tokens` has no matching historical denominator.
    /// While `false`, `cacheHitPct` stays `null` even after new requests add
    /// context tokens: new data alone cannot prove the old history.  There is no
    /// automatic recovery; only a verified migration that reconstructs both
    /// sides may set it back to `true`.
    pub context_tokens_total_complete: bool,
    /// The provider-reported prompt total of the latest response. When present
    /// the panel shows the billed number instead of the host's estimate.
    pub context_usage: Option<u32>,
    pub context_source: &'static str,
    pub protocol: String,
    pub upstream: String,
    fold_events: Vec<FoldEvent>,
    metrics: CompactionMetrics,
    /// True after a nudge reached the model and before a compress tool call
    /// clears it. `nudge_ignored_counted` keeps one unanswered nudge from being
    /// counted on every subsequent prepare.
    nudge_pending: bool,
    nudge_ignored_counted: bool,
    last_prune_requests: Option<u64>,
    last_prune_tokens: Option<u32>,
    raw_input_estimate: Option<u32>,
    effective_projection_estimate: Option<u32>,
    fixed_overhead_estimate: Option<u32>,
    last_projection_estimate: Option<u32>,
    last_raw_input_estimate: Option<u32>,
    prune_trigger_reason: Option<String>,
    /// Set while the host is waiting for the summarizer chain; keeps a second
    /// request from starting a competing fallback fold on the same range.
    fallback_in_flight: bool,
    /// `requests` at the last fallback attempt. One attempt per growth window:
    /// a summarizer that folded nothing must not be retried on every turn.
    fallback_last_requests: Option<u64>,
    /// Archived originals of folded spans (P2). Cleared when a window change
    /// rebuilds the kernel session: block ids and refs are re-allocated, so a
    /// stale archive could no longer be addressed safely.
    fold_archives: Vec<FoldArchive>,
    /// P3 §2.2: fold transaction journal + replay receipts. Both are capped and
    /// persisted so a restart can see an unclosed fold and replay a landed call.
    fold_journal: Vec<FoldJournal>,
    fold_receipts: Vec<FoldReceipt>,
    /// True only while the host is inside `apply_compression` for this entry.
    fold_in_flight: bool,
    /// Phase 3.2 per-model overrides; `None` means the kernel default applies.
    /// Kept on the entry (not the kernel config) so a policy edit takes effect
    /// without rebuilding the session and losing the current folds.
    min_compress_range: Option<usize>,
    max_summary_length: Option<usize>,
    /// Phase 3.3: explicit fresh-window resets, newest last (cap 16).
    window_resets: Vec<WindowReset>,
    /// Phase 3.1: an in-flight speculative fold, and the cooldown cursor that
    /// keeps it from firing on every turn once it has just run.
    speculation_in_flight: bool,
    speculation: Option<SpeculationPlan>,
    speculation_last_requests: Option<u64>,
    /// Task 2: automatic summary dispatch budget/usage for this conversation.
    summary_budget: SummaryBudgetUsage,
}

#[derive(Clone)]
pub struct Store {
    inner: Arc<Mutex<HashMap<String, Entry>>>,
    index: Arc<Mutex<HashMap<String, SnapshotMeta>>>,
    dir: Option<Arc<PathBuf>>,
    artifacts: Option<ArtifactStore>,
    prune_policy: PrunePolicy,
}

/// On-disk snapshot. `state` carries the kernel's blocks/refs; the rest keeps
/// panel counters and the last request shape across a host restart.
#[derive(Serialize, Deserialize)]
struct Persisted {
    version: u32,
    key: String,
    context_limit: u32,
    tokens: u32,
    last_nudge: Option<String>,
    last_seen: u64,
    messages: Vec<Message>,
    requests: u64,
    model: Option<String>,
    input_tokens: u64,
    cached_tokens: u64,
    output_tokens: u64,
    cache_samples: u64,
    #[serde(default)]
    context_tokens_total: u64,
    /// Missing on pre-D-2.2 snapshots => false (denominator history unknown).
    #[serde(default)]
    context_tokens_total_complete: bool,
    context_usage: Option<u32>,
    protocol: String,
    upstream: String,
    #[serde(default)]
    fold_events: Vec<FoldEvent>,
    #[serde(default)]
    metrics: CompactionMetrics,
    #[serde(default)]
    nudge_pending: bool,
    #[serde(default)]
    nudge_ignored_counted: bool,
    #[serde(default)]
    last_prune_requests: Option<u64>,
    #[serde(default)]
    last_prune_tokens: Option<u32>,
    #[serde(default)]
    raw_input_estimate: Option<u32>,
    #[serde(default)]
    effective_projection_estimate: Option<u32>,
    #[serde(default)]
    fixed_overhead_estimate: Option<u32>,
    #[serde(default)]
    last_projection_estimate: Option<u32>,
    #[serde(default)]
    last_raw_input_estimate: Option<u32>,
    #[serde(default)]
    prune_trigger_reason: Option<String>,
    /// Growth-window gate for host-initiated fallback folds. `fallback_in_flight`
    /// is deliberately not persisted: a restart cannot know whether the chain
    /// call is still running, so it starts unblocked.
    #[serde(default)]
    fallback_last_requests: Option<u64>,
    #[serde(default)]
    fold_archives: Vec<FoldArchive>,
    #[serde(default)]
    fold_journal: Vec<FoldJournal>,
    #[serde(default)]
    fold_receipts: Vec<FoldReceipt>,
    #[serde(default)]
    min_compress_range: Option<usize>,
    #[serde(default)]
    max_summary_length: Option<usize>,
    #[serde(default)]
    window_resets: Vec<WindowReset>,
    /// A speculative candidate is a convenience, not durable state: a restart
    /// drops it (the summarizer would have to re-see the blocks anyway).
    #[serde(default)]
    speculation: Option<SpeculationPlan>,
    #[serde(default)]
    speculation_last_requests: Option<u64>,
    #[serde(default)]
    summary_budget: SummaryBudgetUsage,
    state: State,
}

const SNAPSHOT_VERSION: u32 = 1;
const META_VERSION: u32 = 1;

/// Lightweight startup index for a persisted session. The heavy kernel
/// `State` and `messages` stay on disk until the conversation is actually
/// touched, so a long history of idle sessions does not inflate RSS.
#[derive(Clone, Serialize, Deserialize)]
struct SnapshotMeta {
    version: u32,
    key: String,
    context_limit: u32,
    tokens: u32,
    last_seen: u64,
    requests: u64,
    model: Option<String>,
    input_tokens: u64,
    cached_tokens: u64,
    output_tokens: u64,
    cache_samples: u64,
    #[serde(default)]
    context_tokens_total: u64,
    #[serde(default)]
    context_tokens_total_complete: bool,
    context_usage: Option<u32>,
    protocol: String,
    upstream: String,
    blocks: u32,
    active_blocks: u32,
    tokens_compressed: u32,
    gross_saved: i64,
    net_saved: i64,
    summary_cost: i64,
    folded_tokens: u64,
    fold_count: u64,
    has_fold_data: bool,
    #[serde(default)]
    metrics: CompactionMetrics,
}

impl Default for Store {
    fn default() -> Self {
        Self::new()
    }
}

impl Store {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
            index: Arc::new(Mutex::new(HashMap::new())),
            dir: None,
            artifacts: None,
            prune_policy: PrunePolicy::default(),
        }
    }

    /// A store that indexes snapshots under `dir` at startup but materializes
    /// a session only when it is actually used.
    pub fn with_dir(dir: PathBuf) -> Self {
        Self::with_dir_and_policy(dir, PrunePolicy::from_env())
    }

    pub fn with_dir_and_policy(dir: PathBuf, prune_policy: PrunePolicy) -> Self {
        let artifacts = Some(ArtifactStore::new(dir.as_path()));
        let store = Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
            index: Arc::new(Mutex::new(HashMap::new())),
            dir: Some(Arc::new(dir)),
            artifacts,
            prune_policy,
        };
        store.load_index();
        store
    }

    fn snapshot_name(key: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(key.as_bytes());
        format!("{}.json", hex::encode(hasher.finalize()))
    }

    fn meta_name(key: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(key.as_bytes());
        format!("{}.meta.json", hex::encode(hasher.finalize()))
    }

    fn is_meta_path(path: &Path) -> bool {
        path.file_name()
            .and_then(|name| name.to_str())
            .map(|name| name.ends_with(".meta.json"))
            .unwrap_or(false)
    }

    fn meta_from_entry(key: &str, entry: &Entry) -> SnapshotMeta {
        let status = entry.session.status();
        let econ = session_economics(entry);
        SnapshotMeta {
            version: META_VERSION,
            key: key.to_string(),
            context_limit: entry.context_limit,
            tokens: entry.tokens,
            last_seen: entry.last_seen,
            requests: entry.requests,
            model: entry.model.clone(),
            input_tokens: entry.input_tokens,
            cached_tokens: entry.cached_tokens,
            output_tokens: entry.output_tokens,
            cache_samples: entry.cache_samples,
            context_tokens_total: entry.context_tokens_total,
            context_tokens_total_complete: entry.context_tokens_total_complete,
            context_usage: entry.context_usage,
            protocol: entry.protocol.clone(),
            upstream: entry.upstream.clone(),
            blocks: status.blocks as u32,
            active_blocks: status.active_blocks as u32,
            tokens_compressed: status.tokens_compressed as u32,
            gross_saved: econ.gross,
            net_saved: econ.net,
            summary_cost: econ.summary,
            folded_tokens: econ.folded,
            fold_count: econ.folds,
            has_fold_data: econ.has_data,
            metrics: entry.metrics.clone(),
        }
    }

    fn entry_from_snapshot(snapshot: Persisted) -> Entry {
        let context_limit = snapshot.context_limit.max(1);
        let mut session = Session::new(Config::new(context_limit))
            .with_estimator(Self::estimator_for())
            .with_view_mode(billion_context::ViewMode::Send)
            .with_reconcile(true);
        *session.state_mut() = snapshot.state;
        let mut entry = Entry {
            session,
            context_limit,
            tokens: snapshot.tokens,
            ranges: Vec::new(),
            last_nudge: snapshot.last_nudge,
            last_seen: snapshot.last_seen,
            last_messages: snapshot.messages,
            requests: snapshot.requests,
            model: snapshot.model,
            input_tokens: snapshot.input_tokens,
            cached_tokens: snapshot.cached_tokens,
            output_tokens: snapshot.output_tokens,
            cache_samples: snapshot.cache_samples,
            context_tokens_total: snapshot.context_tokens_total,
            context_tokens_total_complete: snapshot.context_tokens_total_complete,
            context_usage: snapshot.context_usage,
            context_source: if snapshot.context_usage.is_some() {
                "usage"
            } else {
                "estimate"
            },
            protocol: snapshot.protocol,
            upstream: snapshot.upstream,
            fold_events: snapshot.fold_events,
            metrics: snapshot.metrics,
            nudge_pending: snapshot.nudge_pending,
            nudge_ignored_counted: snapshot.nudge_ignored_counted,
            last_prune_requests: snapshot.last_prune_requests,
            last_prune_tokens: snapshot.last_prune_tokens,
            raw_input_estimate: snapshot.raw_input_estimate,
            effective_projection_estimate: snapshot.effective_projection_estimate,
            fixed_overhead_estimate: snapshot.fixed_overhead_estimate,
            last_projection_estimate: snapshot.last_projection_estimate,
            last_raw_input_estimate: snapshot.last_raw_input_estimate,
            prune_trigger_reason: snapshot.prune_trigger_reason,
            fallback_in_flight: false,
            fallback_last_requests: snapshot.fallback_last_requests,
            fold_archives: snapshot.fold_archives,
            // P3 §2.2: an entry whose journal never reached `end` is residue of
            // a crash between apply and snapshot. Count it once and hand the
            // entry back unblocked; the next compress refuses to fold on top.
            fold_journal: snapshot.fold_journal,
            fold_receipts: snapshot.fold_receipts,
            fold_in_flight: false,
            min_compress_range: snapshot.min_compress_range,
            max_summary_length: snapshot.max_summary_length,
            window_resets: snapshot.window_resets,
            speculation_in_flight: false,
            speculation: snapshot.speculation,
            speculation_last_requests: snapshot.speculation_last_requests,
            summary_budget: snapshot.summary_budget,
        };
        // P3 §2.2: a journal that never reached `end` is the fingerprint of a
        // crash between `apply_compression` and the snapshot write. Count it
        // once; the next compress refuses to run on top of it.
        if entry
            .fold_journal
            .iter()
            .any(|journal| journal.phase != "end")
        {
            entry.metrics.fold_journal_recovered += 1;
        }
        entry
    }

    fn write_meta(&self, meta: &SnapshotMeta) {
        let Some(dir) = self.dir.as_ref() else { return };
        if fs::create_dir_all(dir.as_ref()).is_err() {
            return;
        }
        let Ok(bytes) = serde_json::to_vec(meta) else {
            return;
        };
        let path = dir.join(Self::meta_name(&meta.key));
        let temp = path.with_extension("meta.json.tmp");
        if fs::write(&temp, bytes).is_ok() {
            let _ = fs::rename(&temp, &path);
        }
    }

    fn persist(&self, key: &str, entry: &Entry) {
        let Some(dir) = self.dir.as_ref() else { return };
        if fs::create_dir_all(dir.as_ref()).is_err() {
            return;
        }
        let snapshot = Persisted {
            version: SNAPSHOT_VERSION,
            key: key.to_string(),
            context_limit: entry.context_limit,
            tokens: entry.tokens,
            last_nudge: entry.last_nudge.clone(),
            last_seen: entry.last_seen,
            messages: entry.last_messages.clone(),
            requests: entry.requests,
            model: entry.model.clone(),
            input_tokens: entry.input_tokens,
            cached_tokens: entry.cached_tokens,
            output_tokens: entry.output_tokens,
            cache_samples: entry.cache_samples,
            context_tokens_total: entry.context_tokens_total,
            context_tokens_total_complete: entry.context_tokens_total_complete,
            context_usage: entry.context_usage,
            protocol: entry.protocol.clone(),
            upstream: entry.upstream.clone(),
            fold_events: entry.fold_events.clone(),
            metrics: entry.metrics.clone(),
            nudge_pending: entry.nudge_pending,
            nudge_ignored_counted: entry.nudge_ignored_counted,
            last_prune_requests: entry.last_prune_requests,
            last_prune_tokens: entry.last_prune_tokens,
            raw_input_estimate: entry.raw_input_estimate,
            effective_projection_estimate: entry.effective_projection_estimate,
            fixed_overhead_estimate: entry.fixed_overhead_estimate,
            last_projection_estimate: entry.last_projection_estimate,
            last_raw_input_estimate: entry.last_raw_input_estimate,
            prune_trigger_reason: entry.prune_trigger_reason.clone(),
            fallback_last_requests: entry.fallback_last_requests,
            fold_archives: entry.fold_archives.clone(),
            fold_journal: entry.fold_journal.clone(),
            fold_receipts: entry.fold_receipts.clone(),
            min_compress_range: entry.min_compress_range,
            max_summary_length: entry.max_summary_length,
            window_resets: entry.window_resets.clone(),
            speculation: entry.speculation.clone(),
            speculation_last_requests: entry.speculation_last_requests,
            summary_budget: entry.summary_budget.clone(),
            state: entry.session.state().clone(),
        };
        let Ok(bytes) = serde_json::to_vec(&snapshot) else {
            return;
        };
        let path = dir.join(Self::snapshot_name(key));
        let temp = path.with_extension("json.tmp");
        let written = fs::write(&temp, bytes).is_ok() && fs::rename(&temp, &path).is_ok();
        if written {
            let meta = Self::meta_from_entry(key, entry);
            self.write_meta(&meta);
            if let Ok(mut index) = self.index.lock() {
                index.insert(key.to_string(), meta);
            }
        }
    }

    fn load_index(&self) {
        let Some(dir) = self.dir.as_ref() else { return };
        let Ok(entries) = fs::read_dir(dir.as_ref()) else {
            return;
        };
        let mut index: HashMap<String, SnapshotMeta> = HashMap::new();
        let mut snapshots: Vec<PathBuf> = Vec::new();
        for file in entries.flatten() {
            let path = file.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            if Self::is_meta_path(&path) {
                let Ok(text) = fs::read_to_string(&path) else {
                    continue;
                };
                let Ok(meta) = serde_json::from_str::<SnapshotMeta>(&text) else {
                    continue;
                };
                if meta.version <= META_VERSION {
                    index.insert(meta.key.clone(), meta);
                }
            } else {
                snapshots.push(path);
            }
        }
        // One-time migration for snapshots written before the sidecar index
        // existed. The heavy state is dropped immediately; only the meta stays.
        for path in snapshots {
            let Ok(text) = fs::read_to_string(&path) else {
                continue;
            };
            let Ok(snapshot) = serde_json::from_str::<Persisted>(&text) else {
                continue;
            };
            if snapshot.version > SNAPSHOT_VERSION {
                continue;
            }
            let key = snapshot.key.clone();
            if index.contains_key(&key) {
                continue;
            }
            let meta = Self::meta_from_entry(&key, &Self::entry_from_snapshot(snapshot));
            self.write_meta(&meta);
            index.insert(key, meta);
        }
        if let Ok(mut guard) = self.index.lock() {
            *guard = index;
        }
    }

    fn artifact_ids_in_text(text: &str) -> Vec<String> {
        let mut ids = Vec::new();
        for segment in text.split("[artifact ").skip(1) {
            let token = segment
                .split(|c: char| c.is_whitespace() || c == ']' || c == '·' || c == ':' || c == '"')
                .next()
                .unwrap_or("");
            if token.starts_with("a_") && token.len() >= 8 {
                ids.push(token.to_string());
            }
        }
        ids
    }

    fn protect_messages(messages: &[Message], protection: &mut ArtifactProtection) {
        for message in messages {
            protection.protect_message(message.id.as_str().to_string());
            if let Content::ToolResult { content, .. } = &message.content {
                for artifact_id in Self::artifact_ids_in_text(content) {
                    protection.protect_artifact(artifact_id);
                }
            }
        }
    }

    /// Collect every reference that could make an artifact live: all message
    /// ids in the in-memory and on-disk sessions, plus every `[artifact a_…]`
    /// stub in those messages. If any snapshot cannot be read or parsed,
    /// `complete=false` and cleanup must refuse to delete anything.
    pub fn artifact_protection(&self) -> ArtifactProtection {
        let mut protection = ArtifactProtection::complete();
        if let Ok(inner) = self.inner.lock() {
            for entry in inner.values() {
                Self::protect_messages(&entry.last_messages, &mut protection);
                // A folded span's archive is addressed by block id, so it is
                // usually absent from `last_messages`; keep it alive explicitly.
                for archive in &entry.fold_archives {
                    protection.protect_artifact(archive.artifact_id.clone());
                }
                // A fresh-window reset drops the block ids but not the archive
                // bodies; the receipt keeps them reachable and protected.
                for reset in &entry.window_resets {
                    for artifact in &reset.artifacts {
                        protection.protect_artifact(artifact.clone());
                    }
                }
            }
        }
        let Some(dir) = self.dir.as_ref() else {
            return protection;
        };
        let entries = match fs::read_dir(dir.as_ref()) {
            Ok(entries) => entries,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return protection,
            Err(_) => {
                protection.complete = false;
                return protection;
            }
        };
        for file in entries.flatten() {
            let path = file.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            if Self::is_meta_path(&path) {
                continue;
            }
            let snapshot = match fs::read_to_string(&path)
                .ok()
                .and_then(|text| serde_json::from_str::<Persisted>(&text).ok())
            {
                Some(snapshot) if snapshot.version <= SNAPSHOT_VERSION => snapshot,
                Some(_) => {
                    // A future schema may carry references this build cannot
                    // recognize. Refusing cleanup is safer than guessing.
                    protection.complete = false;
                    continue;
                }
                None => {
                    protection.complete = false;
                    continue;
                }
            };
            Self::protect_messages(&snapshot.messages, &mut protection);
            for archive in &snapshot.fold_archives {
                protection.protect_artifact(archive.artifact_id.clone());
            }
            for reset in &snapshot.window_resets {
                for artifact in &reset.artifacts {
                    protection.protect_artifact(artifact.clone());
                }
            }
        }
        protection
    }

    /// Explicit maintenance entry point. It derives protection from every
    /// persisted session and never deletes on file age alone.
    pub fn cleanup_artifacts(
        &self,
        now: u64,
        retention_ms: u64,
    ) -> Result<ArtifactCleanupReport, String> {
        let Some(artifacts) = self.artifacts.as_ref() else {
            return Err("artifact store is disabled".into());
        };
        let protection = self.artifact_protection();
        artifacts.cleanup(now, retention_ms, &protection)
    }

    pub fn cleanup_artifacts_default(&self) -> Result<ArtifactCleanupReport, String> {
        self.cleanup_artifacts(now_ms(), DEFAULT_RETENTION_MS)
    }

    fn load_entry(&self, key: &str) -> Option<Entry> {
        let path = self.dir.as_ref()?.join(Self::snapshot_name(key));
        let text = fs::read_to_string(path).ok()?;
        let snapshot = serde_json::from_str::<Persisted>(&text).ok()?;
        if snapshot.version > SNAPSHOT_VERSION {
            return None;
        }
        Some(Self::entry_from_snapshot(snapshot))
    }

    fn ensure_loaded(&self, key: &str) {
        if self
            .inner
            .lock()
            .expect("session store poisoned")
            .contains_key(key)
        {
            return;
        }
        let indexed = self
            .index
            .lock()
            .expect("session store poisoned")
            .contains_key(key);
        if !indexed {
            return;
        }
        let Some(entry) = self.load_entry(key) else {
            return;
        };
        let mut guard = self.inner.lock().expect("session store poisoned");
        guard.entry(key.to_string()).or_insert(entry);
    }

    /// Storage shape for `/__bili/status`: what is actually persisted.
    pub fn storage_json(&self) -> Value {
        let (dir, sessions, bytes) = match self.dir.as_ref() {
            Some(dir) => {
                let mut bytes = 0u64;
                let mut sessions = 0u64;
                if let Ok(entries) = fs::read_dir(dir.as_ref()) {
                    for file in entries.flatten() {
                        let path = file.path();
                        if path.extension().and_then(|e| e.to_str()) != Some("json")
                            || Self::is_meta_path(&path)
                        {
                            continue;
                        }
                        sessions += 1;
                        bytes += fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                    }
                }
                (dir.to_string_lossy().to_string(), sessions, bytes)
            }
            None => (String::new(), 0, 0),
        };
        json!({
            "requested": "full",
            "effective": "full",
            "degraded": false,
            "dir": dir,
            "sessions": sessions,
            "bytes": bytes,
            "loadedSessions": self.inner.lock().expect("session store poisoned").len(),
            "indexedSessions": self.index.lock().expect("session store poisoned").len(),
            "engine": {"name": "json-snapshot", "source": "rust"},
        })
    }

    /// A stable, cheap token estimate matching the Node kernel's default:
    /// CJK 1:1, other bytes ceil(len / 4).
    fn estimator_for() -> FnEstimator<impl Fn(&str) -> u32 + Send + Sync + 'static> {
        FnEstimator(|text: &str| {
            let mut cjk = 0usize;
            let mut other = 0usize;
            for ch in text.chars() {
                let code = ch as u32;
                if (0x4E00..=0x9FFF).contains(&code)
                    || (0x3040..=0x30FF).contains(&code)
                    || (0xAC00..=0xD7AF).contains(&code)
                {
                    cjk += 1;
                } else {
                    other += 1;
                }
            }
            (cjk + other.div_ceil(4)) as u32
        })
    }

    fn with_entry<T>(&self, key: &str, context_limit: u32, f: impl FnOnce(&mut Entry) -> T) -> T {
        self.ensure_loaded(key);
        let mut guard = self.inner.lock().expect("session store poisoned");
        let entry = guard.entry(key.to_string()).or_insert_with(|| Entry {
            session: Session::new(Config::new(context_limit))
                .with_estimator(Self::estimator_for())
                .with_view_mode(billion_context::ViewMode::Send)
                .with_reconcile(true),
            context_limit,
            tokens: 0,
            ranges: Vec::new(),
            last_nudge: None,
            last_seen: 0,
            last_messages: Vec::new(),
            requests: 0,
            model: None,
            input_tokens: 0,
            cached_tokens: 0,
            output_tokens: 0,
            cache_samples: 0,
            context_tokens_total: 0,
            context_tokens_total_complete: true,
            context_usage: None,
            context_source: "estimate",
            protocol: String::new(),
            upstream: String::new(),
            fold_events: Vec::new(),
            metrics: CompactionMetrics::default(),
            nudge_pending: false,
            nudge_ignored_counted: false,
            last_prune_requests: None,
            last_prune_tokens: None,
            raw_input_estimate: None,
            effective_projection_estimate: None,
            fixed_overhead_estimate: None,
            last_projection_estimate: None,
            last_raw_input_estimate: None,
            prune_trigger_reason: None,
            fallback_in_flight: false,
            fallback_last_requests: None,
            fold_archives: Vec::new(),
            fold_journal: Vec::new(),
            fold_receipts: Vec::new(),
            fold_in_flight: false,
            min_compress_range: None,
            max_summary_length: None,
            window_resets: Vec::new(),
            speculation_in_flight: false,
            speculation: None,
            speculation_last_requests: None,
            summary_budget: SummaryBudgetUsage::default(),
        });
        if entry.context_limit != context_limit {
            // A resumed conversation can arrive with a different model window;
            // rebuild rather than run the old window's thresholds forever.
            entry.session = Session::new(Config::new(context_limit))
                .with_estimator(Self::estimator_for())
                .with_view_mode(billion_context::ViewMode::Send)
                .with_reconcile(true);
            entry.context_limit = context_limit;
            entry.tokens = 0;
            entry.ranges.clear();
            entry.last_nudge = None;
            entry.last_seen = 0;
            entry.fold_events.clear();
            entry.metrics = CompactionMetrics::default();
            entry.nudge_pending = false;
            entry.nudge_ignored_counted = false;
            entry.last_prune_requests = None;
            entry.last_prune_tokens = None;
            entry.raw_input_estimate = None;
            entry.effective_projection_estimate = None;
            entry.fixed_overhead_estimate = None;
            entry.last_projection_estimate = None;
            entry.last_raw_input_estimate = None;
            entry.prune_trigger_reason = None;
            entry.fallback_in_flight = false;
            entry.fallback_last_requests = None;
            // Refs and block ids are re-allocated for the new window; an old
            // archive could no longer be resolved to the same span.
            entry.fold_archives.clear();
            // Journal/receipts are keyed by the old window's spans too.
            entry.fold_journal.clear();
            entry.fold_receipts.clear();
            entry.fold_in_flight = false;
            entry.speculation = None;
            entry.speculation_in_flight = false;
            // Policy knobs are model-scoped, not window-scoped; keep them.
        }
        let result = f(entry);
        self.persist(key, entry);
        result
    }

    /// Backwards-compatible entry point: callers that only know the whole
    /// request estimate get zero fixed overhead. Hosts should use
    /// `prepare_with_budget` so system/tools are charged explicitly.
    #[allow(
        dead_code,
        reason = "convenience entry point retained for direct prepare callers"
    )]
    pub fn prepare(
        &self,
        key: &str,
        context_limit: u32,
        messages: &[billion_context::Message],
        token_count: u32,
        model: Option<String>,
    ) -> Prepared {
        self.prepare_with_budget(key, context_limit, messages, token_count, 0, model)
    }

    /// Project one request through the kernel with separated raw input and
    /// fixed request overhead. Hysteresis uses the effective projection
    /// estimate, not the raw request number.
    pub fn prepare_with_budget(
        &self,
        key: &str,
        context_limit: u32,
        messages: &[billion_context::Message],
        raw_input_estimate: u32,
        fixed_overhead_estimate: u32,
        model: Option<String>,
    ) -> Prepared {
        self.prepare_inner(
            key,
            context_limit,
            messages,
            raw_input_estimate,
            fixed_overhead_estimate,
            model,
            false,
        )
    }

    /// Test-only manual one-shot prune for the A group: artifact/stub
    /// mechanics without automatic hysteresis. Subsequent calls must use
    /// `prepare` so A cannot silently become B.
    #[cfg(test)]
    pub fn prepare_manual_prune(
        &self,
        key: &str,
        context_limit: u32,
        messages: &[billion_context::Message],
        raw_input_estimate: u32,
        fixed_overhead_estimate: u32,
        model: Option<String>,
    ) -> Prepared {
        self.prepare_inner(
            key,
            context_limit,
            messages,
            raw_input_estimate,
            fixed_overhead_estimate,
            model,
            true,
        )
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "the projection inputs are independent wire values"
    )]
    fn prepare_inner(
        &self,
        key: &str,
        context_limit: u32,
        messages: &[billion_context::Message],
        raw_input_estimate: u32,
        fixed_overhead_estimate: u32,
        model: Option<String>,
        force_prune: bool,
    ) -> Prepared {
        let now = now_ms();
        self.ensure_loaded(key);
        let (
            protect_ids,
            preserve_recent_messages,
            preserve_recent_tokens,
            requests,
            last_prune_requests,
            last_prune_tokens,
            _last_projection_estimate,
            _last_raw_input_estimate,
        ) = {
            let guard = self.inner.lock().expect("session store poisoned");
            match guard.get(key) {
                Some(entry) => {
                    let mut protect = HashSet::new();
                    for block in &entry.session.state().blocks {
                        if block.active || block.restored_inline {
                            for id in &block.effective {
                                protect.insert(id.base().to_string());
                            }
                        }
                    }
                    (
                        protect,
                        entry.session.config().preserve_recent_messages,
                        entry.session.config().preserve_recent_tokens,
                        entry.requests,
                        entry.last_prune_requests,
                        entry.last_prune_tokens,
                        entry.last_projection_estimate,
                        entry.last_raw_input_estimate,
                    )
                }
                None => (HashSet::new(), 3, 5_000, 0, None, None, None, None),
            }
        };

        // The host may resend the original, unpruned conversation. Re-apply
        // already stored stubs first; this is request assembly, not a new prune.
        // A raw estimate alone must never make a previously-pruned conversation
        // look unpruned again.
        let mut view = messages.to_vec();
        let reapplied = self
            .artifacts
            .as_ref()
            .map(|artifacts| apply_existing_stubs(&mut view, artifacts, &self.prune_policy))
            .unwrap_or(0);
        let assembled_projection = estimate_messages_tokens(&view);
        let effective_current = assembled_projection.saturating_add(fixed_overhead_estimate);
        let decision = if force_prune && self.artifacts.is_some() {
            crate::prune::TriggerDecision {
                triggered: true,
                reason: "manual_test",
                high_watermark: 0,
                growth_tokens: 0,
                cooldown_hit: false,
                growth_bypass: false,
                cooldown_blocked: false,
            }
        } else if self.artifacts.is_some() {
            should_trigger_decision(
                effective_current,
                context_limit,
                requests.saturating_add(1),
                last_prune_requests,
                last_prune_tokens,
                &self.prune_policy,
            )
        } else {
            crate::prune::TriggerDecision {
                triggered: false,
                reason: "no_artifact_store",
                high_watermark: 0,
                growth_tokens: 0,
                cooldown_hit: false,
                growth_bypass: false,
                cooldown_blocked: false,
            }
        };

        let mut report = PruneReport::default();
        if decision.triggered {
            report = prune_tool_results(
                &mut view,
                effective_current,
                context_limit,
                &protect_ids,
                preserve_recent_messages,
                preserve_recent_tokens,
                self.artifacts.as_ref(),
                &self.prune_policy,
                key,
            );
        }
        if !report.triggered && decision.triggered {
            report.reason = decision.reason.to_string();
        }
        let raw_messages = messages.to_vec();
        let pruned_ids = report.pruned_ids.clone();
        let report_for_return = report.clone();
        let projection_after = estimate_messages_tokens(&view);
        let effective_after = projection_after.saturating_add(fixed_overhead_estimate);
        let prune_succeeded = report.pruned_items > 0;

        self.with_entry(key, context_limit, |entry| {
            if entry.nudge_pending && !entry.nudge_ignored_counted {
                entry.metrics.nudge_ignored += 1;
                entry.nudge_ignored_counted = true;
            }
            let turn = entry.session.process_turn(TurnInput {
                messages: &view,
                token_count: projection_after,
                render: RenderStrategy::TextOnly,
                now,
            });
            // Resolve stable refs only after process_turn has allocated them;
            // the artifact metadata then gains source_ref without changing its
            // content-addressed artifact id or the original body.
            for id in &pruned_ids {
                if let Some(reference) = entry
                    .session
                    .state()
                    .refs
                    .ref_of(&billion_context::MessageId::new(id))
                {
                    if let Some(store) = self.artifacts.as_ref() {
                        let _ = store.set_source_ref_for_message(id, &reference.to_string());
                    }
                }
            }
            entry.tokens = effective_after;
            entry.last_seen = now;
            entry.last_messages = raw_messages;
            entry.requests += 1;
            entry.model = model.clone().or_else(|| entry.model.clone());
            entry.ranges = turn.ranges.clone();
            entry.raw_input_estimate = Some(raw_input_estimate);
            entry.fixed_overhead_estimate = Some(fixed_overhead_estimate);
            entry.effective_projection_estimate = Some(effective_after);
            entry.last_projection_estimate = Some(projection_after);
            entry.last_raw_input_estimate = Some(raw_input_estimate);
            entry.prune_trigger_reason = Some(decision.reason.to_string());
            entry.metrics.raw_input_estimate = raw_input_estimate as u64;
            entry.metrics.effective_projection_estimate = effective_after as u64;
            entry.metrics.fixed_overhead_estimate = fixed_overhead_estimate as u64;
            entry.metrics.prune_trigger_reason = decision.reason.to_string();
            entry.metrics.stub_reapply_items = reapplied;
            if decision.cooldown_hit {
                entry.metrics.cooldown_hits += 1;
            }
            if decision.cooldown_blocked {
                entry.metrics.cooldown_blocked += 1;
            }
            if decision.growth_bypass {
                entry.metrics.growth_bypass_triggered += 1;
            }
            if !prune_succeeded {
                entry.metrics.prune_skipped += 1;
            }
            if decision.triggered {
                entry.metrics.prune_attempts += 1;
                entry.metrics.prune_candidates += report.candidates;
                if prune_succeeded {
                    entry.metrics.prune_runs += 1;
                    entry.metrics.prune_items += report.pruned_items;
                    entry.metrics.prune_saved_tokens_est += report.saved_tokens_est;
                    entry.metrics.prune_saved_bytes += report.saved_bytes;
                    entry.metrics.artifact_writes += report.artifact_writes;
                    entry.metrics.artifact_bytes_written += report.artifact_bytes;
                    entry.last_prune_requests = Some(entry.requests);
                    entry.last_prune_tokens = Some(effective_after);
                    entry.metrics.last_change_ref = report
                        .first_changed_message_id
                        .as_deref()
                        .and_then(|id| {
                            entry
                                .session
                                .state()
                                .refs
                                .ref_of(&billion_context::MessageId::new(id))
                        })
                        .map(|reference| reference.index());
                }
            }
            // P0.3: the rule sheet is the bulk of a nudge (KEEP VERBATIM, the
            // objective rules, priorities — usually 2-3K tokens) and it never
            // changes between nudges. Deliver it in full on the first injection
            // of a session, then point back to it instead of re-sending it on
            // every turn the kernel wants to nudge.
            let armed_nudge =
                turn.nudge
                    .as_ref()
                    .filter(|nudge| nudge.should_inject)
                    .map(|nudge| {
                        billion_context::render_nudge_text(
                            nudge,
                            entry.session.state(),
                            &nudge_prompts(entry.metrics.nudge_injected),
                            protocol_named(&entry.protocol),
                        )
                        .text
                    });
            let nudge =
                armed_nudge.filter(|text| entry.last_nudge.as_deref() != Some(text.as_str()));
            if let Some(text) = nudge.clone() {
                entry.last_nudge = Some(text.clone());
                entry.metrics.nudge_injected += 1;
                entry.nudge_pending = true;
                entry.nudge_ignored_counted = false;
            }
            Prepared {
                messages: turn.messages,
                ranges: turn.ranges,
                nudge,
                terminal: turn.terminal.is_some(),
                warnings: turn.warnings,
                pruned_ids,
                prune: Some(report_for_return),
                decision_tokens: projection_after,
                raw_input_estimate,
                assembled_projection_estimate: assembled_projection,
                effective_projection_estimate: effective_after,
                fixed_overhead_estimate,
                prune_decision: decision,
                prune_triggered: prune_succeeded,
                prune_skipped: !prune_succeeded,
                stub_reapply_items: reapplied,
            }
        })
    }

    /// Forget the pending nudge once the model acted on it (or the fold landed),
    /// so the next projection re-injects only when the kernel wants it again.
    #[allow(dead_code, reason = "lifecycle helper retained for host integrations")]
    pub fn clear_nudge(&self, key: &str) {
        let mut guard = self.inner.lock().expect("session store poisoned");
        if let Some(entry) = guard.get_mut(key) {
            entry.last_nudge = None;
            entry.nudge_pending = false;
            entry.nudge_ignored_counted = false;
            self.persist(key, entry);
        }
    }

    /// Whether a host-initiated fold has a candidate right now: foldable ranges
    /// exist and neither a fallback nor a speculation already owns them.
    ///
    /// The compaction path asks this before it pays for a plan: with nothing
    /// foldable the fold-before-refuse attempt would abort on `no_chain` or an
    /// empty input list and only add a summarizer round trip to the refusal.
    pub fn has_foldable_ranges(&self, key: &str) -> bool {
        self.ensure_loaded(key);
        let guard = self.inner.lock().expect("session store poisoned");
        guard.get(key).is_some_and(|entry| {
            !entry.ranges.is_empty() && !entry.fallback_in_flight && !entry.speculation_in_flight
        })
    }

    /// Decide whether this conversation needs a host-initiated fold.
    ///
    /// Called before the request is projected, so the current turn already
    /// benefits from the fold. Marks the attempt in flight and persists the
    /// growth-window gate; the caller must finish with `apply_fallback` or
    /// `abort_fallback`, otherwise the range stays blocked until restart.
    pub fn fallback_plan(&self, key: &str) -> Option<FallbackPlan> {
        self.ensure_loaded(key);
        let mut guard = self.inner.lock().expect("session store poisoned");
        let entry = guard.get_mut(key)?;
        if entry.fallback_in_flight
            || entry.speculation_in_flight
            || entry.requests == 0
            || entry.ranges.is_empty()
        {
            return None;
        }
        if let Some(at) = entry.fallback_last_requests
            && entry.requests.saturating_sub(at) < FALLBACK_WINDOW_REQUESTS
        {
            return None;
        }
        let used = f64::from(entry.tokens) / f64::from(entry.context_limit.max(1));
        let pressure = used >= FALLBACK_PRESSURE_RATIO;
        let unanswered = entry.nudge_pending && entry.metrics.nudge_ignored >= 2;
        let reason = if pressure {
            "pressure"
        } else if unanswered {
            "ignored_nudge"
        } else {
            return None;
        };
        let min_chars = entry
            .min_compress_range
            .unwrap_or_else(|| entry.session.config().compress.min_compress_range)
            .max(1);
        // The oldest foldable range: the part of the conversation least likely
        // to be the current step's working set.
        let ranges: Vec<RequestedRange> = entry
            .ranges
            .iter()
            .find(|range| range.count >= 2 && range.chars >= min_chars)
            .map(|range| {
                vec![RequestedRange {
                    start: Boundary::Message(range.start),
                    end: Boundary::Message(range.end),
                    topic: None,
                    summary: String::new(),
                }]
            })
            .unwrap_or_default();
        if ranges.is_empty() {
            return None;
        }
        entry.fallback_in_flight = true;
        entry.fallback_last_requests = Some(entry.requests);
        entry.metrics.fallback_attempts += 1;
        entry.metrics.fallback_last_reason = reason.to_string();
        self.persist(key, entry);
        Some(FallbackPlan { reason, ranges })
    }

    /// Slice text for the planned ranges, shaped for the summarizer chain.
    pub fn fallback_inputs(
        &self,
        key: &str,
        plan: &FallbackPlan,
    ) -> Vec<crate::summarizer::RangeInput> {
        self.ensure_loaded(key);
        let Ok(guard) = self.inner.lock() else {
            return Vec::new();
        };
        let Some(entry) = guard.get(key) else {
            return Vec::new();
        };
        let mut inputs = Vec::new();
        for range in &plan.ranges {
            let Some(text) = message_slice_text(&entry.last_messages, &range.start, &range.end)
            else {
                continue;
            };
            inputs.push(crate::summarizer::RangeInput {
                label: format!("{}–{}", range.start, range.end),
                topic: None,
                text,
            });
        }
        inputs
    }

    /// Land the fallback fold. Returns `None` when no attempt is in flight, so a
    /// racing request cannot fold the same range twice.
    pub fn apply_fallback(
        &self,
        key: &str,
        context_limit: u32,
        plan: FallbackPlan,
        summaries: &[String],
        now: Timestamp,
    ) -> Option<String> {
        let artifacts = self.artifacts.clone();
        self.with_entry(key, context_limit, |entry| {
            if !entry.fallback_in_flight {
                return None;
            }
            entry.fallback_in_flight = false;
            let mut ranges = plan.ranges.clone();
            for (range, summary) in ranges.iter_mut().zip(summaries.iter()) {
                range.summary = summary.clone();
            }
            // The same gate as a model-requested fold: an external summarizer
            // can echo its instructions or answer with a placeholder, and that
            // must never replace a live range with nothing.
            if let Some((label, defect)) = first_summary_defect(&ranges) {
                entry.metrics.fallback_refused += 1;
                return Some(format!(
                    "[auto-fallback refused: the summary for {label} {defect}; nothing was folded]"
                ));
            }
            // An external summarizer is likelier than the model to drop the
            // user's own words, so the P0.1 carry applies here too.
            let max_summary = entry.max_summary_length.unwrap_or_else(|| entry.session.config().compress.max_summary_length);
            for range in ranges.iter_mut() {
                let (Boundary::Message(start), Boundary::Message(end)) = (&range.start, &range.end)
                else {
                    continue;
                };
                let anchors = summary_carry::user_anchors(
                    &entry.last_messages,
                    (start.index(), end.index()),
                    |id| entry.session.state().refs.ref_of(id).map(|r| r.to_string()),
                );
                let Some(section) = summary_carry::render_carried(
                    &anchors,
                    &range.summary,
                    summary_carry::CarryLimits::default(),
                ) else {
                    continue;
                };
                let Some(augmented) =
                    summary_carry::augment(&range.summary, Some(&section), max_summary)
                else {
                    continue;
                };
                range.summary = augmented;
            }
            let applied = entry.session.apply_compression(CompressRequest {
                ranges,
                now,
                // The host, not a tool call, asked for this fold.
                call_id: None,
            });
            let at_requests = entry.requests;
            let reason = format!("auto:{}", plan.reason);
            for block_id in &applied.blocks {
                if let Some(block) = entry.session.block(*block_id) {
                    let change_ref = block.span.and_then(|(start, _)| match start {
                        Boundary::Message(reference) => Some(reference.index()),
                        Boundary::Block(_) => None,
                    });
                    entry.fold_events.push(FoldEvent {
                        at_requests,
                        s: block.compressed_tokens as u64,
                        sigma: estimate_summary_tokens(&block.summary),
                        block: block.id.to_string(),
                        change_ref,
                        reason: Some(reason.clone()),
                    });
                }
            }
            if entry.fold_events.len() > 512 {
                let excess = entry.fold_events.len() - 512;
                entry.fold_events.drain(0..excess);
            }
            archive_fold_spans(artifacts.as_ref(), entry, key, &applied.blocks);
            entry.metrics.fallback_applied_blocks += applied.blocks.len() as u64;
            entry.metrics.fold_applied += applied.blocks.len() as u64;
            if applied.blocks.is_empty() {
                entry.metrics.fallback_refused += 1;
                entry.metrics.fold_refused += 1;
            }
            entry.last_nudge = None;
            entry.nudge_pending = false;
            entry.nudge_ignored_counted = false;
            let mut text = tool::render_compress(&applied.blocks, &applied.errors);
            text.push_str(&format!(
                "\n[auto-fallback: {reason} — the host folded this range itself because the model left the nudge unanswered. Result: {} block(s), {} error(s)]",
                applied.blocks.len(),
                applied.errors.len()
            ));
            Some(text)
        })
    }

    /// The chain produced nothing (unconfigured, timeout, 5xx): release the
    /// in-flight marker, count the failure, leave the projection untouched.
    pub fn abort_fallback(&self, key: &str, reason: &str) {
        let mut guard = self.inner.lock().expect("session store poisoned");
        let Some(entry) = guard.get_mut(key) else {
            return;
        };
        if !entry.fallback_in_flight {
            return;
        }
        entry.fallback_in_flight = false;
        entry.metrics.fallback_failures += 1;
        entry.metrics.fallback_last_reason = format!("{reason}:failed");
        self.persist(key, entry);
    }

    // ---- Phase 3.1: speculative folds of already-consumed blocks ----------

    /// Open a speculative candidate when the conversation sits in the 45–75%
    /// band, nothing is folding right now, and the cooldown has elapsed. The
    /// candidate is a snapshot (block ids + per-block signatures + the T1
    /// summaries to distill); it is marked in flight so a second request cannot
    /// open a competing plan while the summarizer runs.
    pub fn speculation_plan(&self, key: &str) -> Option<SpeculationPlan> {
        self.ensure_loaded(key);
        let mut guard = self.inner.lock().expect("session store poisoned");
        let entry = guard.get_mut(key)?;
        if entry.speculation_in_flight || entry.fold_in_flight || entry.requests == 0 {
            return None;
        }
        let ratio = f64::from(entry.tokens) / f64::from(entry.context_limit.max(1));
        if !(SPECULATION_BAND_LOW..=SPECULATION_BAND_HIGH).contains(&ratio) {
            return None;
        }
        if let Some(at) = entry.speculation_last_requests
            && entry.requests.saturating_sub(at) < SPECULATION_COOLDOWN_REQUESTS
        {
            return None;
        }
        let blocks: Vec<Block> = entry
            .session
            .state()
            .blocks
            .iter()
            .filter(|block| block.tier == Tier::T1 && block.active)
            .take(SPECULATION_MAX_BLOCKS)
            .cloned()
            .collect();
        if blocks.is_empty() {
            return None;
        }
        let plan = SpeculationPlan {
            at_requests: entry.requests,
            model: entry.model.clone(),
            blocks: blocks.iter().map(|block| block.id.to_string()).collect(),
            signatures: blocks.iter().map(speculation_signature).collect(),
            inputs: blocks.iter().map(|block| block.summary.clone()).collect(),
        };
        entry.speculation_in_flight = true;
        entry.speculation = Some(plan.clone());
        entry.speculation_last_requests = Some(entry.requests);
        entry.metrics.speculation_started += 1;
        self.persist(key, entry);
        Some(plan)
    }

    /// The candidate was abandoned before the summarizer answered (no chain,
    /// timeout, empty output). Release the marker so the next band window can
    /// try again; the abandoned attempt counts as discarded, not adopted.
    pub fn abort_speculation(&self, key: &str, _reason: &str) {
        let mut guard = self.inner.lock().expect("session store poisoned");
        let Some(entry) = guard.get_mut(key) else {
            return;
        };
        if !entry.speculation_in_flight {
            return;
        }
        entry.speculation_in_flight = false;
        entry.speculation = None;
        entry.metrics.speculation_discarded += 1;
        self.persist(key, entry);
    }

    /// Task 2: reserve one automatic summary dispatch before the chain is
    /// called. Returns `Err(reason)` when the shared session budget or the
    /// speculation sub-budget is exhausted; nothing is called in that case, so a
    /// refusal can never turn into a fold. Manual tool-requested compresses skip
    /// this reservation on purpose: an exhausted automation budget must not
    /// break the normal conversation.
    pub fn reserve_summary_call(
        &self,
        key: &str,
        speculative: bool,
        max_calls: u64,
        max_speculation_calls: u64,
    ) -> Result<(), String> {
        self.ensure_loaded(key);
        let mut guard = self.inner.lock().expect("session store poisoned");
        let Some(entry) = guard.get_mut(key) else {
            return Err("unknown conversation".to_string());
        };
        let refusal = if speculative {
            if max_speculation_calls == 0 {
                Some("speculation summaries are disabled (maxSpeculationCalls=0)".to_string())
            } else if entry.summary_budget.calls >= max_calls {
                Some(format!(
                    "session summary budget exhausted ({}/{max_calls})",
                    entry.summary_budget.calls
                ))
            } else if entry.summary_budget.speculation_calls >= max_speculation_calls {
                Some(format!(
                    "speculation budget exhausted ({}/{max_speculation_calls})",
                    entry.summary_budget.speculation_calls
                ))
            } else {
                None
            }
        } else if entry.summary_budget.calls >= max_calls {
            Some(format!(
                "session summary budget exhausted ({}/{max_calls})",
                entry.summary_budget.calls
            ))
        } else {
            None
        };
        if let Some(reason) = refusal {
            entry.summary_budget.refused += 1;
            entry.summary_budget.last_refusal = Some(reason.clone());
            self.persist(key, entry);
            return Err(reason);
        }
        entry.summary_budget.calls += 1;
        if speculative {
            entry.summary_budget.speculation_calls += 1;
        }
        self.persist(key, entry);
        Ok(())
    }

    /// Task 2: charge the tokens/calls one chain batch actually used. The
    /// reservation already counted the first dispatch; a batch that fell back to
    /// a second target spent extra dispatches and is charged here. Provider
    /// `usage` values win; missing ones are counted as estimates.
    pub fn record_summary_usage(
        &self,
        key: &str,
        speculative: bool,
        batch: &crate::summarizer::SummaryBatch,
    ) {
        let mut guard = self.inner.lock().expect("session store poisoned");
        let Some(entry) = guard.get_mut(key) else {
            return;
        };
        let extra = batch.calls.saturating_sub(1);
        entry.summary_budget.calls += extra;
        if speculative {
            entry.summary_budget.speculation_calls += extra;
        }
        entry.summary_budget.input_tokens += batch.input_tokens;
        entry.summary_budget.output_tokens += batch.output_tokens;
        entry.summary_budget.estimated_calls += batch.estimated_calls;
        self.persist(key, entry);
    }

    /// Adopt the candidate: re-derive the block signatures under the lock and
    /// only fold when every planned block is still there, unchanged, under the
    /// same model. Anything else is a discard — a shifted block means the
    /// summaries describe text that has already moved on.
    pub fn apply_speculation(
        &self,
        key: &str,
        context_limit: u32,
        plan: SpeculationPlan,
        summaries: &[String],
        now: Timestamp,
    ) -> Option<String> {
        self.with_entry(key, context_limit, |entry| {
            if !entry.speculation_in_flight {
                return None;
            }
            entry.speculation_in_flight = false;
            let mismatch = if entry.model != plan.model {
                Some("the routed model changed".to_string())
            } else if summaries.len() != plan.blocks.len() {
                Some("the summarizer returned the wrong number of sections".to_string())
            } else {
                let mut mismatch = None;
                for (block_id, signature) in plan.blocks.iter().zip(plan.signatures.iter()) {
                    let current = entry
                        .session
                        .state()
                        .blocks
                        .iter()
                        .find(|block| block.id.to_string() == *block_id)
                        .map(speculation_signature);
                    if current.as_deref() != Some(signature.as_str()) {
                        mismatch = Some(format!("block {block_id} changed under the candidate"));
                        break;
                    }
                }
                mismatch
            };
            if let Some(reason) = mismatch {
                entry.speculation = None;
                entry.metrics.speculation_discarded += 1;
                return Some(format!(
                    "[speculation discarded: {reason}; the candidate was not folded]"
                ));
            }
            // One range per block: folding a block yields one tier above it, so
            // these land as T2 and the T1 summaries they replace stay archived
            // only through the artifact escape hatch when they were consumed.
            let max_chars = entry
                .max_summary_length
                .unwrap_or_else(|| entry.session.config().compress.max_summary_length);
            let ranges: Vec<RequestedRange> = plan
                .blocks
                .iter()
                .zip(summaries.iter())
                .filter_map(|(block_id, summary)| {
                    let start = BlockId::parse(block_id.as_str())?;
                    let mut text = summary.clone();
                    if text.chars().count() > max_chars {
                        text = text.chars().take(max_chars).collect();
                    }
                    Some(RequestedRange {
                        start: Boundary::Block(start),
                        end: Boundary::Block(start),
                        topic: None,
                        summary: text,
                    })
                })
                .collect();
            if let Some((label, defect)) = first_summary_defect(&ranges) {
                entry.speculation = None;
                entry.metrics.speculation_discarded += 1;
                return Some(format!(
                    "[speculation discarded: the summary for {label} {defect}; nothing was folded]"
                ));
            }
            entry.fold_in_flight = true;
            entry.speculation = None;
            let applied = entry.session.apply_compression(CompressRequest {
                ranges,
                now,
                call_id: None,
            });
            entry.fold_in_flight = false;
            let at_requests = entry.requests;
            for block_id in &applied.blocks {
                if let Some(block) = entry.session.block(*block_id) {
                    entry.fold_events.push(FoldEvent {
                        at_requests,
                        s: block.compressed_tokens as u64,
                        sigma: estimate_summary_tokens(&block.summary),
                        block: block.id.to_string(),
                        change_ref: None,
                        reason: Some("speculative:t2".to_string()),
                    });
                }
            }
            if entry.fold_events.len() > 512 {
                let excess = entry.fold_events.len() - 512;
                entry.fold_events.drain(0..excess);
            }
            entry.metrics.fold_applied += applied.blocks.len() as u64;
            if applied.blocks.is_empty() {
                entry.metrics.fold_refused += 1;
                entry.metrics.speculation_discarded += 1;
                return Some(format!(
                    "[speculation refused: {} error(s)]",
                    applied.errors.len()
                ));
            }
            entry.metrics.speculation_adopted += 1;
            let mut text = tool::render_compress(&applied.blocks, &applied.errors);
            text.push_str(&format!(
                "\n[speculation: adopted — {} block(s) folded one tier above the planned T1 blocks, generation check passed]",
                applied.blocks.len()
            ));
            Some(text)
        })
    }

    /// Phase 3.2: apply the routed model's policy to this conversation's entry.
    /// Only `Some` fields are written, so an entry keeps defaults for anything
    /// the policy does not name; the window itself is enforced by `with_entry`
    /// (a changed limit rebuilds the kernel session).
    pub fn apply_model_policy(&self, key: &str, context_limit: u32, policy: &ModelPolicy) {
        if policy.context_window.is_none()
            && policy.min_compress_range.is_none()
            && policy.max_summary_length.is_none()
        {
            return;
        }
        self.with_entry(key, context_limit, |entry| {
            if let Some(min) = policy.min_compress_range {
                entry.min_compress_range = Some(min);
            }
            if let Some(max) = policy.max_summary_length {
                entry.max_summary_length = Some(max);
            }
        });
    }

    pub fn tool(
        &self,
        key: &str,
        context_limit: u32,
        name: &str,
        args: &Value,
    ) -> (String, Option<&'static str>, Option<u32>) {
        let now = now_ms();
        let artifacts = self.artifacts.clone();
        self.with_entry(key, context_limit, |entry| match name {
            "acp_compress" | "compress" => match tool::parse_compress(&normalize_compress_args(args)) {
                Ok(ranges) => {
                    let call_id = args
                        .get("tool_call_id")
                        .and_then(Value::as_str)
                        .map(str::to_string);
                    // P3 §2.2: the same tool_call_id arriving twice is a replay
                    // of an exchange that already landed, not a second fold.
                    if let Some(call_id) = &call_id
                        && let Some(receipt) =
                            entry.fold_receipts.iter().find(|receipt| &receipt.call_id == call_id)
                    {
                        entry.metrics.idempotent_replays += 1;
                        return (
                            format!(
                                "{}\n[idempotent replay: call {call_id} already folded {} block(s) at request {}; nothing was folded again]",
                                receipt.text,
                                receipt.blocks.len(),
                                receipt.requests
                            ),
                            Some("success"),
                            Some(receipt.blocks.len() as u32),
                        );
                    }
                    // A speculative candidate holds a snapshot of the blocks it
                    // plans to fold; a foreground fold now would invalidate it.
                    if entry.speculation_in_flight {
                        entry.metrics.fold_busy_refused += 1;
                        return (
                            "[compress busy: a speculative fold is in flight; retry on the next turn.]".to_string(),
                            Some("busy"),
                            Some(0),
                        );
                    }
                    // An open journal means a fold started and never closed
                    // (crash between apply and snapshot). Never fold on top of
                    // residue: refuse, and close the stale entry so the retry
                    // can run on a clean journal.
                    if entry.fold_in_flight
                        || entry.fold_journal.iter().any(|journal| journal.phase != "end")
                    {
                        entry.metrics.fold_busy_refused += 1;
                        for journal in entry
                            .fold_journal
                            .iter_mut()
                            .filter(|journal| journal.phase != "end")
                        {
                            journal.phase = "end".to_string();
                            journal.note = "closed on busy refusal".to_string();
                        }
                        entry.fold_in_flight = false;
                        return (
                            "[compress busy: a previous fold did not close its journal (the host stopped between apply and snapshot). Nothing was folded on this call; retry on the next turn.]".to_string(),
                            Some("busy"),
                            Some(0),
                        );
                    }
                    // `summaryMaxChars` (Node parity) raises the summary ceiling
                    // for THIS call only, so an over-long summary can still land
                    // instead of the whole batch being rejected (#7762).
                    let per_call_max = summary_max_chars(args);
                    let min = entry.min_compress_range.unwrap_or_else(|| entry.session.config().compress.min_compress_range);
                    if let Some(chars) = message_range_chars(&entry.last_messages, &ranges)
                        && chars > 0
                        && chars < min
                    {
                        return (
                            format!(
                                "[Compression FAILED: Total compressible content too small ({chars} chars across {} range(s), min {min}). Combine more messages into your range(s) to meet the threshold.]",
                                ranges.len()
                            ),
                            Some("refused"),
                            Some(0),
                        );
                    }
                    // P0.2: a summary that is a placeholder, a refusal or an
                    // echo of the instructions would replace the range with
                    // nothing, and the range is already gone from the view.
                    // Refuse before anything folds so the model can rewrite it.
                    if let Some((label, defect)) = first_summary_defect(&ranges) {
                        entry.metrics.summary_quality_refused += 1;
                        return (
                            format!(
                                "[Compression FAILED: the summary for {label} {defect}. Nothing was folded. Rewrite it as dense bullets: what happened in the range, the decisions and their rationale, exact paths/errors, and an `Open objectives:` line for user requests that are still open.]"
                            ),
                            Some("refused"),
                            Some(0),
                        );
                    }
                    // P0.1: the host still holds the raw user turns of the
                    // span, so the ones the summary did not cite are carried
                    // into the folded record verbatim instead of being lost.
                    let mut ranges = ranges;
                    let max_summary = per_call_max
                        .or(entry.max_summary_length)
                        .unwrap_or_else(|| entry.session.config().compress.max_summary_length);
                    let mut carried_turns = 0u64;
                    let mut carried_chars = 0usize;
                    for range in ranges.iter_mut() {
                        let (Boundary::Message(start), Boundary::Message(end)) =
                            (&range.start, &range.end)
                        else {
                            continue;
                        };
                        let anchors = summary_carry::user_anchors(
                            &entry.last_messages,
                            (start.index(), end.index()),
                            |id| entry.session.state().refs.ref_of(id).map(|r| r.to_string()),
                        );
                        let Some(section) = summary_carry::render_carried(
                            &anchors,
                            &range.summary,
                            summary_carry::CarryLimits::default(),
                        ) else {
                            continue;
                        };
                        let Some(augmented) =
                            summary_carry::augment(&range.summary, Some(&section), max_summary)
                        else {
                            continue;
                        };
                        carried_turns += section.matches("- m").count() as u64;
                        carried_chars += section.chars().count();
                        range.summary = augmented;
                    }
                    entry.metrics.intent_anchors_carried += carried_turns;
                    // P3 §2.2 transaction markers: `start` before the kernel
                    // fold, `applied` right after it, `end` once the host-side
                    // bookkeeping (metrics, archives, receipts) is complete.
                    let spans: Vec<String> = ranges
                        .iter()
                        .map(|range| format!("{}–{}", range.start, range.end))
                        .collect();
                    entry.fold_journal.push(FoldJournal {
                        call_id: call_id.clone(),
                        spans,
                        blocks: Vec::new(),
                        phase: "start".to_string(),
                        at_requests: entry.requests,
                        at: now,
                        note: String::new(),
                    });
                    if entry.fold_journal.len() > 32 {
                        let excess = entry.fold_journal.len() - 32;
                        entry.fold_journal.drain(0..excess);
                    }
                    entry.fold_in_flight = true;
                    let applied =
                        entry
                            .session
                            .apply_compression_with_summary_limit(
                                CompressRequest {
                                    ranges,
                                    now,
                                    call_id: call_id.clone(),
                                },
                                max_summary,
                            );
                    if let Some(journal) = entry.fold_journal.last_mut() {
                        journal.phase = "applied".to_string();
                        journal.blocks = applied.blocks.iter().map(ToString::to_string).collect();
                        journal.at = now;
                    }
                    let at_requests = entry.requests;
                    for block_id in &applied.blocks {
                        if let Some(block) = entry.session.block(*block_id) {
                            let change_ref = block.span.and_then(|(start, _)| match start {
                                Boundary::Message(reference) => Some(reference.index()),
                                Boundary::Block(_) => None,
                            });
                            entry.fold_events.push(FoldEvent {
                                at_requests,
                                s: block.compressed_tokens as u64,
                                sigma: estimate_summary_tokens(&block.summary),
                                block: block.id.to_string(),
                                change_ref,
                                reason: None,
                            });
                        }
                    }
                    if entry.fold_events.len() > 512 {
                        let excess = entry.fold_events.len() - 512;
                        entry.fold_events.drain(0..excess);
                    }
                    // P2: keep the raw span addressable before the projection
                    // forgets it; the model-visible record stays the summary.
                    archive_fold_spans(artifacts.as_ref(), entry, key, &applied.blocks);
                    let mut text = tool::render_compress(&applied.blocks, &applied.errors);
                    // The refusal has to name the repair it accepts, or the model
                    // rewrites the summary blind. `summaryMaxChars` is the only
                    // way to keep a summary that the session default is too small
                    // for; the alternative is a shorter summary over more ranges.
                    if applied
                        .errors
                        .iter()
                        .any(|error| error.kind == RangeErrorKind::SummaryTooLong)
                    {
                        text.push_str(&format!(
                            "\n[hint: this summary is longer than the {} char ceiling. Pass `summaryMaxChars` on the next compress call to allow up to {} chars for that call only, or split the range into smaller ones.]",
                            max_summary,
                            SUMMARY_MAX_CHARS_CEIL
                        ));
                    }
                    if carried_turns > 0 {
                        text.push_str(&format!(
                            "\n[host-carried user intent: {carried_turns} user turn(s) inside the folded span, {carried_chars} chars appended because the summary did not cite them]"
                        ));
                    }
                    let outcome = if applied.errors.is_empty() {
                        "applied"
                    } else if applied.blocks.is_empty() {
                        "refused"
                    } else {
                        "partial"
                    };
                    entry.metrics.fold_applied += applied.blocks.len() as u64;
                    if applied.blocks.is_empty() {
                        entry.metrics.fold_refused += 1;
                    } else if !applied.errors.is_empty() {
                        entry.metrics.fold_partial += 1;
                    }
                    entry.last_nudge = None;
                    entry.nudge_pending = false;
                    entry.nudge_ignored_counted = false;
                    // Close the journal, then remember the exchange so a replay
                    // of this call returns exactly this text and block count.
                    if let Some(journal) = entry.fold_journal.last_mut() {
                        journal.phase = "end".to_string();
                    }
                    if let Some(call_id) = &call_id
                        && !applied.blocks.is_empty()
                    {
                        entry.fold_receipts.push(FoldReceipt {
                            call_id: call_id.clone(),
                            text: text.clone(),
                            blocks: applied.blocks.iter().map(ToString::to_string).collect(),
                            requests: entry.requests,
                        });
                        if entry.fold_receipts.len() > 16 {
                            let excess = entry.fold_receipts.len() - 16;
                            entry.fold_receipts.drain(0..excess);
                        }
                    }
                    entry.fold_in_flight = false;
                    (
                        text,
                        Some(outcome),
                        Some(applied.blocks.len() as u32),
                    )
                }
                Err(err) => (format!("[compress refused: {err}]"), Some("refused"), Some(0)),
            },
            "acp_retrieve" | "retrieve" => match tool::parse_retrieve(args) {
                // P2: `mode: "original"` is the documented spelling of
                // `retrieve_raw`; both reach the same explicit raw path.
                Ok(refs) if args.get("mode").and_then(Value::as_str) == Some("original") => {
                    let _ = refs;
                    let (text, outcome) = retrieve_raw(entry, artifacts.as_ref(), key, args);
                    (text, outcome, None)
                }
                Ok(refs) => (tool::render_retrieval(&entry.session.retrieve(&refs)), Some("success"), None),
                Err(err) => (format!("[retrieve refused: {err}]"), Some("refused"), None),
            },
            "acp_retrieve_raw" | "retrieve_raw" => {
                let (text, outcome) = retrieve_raw(entry, artifacts.as_ref(), key, args);
                (text, outcome, None)
            }
            "acp_decompress" | "decompress" => match tool::parse_decompress(args) {
                Ok(id) => {
                    let outcome = entry.session.decompress(&[id]);
                    let restored = outcome
                        .blocks
                        .first()
                        .and_then(|id| entry.session.block(*id).cloned());
                    match restored {
                        Some(block) => (tool::render_decompression(&block), Some("success"), None),
                        None => {
                            // P2: the kernel cannot restore what it no longer
                            // holds. Say so, and point at the archived original
                            // instead of pretending the summary is the text.
                            let text = match entry
                                .fold_archives
                                .iter()
                                .find(|archive| archive.block == id.to_string())
                            {
                                Some(archive) => format!(
                                    "[decompress refused: OriginalUnavailable in this session — block {id}'s folded span survives only as artifact {} ({} bytes). Nothing was re-projected; call retrieve_raw with block \"{}\" if you need the text.]",
                                    archive.artifact_id, archive.byte_len, archive.block
                                ),
                                None => format!(
                                    "[decompress refused: OriginalUnavailable — block {id} is not in this session and no archived original exists for it. Nothing was re-projected.]"
                                ),
                            };
                            (text, Some("refused"), None)
                        }
                    }
                }
                Err(err) => (format!("[decompress refused: {err}]"), Some("refused"), None),
            },
            // Phase 3.3: explicit fresh-window reset. Never automatic: it drops
            // the current window's blocks and refs, so the host hands back a
            // task brief plus the recent tail and keeps every artifact id in a
            // receipt, making the originals reachable through retrieve_raw.
            "acp_reset_window" | "reset_window" => {
                if entry.fold_in_flight {
                    (
                        "[reset refused: a fold is in flight; retry after it lands]".to_string(),
                        Some("busy"),
                        None,
                    )
                } else {
                    let reason = args
                        .get("reason")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    let keep_recent = args
                        .get("keepRecent")
                        .or_else(|| args.get("keep_recent"))
                        .and_then(Value::as_u64)
                        .map(|count| count as usize)
                        .unwrap_or(6)
                        .min(32);
                    let brief = args
                        .get("brief")
                        .and_then(Value::as_str)
                        .map(str::trim)
                        .filter(|brief| !brief.is_empty())
                        .map(str::to_string)
                        .or_else(|| {
                            entry
                                .last_messages
                                .iter()
                                .rev()
                                .find(|message| matches!(message.role, Role::User))
                                .and_then(|message| match &message.content {
                                    Content::Text(text) => Some(text.trim().to_string()),
                                    _ => None,
                                })
                                .filter(|text| !text.is_empty())
                                .map(|text| text.chars().take(2_000).collect::<String>())
                        });
                    let dropped_blocks = entry.session.state().blocks.len();
                    let artifacts: Vec<String> = entry
                        .fold_archives
                        .iter()
                        .map(|archive| archive.artifact_id.clone())
                        .collect();
                    let tail_start = entry.last_messages.len().saturating_sub(keep_recent);
                    let mut header = format!(
                        "## Fresh window (explicit reset)\nreason: {}\ndropped folded blocks: {dropped_blocks}\nkept tail: last {} of {} messages\n",
                        if reason.is_empty() { "(none given)" } else { &reason },
                        entry.last_messages.len().saturating_sub(tail_start),
                        entry.last_messages.len(),
                    );
                    if !artifacts.is_empty() {
                        header.push_str(&format!(
                            "artifacts retained (not re-projected; reach them with artifact_read/retrieve_raw): {}\n",
                            artifacts.join(", ")
                        ));
                    }
                    if let Some(brief) = brief.as_deref() {
                        header.push_str(&format!("task brief:\n{brief}\n"));
                    }
                    let mut messages: Vec<Message> = vec![Message::new(
                        format!("reset-brief-{}", now_ms()),
                        Role::User,
                        Content::Text(header.into()),
                    )];
                    messages.extend(entry.last_messages[tail_start..].iter().cloned());
                    let token_count = estimate_messages_tokens(&messages) as u32;
                    let mut fresh = Session::new(Config::new(entry.context_limit))
                        .with_estimator(Self::estimator_for())
                        .with_view_mode(billion_context::ViewMode::Send)
                        .with_reconcile(true);
                    fresh.process_turn(TurnInput {
                        messages: &messages,
                        token_count,
                        render: RenderStrategy::TextOnly,
                        now: now_ms(),
                    });
                    entry.session = fresh;
                    entry.tokens = token_count;
                    entry.last_messages = messages;
                    entry.ranges.clear();
                    entry.last_nudge = None;
                    entry.last_seen = 0;
                    entry.fold_events.clear();
                    entry.fold_archives.clear();
                    entry.fold_journal.clear();
                    entry.fold_receipts.clear();
                    entry.context_usage = None;
                    entry.last_prune_requests = None;
                    entry.last_prune_tokens = None;
                    entry.raw_input_estimate = None;
                    entry.effective_projection_estimate = None;
                    entry.fixed_overhead_estimate = None;
                    entry.last_projection_estimate = None;
                    entry.last_raw_input_estimate = None;
                    entry.prune_trigger_reason = None;
                    entry.fallback_last_requests = None;
                    entry.speculation = None;
                    entry.speculation_in_flight = false;
                    entry.speculation_last_requests = None;
                    let kept = entry.last_messages.len();
                    let artifact_note = if artifacts.is_empty() {
                        "(none)".to_string()
                    } else {
                        format!(
                            "{} archived original(s) still reachable: {}",
                            artifacts.len(),
                            artifacts.iter().take(8).cloned().collect::<Vec<_>>().join(", ")
                        )
                    };
                    entry.window_resets.push(WindowReset {
                        at_requests: entry.requests,
                        reason: if reason.is_empty() {
                            "-".to_string()
                        } else {
                            reason.clone()
                        },
                        dropped_blocks,
                        kept_messages: kept,
                        artifacts,
                    });
                    if entry.window_resets.len() > 16 {
                        entry.window_resets.remove(0);
                    }
                    let text = format!(
                        "## fresh window reset — dropped {dropped_blocks} folded blocks, kept {kept} messages (1 brief + tail), artifacts retained: {artifact_note}\n[explicit reset: not compression. The dropped folds' summaries are gone; only the archived originals remain, and this is not revoked by a later fold.]"
                    );
                    (text, Some("success"), None)
                }
            },
            "acp_search" | "search" | "search_context" => match tool::parse_search(args) {
                Ok(search) => (
                    tool::render_search(&entry.session.search(&search.query, search.limit)),
                    Some("success"),
                    None,
                ),
                Err(err) => (format!("[search refused: {err}]"), Some("refused"), None),
            },
            "acp_status" | "status" | "acp_cache" => {
                let status = entry.session.status();
                (
                    tool::render_status(&status, entry.tokens as usize, &entry.ranges),
                    Some("success"),
                    None,
                )
            }
            "artifact_info" | "artifact.info" => {
                let result = artifact_target(entry, artifacts.as_ref(), args, key)
                    .and_then(|meta| {
                        serde_json::to_string_pretty(&meta)
                            .map_err(|e| format!("artifact.info serialize: {e}"))
                    });
                match result {
                    Ok(text) => (text, Some("success"), None),
                    Err(err) => (format!("[artifact.info unavailable: {err}]"), Some("refused"), None),
                }
            }
            "artifact_read" | "artifact.read" => {
                let result = artifact_target(entry, artifacts.as_ref(), args, key).and_then(|meta| {
                    let offset = args.get("offset").and_then(Value::as_u64).unwrap_or(0);
                    let limit = args.get("limit").and_then(Value::as_u64).unwrap_or(8_192);
                    artifacts
                        .as_ref()
                        .ok_or_else(|| "no artifact store".to_string())?
                        .read(&meta.artifact_id, offset, limit)
                        .map_err(|e| e.to_string())
                });
                match result {
                    Ok(read) => {
                        entry.metrics.artifact_reads += 1;
                        entry.metrics.artifact_read_bytes += read.text.len() as u64;
                        match serde_json::to_string(&read) {
                            Ok(text) => (text, Some("success"), None),
                            Err(err) => (format!("[artifact.read serialize: {err}]"), Some("refused"), None),
                        }
                    }
                    Err(err) => (format!("[artifact.read unavailable: {err}]"), Some("refused"), None),
                }
            }
            "artifact_search" | "artifact.search" => {
                let result = artifact_target(entry, artifacts.as_ref(), args, key).and_then(|meta| {
                    let query = args
                        .get("query")
                        .and_then(Value::as_str)
                        .ok_or_else(|| "artifact.search needs query".to_string())?;
                    let limit = args
                        .get("limit")
                        .and_then(Value::as_u64)
                        .unwrap_or(5) as usize;
                    let max_bytes = args
                        .get("max_bytes")
                        .and_then(Value::as_u64)
                        .unwrap_or(4_096);
                    artifacts
                        .as_ref()
                        .ok_or_else(|| "no artifact store".to_string())?
                        .search(&meta.artifact_id, query, limit, max_bytes)
                        .map_err(|e| e.to_string())
                });
                match result {
                    Ok(search) => {
                        entry.metrics.artifact_reads += 1;
                        entry.metrics.artifact_read_bytes += search.returned_bytes;
                        match serde_json::to_string(&search) {
                            Ok(text) => (text, Some("success"), None),
                            Err(err) => (format!("[artifact.search serialize: {err}]"), Some("refused"), None),
                        }
                    }
                    Err(err) => (format!("[artifact.search unavailable: {err}]"), Some("refused"), None),
                }
            }
            other => (format!("[unknown tool: {other}]"), Some("refused"), None),
        })
    }

    /// Record which wire and upstream this conversation is being served on,
    /// for the dashboard's session list.
    pub fn set_meta(&self, key: &str, protocol: &str, upstream: &str) {
        let mut guard = self.inner.lock().expect("session store poisoned");
        if let Some(entry) = guard.get_mut(key) {
            if !protocol.is_empty() {
                entry.protocol = protocol.to_string();
            }
            if !upstream.is_empty() {
                entry.upstream = upstream.to_string();
            }
        }
    }

    /// Add one upstream response's billing numbers to the conversation totals.
    /// Only called at end of stream, so one response adds exactly once.
    pub fn record_usage(&self, key: &str, request_model: Option<&str>, usage: Usage) {
        let mut guard = self.inner.lock().expect("session store poisoned");
        let Some(entry) = guard.get_mut(key) else {
            return;
        };
        entry.input_tokens += usage.input;
        entry.cached_tokens += usage.cached;
        entry.output_tokens += usage.output;
        if usage.context > 0 {
            entry.metrics.estimate_vs_provider_delta += usage.context as i64 - entry.tokens as i64;
            entry.context_usage = Some(usage.context);
            entry.context_source = "usage";
            entry.cache_samples += 1;
            entry.context_tokens_total = entry
                .context_tokens_total
                .saturating_add(usage.context as u64);
        }
        if entry.model.is_none() {
            entry.model = request_model.map(str::to_string);
        }
        entry.last_seen = now_ms();
        self.persist(key, entry);
    }

    /// Directory for the per-request usage ledger.  `None` for an in-memory
    /// store (tests), which keeps the ledger off when no session dir exists.
    pub fn requests_dir(&self) -> Option<PathBuf> {
        self.dir.as_ref().map(|dir| dir.join("requests"))
    }

    /// Artifact ids that this request's own prune wrote.
    ///
    /// The evidence is the request's `PruneReport.pruned_ids` — exactly the
    /// messages `prune_tool_results` replaced during this prepare — plus the
    /// artifact metadata's `created_at` falling inside the request window.
    /// Nothing is inferred from the conversation-wide artifact list: when a
    /// pruned id does not resolve to a metadata record written inside the
    /// window, the caller gets `partial` or `unavailable`, never a guess.
    ///
    /// This is a scoped *time-window* association, not a strict causal proof.
    /// Two concurrent requests of one conversation can share a millisecond
    /// window, so an artifact written by one request can be admitted for the
    /// other when both prune the same message id.  `partial`/`unavailable` cover
    /// every case the window cannot account for, and the ledger's
    /// `artifact_refs_attribution` names the weaker basis explicitly.  A strict
    /// receipt would require the artifact ids returned by `put_with_session`
    /// inside `prune_tool_results` (an additive `PruneReport.artifact_ids`
    /// field); that touches prune core and is deliberately not done here.
    pub fn resolve_prune_artifacts(
        &self,
        report: &PruneReport,
        from: u64,
        to: u64,
    ) -> (Vec<String>, &'static str) {
        const MAX_REFS: usize = 64;
        let Some(artifacts) = self.artifacts.as_ref() else {
            return (Vec::new(), "unavailable");
        };
        if report.artifact_writes == 0 {
            return (Vec::new(), "no_artifact_writes");
        }
        let mut ids: Vec<String> = Vec::new();
        let mut unresolved = 0usize;
        for message_id in &report.pruned_ids {
            match artifacts.resolve_by_message_id(message_id) {
                Ok(Some(meta)) if meta.created_at >= from && meta.created_at <= to => {
                    if !ids.iter().any(|known| known == &meta.artifact_id) {
                        ids.push(meta.artifact_id);
                    }
                }
                _ => unresolved = unresolved.saturating_add(1),
            }
        }
        if ids.len() >= MAX_REFS {
            ids.truncate(MAX_REFS);
            return (ids, "proven_truncated");
        }
        if ids.is_empty() {
            return (Vec::new(), "unavailable");
        }
        if unresolved == 0 && ids.len() as u64 == report.artifact_writes {
            // All ids resolved inside the request window and the count matches
            // the writes.  Label retained for schema compatibility; it means
            // "window-scoped match", not exclusive causal proof.
            (ids, "prune_report_proven")
        } else {
            (ids, "partial")
        }
    }

    /// Snapshot for `/__bili/stats`: one row per live conversation, newest
    /// first, in the shape the Node dashboard renders.
    ///
    /// `cacheHitPct` is a conversation-level ratio over every sampled response,
    /// not a per-request hit rate: `cached_tokens / context_tokens_total`.
    /// `cacheHitPctScope` states that explicitly, and the value is `null` when
    /// no sample exists, when the cumulative denominator is missing (pre-D-2.1
    /// snapshot), or when the history is marked incomplete (pre-D-2.2 snapshot):
    /// a consumer can never read a missing or partial baseline as a percentage.
    fn row_from_entry(key: &str, entry: &Entry) -> Value {
        let status = entry.session.status();
        let context_tokens = entry.context_usage.unwrap_or(entry.tokens);
        let cache_hit_pct = if entry.context_tokens_total_complete
            && entry.cache_samples > 0
            && entry.context_tokens_total > 0
        {
            Some(
                ((entry.cached_tokens as f64 * 100.0) / entry.context_tokens_total as f64).round()
                    as u64,
            )
        } else {
            None
        };
        let econ = session_economics(entry);
        json!({
            "id": key,
            "protocol": if entry.protocol.is_empty() { "anthropic" } else { entry.protocol.as_str() },
            "upstream": entry.upstream,
            "label": key,
            "title": key,
            "requests": entry.requests,
            "contextTokens": context_tokens,
            "contextTokensSource": entry.context_source,
            "inputTokens": entry.input_tokens,
            "cachedTokens": entry.cached_tokens,
            "outputTokens": entry.output_tokens,
            "cacheSamples": entry.cache_samples,
            "cacheHitPct": cache_hit_pct,
            "cacheHitPctScope": "cumulative",
            "lastModel": entry.model,
            "blocks": status.blocks,
            "activeBlocks": status.active_blocks,
            "contextWindow": entry.context_limit,
            "lastSeen": entry.last_seen,
            "restored": false,
            "foldCount": econ.folds,
            "grossSaved": econ.gross,
            "netSaved": econ.net,
            "summaryCost": econ.summary,
            "foldedTokens": econ.folded,
            "hasFoldData": econ.has_data,
            "compaction": entry.metrics,
        })
    }

    fn row_from_meta(meta: &SnapshotMeta) -> Value {
        let context_tokens = meta.context_usage.unwrap_or(meta.tokens);
        let cache_hit_pct = if meta.context_tokens_total_complete
            && meta.cache_samples > 0
            && meta.context_tokens_total > 0
        {
            Some(
                ((meta.cached_tokens as f64 * 100.0) / meta.context_tokens_total as f64).round()
                    as u64,
            )
        } else {
            None
        };
        json!({
            "id": meta.key,
            "protocol": if meta.protocol.is_empty() { "anthropic" } else { meta.protocol.as_str() },
            "upstream": meta.upstream,
            "label": meta.key,
            "title": meta.key,
            "requests": meta.requests,
            "contextTokens": context_tokens,
            "contextTokensSource": if meta.context_usage.is_some() { "usage" } else { "estimate" },
            "inputTokens": meta.input_tokens,
            "cachedTokens": meta.cached_tokens,
            "outputTokens": meta.output_tokens,
            "cacheSamples": meta.cache_samples,
            "cacheHitPct": cache_hit_pct,
            "cacheHitPctScope": "cumulative",
            "lastModel": meta.model,
            "blocks": meta.blocks,
            "activeBlocks": meta.active_blocks,
            "contextWindow": meta.context_limit,
            "lastSeen": meta.last_seen,
            "restored": true,
            "foldCount": meta.fold_count,
            "grossSaved": meta.gross_saved,
            "netSaved": meta.net_saved,
            "summaryCost": meta.summary_cost,
            "foldedTokens": meta.folded_tokens,
            "hasFoldData": meta.has_fold_data,
            "tokensCompressed": meta.tokens_compressed,
            "compaction": meta.metrics,
        })
    }

    fn session_rows(&self) -> Vec<Value> {
        let index = self.index.lock().expect("session store poisoned").clone();
        let guard = self.inner.lock().expect("session store poisoned");
        let mut sessions: Vec<Value> = Vec::new();
        for (key, meta) in index.iter() {
            if let Some(entry) = guard.get(key) {
                sessions.push(Self::row_from_entry(key, entry));
            } else {
                sessions.push(Self::row_from_meta(meta));
            }
        }
        for (key, entry) in guard.iter() {
            if !index.contains_key(key) {
                sessions.push(Self::row_from_entry(key, entry));
            }
        }
        sessions
            .sort_by_key(|session| std::cmp::Reverse(session["lastSeen"].as_u64().unwrap_or(0)));
        sessions
    }

    pub fn stats_json(&self) -> Value {
        json!({
            "sessions": self.session_rows(),
            "blindTunnels": [],
            "unrecognizedPaths": {},
            "conflicts": {"sessions": 0, "events": 0},
        })
    }

    /// Aggregate input/cache/folding economics for the overview page. Rust
    /// does not keep Node's cache ledger, so `grossSaved` uses landed fold
    /// events multiplied by requests after each fold; `repay` is not
    /// observable and stays zero until the host records it.
    pub fn overview_json(&self) -> Value {
        let rows = self.session_rows();
        let mut live_sessions = 0u64;
        let mut requests = 0u64;
        let mut input_tokens = 0u64;
        let mut cached_tokens = 0u64;
        let mut output_tokens = 0u64;
        let mut gross = 0i64;
        let mut net = 0i64;
        let mut summary_cost = 0i64;
        let mut folded = 0u64;
        let mut folds = 0u64;
        let mut has_fold_data = false;
        let mut legacy_folded = 0u64;
        let mut protocols: std::collections::BTreeMap<String, (u64, u64, u64, u64, u64, i64)> =
            std::collections::BTreeMap::new();
        for row in &rows {
            let protocol = row["protocol"].as_str().unwrap_or("anthropic").to_string();
            let reqs = row["requests"].as_u64().unwrap_or(0);
            let input = row["inputTokens"].as_u64().unwrap_or(0);
            let cached = row["cachedTokens"].as_u64().unwrap_or(0);
            let output = row["outputTokens"].as_u64().unwrap_or(0);
            let row_folds = row["foldCount"].as_u64().unwrap_or(0);
            let row_gross = row["grossSaved"].as_i64().unwrap_or(0);
            let row_net = row["netSaved"].as_i64().unwrap_or(0);
            let row_summary = row["summaryCost"].as_i64().unwrap_or(0);
            let row_folded = row["foldedTokens"].as_u64().unwrap_or(0);
            let row_has = row["hasFoldData"].as_bool().unwrap_or(false);
            if row["lastSeen"].as_u64().unwrap_or(0) > 0 {
                live_sessions += 1;
            }
            requests += reqs;
            input_tokens += input;
            cached_tokens += cached;
            output_tokens += output;
            gross += row_gross;
            net += row_net;
            summary_cost += row_summary;
            folds += row_folds;
            has_fold_data |= row_has;
            if row_has {
                folded += row_folded;
            } else {
                legacy_folded += row_folded;
            }
            let aggregate = protocols.entry(protocol).or_insert((0, 0, 0, 0, 0, 0));
            aggregate.0 += 1;
            aggregate.1 += reqs;
            aggregate.2 += input;
            aggregate.3 += cached;
            aggregate.4 += row_folds;
            aggregate.5 += row_gross;
        }
        let by_protocol: Vec<Value> = protocols
            .into_iter()
            .map(
                |(protocol, (sessions, reqs, input, cached, fold_count, saved))| {
                    json!({
                        "protocol": protocol,
                        "sessions": sessions,
                        "requests": reqs,
                        "inputTokens": input,
                        "cachedTokens": cached,
                        "cacheHitPct": if input > 0 {
                            ((cached as f64 * 100.0) / input as f64).round() as u64
                        } else { 0 },
                        "foldCount": fold_count,
                        "grossSaved": saved,
                    })
                },
            )
            .collect();
        json!({
            "sessions": rows.len() as u64,
            "live": live_sessions,
            "requests": requests,
            "inputTokens": input_tokens,
            "cachedTokens": cached_tokens,
            "outputTokens": output_tokens,
            "cacheHitPct": if input_tokens > 0 {
                ((cached_tokens as f64 * 100.0) / input_tokens as f64).round() as u64
            } else { 0 },
            "grossSavedTotal": gross + legacy_folded as i64,
            "netSavedTotal": net,
            "summaryCostTotal": summary_cost,
            "repayTotal": 0,
            "foldedTokensTotal": folded + legacy_folded,
            "foldsTotal": folds,
            "hasFoldData": has_fold_data,
            "savedEstimated": legacy_folded > 0,
            "byProtocol": by_protocol,
            "recent": rows,
        })
    }

    /// Snapshot for `/__bili/plugin/status`.
    /// Host-side artifact access for diagnostics and replay. This is not a
    /// model-visible tool surface.
    pub fn artifact_store(&self) -> Option<&ArtifactStore> {
        self.artifacts.as_ref()
    }

    /// The checkpoint the DSH compaction precheck answers with: the snapshot tag
    /// naming this conversation, its ref sequence and every active block's
    /// forged summary. `None` when nothing is folded yet — an empty checkpoint
    /// may not stand in for a summary, and the caller must refuse instead.
    pub fn authored_checkpoint(
        &self,
        key: &str,
    ) -> Option<crate::dsh_compaction::AuthoredCheckpoint> {
        self.ensure_loaded(key);
        let guard = self.inner.lock().expect("session store poisoned");
        let entry = guard.get(key)?;
        let status = entry.session.status();
        let mut blocks = Vec::new();
        for number in 1..=status.blocks {
            let Some(id) = BlockId::new(number as u32) else {
                continue;
            };
            let Some(block) = entry.session.block(id) else {
                continue;
            };
            if !block.active {
                continue;
            }
            blocks.push((
                block.id.to_string(),
                crate::dsh_compaction::render_forged_summary(
                    block.topic.as_deref(),
                    &block.summary,
                ),
            ));
        }
        if blocks.is_empty() {
            return None;
        }
        Some(crate::dsh_compaction::render_checkpoint(
            key,
            status.refs_allocated as u64,
            &blocks,
        ))
    }

    /// The rendered conversation slice for every message range a `compress`
    /// call named, in the order the model listed them.
    ///
    /// The external-summary chain (plan §2.1) needs the *text* behind the refs:
    /// the model only sends labels, and a summarizer cannot summarize a label.
    /// Ranges that name blocks are left to the kernel — those carry summaries
    /// already, so an external model has nothing to add.
    pub fn range_inputs(&self, key: &str, args: &Value) -> Vec<crate::summarizer::RangeInput> {
        self.ensure_loaded(key);
        let Ok(guard) = self.inner.lock() else {
            return Vec::new();
        };
        let Some(entry) = guard.get(key) else {
            return Vec::new();
        };
        let ranges = match tool::parse_compress(&normalize_compress_args(args)) {
            Ok(ranges) => ranges,
            Err(_) => return Vec::new(),
        };
        let mut inputs = Vec::new();
        for range in ranges {
            let Some(text) = message_slice_text(&entry.last_messages, &range.start, &range.end)
            else {
                continue;
            };
            inputs.push(crate::summarizer::RangeInput {
                label: format!("{}–{}", range.start, range.end),
                topic: range.topic.clone(),
                text,
            });
        }
        inputs
    }

    pub fn status_json(&self, key: &str) -> Option<Value> {
        self.ensure_loaded(key);
        let mut body = {
            let guard = self.inner.lock().expect("session store poisoned");
            guard.get(key).map(|entry| status_body(key, entry))?
        };
        if let Some(artifacts) = &self.artifacts
            && let Ok(value) = serde_json::to_value(artifacts.stats())
            && let Some(object) = body.as_object_mut()
        {
            object.insert("artifacts".into(), value);
        }
        Some(body)
    }

    pub fn latest_status_json(&self) -> Option<Value> {
        let latest_key = self
            .index
            .lock()
            .expect("session store poisoned")
            .iter()
            .max_by_key(|(_, meta)| meta.last_seen)
            .map(|(key, _)| key.clone())
            .or_else(|| {
                self.inner
                    .lock()
                    .expect("session store poisoned")
                    .iter()
                    .max_by_key(|(_, entry)| entry.last_seen)
                    .map(|(key, _)| key.clone())
            })?;
        self.ensure_loaded(&latest_key);
        let guard = self.inner.lock().expect("session store poisoned");
        let entry = guard.get(&latest_key)?;
        let mut body = status_body(&latest_key, entry);
        if let Some(object) = body.as_object_mut() {
            object.insert("viaFallback".into(), Value::Bool(true));
        }
        Some(body)
    }

    #[allow(dead_code, reason = "status predicate retained for host integrations")]
    pub fn has(&self, key: &str) -> bool {
        self.inner
            .lock()
            .expect("session store poisoned")
            .contains_key(key)
            || self
                .index
                .lock()
                .expect("session store poisoned")
                .contains_key(key)
    }

    /// The context window this conversation was last prepared with.
    ///
    /// Tool calls (`/__bili/plugin/tool`) carry no `x-bili-plugin-context-window`
    /// header, so a host that falls back to the bare default (200_000) would
    /// hand `with_entry` a *different* limit than the wire request used and
    /// silently rebuild the session (refs, folded blocks and token counters
    /// all reset) — every `mNNNNN` the nudge just printed then becomes
    /// `unknown_ref`. Callers must reuse the stored limit instead.
    pub fn context_limit_of(&self, key: &str) -> Option<u32> {
        self.ensure_loaded(key);
        let guard = self.inner.lock().expect("session store poisoned");
        guard.get(key).map(|entry| entry.context_limit)
    }

    #[cfg(test)]
    fn loaded_count(&self) -> usize {
        self.inner.lock().expect("session store poisoned").len()
    }

    #[cfg(test)]
    fn index_count(&self) -> usize {
        self.index.lock().expect("session store poisoned").len()
    }

    /// Snapshot for `/__bili/pricing`: models seen by live conversations.
    /// The Rust host does not ship a price registry yet, so `price` is null
    /// and `overrides` is empty; the shape matches the Node dashboard.
    pub fn pricing_json(&self, config_path: &str) -> Value {
        let rows = self.session_rows();
        let mut seen: std::collections::BTreeMap<String, Option<String>> =
            std::collections::BTreeMap::new();
        for row in &rows {
            if let Some(model) = row["lastModel"].as_str() {
                let host = row["upstream"].as_str().and_then(upstream_host);
                seen.entry(model.to_string()).or_insert(host);
            }
        }
        let models: Vec<Value> = seen
            .into_iter()
            .map(|(model, host)| {
                let mut row = json!({"model": model, "price": Value::Null});
                if let Some(host) = host {
                    row["host"] = Value::String(host);
                }
                row
            })
            .collect();
        json!({
            "path": config_path,
            "overrides": {},
            "models": models,
        })
    }
}

fn upstream_host(upstream: &str) -> Option<String> {
    let rest = upstream
        .strip_prefix("https://")
        .or_else(|| upstream.strip_prefix("http://"))?;
    let host = rest.split(['/', '?', '#']).next().unwrap_or("").trim();
    (!host.is_empty()).then(|| host.to_string())
}

fn status_body(key: &str, entry: &Entry) -> Value {
    let status = entry.session.status();
    let mut blocks = Vec::new();
    for number in 1..=status.blocks {
        let Some(id) = BlockId::new(number as u32) else {
            continue;
        };
        let Some(block) = entry.session.block(id) else {
            continue;
        };
        blocks.push(json!({
            "id": block.id.to_string(),
            "tier": block.tier as u8,
            "topic": block.topic,
            "summary": block.summary,
            "tokens": block.compressed_tokens,
            "summaryTokens": estimate_summary_tokens(&block.summary),
            "active": block.active,
            "directCount": block.direct.len(),
            "effectiveCount": block.effective.len(),
            "span": block.span.as_ref().map(|(start, end)| json!([start.to_string(), end.to_string()])),
            "directBlocks": block.direct_blocks.iter().map(ToString::to_string).collect::<Vec<_>>(),
            "createdAt": block.created_at,
            "generation": format!("{:?}", block.generation).to_lowercase(),
            "survived": block.survived,
            "expanded": block.expanded,
            "restoredInline": block.restored_inline,
        }));
    }
    let compressible: Vec<Value> = entry
        .ranges
        .iter()
        .map(|range| {
            json!({
                "start": format!("m{:05}", range.start.index()),
                "end": format!("m{:05}", range.end.index()),
                "count": range.count,
                "tokens": range.tokens,
                "chars": range.chars,
            })
        })
        .collect();
    let shown_tokens = entry.context_usage.unwrap_or(entry.tokens);
    let pct = if entry.context_limit > 0 {
        ((shown_tokens as f32 / entry.context_limit as f32) * 100.0).round() as u32
    } else {
        0
    };
    let max_range_chars = compressible
        .iter()
        .filter_map(|range| range.get("chars").and_then(Value::as_u64))
        .max()
        .unwrap_or(0);
    let econ = session_economics(entry);
    json!({
        "ok": true,
        "phase": "ready",
        "viaFallback": false,
        "sessionId": key,
        "conversationId": key,
        "label": key,
        "pluginAgent": "dsh",
        "contextLimit": entry.context_limit,
        "modelContextLimit": entry.context_limit,
        "contextTokens": shown_tokens,
        "tokens": entry.tokens,
        "tokenCount": entry.tokens,
        "contextTokensSource": entry.context_source,
        "requests": entry.requests,
        "blocks": blocks,
        "blockCount": status.blocks,
        "activeBlocks": status.active_blocks,
        "tokensCompressed": status.tokens_compressed,
        "compressionCount": status.compression_count,
        "grossSaved": econ.gross,
        "netSaved": econ.net,
        "summaryCost": econ.summary,
        "foldCount": econ.folds,
        "hasFoldData": econ.has_data,
        "compaction": entry.metrics,
        "nudgePending": entry.nudge_pending,
        "lastPruneRequests": entry.last_prune_requests,
        "lastPruneTokens": entry.last_prune_tokens,
        "rawTokens": entry.raw_input_estimate,
        "effectiveTokens": entry.effective_projection_estimate,
        "fixedOverheadTokens": entry.fixed_overhead_estimate,
        "providerInputUsage": entry.context_usage,
        "pruneTriggerReason": entry.prune_trigger_reason,
        "cooldownHits": entry.metrics.cooldown_hits,
        "cooldownBlocked": entry.metrics.cooldown_blocked,
        "growthBypassTriggered": entry.metrics.growth_bypass_triggered,
        "pruneSkipped": entry.metrics.prune_skipped,
        "stubReapplyItems": entry.metrics.stub_reapply_items,
        "autoFallback": {
            "attempts": entry.metrics.fallback_attempts,
            "appliedBlocks": entry.metrics.fallback_applied_blocks,
            "refused": entry.metrics.fallback_refused,
            "failures": entry.metrics.fallback_failures,
            "reason": entry.metrics.fallback_last_reason,
            "inFlight": entry.fallback_in_flight,
        },
        // P2: which folded spans still have a raw original, and how much has
        // been served on explicit request. Capped: status is a summary, not the
        // artifact index.
        "foldArchives": entry
            .fold_archives
            .iter()
            .rev()
            .take(16)
            .map(|archive| {
                json!({
                    "block": archive.block,
                    "artifact": archive.artifact_id,
                    "bytes": archive.byte_len,
                    "span": archive.span,
                })
            })
            .collect::<Vec<_>>(),
        "artifactRaw": {
            "written": entry.metrics.fold_archive_written,
            "writtenBytes": entry.metrics.fold_archive_bytes,
            "skipped": entry.metrics.fold_archive_skipped,
            "rawCalls": entry.metrics.retrieve_raw_calls,
            "rawBytes": entry.metrics.retrieve_raw_bytes,
            "rawUnavailable": entry.metrics.retrieve_raw_unavailable,
        },
        "foldTxn": {
            "recovered": entry.metrics.fold_journal_recovered,
            "busyRefused": entry.metrics.fold_busy_refused,
            "idempotentReplays": entry.metrics.idempotent_replays,
            "open": entry.fold_journal.iter().filter(|journal| journal.phase != "end").count(),
            "recent": entry
                .fold_journal
                .iter()
                .rev()
                .take(8)
                .map(|journal| {
                    json!({
                        "call": journal.call_id,
                        "phase": journal.phase,
                        "spans": journal.spans,
                        "blocks": journal.blocks,
                        "requests": journal.at_requests,
                        "note": journal.note,
                    })
                })
                .collect::<Vec<_>>(),
        },
        "freshWindow": {
            "resets": entry.window_resets.len(),
            "last": entry.window_resets.last().map(|reset| {
                json!({
                    "requests": reset.at_requests,
                    "reason": reset.reason,
                    "droppedBlocks": reset.dropped_blocks,
                    "keptMessages": reset.kept_messages,
                    "artifacts": reset.artifacts.len(),
                })
            }),
        },
        "speculation": {
            "started": entry.metrics.speculation_started,
            "adopted": entry.metrics.speculation_adopted,
            "discarded": entry.metrics.speculation_discarded,
            "inFlight": entry.speculation_in_flight,
            "lastBandRequests": entry.speculation_last_requests,
            "candidate": entry.speculation.as_ref().map(|plan| json!({
                "blocks": plan.blocks,
                "atRequests": plan.at_requests,
                "model": plan.model,
            })),
        },
        // Task 2: cost of automatic summary dispatches. `limits` is filled by
        // the status handler from the live chain config so the panel shows what
        // the budget is, not only what has been spent.
        "summaryBudget": {
            "calls": entry.summary_budget.calls,
            "speculationCalls": entry.summary_budget.speculation_calls,
            "inputTokens": entry.summary_budget.input_tokens,
            "outputTokens": entry.summary_budget.output_tokens,
            "estimatedCalls": entry.summary_budget.estimated_calls,
            "refused": entry.summary_budget.refused,
            "lastRefusal": entry.summary_budget.last_refusal,
        },
        "refs": status.refs_allocated,
        "tiers": status.tiers,
        "compressibleRanges": compressible,
        "inputTokens": entry.input_tokens,
        "outputTokens": entry.output_tokens,
        "cachedTokens": entry.cached_tokens,
        "cacheSamples": entry.cache_samples,
        "model": entry.model.clone(),
        "lastSeen": entry.last_seen,
        "panel": format!(
            "╭─────────────────────────────────────────────╮\n\
             │           ACP Context Analysis              │\n\
             ╰─────────────────────────────────────────────╯\n\
             billion-context@{} · rust host\n\n\
             Context: {}% ({} / {} tokens, sent-view estimate)\n\
             Sent to LLM (after compression, est.): {}\n\n\
             Blocks: {} total / {} active · {} tokens compressed\n\
             Compressible ranges: {} (max {} chars)\n\
             Nudge: {}\n",
            env!("CARGO_PKG_VERSION"),
            pct,
            shown_tokens,
            entry.context_limit,
            entry.tokens,
            status.blocks,
            status.active_blocks,
            status.tokens_compressed,
            compressible.len(),
            max_range_chars,
            if entry.last_nudge.is_some() { "armed" } else { "idle" },
        ),
    })
}

#[allow(dead_code)]
pub struct Prepared {
    pub messages: Vec<billion_context::Message>,
    pub ranges: Vec<billion_context::Range>,
    pub nudge: Option<String>,
    pub terminal: bool,
    pub warnings: Vec<String>,
    /// Stable message ids whose visible ToolResult content was replaced by an
    /// artifact preview. The wire layer must drop any raw sidecar block for
    /// these ids so the preview is not overwritten on rebuild.
    pub pruned_ids: Vec<String>,
    pub prune: Option<PruneReport>,
    pub decision_tokens: u32,
    pub raw_input_estimate: u32,
    pub assembled_projection_estimate: u32,
    pub effective_projection_estimate: u32,
    pub fixed_overhead_estimate: u32,
    pub prune_decision: crate::prune::TriggerDecision,
    pub prune_triggered: bool,
    pub prune_skipped: bool,
    pub stub_reapply_items: u64,
}

/// Node's compress tool takes ONE `content` string holding either a JSON array
/// of range objects or a single range object (gateways sometimes stringify the
/// array). The vendored kernel parser wants `{ranges:[{start,end,...}]}`, so
/// translate the served shape before handing it over.
/// Node enforces a host-side minimum on the characters inside a fold request
/// (`min_compress_range`, 5000): the kernel's range *finder* respects it, but a
/// model-named range is only gated by the host that translates the tool call.
/// Returns `None` when the request touches a block ref or cannot be mapped —
/// in that case the kernel's own range checks are the authority.
/// One message rendered for an external summarizer: the ref the model saw, the
/// role, and the payload. Tool calls keep their JSON arguments and tool results
/// keep their content — those are exactly the facts a summary must not lose.
fn render_message(message: &Message) -> String {
    let role = match message.role {
        billion_context::Role::User => "user",
        billion_context::Role::Assistant => "assistant",
        billion_context::Role::System => "system",
        billion_context::Role::Tool => "tool",
    };
    let body = match &message.content {
        Content::Text(text) | Content::Reasoning(text) => text.to_string(),
        Content::ToolCall {
            name, arguments, ..
        } => format!("[tool call {}] {arguments}", name),
        Content::ToolResult { name, content, .. } => {
            format!("[tool result {}] {content}", name)
        }
    };
    format!("[{}] {role}: {body}", message.id)
}

/// The raw text of one contiguous range, for the external summarizer chain.
///
/// `last_messages` is the last projected request, so its refs are the message
/// positions (1-based). `None` when the span is outside it or has no content.
fn message_slice_text(messages: &[Message], start: &Boundary, end: &Boundary) -> Option<String> {
    let (Boundary::Message(start), Boundary::Message(end)) = (start, end) else {
        return None;
    };
    if start.index() == 0 || end.index() < start.index() {
        return None;
    }
    let mut text = String::new();
    for (position, message) in messages.iter().enumerate() {
        let reference = position as u32 + 1;
        if reference < start.index() || reference > end.index() {
            continue;
        }
        text.push_str(&render_message(message));
        text.push('\n');
    }
    (!text.trim().is_empty()).then_some(text)
}

/// Archive the raw text of each freshly folded span before the projection
/// forgets it (P2 artifact coordination).
///
/// Runs after `apply_compression` landed and before the next `prepare` replaces
/// `last_messages`, so the span is still addressable by its refs. Failures are
/// counted and swallowed: the fold already happened, and losing only the raw
/// escape hatch must not turn a successful fold into an error.
fn archive_fold_spans(
    artifacts: Option<&ArtifactStore>,
    entry: &mut Entry,
    session_id: &str,
    blocks: &[BlockId],
) {
    for block_id in blocks {
        let Some(block) = entry.session.block(*block_id) else {
            continue;
        };
        let Some((Boundary::Message(start), Boundary::Message(end))) = block.span else {
            continue;
        };
        let span = format!("{start}–{end}");
        let Some(text) = message_slice_text(
            &entry.last_messages,
            &Boundary::Message(start),
            &Boundary::Message(end),
        ) else {
            entry.metrics.fold_archive_skipped += 1;
            continue;
        };
        let Some(artifacts) = artifacts else {
            entry.metrics.fold_archive_skipped += 1;
            continue;
        };
        let Some(source_id) = entry.session.state().refs.raw_of(start) else {
            entry.metrics.fold_archive_skipped += 1;
            continue;
        };
        let block_id_text = block.id.to_string();
        if entry
            .fold_archives
            .iter()
            .any(|archive| archive.block == block_id_text)
        {
            continue;
        }
        match artifacts.put_with_session(
            &text,
            source_id.as_str(),
            Some(&span),
            None,
            None,
            session_id,
        ) {
            Ok(meta) => {
                entry.fold_archives.push(FoldArchive {
                    block: block_id_text,
                    artifact_id: meta.artifact_id,
                    byte_len: meta.byte_len,
                    span,
                });
                entry.metrics.fold_archive_written += 1;
                entry.metrics.fold_archive_bytes += meta.byte_len;
            }
            Err(_) => entry.metrics.fold_archive_skipped += 1,
        }
    }
}

/// Explicit raw retrieval for a deliberate recovery, bounded per ref.
///
/// `refs` accepts message refs only (`parse_retrieve` rejects block refs), so a
/// folded block is addressed with `block: "b3"`. The output never reinjects
/// anything: the caller decides whether to feed it back.
fn retrieve_raw(
    entry: &mut Entry,
    artifacts: Option<&ArtifactStore>,
    _session_id: &str,
    args: &Value,
) -> (String, Option<&'static str>) {
    let Some(artifacts) = artifacts else {
        entry.metrics.retrieve_raw_calls += 1;
        entry.metrics.retrieve_raw_unavailable += 1;
        return (
            "[retrieve_raw unavailable: the artifact store is disabled, so no original survives a fold]"
                .to_string(),
            Some("refused"),
        );
    };
    let offset = args.get("offset").and_then(Value::as_u64).unwrap_or(0);
    let limit = args
        .get("limit")
        .and_then(Value::as_u64)
        .unwrap_or(8_192)
        .clamp(1, crate::artifact::MAX_READ_BYTES);

    let mut targets: Vec<(String, String)> = Vec::new(); // (label, artifact id)
    let mut failures: Vec<String> = Vec::new();
    let mut refs = Vec::new();
    if let Some(raw_refs) = args.get("refs") {
        match tool::parse_retrieve(&json!({ "refs": raw_refs })) {
            Ok(parsed) => refs = parsed,
            Err(err) => failures.push(format!("invalid refs: {}", err)),
        }
    }
    if let Some(block) = args.get("block").and_then(Value::as_str) {
        match entry
            .fold_archives
            .iter()
            .find(|archive| archive.block == block.trim())
        {
            Some(archive) => targets.push((block.trim().to_string(), archive.artifact_id.clone())),
            None => {
                entry.metrics.retrieve_raw_unavailable += 1;
                failures.push(format!(
                    "OriginalUnavailable: block {} has no archived original in this conversation",
                    block.trim()
                ));
            }
        }
    }
    for reference in refs {
        let label = reference.to_string();
        let Some(raw) = entry.session.state().refs.raw_of(reference) else {
            failures.push(format!("{label}: not addressable in this session"));
            continue;
        };
        match artifacts.resolve_by_message_id(raw.as_str()) {
            Ok(Some(meta)) if artifact_belongs_to_entry(entry, &meta) => {
                targets.push((label, meta.artifact_id));
            }
            Ok(Some(_)) => {
                entry.metrics.retrieve_raw_unavailable += 1;
                failures.push(format!(
                    "{label}: OriginalUnavailable — artifact belongs to another conversation"
                ));
            }
            Ok(None) => {
                entry.metrics.retrieve_raw_unavailable += 1;
                failures.push(format!(
                    "{label}: OriginalUnavailable — the span was folded without an archived original (folded before archiving, or too large). Nothing was or will be reinjected."
                ));
            }
            Err(err) => {
                entry.metrics.retrieve_raw_unavailable += 1;
                failures.push(format!(
                    "{label}: OriginalUnavailable — artifact lookup failed: {err}"
                ));
            }
        }
    }

    let mut out = String::new();
    let mut served = 0u64;
    let mut served_bytes = 0u64;
    for (label, artifact_id) in targets {
        entry.metrics.retrieve_raw_calls += 1;
        match artifacts.read(&artifact_id, offset, limit) {
            Ok(read) => {
                served += 1;
                served_bytes += read.text.len() as u64;
                entry.metrics.retrieve_raw_bytes += read.text.len() as u64;
                let span = read
                    .source_ref
                    .as_deref()
                    .map(|span| format!(" · {span}"))
                    .unwrap_or_default();
                let more = if read.next_offset < read.byte_len {
                    format!(" · nextOffset={}", read.next_offset)
                } else {
                    String::new()
                };
                out.push_str(&format!(
                    "## {label} — artifact {artifact_id} · {} bytes stored · showing bytes {}..{}{span}{more}\n{}\n\n",
                    read.byte_len, read.offset, read.next_offset, read.text
                ));
            }
            Err(err) => {
                entry.metrics.retrieve_raw_unavailable += 1;
                failures.push(format!(
                    "{label}: OriginalUnavailable — artifact {artifact_id} unreadable: {err}"
                ));
            }
        }
    }
    if !failures.is_empty() {
        for failure in &failures {
            out.push_str(&format!("[retrieve_raw: {failure}]\n"));
        }
    }
    if served > 0 {
        out.push_str(&format!(
            "\n[retrieve_raw: {served} span(s), {served_bytes} bytes; the projection is unchanged — reinject this text only if the task needs it]"
        ));
    }
    let outcome = if served == 0 && !failures.is_empty() {
        "refused"
    } else {
        "success"
    };
    (out.trim_end().to_string(), Some(outcome))
}

/// Replace each range's `summary` with the text the external chain produced.
///
/// Works on the normalized `{"ranges": [...]}` shape, so it is independent of
/// whether the model sent the JSON surface or the line form. A short chain
/// leaves the remaining ranges untouched — that only happens when a config
/// mixes resolvable and unresolvable ranges, and the kernel's own summary is a
/// better fallback than folding an empty block.
pub fn apply_external_summaries(args: &Value, summaries: &[String]) -> Value {
    let normalized = normalize_compress_args(args);
    let Some(ranges) = normalized.get("ranges").and_then(Value::as_array) else {
        return normalized;
    };
    let mut rewritten = ranges.clone();
    for (index, summary) in summaries.iter().enumerate() {
        let Some(range) = rewritten.get_mut(index) else {
            break;
        };
        if let Some(object) = range.as_object_mut() {
            object.insert("summary".to_string(), Value::String(summary.clone()));
        }
    }
    json!({ "ranges": rewritten })
}

fn message_range_chars(messages: &[Message], ranges: &[RequestedRange]) -> Option<usize> {
    let mut total = 0usize;
    for range in ranges {
        let (Boundary::Message(start), Boundary::Message(end)) = (&range.start, &range.end) else {
            return None;
        };
        if start.index() == 0 || end.index() < start.index() {
            return None;
        }
        for (position, message) in messages.iter().enumerate() {
            let reference = position as u32 + 1;
            if reference >= start.index() && reference <= end.index() {
                total += content_chars(&message.content);
            }
        }
    }
    (total > 0).then_some(total)
}

/// The model-facing label for a requested range: the two boundaries it named.
fn range_label(range: &RequestedRange) -> String {
    format!("{}-{}", range.start, range.end)
}

/// First summary the host refuses to fold. Checking every range up front keeps
/// a placeholder summary from folding a good range in the same call.
fn first_summary_defect(ranges: &[RequestedRange]) -> Option<(String, &'static str)> {
    ranges.iter().find_map(|range| {
        summary_carry::summary_defect(&range.summary).map(|defect| (range_label(range), defect))
    })
}

fn content_chars(content: &Content) -> usize {
    match content {
        Content::Text(text) | Content::Reasoning(text) => text.len(),
        Content::ToolCall { arguments, .. } => arguments.len(),
        Content::ToolResult { content, .. } => content.len(),
    }
}

/// A single-user process still has several conversations in one Store. Direct
/// `artifact_id` is an opaque address, so require the artifact to be referenced
/// by this conversation's own messages or stubs before exposing its bytes.
fn artifact_belongs_to_entry(entry: &Entry, meta: &ArtifactMetadata) -> bool {
    // A fold archive is created for exactly this conversation, and the folded
    // messages leave `last_messages` on the next prepare — the archive list is
    // the durable ownership proof.
    if entry
        .fold_archives
        .iter()
        .any(|archive| archive.artifact_id == meta.artifact_id)
    {
        return true;
    }
    // A fresh-window receipt names the artifacts whose blocks it dropped; the
    // recipe is the only ownership proof left after those refs are gone.
    if entry
        .window_resets
        .iter()
        .any(|reset| reset.artifacts.iter().any(|id| id == &meta.artifact_id))
    {
        return true;
    }
    if entry.last_messages.is_empty() {
        return false;
    }
    for message in &entry.last_messages {
        if meta.source_message_id == message.id.as_str()
            || meta
                .source_message_ids
                .iter()
                .any(|id| id == message.id.as_str())
        {
            return true;
        }
        if let Content::ToolResult { content, .. } = &message.content
            && Store::artifact_ids_in_text(content)
                .iter()
                .any(|id| id == &meta.artifact_id)
        {
            return true;
        }
    }
    false
}

/// Resolve the explicit artifact tool surface to one artifact. `ref` is a
/// convenience address; the model does not need a second raw-retrieval tool.
fn artifact_target(
    entry: &Entry,
    store: Option<&ArtifactStore>,
    args: &Value,
    session_id: &str,
) -> Result<ArtifactMetadata, String> {
    let store = store.ok_or_else(|| "artifact store is disabled".to_string())?;
    let meta = if let Some(id) = args.get("artifact_id").and_then(Value::as_str) {
        store.info(id.trim())?
    } else {
        let reference = args
            .get("ref")
            .and_then(Value::as_str)
            .ok_or_else(|| "artifact tool needs artifact_id or ref".to_string())?;
        let parsed = tool::parse_retrieve(&json!({"refs": [reference]}))
            .map_err(|e| format!("invalid ref {reference}: {e}"))?;
        let raw = parsed
            .first()
            .and_then(|reference| entry.session.state().refs.raw_of(*reference))
            .ok_or_else(|| format!("ref {reference} is not currently addressable"))?;
        store
            .resolve_by_message_id(raw.as_str())?
            .ok_or_else(|| format!("OriginalUnavailable: no artifact for ref {reference}"))?
    };
    if !artifact_belongs_to_entry(entry, &meta) {
        store.audit(
            "refuse_access",
            &meta.artifact_id,
            session_id,
            "fail",
            "artifact not referenced by this conversation",
        );
        return Err(format!(
            "artifact {} is not owned by this conversation",
            meta.artifact_id
        ));
    }
    store.audit(
        "access",
        &meta.artifact_id,
        session_id,
        "success",
        "artifact referenced by this conversation",
    );
    Ok(meta)
}

/// Upper bound a model may ask for with `summaryMaxChars`. A summary that big
/// already swallows most of a 200K window; beyond it the ceiling stops being a
/// repair for an over-long summary and becomes a way to fold nothing but text.
const SUMMARY_MAX_CHARS_CEIL: usize = 250_000;

/// The per-call summary ceiling a `compress` call carries (`summaryMaxChars`,
/// Node parity; `summary_max_chars` is accepted too).
///
/// Read from the **raw** args, before `normalize_compress_args` collapses the
/// `content` form into `{"ranges": …}` — that collapse drops the siblings, and
/// this key travels next to the ranges, not inside them.
///
/// Node passes the value straight to its validator; here it is clamped so a
/// typo cannot turn into a ceiling of zero (`0`, negative and non-integer
/// values are ignored rather than honoured). The session default still applies
/// when the key is absent.
fn summary_max_chars(args: &Value) -> Option<usize> {
    let raw = args
        .get("summaryMaxChars")
        .or_else(|| args.get("summary_max_chars"))?;
    let value = raw.as_u64().or_else(|| {
        raw.as_i64()
            .filter(|value| *value > 0)
            .map(|value| value as u64)
    })?;
    if value == 0 {
        return None;
    }
    Some((value as usize).min(SUMMARY_MAX_CHARS_CEIL))
}

/// Compress arrives in more than one advertised shape. `ranges` is accepted
/// verbatim; `content` may be an array of entry objects, a single entry object,
/// a JSON-encoded string holding either of those, or the documented LINE FORM —
/// one plain string whose blocks each start with an `m00150–m00220 optional
/// topic` header line followed by that block's summary markdown. The line form
/// is what the schema description, the compress prompt and every nudge text
/// tell the model to send, yet it used to reach the tool undecoded and die as
/// "`ranges` must be a non-empty array" on every call that followed that advice
/// (m00527). Decoding it here keeps the advertised contract true instead of
/// telling the model to stop using the shape we publish.
fn normalize_compress_args(args: &Value) -> Value {
    if args.get("ranges").is_some_and(Value::is_array) {
        return args.clone();
    }
    let Some(content) = args.get("content") else {
        return args.clone();
    };
    let ranges = compress_ranges_from_content(content);
    if ranges.is_empty() {
        return args.clone();
    }
    serde_json::json!({"ranges": ranges})
}

/// Decode any accepted `content` shape into canonical range objects.
fn compress_ranges_from_content(content: &Value) -> Vec<Value> {
    match content {
        Value::Array(items) => items.iter().filter_map(compress_range_from_entry).collect(),
        Value::Object(_) => compress_range_from_entry(content).into_iter().collect(),
        Value::String(text) => {
            // Gateways stringify the arguments, models fence them in ```json,
            // and a few hosts double-encode the string. Strip the shells before
            // deciding between JSON and the line form.
            let mut body = text.trim();
            for fence in ["```json", "```"] {
                if let Some(stripped) = body.strip_prefix(fence) {
                    body = stripped.trim_start();
                }
            }
            if let Some(stripped) = body.strip_suffix("```") {
                body = stripped.trim_end();
            }
            if body.starts_with('[') || body.starts_with('{') || body.starts_with('"') {
                if let Ok(parsed) = serde_json::from_str::<Value>(body) {
                    let ranges = compress_ranges_from_content(&parsed);
                    if !ranges.is_empty() {
                        return ranges;
                    }
                }
            }
            compress_ranges_from_line_form(body)
        }
        _ => Vec::new(),
    }
}

fn compress_range_from_entry(entry: &Value) -> Option<Value> {
    if let Value::String(text) = entry {
        return compress_range_from_line(text);
    }
    let start = entry
        .get("startId")
        .or_else(|| entry.get("startRef"))
        .or_else(|| entry.get("start"))
        .and_then(Value::as_str)
        .map(str::to_string);
    let end = entry
        .get("endId")
        .or_else(|| entry.get("endRef"))
        .or_else(|| entry.get("end"))
        .and_then(Value::as_str)
        .map(str::to_string);
    let summary = entry
        .get("summary")
        .and_then(Value::as_str)
        .map(str::to_string);
    let (Some(start), Some(end), Some(summary)) = (start, end, summary) else {
        return None;
    };
    let mut range = serde_json::json!({"start": start, "end": end, "summary": summary});
    if let Some(topic) = entry.get("topic").and_then(Value::as_str) {
        range["topic"] = json!(topic);
    }
    Some(range)
}

/// 原始一行块文本 → 条目列表。只有以 ref 头开始的非空行才开启新块，块之间的
/// 空行与前言（头之前没有 ref 的散文）被丢弃，不会伪造边界。
fn split_line_blocks(text: &str) -> Vec<String> {
    let mut blocks: Vec<String> = Vec::new();
    let mut current: Option<String> = None;
    for line in text.lines() {
        if refs_in_header(line.trim()).is_some() {
            if let Some(block) = current.take() {
                blocks.push(block);
            }
            current = Some(line.to_string());
        } else if let Some(block) = current.as_mut() {
            block.push('\n');
            block.push_str(line);
        }
    }
    if let Some(block) = current.take() {
        blocks.push(block);
    }
    blocks
}

fn compress_ranges_from_line_form(text: &str) -> Vec<Value> {
    split_line_blocks(text)
        .iter()
        .filter_map(|block| compress_range_from_line(block))
        .collect()
}

/// One line-form block: header refs (+ optional topic) on the first line, the
/// summary verbatim on the remaining lines. A block without summary text is
/// dropped rather than guessed — an empty block would fold messages into
/// nothing.
fn compress_range_from_line(block: &str) -> Option<Value> {
    let mut lines = block.lines();
    let head = lines.next()?.trim();
    let summary = lines.collect::<Vec<_>>().join("\n");
    let summary = summary.trim();
    if summary.is_empty() {
        return None;
    }
    let refs = refs_in_header(head)?;
    let topic = head.get(refs.after..).unwrap_or("").trim();
    let mut range = serde_json::json!({"start": refs.start, "end": refs.end, "summary": summary});
    if !topic.is_empty() {
        range["topic"] = json!(topic);
    }
    Some(range)
}

struct HeaderRefs {
    start: String,
    end: String,
    /// Byte offset just past the last ref, so the topic can be sliced off.
    after: usize,
}

/// `m00150–m00220 optional topic` / `b3..b4` / `m00150 to m00220`。Two refs
/// behind an unrecognized divider are refused: narrowing a pair to its first
/// ref would silently fold ONE message of a many-message range.
fn refs_in_header(head: &str) -> Option<HeaderRefs> {
    let (start, after_start) = parse_header_ref(head, 0)?;
    let rest = head.get(after_start..)?;
    if let Some(sep) = separator_len(rest) {
        if let Some((end, after_end)) = parse_header_ref(head, after_start + sep) {
            return Some(HeaderRefs {
                start,
                end,
                after: after_end,
            });
        }
    }
    if contains_ref_token(rest) {
        // Another ref behind a divider we do not recognize (`m00150 x m00220`):
        // a pair was intended, so folding the first ref alone would silently
        // cover ONE message of a many-message range. Refuse instead.
        return None;
    }
    Some(HeaderRefs {
        start: start.clone(),
        end: start,
        after: after_start,
    })
}

/// Does `text` contain a ref token (`m`/`b` + digits)? Used to spot a ref pair
/// whose divider we did not recognize, so the pair is refused instead of
/// silently narrowed to its first ref.
fn contains_ref_token(text: &str) -> bool {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if matches!(bytes[i], b'm' | b'M' | b'b' | b'B') {
            let mut j = i + 1;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            let ends_token = j > i + 1 && bytes.get(j).is_none_or(|b| !b.is_ascii_alphanumeric());
            if ends_token {
                return true;
            }
            i = j.max(i + 1);
        } else {
            i += 1;
        }
    }
    false
}

fn parse_header_ref(text: &str, from: usize) -> Option<(String, usize)> {
    let bytes = text.as_bytes();
    let mut i = from;
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    let marker = match bytes.get(i)? {
        b'm' | b'M' => 'm',
        b'b' | b'B' => 'b',
        _ => return None,
    };
    let mut j = i + 1;
    while j < bytes.len() && bytes[j].is_ascii_digit() {
        j += 1;
    }
    let digits = text.get(i + 1..j)?;
    if digits.is_empty() || digits.len() > 7 {
        return None;
    }
    let normalized = if marker == 'b' {
        format!("b{digits}")
    } else {
        let value: u32 = digits.parse().ok()?;
        format!("m{value:05}")
    };
    Some((normalized, j))
}

/// Length (in bytes, leading blanks included) of a recognized refs divider.
fn separator_len(rest: &str) -> Option<usize> {
    let blanks = rest.len() - rest.trim_start().len();
    let body = rest.trim_start();
    let token = if let Some(dots) = body.strip_prefix("...") {
        body.len() - dots.len()
    } else if let Some(dots) = body.strip_prefix("..") {
        body.len() - dots.len()
    } else if body.len() >= 2 && body.as_bytes()[..2].eq_ignore_ascii_case(b"to") {
        2
    } else {
        let first = body.chars().next()?;
        if matches!(
            first,
            '-' | '\u{2013}' | '\u{2014}' | '\u{2026}' | '~' | '\u{2212}' | '\u{FF0D}'
        ) {
            first.len_utf8()
        } else {
            return None;
        }
    };
    Some(blanks + token)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compress_line_form_string_decodes_into_ranges() {
        // The shape the schema description and every nudge text advertise:
        // ONE plain string, one block per range, header line then summary.
        let args = json!({
            "content": "m00150–m00220 认证迁移\n会话存储换帧格式后重跑。\n\nm00300–m00350 部署\n替换宿主二进制并重启代理。"
        });
        let normalized = normalize_compress_args(&args);
        let ranges = normalized
            .get("ranges")
            .and_then(Value::as_array)
            .expect("line form must decode into ranges");
        assert_eq!(ranges.len(), 2, "every block opens a range: {ranges:?}");
        assert_eq!(ranges[0]["start"], json!("m00150"));
        assert_eq!(ranges[0]["end"], json!("m00220"));
        assert_eq!(ranges[0]["topic"], json!("认证迁移"));
        assert!(
            ranges[0]["summary"]
                .as_str()
                .unwrap()
                .contains("会话存储换帧格式后重跑")
        );
        assert_eq!(ranges[1]["start"], json!("m00300"));
        assert_eq!(ranges[1]["end"], json!("m00350"));
    }

    #[test]
    fn compress_line_form_accepts_single_ref_block_ids_and_ascii_dashes() {
        let single = normalize_compress_args(&json!({"content": "m00200 单条\n摘要"}));
        assert_eq!(single["ranges"][0]["start"], json!("m00200"));
        assert_eq!(single["ranges"][0]["end"], json!("m00200"));
        let blocks = normalize_compress_args(&json!({"content": "b3..b4 蒸馏\n合并两级块。"}));
        assert_eq!(blocks["ranges"][0]["start"], json!("b3"));
        assert_eq!(blocks["ranges"][0]["end"], json!("b4"));
        let dashes = normalize_compress_args(&json!({"content": "m00150 - m00220 范围\n摘要文本"}));
        assert_eq!(dashes["ranges"][0]["end"], json!("m00220"));
    }

    #[test]
    fn external_summaries_replace_the_model_text_in_line_form_args() {
        let args = json!({
            "content": "m00150–m00220 认证迁移\n模型自己写的摘要。"
        });
        let authored = vec!["外部模型写的摘要：保留路径与决定。".to_string()];
        let rewritten = apply_external_summaries(&args, &authored);
        let ranges = rewritten["ranges"].as_array().expect("normalized ranges");
        assert_eq!(ranges.len(), 1);
        assert_eq!(
            ranges[0]["summary"],
            json!("外部模型写的摘要：保留路径与决定。")
        );
        assert_eq!(ranges[0]["start"], json!("m00150"));
    }

    #[test]
    fn external_summaries_leave_ranges_the_chain_skipped_untouched() {
        let args = json!({
            "ranges": [
                {"start": "m00001", "end": "m00002", "summary": "模型一"},
                {"start": "m00003", "end": "m00004", "summary": "模型二"}
            ]
        });
        let authored = vec!["外部一".to_string()];
        let rewritten = apply_external_summaries(&args, &authored);
        let ranges = rewritten["ranges"].as_array().expect("ranges");
        assert_eq!(ranges[0]["summary"], json!("外部一"));
        assert_eq!(
            ranges[1]["summary"],
            json!("模型二"),
            "a short chain must not erase the model's own summary"
        );
    }

    #[test]
    fn compress_line_form_without_bounds_or_summary_is_not_guessed() {
        // No refs header → no bounds; a header without summary → nothing to
        // fold. Both stay unparsed so the tool reports the shape instead of
        // folding the wrong span.
        let no_header =
            normalize_compress_args(&json!({"content": "一段散文，没有 ref 头。\n更多散文。"}));
        assert!(no_header.get("ranges").is_none());
        let header_only = normalize_compress_args(&json!({"content": "m00150–m00220 只有头"}));
        assert!(header_only.get("ranges").is_none());
        let unknown_divider = normalize_compress_args(&json!({"content": "m00150 x m00220\n摘要"}));
        assert!(
            unknown_divider.get("ranges").is_none(),
            "an unknown divider must not narrow the pair to its first ref"
        );
    }

    #[test]
    fn compress_json_content_shapes_still_parse() {
        // Regression guard: the object forms documented in the schema must
        // keep working now that the line form is decoded alongside them.
        let array = normalize_compress_args(&json!({
            "content": [{"startId": "m00001", "endId": "m00002", "summary": "一"}]
        }));
        assert_eq!(array["ranges"][0]["start"], json!("m00001"));
        let single = normalize_compress_args(&json!({
            "content": {"startRef": "m00003", "endRef": "m00004", "summary": "二"}
        }));
        assert_eq!(single["ranges"][0]["endRef"], Value::Null);
        assert_eq!(single["ranges"][0]["start"], json!("m00003"));
        let encoded = normalize_compress_args(&json!({
            "content": "```json\n[{\"startId\":\"m00005\",\"endId\":\"m00006\",\"summary\":\"三\"}]\n```"
        }));
        assert_eq!(encoded["ranges"][0]["start"], json!("m00005"));
    }

    #[test]
    fn first_turn_does_not_inject_a_decision_marked_idle() {
        // The pipeline always returns a decision object so a host can show the
        // pending numbers; only `should_inject` authorizes putting text in front
        // of the model. A tiny first turn must stay byte-clean.
        let body = json!({
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi there"}]}]
        });
        let converted = crate::anthropic::to_core(&body).expect("valid test fixture");
        let store = Store::new();
        let prepared = store.prepare("tiny", 200_000, &converted.messages, 80, None);
        assert!(
            prepared.nudge.is_none(),
            "kernel decision should_inject=false must not reach the wire"
        );
    }

    #[test]
    fn tool_call_reuses_the_stored_context_limit_instead_of_defaulting() {
        // Wire requests prepare this conversation with the real model window
        // (262_144 for DSH), while tool calls carry no context-window header and
        // used to fall back to the bare default 200_000. `with_entry` rebuilds
        // the session on a limit mismatch, so that fallback wiped refs/folds and
        // every `mNNNNN` the nudge printed became `unknown_ref`. The tool route
        // must reuse the stored limit instead.
        let store = Store::new();
        store.with_entry("c1", 262_144, |entry| entry.requests += 1);
        assert_eq!(store.context_limit_of("c1"), Some(262_144));

        let limit = store.context_limit_of("c1").expect("wire limit persisted");
        store.with_entry("c1", limit, |entry| entry.requests += 1);
        assert_eq!(
            store.context_limit_of("c1"),
            Some(262_144),
            "a tool call must not rebuild the session on a limit mismatch"
        );
        assert_eq!(store.with_entry("c1", 262_144, |entry| entry.requests), 2);
        assert_eq!(store.context_limit_of("unseen"), None);
    }

    #[test]
    fn fold_economics_multiply_by_requests_after_each_fold() {
        let mut entry = Entry {
            session: Session::new(Config::new(1_000)),
            context_limit: 1_000,
            tokens: 0,
            ranges: Vec::new(),
            last_nudge: None,
            last_seen: 0,
            last_messages: Vec::new(),
            requests: 10,
            model: None,
            input_tokens: 0,
            cached_tokens: 0,
            output_tokens: 0,
            cache_samples: 0,
            context_tokens_total: 0,
            context_tokens_total_complete: true,
            context_usage: None,
            context_source: "estimate",
            protocol: String::new(),
            upstream: String::new(),
            fold_events: vec![
                FoldEvent {
                    at_requests: 3,
                    s: 100,
                    sigma: 20,
                    block: "b1".into(),
                    change_ref: None,
                    reason: None,
                },
                FoldEvent {
                    at_requests: 8,
                    s: 60,
                    sigma: 10,
                    block: "b2".into(),
                    change_ref: None,
                    reason: None,
                },
            ],
            metrics: CompactionMetrics::default(),
            nudge_pending: false,
            nudge_ignored_counted: false,
            last_prune_requests: None,
            last_prune_tokens: None,
            raw_input_estimate: None,
            effective_projection_estimate: None,
            fixed_overhead_estimate: None,
            last_projection_estimate: None,
            last_raw_input_estimate: None,
            prune_trigger_reason: None,
            fallback_in_flight: false,
            fallback_last_requests: None,
            fold_archives: Vec::new(),
            fold_journal: Vec::new(),
            fold_receipts: Vec::new(),
            fold_in_flight: false,
            min_compress_range: None,
            max_summary_length: None,
            window_resets: Vec::new(),
            speculation_in_flight: false,
            speculation: None,
            speculation_last_requests: None,
            summary_budget: SummaryBudgetUsage::default(),
        };
        let econ = session_economics(&entry);
        assert_eq!(econ.gross, 80 * 7 + 50 * 2);
        assert_eq!(econ.summary, 30);
        assert_eq!(econ.net, 630);
        assert_eq!(econ.folds, 2);
        assert!(econ.has_data);

        // A fold that has not yet been followed by another request still
        // contributes its summary cost, never a negative count of requests.
        entry.requests = 8;
        let econ = session_economics(&entry);
        assert_eq!(econ.gross, (80 * 5));
        assert_eq!(econ.net, 400 - 30);
    }

    #[test]
    fn a_landed_compress_call_records_a_fold_event() {
        let long = "The archive explains the deployment constraint in detail. ".repeat(90);
        let body = json!({
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": long}]},
                {"role": "assistant", "content": [{"type": "text", "text": long}]},
                {"role": "user", "content": [{"type": "text", "text": long}]},
                {"role": "assistant", "content": [{"type": "text", "text": long}]},
                {"role": "user", "content": [{"type": "text", "text": long}]},
                {"role": "assistant", "content": [{"type": "text", "text": long}]}
            ]
        });
        let converted = crate::anthropic::to_core(&body).expect("valid test fixture");
        let store = Store::new();
        let first = store.prepare("ledger-live", 200_000, &converted.messages, 12_000, None);
        assert!(!first.ranges.is_empty() || first.messages.len() == converted.messages.len());
        let summary = "This summary keeps the deployment decision, the rejected alternative, and the files touched.";
        let (_, outcome, blocks) = store.tool(
            "ledger-live",
            200_000,
            "compress",
            &json!({"ranges": [{"start": "m00001", "end": "m00003", "summary": summary}]}),
        );
        assert_eq!(outcome, Some("applied"), "compress should land");
        assert_eq!(blocks, Some(1));
        let status = store.status_json("ledger-live").expect("session exists");
        assert_eq!(status["foldCount"], 1);
        assert_eq!(status["hasFoldData"], true);
        assert!(status["grossSaved"].as_i64().unwrap_or(0) >= 0);
        assert!(status["blocks"].as_array().map(|b| b.len()).unwrap_or(0) >= 1);

        // One later request turns the landed fold into accumulated gross
        // savings: (S - sigma) × requestsAfter.
        store.prepare("ledger-live", 200_000, &converted.messages, 12_000, None);
        let overview = store.overview_json();
        assert_eq!(overview["foldsTotal"], 1);
        assert_eq!(overview["hasFoldData"], true);
        assert!(
            overview["grossSavedTotal"].as_i64().unwrap_or(0) > 0,
            "a fold followed by a request must show positive gross savings"
        );
    }

    #[test]
    fn a_per_call_summary_ceiling_lets_an_over_long_summary_land() {
        let long = "The archive explains the deployment constraint in detail. ".repeat(90);
        let body = json!({
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": long}]},
                {"role": "assistant", "content": [{"type": "text", "text": long}]},
                {"role": "user", "content": [{"type": "text", "text": long}]},
                {"role": "assistant", "content": [{"type": "text", "text": long}]},
                {"role": "user", "content": [{"type": "text", "text": long}]},
                {"role": "assistant", "content": [{"type": "text", "text": long}]}
            ]
        });
        let converted = crate::anthropic::to_core(&body).expect("valid test fixture");
        let store = Store::new();
        store.prepare("ceiling-live", 200_000, &converted.messages, 12_000, None);
        // Over the kernel default of 20 000 chars: exactly the shape the Node
        // host rejected as a whole batch (bili #7762).
        let summary = "- The archive recorded the deployment constraint and the run's file list. "
            .repeat(400);

        let (text, outcome, _) = store.tool(
            "ceiling-live",
            200_000,
            "compress",
            &json!({"ranges": [{"start": "m00001", "end": "m00003", "summary": summary}]}),
        );
        assert_eq!(outcome, Some("refused"), "{text}");
        assert!(text.contains("maximum is 20000"), "{text}");
        assert!(
            text.contains("summaryMaxChars"),
            "the refusal must name the parameter that repairs it: {text}"
        );

        // Same summary under an explicit per-call ceiling: the fold lands.
        let (text, outcome, blocks) = store.tool(
            "ceiling-live",
            200_000,
            "compress",
            &json!({
                "summaryMaxChars": 40_000,
                "ranges": [{"start": "m00001", "end": "m00003", "summary": summary}]
            }),
        );
        assert_eq!(outcome, Some("applied"), "{text}");
        assert_eq!(blocks, Some(1));
        let status = store.status_json("ceiling-live").expect("session exists");
        assert_eq!(status["foldCount"], 1);

        // The ceiling is per call, not per session: a later fold is back under
        // the config default, so the same summary is refused again.
        store.prepare("ceiling-live", 200_000, &converted.messages, 12_000, None);
        let (text, outcome, _) = store.tool(
            "ceiling-live",
            200_000,
            "compress",
            &json!({"ranges": [{"start": "m00004", "end": "m00006", "summary": summary}]}),
        );
        assert_eq!(outcome, Some("refused"), "{text}");
        assert!(text.contains("maximum is 20000"), "{text}");
    }

    #[test]
    fn summary_max_chars_reads_the_raw_args_and_ignores_nonsense() {
        assert_eq!(
            summary_max_chars(&json!({"summaryMaxChars": 9_000})),
            Some(9_000)
        );
        assert_eq!(
            summary_max_chars(&json!({"summary_max_chars": 9_000})),
            Some(9_000)
        );
        // The `content` form is normalised into `{"ranges": …}`, so the sibling
        // key must be read before that collapse.
        assert_eq!(
            summary_max_chars(&json!({"content": "m00001–m00003 keep", "summaryMaxChars": 9_000})),
            Some(9_000)
        );
        assert_eq!(summary_max_chars(&json!({"summaryMaxChars": 0})), None);
        assert_eq!(summary_max_chars(&json!({"summaryMaxChars": -5})), None);
        assert_eq!(summary_max_chars(&json!({"summaryMaxChars": "9000"})), None);
        assert_eq!(
            summary_max_chars(&json!({"summaryMaxChars": 300_000})),
            Some(SUMMARY_MAX_CHARS_CEIL)
        );
        assert_eq!(summary_max_chars(&json!({})), None);
    }

    #[test]
    fn a_repeated_nudge_carries_the_compact_rules_instead_of_the_full_sheet() {
        let full = nudge_prompts(0);
        assert!(
            full.how_to_compress_rules.chars().count() > 1_000,
            "the canonical sheet is the long one"
        );
        assert!(full.how_to_compress_rules.contains("KEEP VERBATIM"));
        let repeat = nudge_prompts(1);
        assert_eq!(repeat.how_to_compress_rules, NUDGE_REPEAT_RULES);
        assert!(
            repeat.how_to_compress_rules.chars().count() < 600,
            "the repeat reminder must stay compact"
        );
        assert_eq!(repeat.compress_philosophy, full.compress_philosophy);
    }

    #[test]
    fn protocol_names_map_onto_the_kernel_protocol() {
        assert!(matches!(
            protocol_named("anthropic"),
            Some(billion_context::Protocol::Anthropic)
        ));
        assert!(matches!(
            protocol_named("openai"),
            Some(billion_context::Protocol::Openai)
        ));
        assert!(matches!(
            protocol_named("responses"),
            Some(billion_context::Protocol::Responses)
        ));
        assert!(protocol_named("unknown").is_none());
    }

    #[test]
    fn a_fold_carries_the_user_intent_its_summary_dropped() {
        let long = "The archive explains the deployment constraint in detail. ".repeat(90);
        let intent = "Keep the export budget at 25 USD per run. Never raise it without asking.";
        let body = json!({
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": intent}]},
                {"role": "assistant", "content": [{"type": "text", "text": long}]},
                {"role": "assistant", "content": [{"type": "text", "text": long}]}
            ]
        });
        let converted = crate::anthropic::to_core(&body).expect("valid test fixture");
        let store = Store::new();
        store.prepare("carry-live", 200_000, &converted.messages, 12_000, None);
        let summary = "- Folded the export run: the ledger wrote three rows and the new store path was exercised end to end.";
        let (text, outcome, blocks) = store.tool(
            "carry-live",
            200_000,
            "compress",
            &json!({"ranges": [{"start": "m00001", "end": "m00003", "summary": summary}]}),
        );
        assert_eq!(outcome, Some("applied"), "{text}");
        assert_eq!(blocks, Some(1));
        assert!(text.contains("host-carried user intent: 1"), "{text}");
        let status = store.status_json("carry-live").expect("session exists");
        let block_summary = status["blocks"][0]["summary"].as_str().expect("summary");
        assert!(block_summary.starts_with(summary), "{block_summary}");
        assert!(
            block_summary.contains("[Host-carried user intent]"),
            "{block_summary}"
        );
        assert!(block_summary.contains("m00001"), "{block_summary}");
        assert!(
            block_summary.contains("Never raise it without asking"),
            "{block_summary}"
        );
        assert_eq!(status["compaction"]["intent_anchors_carried"], 1);
    }

    #[test]
    fn a_summary_that_already_cites_the_user_turn_gains_no_carried_section() {
        let long = "The archive explains the deployment constraint in detail. ".repeat(90);
        let body = json!({
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": long}]},
                {"role": "assistant", "content": [{"type": "text", "text": long}]},
                {"role": "assistant", "content": [{"type": "text", "text": long}]}
            ]
        });
        let converted = crate::anthropic::to_core(&body).expect("valid test fixture");
        let store = Store::new();
        store.prepare("cite-live", 200_000, &converted.messages, 12_000, None);
        let summary =
            "- The user (m00001) fixed the deployment constraint; the exit path was traced twice.";
        let (_, outcome, blocks) = store.tool(
            "cite-live",
            200_000,
            "compress",
            &json!({"ranges": [{"start": "m00001", "end": "m00003", "summary": summary}]}),
        );
        assert_eq!(outcome, Some("applied"));
        assert_eq!(blocks, Some(1));
        let status = store.status_json("cite-live").expect("session exists");
        let block_summary = status["blocks"][0]["summary"].as_str().expect("summary");
        assert_eq!(block_summary, summary);
        assert_eq!(status["compaction"]["intent_anchors_carried"], 0);
    }

    #[test]
    fn a_summary_that_echoes_the_instructions_is_refused_before_anything_folds() {
        let long = "The archive explains the deployment constraint in detail. ".repeat(90);
        let body = json!({
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": long}]},
                {"role": "assistant", "content": [{"type": "text", "text": long}]},
                {"role": "assistant", "content": [{"type": "text", "text": long}]}
            ]
        });
        let converted = crate::anthropic::to_core(&body).expect("valid test fixture");
        let store = Store::new();
        store.prepare("echo-live", 200_000, &converted.messages, 12_000, None);
        let echo = "- Follow the rules: keep verbatim paths, name user intent with a ref, and list open objectives whenever a summary is written.";
        let (text, outcome, blocks) = store.tool(
            "echo-live",
            200_000,
            "compress",
            &json!({"ranges": [{"start": "m00001", "end": "m00003", "summary": echo}]}),
        );
        assert_eq!(outcome, Some("refused"), "{text}");
        assert_eq!(blocks, Some(0));
        assert!(text.contains("Compression FAILED"), "{text}");
        assert!(
            text.contains("echoes the compression instructions"),
            "{text}"
        );
        let status = store.status_json("echo-live").expect("session exists");
        assert_eq!(status["blockCount"], 0);
        assert_eq!(status["compaction"]["summary_quality_refused"], 1);
    }

    #[test]
    fn the_authored_checkpoint_needs_an_active_block_and_renders_its_summary() {
        let long = "The archive explains the deployment constraint in detail. ".repeat(90);
        let body = json!({
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": long}]},
                {"role": "assistant", "content": [{"type": "text", "text": long}]},
                {"role": "user", "content": [{"type": "text", "text": long}]},
                {"role": "assistant", "content": [{"type": "text", "text": long}]},
                {"role": "user", "content": [{"type": "text", "text": long}]},
                {"role": "assistant", "content": [{"type": "text", "text": long}]}
            ]
        });
        let converted = crate::anthropic::to_core(&body).expect("valid test fixture");
        let store = Store::new();
        store.prepare("precheck-live", 200_000, &converted.messages, 12_000, None);
        assert!(
            store.authored_checkpoint("precheck-live").is_none(),
            "nothing folded yet means no checkpoint to answer with"
        );

        let summary = "Deployment decision: keep the JSON snapshot store and the proven ledger.";
        let (_, outcome, blocks) = store.tool(
            "precheck-live",
            200_000,
            "compress",
            &json!({"ranges": [{"start": "m00001", "end": "m00003", "summary": summary}]}),
        );
        assert_eq!(outcome, Some("applied"), "compress should land");
        assert_eq!(blocks, Some(1));

        let checkpoint = store
            .authored_checkpoint("precheck-live")
            .expect("a landed fold is answerable");
        assert_eq!(checkpoint.blocks, vec!["b1".to_string()]);
        assert!(
            checkpoint
                .text
                .starts_with("<bili-snapshot v1 conv=\"precheck-live\" seq=\"")
        );
        assert!(checkpoint.text.contains("blocks=\"b1\""));
        assert!(checkpoint.text.contains(summary));
        assert!(
            checkpoint
                .text
                .contains("[Compressed conversation section]")
        );
    }

    #[test]
    fn prune_artifact_refs_are_scoped_to_the_request_window() {
        let dir = std::env::temp_dir().join(format!("bili-rust-prune-refs-{}", now_ms()));
        let store = Store::with_dir(dir.clone());
        let artifacts = store.artifacts.as_ref().expect("artifact store");
        let from = now_ms();
        let meta = artifacts
            .put(
                "tool result bytes",
                "m3",
                None,
                Some("tool"),
                Some("call-1"),
            )
            .expect("artifact write");
        let to = now_ms();
        let report = PruneReport {
            triggered: true,
            candidates: 3,
            pruned_items: 1,
            artifact_writes: 1,
            pruned_ids: vec!["m3".to_string()],
            ..PruneReport::default()
        };
        let (refs, status) = store.resolve_prune_artifacts(&report, from, to);
        assert_eq!(refs, vec![meta.artifact_id.clone()]);
        assert_eq!(status, "prune_report_proven");

        // The message mapping exists, but a window before the write cannot
        // prove this request created the artifact.
        let (refs, status) = store.resolve_prune_artifacts(&report, 0, from.saturating_sub(1));
        assert!(refs.is_empty());
        assert_eq!(status, "unavailable");

        // A request that wrote nothing must not borrow session artifacts.
        let quiet = PruneReport {
            artifact_writes: 0,
            ..PruneReport::default()
        };
        let (refs, status) = store.resolve_prune_artifacts(&quiet, from, to);
        assert!(refs.is_empty());
        assert_eq!(status, "no_artifact_writes");
    }

    #[test]
    fn snapshots_are_indexed_without_loading_state() {
        let body = json!({"messages": [{"role": "user", "content": [{"type": "text", "text": "lazy restore"}]}]});
        let converted = crate::anthropic::to_core(&body).expect("valid test fixture");
        let dir = std::env::temp_dir().join(format!("bili-rust-lazy-{}", now_ms()));
        {
            let store = Store::with_dir(dir.clone());
            store.prepare("lazy-key", 200_000, &converted.messages, 120, None);
        }
        let store = Store::with_dir(dir.clone());
        assert_eq!(
            store.loaded_count(),
            0,
            "startup must not materialize state"
        );
        assert_eq!(store.index_count(), 1);
        let stats = store.stats_json();
        assert_eq!(stats["sessions"].as_array().map(Vec::len), Some(1));
        assert_eq!(store.loaded_count(), 0, "stats must stay lazy");
        let status = store.status_json("lazy-key").expect("indexed session");
        assert_eq!(status["ok"], true);
        assert_eq!(store.loaded_count(), 1, "status materializes one session");
        assert_eq!(store.storage_json()["sessions"], 1);

        // A pre-index snapshot is migrated once, then dropped again.
        let _ = fs::remove_file(dir.join(Store::meta_name("lazy-key")));
        let migrated = Store::with_dir(dir.clone());
        assert_eq!(migrated.loaded_count(), 0);
        assert_eq!(migrated.index_count(), 1);
        assert!(dir.join(Store::meta_name("lazy-key")).exists());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn cache_hit_pct_is_cumulative_not_a_single_request_rate() {
        let store = Store::new();
        store.with_entry("hit-rate", 200_000, |entry| {
            entry.protocol = "anthropic".to_string();
            entry.upstream = "http://127.0.0.1:9".to_string();
        });
        // Two responses: request 1 hit 900 of 1000 prompt tokens, request 2 hit
        // 10 of 100. The conversation ratio is 910 / 1100 = 82.7 -> 83%.
        store.record_usage(
            "hit-rate",
            Some("m"),
            Usage {
                input: 100,
                cached: 900,
                output: 5,
                context: 1000,
            },
        );
        store.record_usage(
            "hit-rate",
            Some("m"),
            Usage {
                input: 90,
                cached: 10,
                output: 1,
                context: 100,
            },
        );
        let stats = store.stats_json();
        let row = stats["sessions"]
            .as_array()
            .expect("stats rows")
            .iter()
            .find(|row| row["id"] == "hit-rate")
            .expect("hit-rate row");
        assert_eq!(row["cacheHitPctScope"], "cumulative");
        assert_eq!(row["cacheSamples"], 2);
        assert_eq!(row["cachedTokens"], 910);
        // Neither request's own rate (90% and 10%), and not the distorted
        // legacy ratio cached_total / latest_context (910%).  The aggregate is
        // visibly an aggregate.
        assert_eq!(row["cacheHitPct"], 83);

        // A conversation without a usage sample must not read as 0%.
        store.with_entry("no-samples", 200_000, |entry| {
            entry.protocol = "anthropic".to_string();
            entry.upstream = "http://127.0.0.1:9".to_string();
        });
        let stats = store.stats_json();
        let empty = stats["sessions"]
            .as_array()
            .expect("stats rows")
            .iter()
            .find(|row| row["id"] == "no-samples")
            .expect("no-samples row");
        assert_eq!(empty["cacheHitPct"], serde_json::Value::Null);
        assert_eq!(empty["cacheHitPctScope"], "cumulative");
    }

    #[test]
    fn legacy_snapshot_stays_null_after_a_new_request() {
        let dir = std::env::temp_dir().join(format!("bili-rust-legacy-hit-{}", now_ms()));
        let key = "legacy-hit";
        {
            let store = Store::with_dir(dir.clone());
            store.with_entry(key, 200_000, |entry| {
                entry.protocol = "anthropic".to_string();
                entry.upstream = "http://127.0.0.1:9".to_string();
            });
            store.record_usage(
                key,
                Some("m"),
                Usage {
                    input: 100,
                    cached: 900,
                    output: 5,
                    context: 1000,
                },
            );
        }
        // Simulate files written before D-2.2: the cumulative denominator and
        // its completeness marker are absent from both the snapshot and the
        // lightweight meta index.
        for path in [
            dir.join(Store::snapshot_name(key)),
            dir.join(Store::meta_name(key)),
        ] {
            let text = fs::read_to_string(&path).expect("prepared snapshot exists");
            let mut value: Value = serde_json::from_str(&text).expect("snapshot json");
            let object = value.as_object_mut().expect("snapshot object");
            object.remove("context_tokens_total");
            object.remove("context_tokens_total_complete");
            fs::write(&path, serde_json::to_vec(&value).expect("snapshot bytes"))
                .expect("rewrite old-format snapshot");
        }

        let store = Store::with_dir(dir.clone());
        store.with_entry(key, 200_000, |_| ());
        // New data alone cannot prove the pre-D-2.2 denominator history.
        store.record_usage(
            key,
            Some("m"),
            Usage {
                input: 90,
                cached: 10,
                output: 1,
                context: 100,
            },
        );
        let stats = store.stats_json();
        let row = stats["sessions"]
            .as_array()
            .expect("stats rows")
            .iter()
            .find(|row| row["id"] == key)
            .expect("legacy row");
        assert_eq!(row["cacheHitPctScope"], "cumulative");
        assert_eq!(row["cacheHitPct"], Value::Null);
        assert_eq!(row["cacheSamples"], 2);
        assert_eq!(row["cachedTokens"], 910);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn session_survives_a_restart_through_snapshots() {
        let body = json!({
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hello persistence"}]}]
        });
        let converted = crate::anthropic::to_core(&body).expect("valid test fixture");
        let dir = std::env::temp_dir().join(format!("bili-rust-persist-{}", now_ms()));
        {
            let store = Store::with_dir(dir.clone());
            store.prepare(
                "restart-key",
                200_000,
                &converted.messages,
                40,
                Some("gpt-test".into()),
            );
            store.record_usage(
                "restart-key",
                Some("gpt-test"),
                Usage {
                    input: 120,
                    cached: 40,
                    output: 8,
                    context: 120,
                },
            );
            let status = store.status_json("restart-key").unwrap();
            assert_eq!(status["requests"], 1);
            assert_eq!(status["contextTokens"], 120);
            assert_eq!(status["contextTokensSource"], "usage");
        }
        {
            let store = Store::with_dir(dir.clone());
            let status = store.status_json("restart-key").expect("snapshot restored");
            assert_eq!(status["requests"], 1);
            assert_eq!(status["contextTokens"], 120);
            assert_eq!(status["contextTokensSource"], "usage");
            assert_eq!(store.storage_json()["sessions"], 1);
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn store_level_timeline_separates_cooldown_bypass_and_expiry() {
        fn conversation(tools: usize) -> Vec<Message> {
            let mut messages = vec![Message::text(
                "m-large-user",
                billion_context::Role::User,
                "background context ".repeat(6_000),
            )];
            for index in 1..=tools {
                messages.push(Message::tool_result(
                    format!("m-tool-{index}"),
                    "Bash",
                    format!("c{index}"),
                    "tool output line\n".repeat(2_500),
                ));
            }
            for index in 0..5 {
                messages.push(Message::text(
                    format!("m-tail-{index}"),
                    billion_context::Role::User,
                    "recent small message",
                ));
            }
            messages
        }

        let dir = std::env::temp_dir().join(format!("bili-rust-timeline-{}", now_ms()));
        let policy = PrunePolicy {
            enabled: true,
            min_tokens: 1,
            cooldown_requests: 3,
            high_ratio: 0.60,
            growth_ratio: 0.05,
            ..PrunePolicy::default()
        };
        let store = Store::with_dir_and_policy(dir.clone(), policy);
        let context_limit = 40_000;
        let fixed_overhead = 800;
        let run = |tools: usize| {
            let messages = conversation(tools);
            let raw = estimate_messages_tokens(&messages);
            store.prepare_with_budget(
                "timeline",
                context_limit,
                &messages,
                raw,
                fixed_overhead,
                None,
            )
        };

        let first = run(2);
        assert!(first.prune_triggered);
        assert_eq!(first.prune.as_ref().unwrap().pruned_items, 2);
        assert_eq!(first.prune_decision.reason, "high_watermark");

        // Same raw transcript: the host resend is restubbed, but the cooldown
        // must still block a second prune.
        let second = run(2);
        assert!(!second.prune_triggered);
        assert!(second.prune_decision.cooldown_blocked);
        assert!(!second.prune_decision.growth_bypass);

        // A genuinely new oversized result grows the effective projection past
        // the growth threshold, which legitimately bypasses the cooldown.
        let third = run(3);
        assert!(third.prune_triggered);
        assert!(third.prune_decision.growth_bypass);
        assert!(!third.prune_decision.cooldown_blocked);
        assert!(
            !third.prune_decision.cooldown_hit,
            "a bypassed cooldown is recorded as growth_bypass, not cooldown_hit"
        );

        // A second new result does the same on the next request.
        let fourth = run(4);
        assert!(fourth.prune_triggered);
        assert!(fourth.prune_decision.growth_bypass);

        // Without fresh growth the cooldown blocks twice more.
        assert!(!run(4).prune_triggered);
        assert!(!run(4).prune_triggered);

        // Three requests after the last landed prune, the cooldown expires and
        // the high-water check runs again even with zero growth.
        let expired = run(4);
        assert!(expired.prune_decision.triggered);
        assert!(!expired.prune_decision.cooldown_blocked);
        assert!(!expired.prune_decision.growth_bypass);
        assert!(!expired.prune_triggered, "all remaining tools are stubs");

        let status = store.status_json("timeline").expect("status");
        assert_eq!(status["compaction"]["prune_runs"], 3);
        assert_eq!(status["compaction"]["prune_items"], 4);
        assert_eq!(status["cooldownBlocked"], 3);
        assert_eq!(status["growthBypassTriggered"], 2);
        assert_eq!(status["pruneSkipped"], 4);
        assert_eq!(status["requests"], 7);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn host_prune_records_metrics_and_keeps_stable_refs() {
        let dir = std::env::temp_dir().join(format!("bili-rust-prune-{}", now_ms()));
        let policy = PrunePolicy {
            enabled: true,
            min_tokens: 1,
            ..PrunePolicy::production()
        };
        let store = Store::with_dir_and_policy(dir.clone(), policy);
        let mut messages = vec![
            Message::text("m1", billion_context::Role::User, "run it"),
            Message::tool_call("m2", "Bash", "c1", "{}"),
            Message::tool_result("m3", "Bash", "c1", "q".repeat(80_000)),
        ];
        // Keep the oversized result outside the protected recent-token tail.
        for index in 4..10 {
            messages.push(Message::text(
                format!("m{index}"),
                billion_context::Role::User,
                "tail".repeat(1_000),
            ));
        }
        // Warm-up by itself must not hit the emergency truncation latch; use a
        // generous limit so the later 20k prune is the only pressure event.
        store.prepare("prune-key", 200_000, &messages, 1_000, None);
        let reference_before = {
            let guard = store.inner.lock().expect("session store poisoned");
            let entry = guard.get("prune-key").expect("session");
            entry
                .session
                .state()
                .refs
                .ref_of(&billion_context::MessageId::new("m3"))
        };
        let prepared = store.prepare("prune-key", 20_000, &messages, 20_000, None);
        assert_eq!(prepared.pruned_ids, vec!["m3".to_string()]);
        let reference_after = {
            let guard = store.inner.lock().expect("session store poisoned");
            let entry = guard.get("prune-key").expect("session");
            entry
                .session
                .state()
                .refs
                .ref_of(&billion_context::MessageId::new("m3"))
        };
        assert_eq!(
            reference_before, reference_after,
            "prune must not move refs"
        );
        assert!(prepared.decision_tokens < 20_000);
        let pruned = prepared
            .messages
            .iter()
            .find(|message| message.id.as_str() == "m3")
            .expect("pruned message remains in projection");
        let Content::ToolResult {
            call_id, content, ..
        } = &pruned.content
        else {
            panic!("expected tool result");
        };
        assert_eq!(&**call_id, "c1");
        assert!(content.starts_with("[artifact "));
        let status = store.status_json("prune-key").expect("status");
        assert_eq!(status["compaction"]["prune_items"], 1);
        assert_eq!(status["compaction"]["artifact_writes"], 1);
        let (text, outcome, _) =
            store.tool("prune-key", 20_000, "artifact_info", &json!({"ref": "m3"}));
        assert_eq!(outcome, Some("success"));
        assert!(text.contains("a_"));
        let (text, outcome, _) = store.tool(
            "prune-key",
            20_000,
            "artifact_read",
            &json!({"ref": "m3", "offset": 0, "limit": 64}),
        );
        assert_eq!(outcome, Some("success"));
        assert!(text.contains("next_offset"));
        let (text, outcome, _) = store.tool(
            "prune-key",
            20_000,
            "artifact_search",
            &json!({"ref": "m3", "query": "q", "limit": 2}),
        );
        assert_eq!(outcome, Some("success"));
        assert!(text.contains("hits"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn artifact_id_from_another_conversation_is_refused() {
        let dir = std::env::temp_dir().join(format!("bili-rust-owner-{}", now_ms()));
        let store = Store::with_dir(dir.clone());
        let artifacts = store.artifacts.as_ref().expect("artifact store");
        let a = artifacts
            .put("A payload", "m-a", None, Some("Bash"), None)
            .expect("a");
        let b = artifacts
            .put("B payload", "m-b", None, Some("Bash"), None)
            .expect("b");
        let messages_a = vec![Message::tool_result(
            "m-a",
            "Bash",
            "c1",
            format!("[artifact {}] source m-a", a.artifact_id),
        )];
        let messages_b = vec![Message::tool_result(
            "m-b",
            "Bash",
            "c1",
            format!("[artifact {}] source m-b", b.artifact_id),
        )];
        store.prepare("conv-a", 200_000, &messages_a, 1, None);
        store.prepare("conv-b", 200_000, &messages_b, 1, None);

        let (text, outcome, _) = store.tool(
            "conv-a",
            200_000,
            "artifact_info",
            &json!({"artifact_id": b.artifact_id}),
        );
        assert_ne!(outcome, Some("success"));
        assert!(text.contains("not owned by this conversation"), "{text}");
        let audit =
            std::fs::read_to_string(dir.join("artifacts").join("audit.jsonl")).expect("audit");
        assert!(audit.contains("refuse_access"), "{audit}");
        assert!(audit.contains("conv-a"), "{audit}");
        let (text, outcome, _) = store.tool(
            "conv-a",
            200_000,
            "artifact_info",
            &json!({"artifact_id": a.artifact_id}),
        );
        assert_eq!(outcome, Some("success"), "{text}");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn quota_rejection_during_prune_leaves_no_artifact_stub() {
        let dir = std::env::temp_dir().join(format!("bili-rust-quota-stub-{}", now_ms()));
        let policy = PrunePolicy {
            enabled: true,
            high_ratio: 0.01,
            low_ratio: 0.0,
            min_tokens: 1,
            head_chars: 16,
            tail_chars: 16,
            cooldown_requests: 0,
            growth_ratio: 0.0,
        };
        let mut store = Store::with_dir_and_policy(dir.clone(), policy);
        // Force a store that cannot accept the 5KB tool result.
        store.artifacts = Some(ArtifactStore::with_quota(&dir, 8));
        let mut messages = vec![Message::tool_result(
            "m-big",
            "Bash",
            "c1",
            "oversized output ".repeat(500),
        )];
        for index in 0..5 {
            messages.push(Message::text(
                format!("m-tail-{index}"),
                billion_context::Role::User,
                "recent",
            ));
        }
        let prepared = store.prepare("quota-key", 1_000, &messages, 1, None);
        for message in &prepared.messages {
            if let Content::ToolResult { content, .. } = &message.content {
                assert!(
                    !content.starts_with("[artifact "),
                    "quota failure must not leave a dangling stub: {}",
                    &content[..content.len().min(80)]
                );
            }
        }
        let status = store.status_json("quota-key").expect("status");
        assert_eq!(status["compaction"]["artifact_writes"], 0);
        assert_eq!(status["compaction"]["prune_items"], 0);
        let _ = fs::remove_dir_all(dir);
    }

    fn emergency_latched(store: &Store, key: &str) -> bool {
        let guard = store.inner.lock().expect("session store poisoned");
        guard
            .get(key)
            .is_some_and(|entry| !entry.session.state().nudge.truncate_latch.is_empty())
    }

    fn text_of(prepared: &Prepared, id: &str) -> String {
        prepared
            .messages
            .iter()
            .find(|message| message.id.as_str() == id)
            .and_then(|message| message.content.body())
            .map(str::to_string)
            .unwrap_or_default()
    }

    #[test]
    fn emergency_truncate_alone_latches_and_rewrites_the_oversized_body() {
        let dir = std::env::temp_dir().join(format!("bili-rust-emergency-{}", now_ms()));
        let policy = PrunePolicy {
            enabled: false,
            ..PrunePolicy::default()
        };
        let store = Store::with_dir_and_policy(dir.clone(), policy);
        let raw = "raw payload line\n".repeat(4_000);
        let mut messages = vec![Message::tool_result("m1", "Bash", "c1", &raw)];
        for index in 0..4 {
            messages.push(Message::text(
                format!("t{index}"),
                billion_context::Role::User,
                "recent",
            ));
        }
        let prepared = store.prepare("emergency", 1_000, &messages, 1, None);
        assert!(emergency_latched(&store, "emergency"));
        assert!(!text_of(&prepared, "m1").contains(&raw));
        assert!(text_of(&prepared, "m1").contains("[truncated for context space]"));
        assert!(!prepared.prune_triggered);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn hysteresis_prune_alone_does_not_arm_the_truncate_latch() {
        let dir = std::env::temp_dir().join(format!("bili-rust-hysteresis-{}", now_ms()));
        let policy = PrunePolicy {
            enabled: true,
            high_ratio: 0.60,
            low_ratio: 0.0,
            min_tokens: 1,
            head_chars: 32,
            tail_chars: 32,
            ..PrunePolicy::default()
        };
        let store = Store::with_dir_and_policy(dir.clone(), policy);
        let messages = vec![
            Message::tool_result("m1", "Bash", "c1", "q".repeat(100_000)),
            Message::text("m2", billion_context::Role::User, "tail"),
            Message::text("m3", billion_context::Role::User, "tail"),
            Message::text("m4", billion_context::Role::User, "tail"),
        ];
        let prepared = store.prepare("hysteresis", 20_000, &messages, 20_000, None);
        assert!(prepared.prune_triggered);
        assert_eq!(prepared.pruned_ids, vec!["m1".to_string()]);
        assert!(text_of(&prepared, "m1").starts_with("[artifact "));
        assert!(!emergency_latched(&store, "hysteresis"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn hysteresis_then_emergency_both_trigger_in_one_round() {
        let dir = std::env::temp_dir().join(format!("bili-rust-combined-{}", now_ms()));
        let policy = PrunePolicy {
            enabled: true,
            high_ratio: 0.01,
            low_ratio: 0.0,
            min_tokens: 1,
            head_chars: 16,
            tail_chars: 16,
            ..PrunePolicy::default()
        };
        let store = Store::with_dir_and_policy(dir.clone(), policy);
        let mut messages = vec![
            Message::tool_call("m0", "Bash", "c1", "{}"),
            Message::tool_result("m1", "Bash", "c1", "tool payload ".repeat(4_000)),
            Message::tool_result("m2", "Bash", "c2", "unrecoverable payload ".repeat(4_000))
                .with_unrecoverable_payload(true),
        ];
        for index in 0..4 {
            messages.push(Message::text(
                format!("t{index}"),
                billion_context::Role::User,
                "recent",
            ));
        }
        let prepared = store.prepare("combined", 1_000, &messages, 1, None);
        assert!(prepared.prune_triggered, "host Hysteresis must run");
        assert!(text_of(&prepared, "m1").starts_with("[artifact "));
        assert!(
            emergency_latched(&store, "combined"),
            "kernel emergency must run"
        );
        assert!(
            text_of(&prepared, "m2").contains("[truncated for context space]"),
            "the host-unrecoverable payload is handled by emergency truncate"
        );
        messages.clear();
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn raw_resend_keeps_existing_stub_and_does_not_duplicate_artifact() {
        let dir = std::env::temp_dir().join(format!("bili-rust-resend-{}", now_ms()));
        let policy = PrunePolicy {
            enabled: true,
            high_ratio: 0.01,
            low_ratio: 0.0,
            min_tokens: 1,
            head_chars: 16,
            tail_chars: 16,
            ..PrunePolicy::default()
        };
        let store = Store::with_dir_and_policy(dir.clone(), policy);
        let messages = vec![
            Message::tool_call("m0", "Bash", "c1", "{}"),
            Message::tool_result("m1", "Bash", "c1", "raw payload ".repeat(5_000)),
            Message::text("m2", billion_context::Role::User, "tail"),
        ];
        let first = store.prepare("resend", 1_000, &messages, 1, None);
        assert!(first.prune_triggered);
        let stub = text_of(&first, "m1");
        assert!(stub.starts_with("[artifact "));
        let second = store.prepare("resend", 1_000, &messages, 1, None);
        assert_eq!(text_of(&second, "m1"), stub);
        assert!(text_of(&second, "m1").starts_with("[artifact "));
        let status = store.status_json("resend").expect("status");
        assert_eq!(status["compaction"]["artifact_writes"], 1);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn stale_latch_does_not_overwrite_a_stub() {
        let dir = std::env::temp_dir().join(format!("bili-rust-stale-{}", now_ms()));
        let policy = PrunePolicy {
            enabled: false,
            ..PrunePolicy::default()
        };
        let store = Store::with_dir_and_policy(dir.clone(), policy);
        let raw = "raw payload ".repeat(4_000);
        let mut messages = vec![Message::tool_result("m1", "Bash", "c1", &raw)];
        let mut tailed = vec![Message::tool_result("m1", "Bash", "c1", "")];
        for index in 0..4 {
            let tail = Message::text(format!("t{index}"), billion_context::Role::User, "recent");
            messages.push(tail.clone());
            tailed.push(tail);
        }
        let _ = store.prepare("stale", 1_000, &messages, 1, None);
        assert!(emergency_latched(&store, "stale"));
        let stub = format!(
            "[artifact a_0123456789abcdef01234567] source m1 (Bash, 123 bytes)\n{}",
            "st ub ".repeat(2_000)
        );
        if let Content::ToolResult { content, .. } = &mut tailed[0].content {
            *content = stub.clone().into();
        }
        let prepared = store.prepare("stale", 1_000, &tailed, 1, None);
        assert_eq!(text_of(&prepared, "m1"), stub);
        assert!(!emergency_latched(&store, "stale"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn artifact_write_failure_with_truncate_pressure_leaves_no_stub() {
        let dir = std::env::temp_dir().join(format!("bili-rust-fail-both-{}", now_ms()));
        let policy = PrunePolicy {
            enabled: true,
            high_ratio: 0.01,
            low_ratio: 0.0,
            min_tokens: 1,
            head_chars: 16,
            tail_chars: 16,
            ..PrunePolicy::default()
        };
        let mut store = Store::with_dir_and_policy(dir.clone(), policy);
        store.artifacts = Some(ArtifactStore::with_quota(&dir, 8));
        let mut messages = vec![
            Message::tool_call("m0", "Bash", "c1", "{}"),
            Message::tool_result("m1", "Bash", "c1", "raw payload ".repeat(5_000)),
            Message::text(
                "m2",
                billion_context::Role::User,
                "text payload ".repeat(3_000),
            ),
        ];
        for index in 0..4 {
            messages.push(Message::text(
                format!("t{index}"),
                billion_context::Role::User,
                "recent",
            ));
        }
        let prepared = store.prepare("fail-both", 1_000, &messages, 1, None);
        eprintln!("C2_TIMELINE 1 hysteresis_condition=true reason=high_watermark");
        assert!(
            prepared.prune_decision.triggered,
            "Hysteresis must be triggered"
        );
        assert_eq!(prepared.prune_decision.reason, "high_watermark");
        eprintln!("C2_TIMELINE 2 hysteresis_write=false artifact_writes=0");
        assert!(
            !prepared.prune_triggered,
            "Hysteresis must not claim a write"
        );
        assert_eq!(
            store.artifacts.as_ref().unwrap().stats().artifacts,
            0,
            "write failure must leave no artifact"
        );
        eprintln!("C2_TIMELINE 3 emergency_triggered=true latch=true");
        assert!(
            emergency_latched(&store, "fail-both"),
            "emergency must be triggered"
        );
        eprintln!("C2_TIMELINE 4 projection=m1_truncated_no_stub");
        assert!(
            !text_of(&prepared, "m1").starts_with("[artifact "),
            "write failure must not create a stub"
        );
        assert!(text_of(&prepared, "m1").contains("[truncated for context space]"));
        assert!(
            text_of(&prepared, "m1").len() < 10_000,
            "final projection must be the clipped view, not the raw body"
        );
        eprintln!("C2_TIMELINE 5 latch_after_prepare=true");
        assert!(emergency_latched(&store, "fail-both"));
        eprintln!("C2_TIMELINE 6 artifact_and_index_empty=true");
        assert_eq!(store.artifacts.as_ref().unwrap().stats().artifacts, 0);
        assert_eq!(
            store
                .artifacts
                .as_ref()
                .unwrap()
                .index_entry_count()
                .unwrap(),
            0
        );
        eprintln!("C2_TIMELINE 7 tool_pair=m0:m1 call_id=c1");
        assert_eq!(messages[0].id.to_string(), "m0");
        assert_eq!(messages[1].id.to_string(), "m1");
        let Content::ToolResult { call_id, .. } = &messages[1].content else {
            panic!("expected tool result");
        };
        assert_eq!(&**call_id, "c1");
        let audit = fs::read_to_string(dir.join("artifacts").join("audit.jsonl"))
            .expect("quota rejection audit");
        assert!(audit.contains("quota_reject"), "{audit}");
        assert!(!audit.contains("raw payload"), "{audit}");
        eprintln!("C2_TIMELINE 8 repeated_prepare_stable=true");
        let repeated = store.prepare("fail-both", 1_000, &messages, 2, None);
        assert!(!repeated.prune_triggered);
        assert!(!text_of(&repeated, "m1").starts_with("[artifact "));
        assert!(text_of(&repeated, "m1").contains("[truncated for context space]"));
        assert!(emergency_latched(&store, "fail-both"));
        drop(store);
        eprintln!("C2_TIMELINE 9 restart_preserved=true");
        let mut reopened = Store::with_dir_and_policy(
            dir.clone(),
            PrunePolicy {
                enabled: true,
                high_ratio: 0.01,
                low_ratio: 0.0,
                min_tokens: 1,
                head_chars: 16,
                tail_chars: 16,
                ..PrunePolicy::default()
            },
        );
        reopened.artifacts = Some(ArtifactStore::with_quota(&dir, 8));
        let restarted = reopened.prepare("fail-both", 1_000, &messages, 1, None);
        assert!(!restarted.prune_triggered);
        assert!(!text_of(&restarted, "m1").starts_with("[artifact "));
        assert!(text_of(&restarted, "m1").contains("[truncated for context space]"));
        assert!(emergency_latched(&reopened, "fail-both"));
        assert_eq!(reopened.artifacts.as_ref().unwrap().stats().artifacts, 0);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn restart_preserves_stub_and_truncate_state_without_dangling_ref() {
        let dir = std::env::temp_dir().join(format!("bili-rust-restart-both-{}", now_ms()));
        let policy = PrunePolicy {
            enabled: true,
            high_ratio: 0.01,
            low_ratio: 0.0,
            min_tokens: 1,
            head_chars: 16,
            tail_chars: 16,
            ..PrunePolicy::default()
        };
        let messages = vec![
            Message::tool_call("m0", "Bash", "c1", "{}"),
            Message::tool_result("m1", "Bash", "c1", "raw payload ".repeat(5_000)),
            Message::text(
                "m2",
                billion_context::Role::User,
                "text payload ".repeat(3_000),
            ),
        ];
        {
            let store = Store::with_dir_and_policy(dir.clone(), policy.clone());
            let first = store.prepare("restart-both", 1_000, &messages, 1, None);
            assert!(first.prune_triggered);
        }
        let reopened = Store::with_dir_and_policy(dir.clone(), policy);
        let second = reopened.prepare("restart-both", 1_000, &messages, 1, None);
        assert!(text_of(&second, "m1").starts_with("[artifact "));
        assert!(
            !reopened
                .artifacts
                .as_ref()
                .unwrap()
                .info(
                    text_of(&second, "m1")
                        .split(' ')
                        .nth(1)
                        .unwrap_or("")
                        .trim_matches(['[', ']'])
                )
                .is_err(),
            "stub must point at a persisted artifact"
        );
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn combination_keeps_tool_call_and_result_paired() {
        let dir = std::env::temp_dir().join(format!("bili-rust-pair-both-{}", now_ms()));
        let policy = PrunePolicy {
            enabled: true,
            high_ratio: 0.01,
            low_ratio: 0.0,
            min_tokens: 1,
            head_chars: 16,
            tail_chars: 16,
            ..PrunePolicy::default()
        };
        let store = Store::with_dir_and_policy(dir.clone(), policy);
        let messages = vec![
            Message::tool_call("m0", "Bash", "c1", "{}"),
            Message::tool_result("m1", "Bash", "c1", "raw payload ".repeat(5_000)),
            Message::text(
                "m2",
                billion_context::Role::User,
                "text payload ".repeat(3_000),
            ),
        ];
        let prepared = store.prepare("pair-both", 1_000, &messages, 1, None);
        let call = prepared
            .messages
            .iter()
            .find(|message| message.id.as_str() == "m0")
            .and_then(|message| match &message.content {
                Content::ToolCall { call_id, .. } => Some(call_id.to_string()),
                _ => None,
            });
        let result = prepared
            .messages
            .iter()
            .find(|message| message.id.as_str() == "m1")
            .and_then(|message| match &message.content {
                Content::ToolResult { call_id, .. } => Some(call_id.to_string()),
                _ => None,
            });
        assert_eq!(call, result);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn persisted_stub_protects_its_artifact_across_restart_and_cleanup() {
        let dir = std::env::temp_dir().join(format!("bili-rust-protect-{}", now_ms()));
        let artifact_id;
        {
            let store = Store::with_dir(dir.clone());
            let artifacts = store.artifacts.as_ref().expect("artifact store");
            let live = artifacts
                .put("protected payload", "m-src", None, Some("Bash"), None)
                .expect("live artifact");
            artifact_id = live.artifact_id;
            let dead = artifacts
                .put("dead payload", "m-dead", None, Some("Bash"), None)
                .expect("dead artifact");
            let messages = vec![Message::tool_result(
                "m-stub",
                "Bash",
                "c1",
                format!(
                    "[artifact {artifact_id}] source m-src (Bash, {} bytes)\npreview",
                    live.byte_len
                ),
            )];
            store.prepare("protect-key", 200_000, &messages, 1, None);
            let report = store
                .cleanup_artifacts(now_ms() + 1_000, 0)
                .expect("cleanup");
            assert!(report.protected_artifacts >= 1, "{report:?}");
            assert_eq!(report.expired_removed, 1);
            assert!(artifacts.info(&artifact_id).is_ok());
            assert!(artifacts.info(&dead.artifact_id).is_err());
        }
        let reopened = Store::with_dir(dir.clone());
        let protection = reopened.artifact_protection();
        assert!(protection.complete);
        assert!(protection.artifact_ids.contains(&artifact_id));
        assert!(protection.message_ids.contains("m-stub"));
        let second = reopened
            .cleanup_artifacts(now_ms() + 10_000, 0)
            .expect("restart cleanup");
        assert_eq!(second.expired_removed, 0);
        assert!(second.protected_artifacts >= 1);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn unreadable_session_snapshot_blocks_artifact_cleanup() {
        let dir = std::env::temp_dir().join(format!("bili-rust-protect-corrupt-{}", now_ms()));
        let store = Store::with_dir(dir.clone());
        let artifacts = store.artifacts.as_ref().expect("artifact store");
        let meta = artifacts
            .put("must survive", "m1", None, Some("Bash"), None)
            .expect("artifact");
        fs::write(dir.join("broken-session.json"), b"{ not a snapshot").expect("corrupt snapshot");
        let err = store
            .cleanup_artifacts(now_ms() + 10_000, 0)
            .expect_err("cleanup must refuse");
        assert!(err.contains("incomplete"), "{err}");
        assert!(artifacts.info(&meta.artifact_id).is_ok());
        let _ = std::fs::remove_dir_all(dir);
    }

    // -----------------------------------------------------------------------
    // P1 auto fallback
    // -----------------------------------------------------------------------

    /// Prepare a conversation and then set the token counter the way `prepare`
    /// would have: the tests need the ratio, not a huge literal body. A small
    /// window keeps the protected tail below the transcript, so the kernel
    /// still advertises foldable ranges.
    fn prepared_at(key: &str, store: &Store, limit: u32, tokens: u32) {
        let long = "The archive explains the deployment constraint in detail. ".repeat(90);
        let turn =
            |text: &str| json!({"role": "assistant", "content": [{"type": "text", "text": text}]});
        let body = json!({
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Keep the export budget at 25 USD per run."}]},
                turn(&long),
                turn(&long),
                turn(&long),
                turn(&long),
                turn(&long),
                turn(&long)
            ]
        });
        let converted = crate::anthropic::to_core(&body).expect("valid test fixture");
        store.prepare(key, limit, &converted.messages, 12_000, None);
        store.with_entry(key, limit, |entry| entry.tokens = tokens);
    }

    /// A summary that passes every gate: dense bullets, no refs, no rule echo.
    const FALLBACK_SUMMARY: &str = "- Folded the deployment run: the archive script read three tables, wrote the JSON snapshot, and verified the ledger row.\n- Decision: keep the snapshot store and the proven ledger path; the export budget of 25 USD per run stays unchanged.\n- Next: rerun the audit after the store rename lands.";

    #[test]
    fn a_fallback_plan_needs_pressure_or_two_ignored_nudges() {
        let store = Store::new();
        prepared_at("calm-live", &store, 200_000, 12_000);
        assert!(
            store.fallback_plan("calm-live").is_none(),
            "12% of the window and no ignored nudge is not a trigger"
        );
        prepared_at("press-live", &store, 8_000, 7_300);
        let plan = store.fallback_plan("press-live").expect("pressure plan");
        assert_eq!(plan.reason, "pressure");
        assert!(!plan.ranges.is_empty(), "a plan must name a range");
        assert!(
            store.fallback_plan("press-live").is_none(),
            "one attempt in flight blocks the next one"
        );
        prepared_at("nudge-live", &store, 200_000, 12_000);
        store.with_entry("nudge-live", 200_000, |entry| {
            entry.nudge_pending = true;
            entry.metrics.nudge_ignored = 2;
        });
        let plan = store
            .fallback_plan("nudge-live")
            .expect("two ignored nudges are a trigger");
        assert_eq!(plan.reason, "ignored_nudge");
    }

    #[test]
    fn has_foldable_ranges_answers_the_compaction_preflight_question() {
        let store = Store::new();
        assert!(
            !store.has_foldable_ranges("preflight-none"),
            "an unknown conversation has nothing to fold"
        );
        prepared_at("preflight-live", &store, 8_000, 7_300);
        assert!(
            store.has_foldable_ranges("preflight-live"),
            "under pressure the range is foldable without a compress call"
        );
        let plan = store
            .fallback_plan("preflight-live")
            .expect("pressure plan");
        assert!(!plan.ranges.is_empty());
        assert!(
            !store.has_foldable_ranges("preflight-live"),
            "a planned fold owns the range until it lands or aborts"
        );
        store.abort_fallback("preflight-live", "test");
        assert!(
            store.has_foldable_ranges("preflight-live"),
            "an aborted plan releases the range for the next attempt"
        );
    }

    #[test]
    fn a_pressure_fallback_folds_the_oldest_range_without_a_tool_call() {
        let store = Store::new();
        prepared_at("fold-live", &store, 8_000, 7_300);
        let plan = store.fallback_plan("fold-live").expect("pressure plan");
        let inputs = store.fallback_inputs("fold-live", &plan);
        assert_eq!(inputs.len(), 1, "one range, one summarizer input");
        assert!(
            !inputs[0].text.trim().is_empty(),
            "the slice must have text"
        );
        let text = store
            .apply_fallback(
                "fold-live",
                8_000,
                plan,
                &[FALLBACK_SUMMARY.to_string()],
                now_ms(),
            )
            .expect("the attempt was in flight");
        assert!(text.contains("auto-fallback: auto:pressure"), "{text}");
        let status = store.status_json("fold-live").expect("session exists");
        assert_eq!(status["blockCount"], 1);
        assert_eq!(status["compaction"]["fallback_attempts"], 1);
        assert_eq!(status["compaction"]["fallback_applied_blocks"], 1);
        assert_eq!(status["autoFallback"]["appliedBlocks"], 1);
        assert_eq!(status["autoFallback"]["inFlight"], false);
        assert_eq!(status["nudgePending"], false);
        let block_summary = status["blocks"][0]["summary"].as_str().expect("summary");
        assert!(
            block_summary.starts_with(FALLBACK_SUMMARY),
            "{block_summary}"
        );
        // One attempt per growth window: the next turn must not bill again.
        assert!(
            store.fallback_plan("fold-live").is_none(),
            "the window holds the next attempt"
        );
        store.with_entry("fold-live", 8_000, |entry| entry.requests += 3);
        assert!(
            store.fallback_plan("fold-live").is_some(),
            "a grown window may try again"
        );
    }

    #[test]
    fn a_fallback_summary_that_echoes_the_rules_is_refused() {
        let store = Store::new();
        prepared_at("echo-auto", &store, 8_000, 7_300);
        let plan = store.fallback_plan("echo-auto").expect("pressure plan");
        let echo = "- Follow the rules: keep verbatim paths, name user intent with a ref, and list open objectives whenever a summary is written.";
        let text = store
            .apply_fallback("echo-auto", 8_000, plan, &[echo.to_string()], now_ms())
            .expect("the attempt was in flight");
        assert!(text.contains("auto-fallback refused"), "{text}");
        assert!(
            text.contains("echoes the compression instructions"),
            "{text}"
        );
        let status = store.status_json("echo-auto").expect("session exists");
        assert_eq!(status["blockCount"], 0, "a refused summary folds nothing");
        assert_eq!(status["compaction"]["fallback_refused"], 1);
        assert_eq!(status["autoFallback"]["inFlight"], false);
        // A released attempt cannot be landed twice.
        assert!(
            !store
                .apply_fallback(
                    "echo-auto",
                    8_000,
                    FallbackPlan {
                        reason: "pressure",
                        ranges: Vec::new(),
                    },
                    &[FALLBACK_SUMMARY.to_string()],
                    now_ms()
                )
                .is_some(),
            "no attempt is in flight"
        );
    }

    #[test]
    fn a_dead_summarizer_chain_releases_the_attempt_and_waits_one_window() {
        let store = Store::new();
        prepared_at("chain-live", &store, 8_000, 7_300);
        assert!(store.fallback_plan("chain-live").is_some());
        store.abort_fallback("chain-live", "no_chain");
        let status = store.status_json("chain-live").expect("session exists");
        assert_eq!(status["autoFallback"]["failures"], 1);
        assert_eq!(status["autoFallback"]["reason"], "no_chain:failed");
        assert_eq!(status["autoFallback"]["inFlight"], false);
        assert_eq!(
            status["blockCount"], 0,
            "a dead chain must not change the view"
        );
        assert!(store.fallback_plan("chain-live").is_none());
        store.with_entry("chain-live", 8_000, |entry| entry.requests += 2);
        assert!(
            store.fallback_plan("chain-live").is_none(),
            "two requests is still inside the window"
        );
        store.with_entry("chain-live", 8_000, |entry| entry.requests += 1);
        assert!(
            store.fallback_plan("chain-live").is_some(),
            "three requests opens a new window"
        );
    }

    // ---- P2: fold archives and explicit raw retrieval -------------------

    fn archive_test_store() -> (Store, PathBuf) {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "bili-rust-archive-{}-{}-{}",
            std::process::id(),
            now_ms(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        (Store::with_dir(dir.clone()), dir)
    }

    fn six_long_messages() -> Vec<Message> {
        let long = "The archive explains the deployment constraint in detail. ".repeat(90);
        let body = json!({
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Keep the export budget at 25 USD per run."}]},
                {"role": "assistant", "content": [{"type": "text", "text": long}]},
                {"role": "assistant", "content": [{"type": "text", "text": long}]},
                {"role": "assistant", "content": [{"type": "text", "text": long}]},
                {"role": "assistant", "content": [{"type": "text", "text": long}]},
                {"role": "assistant", "content": [{"type": "text", "text": long}]}
            ]
        });
        crate::anthropic::to_core(&body)
            .expect("valid test fixture")
            .messages
    }

    #[test]
    fn a_landed_fold_archives_the_raw_span_and_retrieve_raw_serves_it() {
        let (store, dir) = archive_test_store();
        let messages = six_long_messages();
        store.prepare("archive-live", 200_000, &messages, 30_000, None);
        let summary = "- Folded the deployment run: three config rows moved, the rejected vendor choice and the exact path stay recorded.";
        let (text, outcome, _) = store.tool(
            "archive-live",
            200_000,
            "compress",
            &json!({"ranges": [{"start": "m00002", "end": "m00003", "summary": summary}]}),
        );
        assert_eq!(outcome, Some("applied"), "{text}");
        let status = store.status_json("archive-live").expect("session exists");
        assert_eq!(status["compaction"]["fold_archive_written"], 1, "{status}");
        let archive = &status["foldArchives"][0];
        let block = archive["block"].as_str().expect("block id");
        assert_eq!(archive["span"], "m00002–m00003");
        assert!(
            archive["bytes"].as_u64().unwrap_or(0) > 5_000,
            "the raw span is archived with its real size: {archive}"
        );

        // The block id is enough to reach the original without a live ref.
        let (text, outcome, _) = store.tool(
            "archive-live",
            200_000,
            "retrieve_raw",
            &json!({"block": block}),
        );
        assert_eq!(outcome, Some("success"), "{text}");
        assert!(text.contains("bytes stored"), "{text}");
        assert!(
            text.contains("The archive explains the deployment constraint"),
            "the raw text comes back: {text}"
        );
        let status = store.status_json("archive-live").expect("session exists");
        assert_eq!(status["compaction"]["retrieve_raw_calls"], 1);
        assert!(
            status["compaction"]["retrieve_raw_bytes"]
                .as_u64()
                .unwrap_or(0)
                > 0
        );

        // The documented spelling: retrieve with mode=original, bounded by limit.
        let (text, outcome, _) = store.tool(
            "archive-live",
            200_000,
            "retrieve",
            &json!({"refs": ["m00002"], "mode": "original", "limit": 512}),
        );
        assert_eq!(outcome, Some("success"), "{text}");
        assert!(
            text.contains("nextOffset="),
            "the excerpt is bounded: {text}"
        );
        assert!(
            text.chars().count() < 4_000,
            "the default surface never dumps the full span"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn retrieve_raw_says_original_unavailable_without_an_archive() {
        let (store, dir) = archive_test_store();
        let messages = six_long_messages();
        store.prepare("raw-miss", 200_000, &messages, 30_000, None);
        let (text, outcome, _) = store.tool(
            "raw-miss",
            200_000,
            "retrieve_raw",
            &json!({"refs": ["m00001"]}),
        );
        assert_eq!(outcome, Some("refused"), "{text}");
        assert!(text.contains("OriginalUnavailable"), "{text}");
        let status = store.status_json("raw-miss").expect("session exists");
        assert_eq!(status["compaction"]["retrieve_raw_unavailable"], 1);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn decompress_of_a_missing_block_names_original_unavailable() {
        let (store, dir) = archive_test_store();
        let messages = six_long_messages();
        store.prepare("decompress-miss", 200_000, &messages, 30_000, None);
        let (text, outcome, _) = store.tool(
            "decompress-miss",
            200_000,
            "decompress",
            &json!({"blockId": "b99"}),
        );
        assert_eq!(outcome, Some("refused"), "{text}");
        assert!(text.contains("OriginalUnavailable"), "{text}");
        assert!(text.contains("Nothing was re-projected"), "{text}");
        let _ = std::fs::remove_dir_all(dir);
    }

    // ---- P3: fold journal, busy guard, call_id idempotency --------------

    const P3_SUMMARY: &str = "- Folded the deployment run: three config rows moved, the rejected vendor choice and the exact path stay recorded.";

    fn open_journal(entry: &mut Entry, call_id: Option<String>) {
        entry.fold_journal.push(FoldJournal {
            call_id,
            spans: vec!["m00002–m00003".to_string()],
            blocks: Vec::new(),
            phase: "start".to_string(),
            at_requests: entry.requests,
            at: now_ms(),
            note: String::new(),
        });
        entry.fold_in_flight = true;
    }

    #[test]
    fn a_replayed_compress_call_returns_the_same_exchange_without_folding_again() {
        let (store, dir) = archive_test_store();
        let messages = six_long_messages();
        store.prepare("idem-live", 200_000, &messages, 30_000, None);
        let args = json!({
            "ranges": [{"start": "m00002", "end": "m00003", "summary": P3_SUMMARY}],
            "tool_call_id": "call-77"
        });
        let (first, outcome, blocks) = store.tool("idem-live", 200_000, "compress", &args);
        assert_eq!(outcome, Some("applied"), "{first}");
        assert_eq!(blocks, Some(1));
        let (second, outcome, blocks) = store.tool("idem-live", 200_000, "compress", &args);
        assert_eq!(outcome, Some("success"), "{second}");
        assert!(second.contains("idempotent replay"), "{second}");
        assert_eq!(
            blocks,
            Some(1),
            "the replayed exchange keeps its block count"
        );
        let status = store.status_json("idem-live").expect("session exists");
        assert_eq!(status["blockCount"], 1, "the view folded exactly once");
        assert_eq!(status["compaction"]["fold_applied"], 1);
        assert_eq!(status["compaction"]["idempotent_replays"], 1);
        assert_eq!(status["foldTxn"]["open"], 0);
        assert_eq!(status["foldTxn"]["recent"][0]["phase"], "end");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn an_open_fold_journal_refuses_the_next_fold_then_clears() {
        let (store, dir) = archive_test_store();
        let messages = six_long_messages();
        store.prepare("journal-live", 200_000, &messages, 30_000, None);
        store.with_entry("journal-live", 200_000, |entry| {
            open_journal(entry, Some("call-9".to_string()))
        });
        let (text, outcome, blocks) = store.tool(
            "journal-live",
            200_000,
            "compress",
            &json!({
                "ranges": [{"start": "m00002", "end": "m00003", "summary": P3_SUMMARY}],
                "tool_call_id": "call-10"
            }),
        );
        assert_eq!(outcome, Some("busy"), "{text}");
        assert_eq!(blocks, Some(0));
        assert!(text.contains("compress busy"), "{text}");
        let status = store.status_json("journal-live").expect("session exists");
        assert_eq!(status["blockCount"], 0, "residue must not be folded over");
        assert_eq!(status["compaction"]["fold_busy_refused"], 1);
        assert_eq!(status["foldTxn"]["open"], 0, "the stale entry was closed");
        // The next turn runs on a clean journal.
        let (text, outcome, _) = store.tool(
            "journal-live",
            200_000,
            "compress",
            &json!({"ranges": [{"start": "m00002", "end": "m00003", "summary": P3_SUMMARY}]}),
        );
        assert_eq!(outcome, Some("applied"), "{text}");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn an_open_journal_survives_a_restart_as_one_recovery_count() {
        let (store, dir) = archive_test_store();
        let messages = six_long_messages();
        store.prepare("restart-live", 200_000, &messages, 30_000, None);
        store.with_entry("restart-live", 200_000, |entry| {
            open_journal(entry, Some("call-11".to_string()))
        });
        drop(store);
        let reopened = Store::with_dir(dir.clone());
        let status = reopened.status_json("restart-live").expect("restored");
        assert_eq!(
            status["compaction"]["fold_journal_recovered"], 1,
            "{status}"
        );
        assert_eq!(
            status["foldTxn"]["open"], 1,
            "residue stays visible until the next fold"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    // ---- Phase 3.3: explicit fresh-window reset ------------------------

    #[test]
    fn fresh_window_reset_keeps_tail_and_keeps_archived_originals_reachable() {
        let (store, dir) = archive_test_store();
        let messages = six_long_messages();
        store.prepare("reset-live", 200_000, &messages, 30_000, None);
        let (text, outcome, _) = store.tool(
            "reset-live",
            200_000,
            "compress",
            &json!({"ranges": [{"start": "m00002", "end": "m00003", "summary": P3_SUMMARY}]}),
        );
        assert_eq!(outcome, Some("applied"), "{text}");
        let status = store.status_json("reset-live").expect("session exists");
        let artifact = status["foldArchives"][0]["artifact"]
            .as_str()
            .expect("archive artifact")
            .to_string();
        let block = status["foldArchives"][0]["block"]
            .as_str()
            .expect("archive block")
            .to_string();
        let (text, outcome, _) = store.tool(
            "reset-live",
            200_000,
            "reset_window",
            &json!({"reason": "deployment brief changed", "keepRecent": 4}),
        );
        assert_eq!(outcome, Some("success"), "{text}");
        assert!(text.contains("fresh window reset"), "{text}");
        assert!(
            text.contains(&artifact),
            "the reset receipt must name the archived original: {text}"
        );
        let status = store.status_json("reset-live").expect("session exists");
        assert_eq!(status["blockCount"], 0, "folds dropped: {status}");
        assert_eq!(status["freshWindow"]["resets"], 1, "{status}");
        assert_eq!(
            status["freshWindow"]["last"]["droppedBlocks"], 1,
            "{status}"
        );
        assert_eq!(
            status["foldArchives"].as_array().map(Vec::len),
            Some(0),
            "block addressing is gone with the old window: {status}"
        );
        // The archived span survives the reset; the receipt is the ownership proof.
        let (raw, outcome, _) = store.tool(
            "reset-live",
            200_000,
            "artifact_read",
            &json!({"artifact_id": artifact, "limit": 512}),
        );
        assert_eq!(outcome, Some("success"), "{raw}");
        // The dropped block id no longer resolves in the new window.
        let (raw, outcome, _) = store.tool(
            "reset-live",
            200_000,
            "decompress",
            &json!({"blockId": block}),
        );
        assert_eq!(outcome, Some("refused"), "{raw}");
        assert!(raw.contains("OriginalUnavailable"), "{raw}");
        let _ = std::fs::remove_dir_all(dir);
    }

    // ---- Phase 3.1: speculative folds --------------------------------

    #[test]
    fn speculation_folds_consumed_blocks_only_when_the_generation_still_matches() {
        let (store, dir) = archive_test_store();
        let messages = six_long_messages();
        store.prepare(
            "spec-live",
            100_000,
            &messages,
            30_000,
            Some("probe-model".to_string()),
        );
        let (text, outcome, _) = store.tool(
            "spec-live",
            100_000,
            "compress",
            &json!({"ranges": [{"start": "m00002", "end": "m00003", "summary": P3_SUMMARY}]}),
        );
        assert_eq!(outcome, Some("applied"), "{text}");
        // Put the conversation in the 45–75% band without another request.
        store.with_entry("spec-live", 100_000, |entry| {
            entry.tokens = 60_000;
        });
        let plan = store
            .speculation_plan("spec-live")
            .expect("band + block should open a candidate");
        assert_eq!(plan.blocks.len(), 1, "{plan:?}");
        assert!(plan.signatures[0].contains("T1"), "{plan:?}");
        let summaries = vec![
            "- T2 digest: the deployment run merged into one paragraph, no detail was lost."
                .to_string(),
        ];
        let note = store
            .apply_speculation("spec-live", 100_000, plan, &summaries, now_ms())
            .expect("an adopted candidate answers");
        assert!(note.contains("adopted"), "{note}");
        let status = store.status_json("spec-live").expect("session exists");
        assert_eq!(status["speculation"]["started"], 1, "{status}");
        assert_eq!(status["speculation"]["adopted"], 1, "{status}");
        assert_eq!(status["speculation"]["discarded"], 0, "{status}");
        // A second candidate that changes under the lock must be discarded.
        store.with_entry("spec-live", 100_000, |entry| {
            entry.requests += 4;
            entry.speculation_last_requests = Some(entry.requests.saturating_sub(4));
        });
        // Folding the only T1 block left no T1 target: fold another range so a
        // fresh candidate exists to drift from.
        let (text, outcome, _) = store.tool(
            "spec-live",
            100_000,
            "compress",
            &json!({"ranges": [{"start": "m00004", "end": "m00005", "summary": P3_SUMMARY}]}),
        );
        assert_eq!(outcome, Some("applied"), "{text}");
        let plan = store
            .speculation_plan("spec-live")
            .expect("cooldown elapsed, another candidate opens");
        let block = BlockId::parse(plan.blocks[0].as_str()).expect("planned block parses");
        store.with_entry("spec-live", 100_000, |entry| {
            if let Some(block) = entry
                .session
                .state_mut()
                .blocks
                .iter_mut()
                .find(|candidate| candidate.id == block)
            {
                block.summary.push_str(" (drifted)");
            }
        });
        let note = store
            .apply_speculation("spec-live", 100_000, plan, &summaries, now_ms())
            .expect("a discarded candidate still answers");
        assert!(note.contains("discarded"), "{note}");
        let status = store.status_json("spec-live").expect("session exists");
        assert_eq!(status["speculation"]["adopted"], 1, "{status}");
        assert_eq!(status["speculation"]["discarded"], 1, "{status}");
        assert_eq!(
            status["speculation"]["inFlight"], false,
            "the marker is released either way: {status}"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    // ---- Phase 3.2: per-model policy ---------------------------------

    #[test]
    fn model_policy_resolves_exact_then_case_insensitive_windows() {
        let config = r#"{
            "modelPolicies": {
                "claude-sonnet-4": {"contextWindow": 200000},
                "DeepSeek-V4": {"contextWindow": 128000},
                "gpt-5": {"context_window": 400000},
                "tight": {"contextWindow": 128000, "minCompressRange": 200, "maxSummaryLength": 9000},
                "bad-low": {"contextWindow": 10},
                "bad-high": {"contextWindow": 9000000}
            }
        }"#;
        assert_eq!(
            model_context_window(config, "claude-sonnet-4"),
            Some(200_000)
        );
        assert_eq!(
            model_context_window(config, "deepseek-v4"),
            Some(128_000),
            "model ids are matched case-insensitively"
        );
        assert_eq!(model_context_window(config, "gpt-5"), Some(400_000));
        assert_eq!(
            model_context_window(config, "deepseek-v4-1-flash"),
            None,
            "no prefix matching: an unknown model keeps the default window"
        );
        assert_eq!(model_context_window(config, "bad-low"), None);
        assert_eq!(model_context_window(config, "bad-high"), None);
        assert_eq!(model_context_window(config, ""), None);
        assert_eq!(model_context_window("not json", "claude-sonnet-4"), None);
        let tight = model_policy(config, "tight");
        assert_eq!(tight.min_compress_range, Some(200));
        assert_eq!(tight.max_summary_length, Some(9_000));
    }

    #[test]
    fn model_policy_compress_knobs_reach_the_compress_path() {
        let (store, dir) = archive_test_store();
        let messages = six_long_messages();
        store.prepare("policy-live", 200_000, &messages, 30_000, None);
        // A policy raising the minimum makes a span that normally folds refuse.
        store.apply_model_policy(
            "policy-live",
            200_000,
            &ModelPolicy {
                context_window: Some(300_000),
                min_compress_range: Some(500_000),
                max_summary_length: Some(9_000),
            },
        );
        let (text, outcome, _) = store.tool(
            "policy-live",
            200_000,
            "compress",
            &json!({"ranges": [{"start": "m00002", "end": "m00003", "summary": P3_SUMMARY}]}),
        );
        assert_eq!(outcome, Some("refused"), "{text}");
        assert!(text.contains("too small"), "policy minimum applied: {text}");
        assert_eq!(status_block_count(&store, "policy-live"), 0);
        let _ = std::fs::remove_dir_all(dir);
    }

    fn status_block_count(store: &Store, key: &str) -> u64 {
        store
            .status_json(key)
            .and_then(|status| status["blockCount"].as_u64())
            .unwrap_or(u64::MAX)
    }

    // ---- Task 2: automatic summary budget -------------------------------

    #[test]
    fn a_refused_budget_releases_the_fallback_and_never_folds() {
        let store = Store::new();
        prepared_at("budget-fallback", &store, 8_000, 7_300);
        assert!(store.fallback_plan("budget-fallback").is_some());
        // Pretend the session already spent its whole automatic budget; the
        // fallback must release its in-flight flag instead of staying stuck.
        store.with_entry("budget-fallback", 8_000, |entry| {
            entry.summary_budget.calls = 2;
            entry.metrics.fallback_attempts += 1;
        });
        let refusal = store
            .reserve_summary_call("budget-fallback", false, 2, 16)
            .expect_err("budget is exhausted");
        assert!(refusal.contains("exhausted"), "{refusal}");
        store.abort_fallback("budget-fallback", "budget");
        let status = store.status_json("budget-fallback").expect("status");
        assert_eq!(status["autoFallback"]["inFlight"], false);
        assert_eq!(status["summaryBudget"]["calls"], 2);
        assert_eq!(status["summaryBudget"]["refused"], 1);
        assert_eq!(status["summaryBudget"]["lastRefusal"], refusal);
        assert_eq!(status["blockCount"], 0, "a refusal never lands a fold");
        // Ordinary dialogue is not gated: a manual compress still runs after the
        // automatic budget is gone.
        let (text, outcome, _) = store.tool(
            "budget-fallback",
            8_000,
            "compress",
            &json!({"ranges": [{"start": "m00002", "end": "m00003", "summary": P3_SUMMARY}]}),
        );
        assert_eq!(outcome, Some("applied"), "{text}");
        assert_eq!(status_block_count(&store, "budget-fallback"), 1);
        assert_eq!(
            status["summaryBudget"]["calls"], 2,
            "manual work spends no budget"
        );
    }

    #[test]
    fn reserved_calls_and_real_usage_are_charged_to_the_budget() {
        let store = Store::new();
        prepared_at("budget-usage", &store, 8_000, 4_000);
        store
            .reserve_summary_call("budget-usage", false, 4, 2)
            .expect("first automatic call");
        store
            .reserve_summary_call("budget-usage", true, 4, 2)
            .expect("first speculative call");
        let batch = crate::summarizer::SummaryBatch {
            summaries: vec!["- one fact".to_string()],
            calls: 2,
            input_tokens: 120,
            output_tokens: 30,
            estimated_calls: 1,
        };
        // The reservation covers the first dispatch; the second fallback target
        // is charged here together with provider (or estimated) usage.
        store.record_summary_usage("budget-usage", true, &batch);
        let status = store.status_json("budget-usage").expect("status");
        assert_eq!(status["summaryBudget"]["calls"], 3);
        assert_eq!(status["summaryBudget"]["speculationCalls"], 2);
        assert_eq!(status["summaryBudget"]["inputTokens"], 120);
        assert_eq!(status["summaryBudget"]["outputTokens"], 30);
        assert_eq!(status["summaryBudget"]["estimatedCalls"], 1);
        let refusal = store
            .reserve_summary_call("budget-usage", true, 4, 2)
            .expect_err("the speculation sub-budget is spent");
        assert!(refusal.contains("speculation budget"), "{refusal}");
        assert_eq!(status_block_count(&store, "budget-usage"), 0);
    }
}

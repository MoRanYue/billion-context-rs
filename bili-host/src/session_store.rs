//! Per-conversation kernel sessions behind the proxy.
//!
//! The host learns the conversation id from the
//! `x-bili-plugin-conversation` header every turn and re-projects the body
//! through the current kernel state. Session state lives here, keyed by that
//! id. It is snapshotted atomically to `rust-sessions/<sha256(key)>.json`
//! (kernel `State` plus host counters) so a restart resumes folds and refs.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use billion_context::{
    BlockId, Boundary, CompressRequest, Config, Content, FnEstimator, Message, RenderStrategy,
    RequestedRange, Session, State, Timestamp, TurnInput,
};
use billion_context_tool as tool;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

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
    /// The provider-reported prompt total of the latest response. When present
    /// the panel shows the billed number instead of the host's estimate.
    pub context_usage: Option<u32>,
    pub context_source: &'static str,
    pub protocol: String,
    pub upstream: String,
    fold_events: Vec<FoldEvent>,
}

#[derive(Clone)]
pub struct Store {
    inner: Arc<Mutex<HashMap<String, Entry>>>,
    index: Arc<Mutex<HashMap<String, SnapshotMeta>>>,
    dir: Option<Arc<PathBuf>>,
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
    context_usage: Option<u32>,
    protocol: String,
    upstream: String,
    #[serde(default)]
    fold_events: Vec<FoldEvent>,
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
        }
    }

    /// A store that indexes snapshots under `dir` at startup but materializes
    /// a session only when it is actually used.
    pub fn with_dir(dir: PathBuf) -> Self {
        let store = Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
            index: Arc::new(Mutex::new(HashMap::new())),
            dir: Some(Arc::new(dir)),
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
        }
    }

    fn entry_from_snapshot(snapshot: Persisted) -> Entry {
        let context_limit = snapshot.context_limit.max(1);
        let mut session = Session::new(Config::new(context_limit))
            .with_estimator(Self::estimator_for())
            .with_view_mode(billion_context::ViewMode::Send)
            .with_reconcile(true);
        *session.state_mut() = snapshot.state;
        Entry {
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
            context_usage: snapshot.context_usage,
            context_source: if snapshot.context_usage.is_some() {
                "usage"
            } else {
                "estimate"
            },
            protocol: snapshot.protocol,
            upstream: snapshot.upstream,
            fold_events: snapshot.fold_events,
        }
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
            context_usage: entry.context_usage,
            protocol: entry.protocol.clone(),
            upstream: entry.upstream.clone(),
            fold_events: entry.fold_events.clone(),
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
            context_usage: None,
            context_source: "estimate",
            protocol: String::new(),
            upstream: String::new(),
            fold_events: Vec::new(),
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
        }
        let result = f(entry);
        self.persist(key, entry);
        result
    }

    /// Project one request through the kernel. Returns the rebuilt message
    /// array, the ranges a pending nudge points at, and the nudge text (only
    /// when it differs from the one already injected).
    pub fn prepare(
        &self,
        key: &str,
        context_limit: u32,
        messages: &[billion_context::Message],
        token_count: u32,
        model: Option<String>,
    ) -> Prepared {
        let now = now_ms();
        self.with_entry(key, context_limit, |entry| {
            let turn = entry.session.process_turn(TurnInput {
                messages,
                token_count,
                render: RenderStrategy::TextOnly,
                now,
            });
            entry.tokens = token_count;
            entry.last_seen = now;
            entry.last_messages = messages.to_vec();
            entry.requests += 1;
            entry.model = model.clone().or_else(|| entry.model.clone());
            entry.ranges = turn.ranges.clone();
            let armed_nudge = turn
                .nudge
                .as_ref()
                .filter(|nudge| nudge.should_inject)
                .map(billion_context::nudge_instruction);
            let nudge =
                armed_nudge.filter(|text| entry.last_nudge.as_deref() != Some(text.as_str()));
            if let Some(text) = nudge.clone() {
                entry.last_nudge = Some(text.clone());
            }
            Prepared {
                messages: turn.messages,
                ranges: turn.ranges,
                nudge,
                terminal: turn.terminal.is_some(),
                warnings: turn.warnings,
            }
        })
    }

    /// Forget the pending nudge once the model acted on it (or the fold landed),
    /// so the next projection re-injects only when the kernel wants it again.
    pub fn clear_nudge(&self, key: &str) {
        let mut guard = self.inner.lock().expect("session store poisoned");
        if let Some(entry) = guard.get_mut(key) {
            entry.last_nudge = None;
            self.persist(key, entry);
        }
    }

    pub fn tool(
        &self,
        key: &str,
        context_limit: u32,
        name: &str,
        args: &Value,
    ) -> (String, Option<&'static str>, Option<u32>) {
        let now = now_ms();
        self.with_entry(key, context_limit, |entry| match name {
            "acp_compress" | "compress" => match tool::parse_compress(&normalize_compress_args(args)) {
                Ok(ranges) => {
                    let min = entry.session.config().compress.min_compress_range;
                    if let Some(chars) = message_range_chars(&entry.last_messages, &ranges) {
                        if chars > 0 && chars < min {
                            return (
                                format!(
                                    "[Compression FAILED: Total compressible content too small ({chars} chars across {} range(s), min {min}). Combine more messages into your range(s) to meet the threshold.]",
                                    ranges.len()
                                ),
                                Some("refused"),
                                Some(0),
                            );
                        }
                    }
                    let applied = entry.session.apply_compression(CompressRequest {
                        ranges,
                        now,
                        call_id: args
                            .get("tool_call_id")
                            .and_then(Value::as_str)
                            .map(str::to_string),
                    });
                    let at_requests = entry.requests;
                    for block_id in &applied.blocks {
                        if let Some(block) = entry.session.block(*block_id) {
                            entry.fold_events.push(FoldEvent {
                                at_requests,
                                s: block.compressed_tokens as u64,
                                sigma: estimate_summary_tokens(&block.summary),
                                block: block.id.to_string(),
                            });
                        }
                    }
                    if entry.fold_events.len() > 512 {
                        let excess = entry.fold_events.len() - 512;
                        entry.fold_events.drain(0..excess);
                    }
                    let text = tool::render_compress(&applied.blocks, &applied.errors);
                    let outcome = if applied.errors.is_empty() {
                        "applied"
                    } else if applied.blocks.is_empty() {
                        "refused"
                    } else {
                        "partial"
                    };
                    entry.last_nudge = None;
                    (
                        text,
                        Some(outcome),
                        Some(applied.blocks.len() as u32),
                    )
                }
                Err(err) => (format!("[compress refused: {err}]"), Some("refused"), Some(0)),
            },
            "acp_retrieve" | "retrieve" => match tool::parse_retrieve(args) {
                Ok(refs) => (tool::render_retrieval(&entry.session.retrieve(&refs)), Some("success"), None),
                Err(err) => (format!("[retrieve refused: {err}]"), Some("refused"), None),
            },
            "acp_decompress" | "decompress" => match tool::parse_decompress(args) {
                Ok(id) => {
                    let outcome = entry.session.decompress(&[id]);
                    let restored = outcome
                        .blocks
                        .first()
                        .and_then(|id| entry.session.block(*id).cloned());
                    match restored {
                        Some(block) => (tool::render_decompression(&block), Some("success"), None),
                        None => (format!("[decompress refused: block {id} not found]"), Some("refused"), None),
                    }
                }
                Err(err) => (format!("[decompress refused: {err}]"), Some("refused"), None),
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
            entry.context_usage = Some(usage.context);
            entry.context_source = "usage";
            entry.cache_samples += 1;
        }
        if entry.model.is_none() {
            entry.model = request_model.map(str::to_string);
        }
        entry.last_seen = now_ms();
        self.persist(key, entry);
    }

    /// Snapshot for `/__bili/stats`: one row per live conversation, newest
    /// first, in the shape the Node dashboard renders.
    fn row_from_entry(key: &str, entry: &Entry) -> Value {
        let status = entry.session.status();
        let context_tokens = entry.context_usage.unwrap_or(entry.tokens);
        let cache_hit_pct = if let Some(context) = entry.context_usage {
            if context > 0 {
                ((entry.cached_tokens as f64 * 100.0) / context as f64).round() as u64
            } else {
                0
            }
        } else {
            0
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
        })
    }

    fn row_from_meta(meta: &SnapshotMeta) -> Value {
        let context_tokens = meta.context_usage.unwrap_or(meta.tokens);
        let cache_hit_pct = if let Some(context) = meta.context_usage {
            if context > 0 {
                ((meta.cached_tokens as f64 * 100.0) / context as f64).round() as u64
            } else {
                0
            }
        } else {
            0
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
    pub fn status_json(&self, key: &str) -> Option<Value> {
        self.ensure_loaded(key);
        let guard = self.inner.lock().expect("session store poisoned");
        guard.get(key).map(|entry| status_body(key, entry))
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

pub struct Prepared {
    pub messages: Vec<billion_context::Message>,
    pub ranges: Vec<billion_context::Range>,
    pub nudge: Option<String>,
    pub terminal: bool,
    pub warnings: Vec<String>,
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

fn content_chars(content: &Content) -> usize {
    match content {
        Content::Text(text) | Content::Reasoning(text) => text.len(),
        Content::ToolCall { arguments, .. } => arguments.len(),
        Content::ToolResult { content, .. } => content.len(),
    }
}

fn normalize_compress_args(args: &Value) -> Value {
    if args.get("ranges").is_some_and(Value::is_array) {
        return args.clone();
    }
    let content = match args.get("content") {
        Some(Value::String(text)) => serde_json::from_str::<Value>(text).ok(),
        Some(value) => Some(value.clone()),
        None => None,
    };
    let Some(content) = content else {
        return args.clone();
    };
    let entries: Vec<Value> = match content {
        Value::Array(items) => items,
        Value::Object(_) => vec![content],
        _ => return args.clone(),
    };
    let mut ranges = Vec::new();
    for entry in entries {
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
            continue;
        };
        let mut range = serde_json::json!({"start": start, "end": end, "summary": summary});
        if let Some(topic) = entry.get("topic").and_then(Value::as_str) {
            range["topic"] = json!(topic);
        }
        ranges.push(range);
    }
    if ranges.is_empty() {
        return args.clone();
    }
    serde_json::json!({"ranges": ranges})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_turn_does_not_inject_a_decision_marked_idle() {
        // The pipeline always returns a decision object so a host can show the
        // pending numbers; only `should_inject` authorizes putting text in front
        // of the model. A tiny first turn must stay byte-clean.
        let body = json!({
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi there"}]}]
        });
        let converted = crate::anthropic::to_core(&body);
        let store = Store::new();
        let prepared = store.prepare("tiny", 200_000, &converted.messages, 80, None);
        assert!(
            prepared.nudge.is_none(),
            "kernel decision should_inject=false must not reach the wire"
        );
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
                },
                FoldEvent {
                    at_requests: 8,
                    s: 60,
                    sigma: 10,
                    block: "b2".into(),
                },
            ],
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
        assert_eq!(econ.gross, 80 * 5 + 50 * 0);
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
        let converted = crate::anthropic::to_core(&body);
        let store = Store::new();
        let first = store.prepare("ledger-live", 200_000, &converted.messages, 12_000, None);
        assert!(first.ranges.len() > 0 || first.messages.len() == converted.messages.len());
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
    fn snapshots_are_indexed_without_loading_state() {
        let body = json!({"messages": [{"role": "user", "content": [{"type": "text", "text": "lazy restore"}]}]});
        let converted = crate::anthropic::to_core(&body);
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
    fn session_survives_a_restart_through_snapshots() {
        let body = json!({
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hello persistence"}]}]
        });
        let converted = crate::anthropic::to_core(&body);
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
}

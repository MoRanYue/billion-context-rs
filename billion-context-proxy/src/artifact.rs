//! Content-addressed storage for oversized tool results.
//!
//! Artifacts are the host's authoritative copy of bytes the model only sees as a
//! preview. The store deliberately has no knowledge of kernel refs: callers pass
//! the stable host message id and resolve refs through the session state.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const ARTIFACT_SCHEMA_VERSION: u32 = 1;
pub const PREVIEW_POLICY_VERSION: u32 = 2;
pub const MAX_ARTIFACT_BYTES: u64 = 32 * 1024 * 1024;
pub const MAX_READ_BYTES: u64 = 64 * 1024;
/// Whole-store ceiling. A single tool result may be small but a long session
/// can still exhaust the disk; this is the backstop before any write.
pub const DEFAULT_STORE_QUOTA_BYTES: u64 = 512 * 1024 * 1024;
/// Default retention for unreferenced session artifacts. Cleanup is explicit;
/// nothing is removed just because a file is old.
pub const DEFAULT_RETENTION_MS: u64 = 14 * 24 * 60 * 60 * 1000;
const HASH_PREFIX_CHARS: usize = 24;
const MESSAGE_INDEX_FILE: &str = "message-index.json";
const AUDIT_FILE: &str = "audit.jsonl";
const MESSAGE_INDEX_VERSION: u32 = 1;

fn configured_store_quota() -> u64 {
    std::env::var("BILI_ARTIFACT_STORE_BYTES")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_STORE_QUOTA_BYTES)
}

fn artifact_id_for(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    let sha256 = hex::encode(hasher.finalize());
    format!("a_{}", &sha256[..HASH_PREFIX_CHARS])
}

/// Content-addressed ids are always `a_` plus a lowercase hex prefix. Validate
/// before joining a path: a tool-supplied id must never become a path
/// separator, even for a transient file-existence oracle.
pub fn is_valid_artifact_id(id: &str) -> bool {
    id.len() == HASH_PREFIX_CHARS + 2
        && id.starts_with("a_")
        && id[2..]
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn push_once(ids: &mut Vec<String>, id: &str) {
    if !ids.iter().any(|existing| existing == id) {
        ids.push(id.to_string());
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArtifactMetadata {
    pub schema_version: u32,
    pub artifact_id: String,
    pub sha256: String,
    pub byte_len: u64,
    pub media_type: String,
    pub source_message_id: String,
    /// Other stable host ids that resolved to this same content-addressed
    /// artifact. The primary id stays in `source_message_id`; aliases make an
    /// index rebuild recover every duplicate-content mapping instead of only the
    /// first one.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub source_message_ids: Vec<String>,
    pub source_ref: Option<String>,
    pub created_at: u64,
    pub retention_class: String,
    pub tool_name: Option<String>,
    pub call_id: Option<String>,
    pub preview_policy_version: u32,
}

impl ArtifactMetadata {
    fn message_ids(&self) -> Vec<&str> {
        let mut ids = Vec::with_capacity(self.source_message_ids.len() + 1);
        if !self.source_message_id.is_empty() {
            ids.push(self.source_message_id.as_str());
        }
        for id in &self.source_message_ids {
            if !id.is_empty() && !ids.contains(&id.as_str()) {
                ids.push(id.as_str());
            }
        }
        ids
    }

    fn has_message_id(&self, message_id: &str) -> bool {
        self.message_ids().contains(&message_id)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ArtifactRead {
    pub artifact_id: String,
    pub source_message_id: String,
    pub source_ref: Option<String>,
    pub sha256: String,
    pub byte_len: u64,
    pub offset: u64,
    pub limit: u64,
    pub next_offset: u64,
    pub text: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct ArtifactHit {
    pub offset: u64,
    pub line: u64,
    pub text: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct ArtifactSearch {
    pub artifact_id: String,
    pub source_message_id: String,
    pub source_ref: Option<String>,
    pub query: String,
    pub hits: Vec<ArtifactHit>,
    pub returned_bytes: u64,
    pub truncated: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct ArtifactStats {
    pub artifacts: u64,
    pub bytes: u64,
    pub quota_bytes: u64,
    pub remaining_bytes: u64,
    pub dir: String,
}

/// A conservative set of live references collected from every persisted
/// session before cleanup. `complete=false` means at least one reference
/// source could not be read; cleanup then refuses to delete anything.
#[derive(Clone, Debug, Default)]
pub struct ArtifactProtection {
    pub complete: bool,
    pub artifact_ids: BTreeSet<String>,
    pub message_ids: BTreeSet<String>,
}

impl ArtifactProtection {
    pub fn complete() -> Self {
        Self {
            complete: true,
            ..Self::default()
        }
    }

    pub fn protect_artifact(&mut self, artifact_id: impl Into<String>) {
        self.artifact_ids.insert(artifact_id.into());
    }

    pub fn protect_message(&mut self, message_id: impl Into<String>) {
        self.message_ids.insert(message_id.into());
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ArtifactCleanupReport {
    pub scanned_artifacts: u64,
    pub protected_artifacts: u64,
    pub expired_removed: u64,
    pub orphan_bodies_removed: u64,
    pub orphan_metadata_removed: u64,
    pub kept_uncertain: u64,
    pub failed: u64,
    pub bytes_reclaimed: u64,
    pub index_entries: usize,
    /// Concrete ids behind `kept_uncertain`/`protected_artifacts`, so an operator
    /// can inspect a stale ref instead of only seeing a counter.
    pub uncertain_ids: Vec<String>,
    pub protected_ids: Vec<String>,
}

/// Audit record deliberately excludes artifact bytes and message text. It
/// records identifiers, outcome and a bounded reason only.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArtifactAuditEvent {
    pub ts: u64,
    pub operation: String,
    pub artifact_id: String,
    pub session_id: String,
    pub outcome: String,
    pub reason: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct PersistedMessageIndex {
    schema_version: u32,
    #[serde(default)]
    entries: BTreeMap<String, Vec<String>>,
}

#[derive(Default)]
struct IndexCache {
    loaded: bool,
    entries: BTreeMap<String, Vec<String>>,
}

/// Test-only fault points at the real write/commit stages. This does not
/// abstract the production filesystem and is absent from release builds.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FaultStage {
    BodyWrite,
    BodyRename,
    MetaWrite,
    MetaRename,
    IndexWrite,
    IndexRename,
}

#[derive(Clone)]
pub struct ArtifactStore {
    dir: Arc<PathBuf>,
    index: Arc<Mutex<IndexCache>>,
    /// Serializes the quota check with the body write. The index mutex is not
    /// enough because the body file is written before the index is touched.
    write: Arc<Mutex<()>>,
    max_store_bytes: u64,
    /// One-shot ENOSPC injection, compiled only for tests.
    #[cfg(test)]
    fault: Arc<Mutex<Option<FaultStage>>>,
}

impl ArtifactStore {
    pub fn new(session_dir: &Path) -> Self {
        Self::with_quota(session_dir, configured_store_quota())
    }

    pub fn with_quota(session_dir: &Path, max_store_bytes: u64) -> Self {
        Self {
            dir: Arc::new(session_dir.join("artifacts")),
            index: Arc::new(Mutex::new(IndexCache::default())),
            write: Arc::new(Mutex::new(())),
            max_store_bytes,
            #[cfg(test)]
            fault: Arc::new(Mutex::new(None)),
        }
    }

    #[allow(dead_code, reason = "quota introspection for host status endpoints")]
    pub fn quota_bytes(&self) -> u64 {
        self.max_store_bytes
    }

    /// Arm a one-shot write-stage failure. The next call at that exact stage
    /// returns a synthetic `ENOSPC`; later calls proceed normally.
    #[cfg(test)]
    pub(crate) fn inject_fault(&self, stage: FaultStage) {
        *self.fault.lock().expect("artifact fault poisoned") = Some(stage);
    }

    #[cfg(test)]
    fn maybe_fail(&self, stage: FaultStage) -> io::Result<()> {
        let mut slot = self.fault.lock().expect("artifact fault poisoned");
        if slot.as_ref() == Some(&stage) {
            *slot = None;
            return Err(io::Error::new(
                io::ErrorKind::StorageFull,
                format!("injected ENOSPC at {stage:?}"),
            ));
        }
        Ok(())
    }

    fn body_path(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{id}.txt"))
    }

    fn meta_path(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{id}.json"))
    }

    fn ensure_dir(&self) -> io::Result<()> {
        fs::create_dir_all(self.dir.as_ref())
    }

    /// Count every body, including orphans, so a crashed write cannot make the
    /// store look emptier than the disk really is.
    fn total_body_bytes(&self) -> u64 {
        let mut bytes = 0u64;
        if let Ok(entries) = fs::read_dir(self.dir.as_ref()) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("txt") && path.is_file() {
                    bytes = bytes.saturating_add(fs::metadata(&path).map(|m| m.len()).unwrap_or(0));
                }
            }
        }
        bytes
    }

    fn index_path(&self) -> PathBuf {
        self.dir.join(MESSAGE_INDEX_FILE)
    }

    fn audit_path(&self) -> PathBuf {
        self.dir.join(AUDIT_FILE)
    }

    /// Best-effort audit append. Artifact correctness must not depend on the
    /// audit file being writable; a permission failure is reported on stderr
    /// while the caller still gets its normal result.
    pub fn audit(
        &self,
        operation: &str,
        artifact_id: &str,
        session_id: &str,
        outcome: &str,
        reason: &str,
    ) {
        let event = ArtifactAuditEvent {
            ts: now_ms(),
            operation: operation.to_string(),
            artifact_id: artifact_id.to_string(),
            session_id: session_id.to_string(),
            outcome: outcome.to_string(),
            reason: reason.chars().take(240).collect(),
        };
        let line = match serde_json::to_string(&event) {
            Ok(line) => line,
            Err(err) => {
                eprintln!("[artifact] audit serialize failed: {err}");
                return;
            }
        };
        if let Err(err) = self.ensure_dir().and_then(|_| {
            use std::io::Write;
            let path = self.audit_path();
            let mut file = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .or_else(|_| fs::OpenOptions::new().append(true).open(&path))?;
            writeln!(file, "{line}")
        }) {
            eprintln!("[artifact] audit append failed: {err}");
        }
    }

    fn index_temp_path(&self) -> PathBuf {
        self.dir.join(format!(
            ".{MESSAGE_INDEX_FILE}.{}.{}.tmp",
            now_ms(),
            std::process::id()
        ))
    }

    fn record_mapping_locked(
        entries: &mut BTreeMap<String, Vec<String>>,
        message_id: &str,
        artifact_id: &str,
    ) {
        for key in [
            message_id.to_string(),
            message_id
                .split('#')
                .next()
                .unwrap_or(message_id)
                .to_string(),
        ] {
            let ids = entries.entry(key).or_default();
            if !ids.iter().any(|id| id == artifact_id) {
                ids.push(artifact_id.to_string());
            }
        }
    }

    fn save_index_locked(&self, cache: &IndexCache) -> Result<(), String> {
        let persisted = PersistedMessageIndex {
            schema_version: MESSAGE_INDEX_VERSION,
            entries: cache.entries.clone(),
        };
        let bytes = serde_json::to_vec(&persisted).map_err(|e| format!("artifact index: {e}"))?;
        let temp = self.index_temp_path();
        #[cfg(test)]
        if let Err(err) = self.maybe_fail(FaultStage::IndexWrite) {
            let _ = fs::remove_file(&temp);
            return Err(format!("artifact index write: {err}"));
        }
        if let Err(err) = fs::write(&temp, bytes) {
            let _ = fs::remove_file(&temp);
            return Err(format!("artifact index write: {err}"));
        }
        #[cfg(test)]
        if let Err(err) = self.maybe_fail(FaultStage::IndexRename) {
            let _ = fs::remove_file(&temp);
            return Err(format!("artifact index rename: {err}"));
        }
        if let Err(err) = fs::rename(&temp, self.index_path()) {
            let _ = fs::remove_file(&temp);
            return Err(format!("artifact index rename: {err}"));
        }
        Ok(())
    }

    fn rebuild_index_locked(&self, cache: &mut IndexCache) -> Result<u64, String> {
        let mut entries: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let read = fs::read_dir(self.dir.as_ref());
        let mut rebuilt = 0u64;
        if let Ok(entries_iter) = read {
            for entry in entries_iter.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) != Some("json") {
                    continue;
                }
                if path.file_name().and_then(|n| n.to_str()) == Some(MESSAGE_INDEX_FILE) {
                    continue;
                }
                if path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.ends_with(".json.tmp"))
                {
                    continue;
                }
                let Ok(text) = fs::read_to_string(&path) else {
                    continue;
                };
                let Ok(meta) = serde_json::from_str::<ArtifactMetadata>(&text) else {
                    continue;
                };
                for message_id in meta.message_ids() {
                    Self::record_mapping_locked(&mut entries, message_id, &meta.artifact_id);
                }
                rebuilt += 1;
            }
        }
        cache.entries = entries;
        cache.loaded = true;
        self.save_index_locked(cache)?;
        Ok(rebuilt)
    }

    fn ensure_index_locked(&self, cache: &mut IndexCache) -> Result<(), String> {
        if cache.loaded {
            return Ok(());
        }
        match fs::read_to_string(self.index_path()) {
            Ok(text) => match serde_json::from_str::<PersistedMessageIndex>(&text) {
                Ok(index) if index.schema_version == MESSAGE_INDEX_VERSION => {
                    cache.entries = index.entries;
                    cache.loaded = true;
                    Ok(())
                }
                Ok(_) | Err(_) => self.rebuild_index_locked(cache).map(|_| ()),
            },
            Err(err) if err.kind() == io::ErrorKind::NotFound => {
                self.rebuild_index_locked(cache).map(|_| ())
            }
            Err(_) => self.rebuild_index_locked(cache).map(|_| ()),
        }
    }

    #[allow(
        dead_code,
        reason = "index maintenance hook retained for recovery tools"
    )]
    fn ensure_index(&self) -> Result<(), String> {
        let mut cache = self.index.lock().expect("artifact index poisoned");
        self.ensure_index_locked(&mut cache)
    }

    /// Rebuild the message-id index from metadata files. This is the recovery
    /// path when the index is missing, corrupt, or known to lag artifacts.
    #[allow(
        dead_code,
        reason = "operator-facing index rebuild retained for recovery"
    )]
    pub fn rebuild_index(&self) -> Result<u64, String> {
        let mut cache = self.index.lock().expect("artifact index poisoned");
        self.rebuild_index_locked(&mut cache)
    }

    #[allow(
        dead_code,
        reason = "index size introspection for host status endpoints"
    )]
    pub fn index_entry_count(&self) -> Result<usize, String> {
        let mut cache = self.index.lock().expect("artifact index poisoned");
        self.ensure_index_locked(&mut cache)?;
        Ok(cache.entries.len())
    }

    /// Resolve a stable host message id without a bounded directory scan.
    /// `Err` is returned for ambiguous or stale mappings; callers must not turn
    /// that into an arbitrary artifact selection.
    pub fn resolve_by_message_id(
        &self,
        message_id: &str,
    ) -> Result<Option<ArtifactMetadata>, String> {
        let mut had_mapping = false;
        for attempt in 0..2 {
            let ids = {
                let mut cache = self.index.lock().expect("artifact index poisoned");
                self.ensure_index_locked(&mut cache)?;
                let mut ids = cache.entries.get(message_id).cloned().unwrap_or_default();
                if ids.is_empty() {
                    // A stale index is recoverable; rebuild once and retry.
                    self.rebuild_index_locked(&mut cache)?;
                    ids = cache.entries.get(message_id).cloned().unwrap_or_default();
                }
                had_mapping |= !ids.is_empty();
                ids
            };
            let mut distinct = ids.clone();
            distinct.sort();
            distinct.dedup();
            if distinct.len() > 1 {
                return Err(format!(
                    "artifact index ambiguity for message {message_id}: {} artifacts",
                    distinct.len()
                ));
            }
            let Some(artifact_id) = distinct.first() else {
                if had_mapping {
                    return Err(format!(
                        "artifact index is stale for message {message_id}: mapped artifact is unavailable"
                    ));
                }
                return Ok(None);
            };
            match self.info(artifact_id) {
                Ok(meta) if meta.artifact_id == *artifact_id => return Ok(Some(meta)),
                _ if attempt == 0 => {
                    let mut cache = self.index.lock().expect("artifact index poisoned");
                    self.rebuild_index_locked(&mut cache)?;
                }
                Err(err) => return Err(err),
                _ => return Err(format!("artifact {artifact_id} unavailable")),
            }
        }
        Err(format!("artifact index unstable for message {message_id}"))
    }

    /// Persist one tool output. The id is derived from the full content hash,
    /// so identical outputs share one blob. Returns the existing metadata when
    /// the content is already present.
    #[allow(
        dead_code,
        reason = "artifact write entry point retained for host integrations"
    )]
    pub fn put(
        &self,
        content: &str,
        source_message_id: &str,
        source_ref: Option<&str>,
        tool_name: Option<&str>,
        call_id: Option<&str>,
    ) -> Result<ArtifactMetadata, String> {
        self.put_with_session(
            content,
            source_message_id,
            source_ref,
            tool_name,
            call_id,
            "",
        )
    }

    pub fn put_with_session(
        &self,
        content: &str,
        source_message_id: &str,
        source_ref: Option<&str>,
        tool_name: Option<&str>,
        call_id: Option<&str>,
        session_id: &str,
    ) -> Result<ArtifactMetadata, String> {
        let result = self.put_inner(content, source_message_id, source_ref, tool_name, call_id);
        let (operation, outcome, reason) = match &result {
            Ok(meta) => ("write", "success", format!("bytes={}", meta.byte_len)),
            Err(err) if err.contains("quota exceeded") => ("quota_reject", "fail", err.clone()),
            Err(err) => ("write", "fail", err.clone()),
        };
        let artifact_id = match &result {
            Ok(meta) => meta.artifact_id.clone(),
            Err(_) => artifact_id_for(content),
        };
        self.audit(operation, &artifact_id, session_id, outcome, &reason);
        result
    }

    fn put_inner(
        &self,
        content: &str,
        source_message_id: &str,
        source_ref: Option<&str>,
        tool_name: Option<&str>,
        call_id: Option<&str>,
    ) -> Result<ArtifactMetadata, String> {
        if content.len() as u64 > MAX_ARTIFACT_BYTES {
            return Err(format!(
                "artifact too large: {} bytes (limit {})",
                content.len(),
                MAX_ARTIFACT_BYTES
            ));
        }
        let _write_guard = self.write.lock().expect("artifact write lock poisoned");
        self.ensure_dir()
            .map_err(|e| format!("artifact dir: {e}"))?;
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        let sha256 = hex::encode(hasher.finalize());
        let artifact_id = format!("a_{}", &sha256[..HASH_PREFIX_CHARS]);
        let body = self.body_path(&artifact_id);
        let meta_path = self.meta_path(&artifact_id);
        if let Ok(mut meta) = self.info(&artifact_id)
            && meta.sha256 == sha256
            && body.is_file()
        {
            if !meta.has_message_id(source_message_id) {
                meta.source_message_ids.push(source_message_id.to_string());
                let bytes = serde_json::to_vec(&meta).map_err(|e| format!("artifact meta: {e}"))?;
                let temp = self.dir.join(format!(
                    ".{}.{}.{}.json.tmp",
                    artifact_id,
                    now_ms(),
                    std::process::id()
                ));
                #[cfg(test)]
                self.maybe_fail(FaultStage::MetaWrite)
                    .map_err(|e| format!("artifact meta update: {e}"))?;
                fs::write(&temp, bytes).map_err(|e| format!("artifact meta update: {e}"))?;
                #[cfg(test)]
                if let Err(err) = self.maybe_fail(FaultStage::MetaRename) {
                    let _ = fs::remove_file(&temp);
                    return Err(format!("artifact meta update rename: {err}"));
                }
                fs::rename(&temp, &meta_path)
                    .map_err(|e| format!("artifact meta update rename: {e}"))?;
            }
            let mut cache = self.index.lock().expect("artifact index poisoned");
            self.ensure_index_locked(&mut cache)?;
            Self::record_mapping_locked(&mut cache.entries, source_message_id, &meta.artifact_id);
            self.save_index_locked(&cache)?;
            return Ok(meta);
        }
        let new_bytes = if body.is_file() {
            0
        } else {
            content.len() as u64
        };
        let current_bytes = self.total_body_bytes();
        if current_bytes.saturating_add(new_bytes) > self.max_store_bytes {
            return Err(format!(
                "artifact store quota exceeded: current {} + new {} > limit {}",
                current_bytes, new_bytes, self.max_store_bytes
            ));
        }
        let temp = self.dir.join(format!(
            ".{}.{}.{}.tmp",
            artifact_id,
            now_ms(),
            std::process::id()
        ));
        #[cfg(test)]
        if let Err(err) = self.maybe_fail(FaultStage::BodyWrite) {
            let _ = fs::remove_file(&temp);
            return Err(format!("artifact write: {err}"));
        }
        if let Err(err) = fs::write(&temp, content.as_bytes()) {
            let _ = fs::remove_file(&temp);
            return Err(format!("artifact write: {err}"));
        }
        let written = if body.is_file() {
            let _ = fs::remove_file(&temp);
            false
        } else {
            #[cfg(test)]
            if let Err(err) = self.maybe_fail(FaultStage::BodyRename) {
                let _ = fs::remove_file(&temp);
                return Err(format!("artifact rename: {err}"));
            }
            match fs::rename(&temp, &body) {
                Ok(()) => true,
                Err(err) => {
                    let _ = fs::remove_file(&temp);
                    if body.is_file() {
                        false
                    } else {
                        return Err(format!("artifact rename: {err}"));
                    }
                }
            }
        };
        if !written && !body.is_file() {
            return Err("artifact body missing after write".into());
        }
        let meta = ArtifactMetadata {
            schema_version: ARTIFACT_SCHEMA_VERSION,
            artifact_id: artifact_id.clone(),
            sha256,
            byte_len: content.len() as u64,
            media_type: "text/plain".into(),
            source_message_id: source_message_id.to_string(),
            source_message_ids: Vec::new(),
            source_ref: source_ref.map(str::to_string),
            created_at: now_ms(),
            retention_class: "session".into(),
            tool_name: tool_name.map(str::to_string),
            call_id: call_id.map(str::to_string),
            preview_policy_version: PREVIEW_POLICY_VERSION,
        };
        let meta_bytes = serde_json::to_vec(&meta).map_err(|e| format!("artifact meta: {e}"))?;
        let meta_temp = self.dir.join(format!(
            ".{}.{}.{}.json.tmp",
            artifact_id,
            now_ms(),
            std::process::id()
        ));
        #[cfg(test)]
        if let Err(err) = self.maybe_fail(FaultStage::MetaWrite) {
            let _ = fs::remove_file(&meta_temp);
            if written {
                let _ = fs::remove_file(&body);
            }
            return Err(format!("artifact meta write: {err}"));
        }
        if let Err(err) = fs::write(&meta_temp, meta_bytes) {
            let _ = fs::remove_file(&meta_temp);
            if written {
                let _ = fs::remove_file(&body);
            }
            return Err(format!("artifact meta write: {err}"));
        }
        #[cfg(test)]
        if let Err(err) = self.maybe_fail(FaultStage::MetaRename) {
            let _ = fs::remove_file(&meta_temp);
            if written {
                let _ = fs::remove_file(&body);
            }
            return Err(format!("artifact meta rename: {err}"));
        }
        if let Err(err) = fs::rename(&meta_temp, &meta_path) {
            let _ = fs::remove_file(&meta_temp);
            if written {
                let _ = fs::remove_file(&body);
            }
            return Err(format!("artifact meta rename: {err}"));
        }
        // Body then metadata then index. A crash before the index write is
        // recoverable by rebuild_index from metadata; lookup never trusts a
        // stale mapping without verifying the metadata file.
        let mut cache = self.index.lock().expect("artifact index poisoned");
        self.ensure_index_locked(&mut cache)?;
        Self::record_mapping_locked(&mut cache.entries, source_message_id, &meta.artifact_id);
        self.save_index_locked(&cache)?;
        Ok(meta)
    }

    pub fn info(&self, artifact_id: &str) -> Result<ArtifactMetadata, String> {
        if !is_valid_artifact_id(artifact_id) {
            return Err("invalid artifact id".to_string());
        }
        let text = fs::read_to_string(self.meta_path(artifact_id))
            .map_err(|e| format!("artifact {artifact_id} metadata: {e}"))?;
        serde_json::from_str(&text).map_err(|e| format!("artifact {artifact_id} metadata: {e}"))
    }

    /// Attach the stable `mNNNNN` ref after the kernel has allocated it. The
    /// content-addressed `a_...` id is never changed; only metadata gains the
    /// host reference. Atomic temp+rename keeps readers from seeing half a
    /// metadata update.
    pub fn set_source_ref(
        &self,
        artifact_id: &str,
        source_ref: &str,
    ) -> Result<ArtifactMetadata, String> {
        let mut meta = self.info(artifact_id)?;
        if meta.source_ref.as_deref() == Some(source_ref) {
            return Ok(meta);
        }
        meta.source_ref = Some(source_ref.to_string());
        let bytes = serde_json::to_vec(&meta).map_err(|e| format!("artifact meta: {e}"))?;
        let temp = self.dir.join(format!(
            ".{}.{}.{}.json.tmp",
            artifact_id,
            now_ms(),
            std::process::id()
        ));
        fs::write(&temp, bytes).map_err(|e| format!("artifact meta update: {e}"))?;
        fs::rename(&temp, self.meta_path(artifact_id))
            .map_err(|e| format!("artifact meta update rename: {e}"))?;
        Ok(meta)
    }

    /// Scan by stable message id and attach a ref. Returns the number of
    /// metadata files updated; duplicate content has one canonical artifact.
    pub fn set_source_ref_for_message(
        &self,
        message_id: &str,
        source_ref: &str,
    ) -> Result<u64, String> {
        let Some(meta) = self.resolve_by_message_id(message_id)? else {
            return Ok(0);
        };
        if meta.source_ref.as_deref() == Some(source_ref) {
            return Ok(0);
        }
        self.set_source_ref(&meta.artifact_id, source_ref)?;
        Ok(1)
    }

    /// Resolve a stable host message id to the artifact that stored its bytes.
    /// Compatibility wrapper: ambiguity/staleness is refused by
    /// `resolve_by_message_id`, never silently replaced with another artifact.
    pub fn find_by_message_id(&self, message_id: &str) -> Option<ArtifactMetadata> {
        self.resolve_by_message_id(message_id).ok().flatten()
    }

    pub fn read(&self, artifact_id: &str, offset: u64, limit: u64) -> Result<ArtifactRead, String> {
        let meta = self.info(artifact_id)?;
        let bytes = fs::read(self.body_path(artifact_id))
            .map_err(|e| format!("artifact {artifact_id} body: {e}"))?;
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let actual = hex::encode(hasher.finalize());
        if actual != meta.sha256 {
            return Err(format!(
                "artifact {artifact_id} hash mismatch: {} != {}",
                actual, meta.sha256
            ));
        }
        let byte_len = bytes.len() as u64;
        let full = String::from_utf8(bytes)
            .map_err(|e| format!("artifact {artifact_id} is not UTF-8: {e}"))?;
        let limit = limit.clamp(1, MAX_READ_BYTES);
        let mut start = (offset.min(byte_len) as usize).min(full.len());
        while start < full.len() && !full.is_char_boundary(start) {
            start += 1;
        }
        let wanted = start.saturating_add(limit as usize).min(full.len());
        let mut end = wanted;
        while end > start && !full.is_char_boundary(end) {
            end -= 1;
        }
        let text = full[start..end].to_string();
        Ok(ArtifactRead {
            artifact_id: artifact_id.to_string(),
            source_message_id: meta.source_message_id.clone(),
            source_ref: meta.source_ref.clone(),
            sha256: meta.sha256,
            byte_len,
            offset: start as u64,
            limit,
            next_offset: end as u64,
            text,
        })
    }

    /// Bounded literal search over one artifact. Returns at most `limit` hits
    /// and stops after `max_bytes` of returned text.
    pub fn search(
        &self,
        artifact_id: &str,
        query: &str,
        limit: usize,
        max_bytes: u64,
    ) -> Result<ArtifactSearch, String> {
        let meta = self.info(artifact_id)?;
        let bytes = fs::read(self.body_path(artifact_id))
            .map_err(|e| format!("artifact {artifact_id} body: {e}"))?;
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let actual = hex::encode(hasher.finalize());
        if actual != meta.sha256 {
            return Err(format!(
                "artifact {artifact_id} hash mismatch: {} != {}",
                actual, meta.sha256
            ));
        }
        let text = String::from_utf8(bytes)
            .map_err(|e| format!("artifact {artifact_id} is not UTF-8: {e}"))?;
        let query = query.trim();
        if query.is_empty() {
            return Err("artifact.search query is empty".into());
        }
        let needle = query.to_lowercase();
        let max_hits = limit.clamp(1, 20);
        let max_bytes = max_bytes.clamp(256, 32 * 1024);
        let mut hits = Vec::new();
        let mut returned_bytes = 0u64;
        let mut truncated = false;
        let mut byte_offset = 0u64;
        for (line_no, line) in (1u64..).zip(text.lines()) {
            let lower = line.to_lowercase();
            if lower.contains(&needle) {
                let remaining = max_bytes.saturating_sub(returned_bytes);
                if remaining == 0 || hits.len() >= max_hits {
                    truncated = true;
                    break;
                }
                let snippet: String = line.chars().take(remaining.min(512) as usize).collect();
                returned_bytes += snippet.len() as u64;
                hits.push(ArtifactHit {
                    offset: byte_offset,
                    line: line_no,
                    text: snippet,
                });
            }
            byte_offset += line.len() as u64 + 1;
        }
        Ok(ArtifactSearch {
            artifact_id: artifact_id.to_string(),
            source_message_id: meta.source_message_id.clone(),
            source_ref: meta.source_ref.clone(),
            query: query.to_string(),
            hits,
            returned_bytes,
            truncated,
        })
    }

    pub fn stats(&self) -> ArtifactStats {
        let mut artifacts = 0u64;
        let mut bytes = 0u64;
        if let Ok(entries) = fs::read_dir(self.dir.as_ref()) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("txt") && path.is_file() {
                    artifacts += 1;
                    bytes = bytes.saturating_add(fs::metadata(&path).map(|m| m.len()).unwrap_or(0));
                }
            }
        }
        ArtifactStats {
            artifacts,
            bytes,
            quota_bytes: self.max_store_bytes,
            remaining_bytes: self.max_store_bytes.saturating_sub(bytes),
            dir: self.dir.to_string_lossy().to_string(),
        }
    }

    /// Explicit, conservative cleanup. It only removes:
    /// - expired metadata that no live reference protects, then its body;
    /// - body files with no metadata and no live/index reference;
    /// - metadata files whose body is already missing and that no reference
    ///   protects.
    ///
    /// A body is never deleted from file age alone. The caller must collect
    /// references from every session; `complete=false` aborts the whole run.
    pub fn cleanup(
        &self,
        now: u64,
        retention_ms: u64,
        protection: &ArtifactProtection,
    ) -> Result<ArtifactCleanupReport, String> {
        if !protection.complete {
            self.audit(
                "cleanup_refuse",
                "",
                "__cleanup__",
                "fail",
                "reference scan incomplete",
            );
            return Err(
                "artifact cleanup refused: reference scan was incomplete; nothing deleted".into(),
            );
        }
        let _write_guard = self.write.lock().expect("artifact write lock poisoned");
        self.ensure_dir()
            .map_err(|e| format!("artifact dir: {e}"))?;
        let mut cache = self.index.lock().expect("artifact index poisoned");
        self.ensure_index_locked(&mut cache)?;
        let index_artifact_ids: BTreeSet<String> = cache
            .entries
            .values()
            .flat_map(|ids| ids.iter().cloned())
            .collect();

        let mut report = ArtifactCleanupReport::default();
        let mut metadata: BTreeMap<String, Result<ArtifactMetadata, String>> = BTreeMap::new();
        if let Ok(entries) = fs::read_dir(self.dir.as_ref()) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) != Some("json") {
                    continue;
                }
                let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                    continue;
                };
                if name == MESSAGE_INDEX_FILE
                    || name.ends_with(".meta.json")
                    || name.ends_with(".json.tmp")
                {
                    continue;
                }
                let artifact_id = name.trim_end_matches(".json").to_string();
                let parsed = fs::read_to_string(&path)
                    .map_err(|e| format!("artifact {artifact_id} metadata: {e}"))
                    .and_then(|text| {
                        serde_json::from_str::<ArtifactMetadata>(&text)
                            .map_err(|e| format!("artifact {artifact_id} metadata: {e}"))
                    });
                report.scanned_artifacts = report.scanned_artifacts.saturating_add(1);
                metadata.insert(artifact_id, parsed);
            }
        }

        let mut bodies: BTreeSet<String> = BTreeSet::new();
        if let Ok(entries) = fs::read_dir(self.dir.as_ref()) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("txt")
                    && let Some(id) = path.file_stem().and_then(|n| n.to_str())
                {
                    bodies.insert(id.to_string());
                }
            }
        }

        let mut changed = false;
        for (artifact_id, parsed) in &metadata {
            let body = self.body_path(artifact_id);
            let protected = protection.artifact_ids.contains(artifact_id);
            let parsed = match parsed {
                Ok(meta) => meta,
                Err(_) => {
                    // Corrupt metadata means the reference state is unknown.
                    // Keep both sides; a later explicit repair can inspect it.
                    report.kept_uncertain = report.kept_uncertain.saturating_add(1);
                    push_once(&mut report.uncertain_ids, artifact_id);
                    self.audit(
                        "cleanup_uncertain",
                        artifact_id,
                        "__cleanup__",
                        "kept",
                        "corrupt metadata; reference state unknown",
                    );
                    continue;
                }
            };
            let protected = protected
                || protection
                    .message_ids
                    .iter()
                    .any(|message_id| parsed.has_message_id(message_id));
            if !body.exists() {
                if protected {
                    report.protected_artifacts = report.protected_artifacts.saturating_add(1);
                    push_once(&mut report.protected_ids, artifact_id);
                    continue;
                }
                if fs::remove_file(self.meta_path(artifact_id)).is_ok() {
                    report.orphan_metadata_removed =
                        report.orphan_metadata_removed.saturating_add(1);
                    changed = true;
                    self.audit(
                        "delete",
                        artifact_id,
                        "__cleanup__",
                        "success",
                        "orphan metadata removed",
                    );
                } else {
                    report.failed = report.failed.saturating_add(1);
                    self.audit(
                        "delete",
                        artifact_id,
                        "__cleanup__",
                        "fail",
                        "orphan metadata removal failed",
                    );
                }
                continue;
            }
            if protected {
                report.protected_artifacts = report.protected_artifacts.saturating_add(1);
                push_once(&mut report.protected_ids, artifact_id);
                continue;
            }
            if parsed.retention_class == "immutable" {
                report.protected_artifacts = report.protected_artifacts.saturating_add(1);
                push_once(&mut report.protected_ids, artifact_id);
                continue;
            }
            let expired =
                parsed.created_at <= now && now.saturating_sub(parsed.created_at) >= retention_ms;
            if !expired {
                continue;
            }
            // Metadata first: a crash between the two removes leaves an
            // orphan body (safe) rather than metadata pointing at absent bytes.
            if fs::remove_file(self.meta_path(artifact_id)).is_err() {
                report.failed = report.failed.saturating_add(1);
                self.audit(
                    "delete",
                    artifact_id,
                    "__cleanup__",
                    "fail",
                    "expired metadata removal failed; body kept",
                );
                continue;
            }
            match fs::remove_file(&body) {
                Ok(()) => {
                    report.expired_removed = report.expired_removed.saturating_add(1);
                    report.bytes_reclaimed = report.bytes_reclaimed.saturating_add(parsed.byte_len);
                    changed = true;
                    self.audit(
                        "delete",
                        artifact_id,
                        "__cleanup__",
                        "success",
                        "expired artifact removed",
                    );
                }
                Err(_) => {
                    report.failed = report.failed.saturating_add(1);
                    report.kept_uncertain = report.kept_uncertain.saturating_add(1);
                    push_once(&mut report.uncertain_ids, artifact_id);
                    self.audit(
                        "delete",
                        artifact_id,
                        "__cleanup__",
                        "fail",
                        "expired body removal failed; metadata already removed",
                    );
                }
            }
        }

        let metadata_ids: BTreeSet<String> = metadata.keys().cloned().collect();
        for artifact_id in bodies {
            if metadata_ids.contains(&artifact_id) || protection.artifact_ids.contains(&artifact_id)
            {
                if protection.artifact_ids.contains(&artifact_id)
                    && !metadata_ids.contains(&artifact_id)
                {
                    report.protected_artifacts = report.protected_artifacts.saturating_add(1);
                    push_once(&mut report.protected_ids, &artifact_id);
                }
                continue;
            }
            if index_artifact_ids.contains(&artifact_id) {
                // A stale index may still point at this body; deleting it would
                // turn a recoverable index inconsistency into data loss.
                report.kept_uncertain = report.kept_uncertain.saturating_add(1);
                push_once(&mut report.uncertain_ids, &artifact_id);
                continue;
            }
            match fs::remove_file(self.body_path(&artifact_id)) {
                Ok(()) => {
                    report.orphan_bodies_removed = report.orphan_bodies_removed.saturating_add(1);
                    changed = true;
                    self.audit(
                        "delete",
                        &artifact_id,
                        "__cleanup__",
                        "success",
                        "orphan body removed",
                    );
                }
                Err(_) => {
                    report.failed = report.failed.saturating_add(1);
                    self.audit(
                        "delete",
                        &artifact_id,
                        "__cleanup__",
                        "fail",
                        "orphan body removal failed",
                    );
                }
            }
        }

        if changed {
            self.rebuild_index_locked(&mut cache)?;
        }
        report.index_entries = cache.entries.len();
        self.audit(
            "cleanup_summary",
            "",
            "__cleanup__",
            if report.failed == 0 { "success" } else { "partial" },
            &format!(
                "scanned={} protected={} expired={} orphan_body={} orphan_meta={} uncertain={} failed={}",
                report.scanned_artifacts,
                report.protected_artifacts,
                report.expired_removed,
                report.orphan_bodies_removed,
                report.orphan_metadata_removed,
                report.kept_uncertain,
                report.failed
            ),
        );
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_store() -> (ArtifactStore, PathBuf) {
        temp_store_with_quota(DEFAULT_STORE_QUOTA_BYTES)
    }

    fn temp_store_with_quota(quota: u64) -> (ArtifactStore, PathBuf) {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let seq = NEXT.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "bili-artifact-test-{}-{}-{}",
            now_ms(),
            std::process::id(),
            seq
        ));
        (ArtifactStore::with_quota(&dir, quota), dir)
    }

    fn artifact_id_for(content: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        let sha256 = hex::encode(hasher.finalize());
        format!("a_{}", &sha256[..HASH_PREFIX_CHARS])
    }

    #[test]
    fn metadata_lookup_rejects_ids_that_are_not_content_addressed() {
        let (store, root) = temp_store();
        let outside = root
            .parent()
            .unwrap_or(&root)
            .join("bili-artifact-oracle.json");
        fs::write(&outside, r#"{"artifact_id":"a_outside"}"#).expect("write outside");
        for id in [
            "../../../bili-artifact-oracle",
            "a_/../../x",
            "a_",
            "a_zzzz",
        ] {
            assert!(store.info(id).is_err(), "{id} must be refused");
        }
        let missing = artifact_id_for("never stored");
        assert!(store.info(&missing).is_err());
        let _ = fs::remove_file(outside);
    }

    #[test]
    fn audit_records_operations_without_body_text() {
        let (store, root) = temp_store_with_quota(4);
        store
            .put_with_session("AAAA", "m-audit-a", None, Some("Bash"), None, "s1")
            .expect("first");
        let rejected = store.put_with_session("BBBB", "m-audit-b", None, Some("Bash"), None, "s1");
        assert!(rejected.is_err());
        let report = store
            .cleanup(now_ms() + 1_000, 0, &ArtifactProtection::complete())
            .expect("cleanup");
        assert_eq!(report.expired_removed, 1);
        let audit = root.join("artifacts").join(AUDIT_FILE);
        let text = fs::read_to_string(audit).expect("audit");
        let events: Vec<ArtifactAuditEvent> = text
            .lines()
            .map(|line| serde_json::from_str(line).expect("audit line"))
            .collect();
        assert!(
            events
                .iter()
                .any(|e| e.operation == "write" && e.outcome == "success")
        );
        assert!(events.iter().any(|e| e.operation == "quota_reject"));
        assert!(
            events
                .iter()
                .any(|e| e.operation == "delete" && e.outcome == "success")
        );
        assert!(events.iter().any(|e| e.session_id == "s1"));
        assert!(!text.contains("AAAA"));
        assert!(!text.contains("BBBB"));
        let _ = fs::remove_dir_all(root);
    }

    fn assert_audit_failure(root: &Path, stage: FaultStage, secret: &str) {
        let audit = fs::read_to_string(root.join("artifacts").join(AUDIT_FILE))
            .expect("failure must be audited");
        assert!(audit.contains("injected ENOSPC"), "{audit}");
        assert!(audit.contains(&format!("{stage:?}")), "{audit}");
        assert!(!audit.contains(secret), "audit must not contain body bytes");
    }

    fn assert_clean_twice(store: &ArtifactStore, root: &Path) {
        let report = store
            .cleanup(
                now_ms() + 10_000,
                u64::MAX / 2,
                &ArtifactProtection::complete(),
            )
            .expect("cleanup after failure");
        assert_eq!(report.expired_removed, 0);
        assert_eq!(report.orphan_bodies_removed, 0);
        assert_eq!(report.orphan_metadata_removed, 0);
        let second = store
            .cleanup(
                now_ms() + 20_000,
                u64::MAX / 2,
                &ArtifactProtection::complete(),
            )
            .expect("repeated cleanup");
        assert_eq!(second.expired_removed, 0);
        assert_eq!(second.orphan_bodies_removed, 0);
        assert_eq!(second.orphan_metadata_removed, 0);
        assert!(
            !root.join("artifacts").join(MESSAGE_INDEX_FILE).exists()
                || fs::read_to_string(root.join("artifacts").join(MESSAGE_INDEX_FILE))
                    .map(|text| text.contains("schema_version"))
                    .unwrap_or(false)
        );
    }

    #[test]
    fn enospc_body_write_leaves_no_artifact_and_preserves_existing() {
        let (store, root) = temp_store();
        let existing = store
            .put("existing-valid", "m-existing", None, Some("Bash"), None)
            .expect("seed existing");
        let secret = "BODY-SECRET ".repeat(1_000);
        let id = artifact_id_for(&secret);
        store.inject_fault(FaultStage::BodyWrite);
        let err = store
            .put_with_session(&secret, "m-new", None, Some("Bash"), None, "s-enospc")
            .expect_err("body ENOSPC");
        assert!(err.contains("injected ENOSPC at BodyWrite"), "{err}");
        assert!(store.info(&id).is_err(), "no metadata may validate");
        assert!(!store.body_path(&id).exists(), "no body may survive");
        assert!(store.read(&existing.artifact_id, 0, 64).is_ok());
        assert!(store.info(&existing.artifact_id).is_ok());
        assert_audit_failure(&root, FaultStage::BodyWrite, "BODY-SECRET");
        assert_clean_twice(&store, &root);
        drop(store);
        let reopened = ArtifactStore::new(&root);
        assert!(reopened.info(&id).is_err());
        assert!(reopened.read(&existing.artifact_id, 0, 64).is_ok());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn enospc_metadata_write_rolls_back_body_and_preserves_existing() {
        let (store, root) = temp_store();
        let existing = store
            .put("existing-meta", "m-existing-meta", None, Some("Bash"), None)
            .expect("seed existing");
        let secret = "META-SECRET ".repeat(1_000);
        let id = artifact_id_for(&secret);
        store.inject_fault(FaultStage::MetaWrite);
        let err = store
            .put_with_session(&secret, "m-new-meta", None, Some("Bash"), None, "s-enospc")
            .expect_err("metadata ENOSPC");
        assert!(err.contains("injected ENOSPC at MetaWrite"), "{err}");
        assert!(store.info(&id).is_err(), "no metadata may validate");
        assert!(
            !store.body_path(&id).exists(),
            "new body must be rolled back"
        );
        assert!(store.read(&existing.artifact_id, 0, 64).is_ok());
        assert_audit_failure(&root, FaultStage::MetaWrite, "META-SECRET");
        assert_clean_twice(&store, &root);
        drop(store);
        let reopened = ArtifactStore::new(&root);
        assert!(reopened.info(&id).is_err());
        assert!(reopened.read(&existing.artifact_id, 0, 64).is_ok());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn enospc_index_write_keeps_valid_artifact_and_rebuilds_on_restart() {
        let (store, root) = temp_store();
        let existing = store
            .put(
                "existing-index",
                "m-existing-index",
                None,
                Some("Bash"),
                None,
            )
            .expect("seed existing");
        let secret = "INDEX-SECRET ".repeat(1_000);
        let id = artifact_id_for(&secret);
        store.inject_fault(FaultStage::IndexWrite);
        let err = store
            .put_with_session(&secret, "m-new-index", None, Some("Bash"), None, "s-enospc")
            .expect_err("index ENOSPC");
        assert!(err.contains("injected ENOSPC at IndexWrite"), "{err}");
        // Body and metadata are valid; only the derived index failed.
        assert!(store.info(&id).is_ok());
        assert!(store.read(&id, 0, 64).is_ok());
        assert!(store.read(&existing.artifact_id, 0, 64).is_ok());
        assert_audit_failure(&root, FaultStage::IndexWrite, "INDEX-SECRET");
        drop(store);
        let reopened = ArtifactStore::new(&root);
        assert!(reopened.info(&id).is_ok());
        let rebuilt = reopened.rebuild_index().expect("rebuild after restart");
        assert!(
            rebuilt >= 2,
            "both valid artifacts must be indexed: {rebuilt}"
        );
        assert!(reopened.index_entry_count().expect("index count") >= 2);
        assert_clean_twice(&reopened, &root);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn enospc_at_commit_rename_stages_leaves_no_dangling_state() {
        for (stage, secret) in [
            (FaultStage::BodyRename, "BODY-RENAME-SECRET ".repeat(1_000)),
            (FaultStage::MetaRename, "META-RENAME-SECRET ".repeat(1_000)),
            (
                FaultStage::IndexRename,
                "INDEX-RENAME-SECRET ".repeat(1_000),
            ),
        ] {
            let (store, root) = temp_store();
            let existing = store
                .put(
                    "existing-rename",
                    "m-existing-rename",
                    None,
                    Some("Bash"),
                    None,
                )
                .expect("seed existing");
            let id = artifact_id_for(&secret);
            store.inject_fault(stage);
            let err = store
                .put_with_session(
                    &secret,
                    "m-new-rename",
                    None,
                    Some("Bash"),
                    None,
                    "s-enospc",
                )
                .expect_err("commit ENOSPC");
            assert!(
                err.contains(&format!("injected ENOSPC at {stage:?}")),
                "{err}"
            );
            let body_exists = store.body_path(&id).exists();
            let meta_exists = store.meta_path(&id).exists();
            match stage {
                FaultStage::BodyRename | FaultStage::MetaRename => {
                    assert!(!body_exists, "{stage:?} must not leave a body");
                    assert!(!meta_exists, "{stage:?} must not leave metadata");
                    assert!(store.info(&id).is_err());
                }
                FaultStage::IndexRename => {
                    assert!(body_exists, "index rename failure keeps durable body");
                    assert!(meta_exists, "index rename failure keeps durable metadata");
                    assert!(store.info(&id).is_ok());
                }
                _ => unreachable!(),
            }
            assert!(store.read(&existing.artifact_id, 0, 64).is_ok());
            assert_audit_failure(&root, stage, secret.split(' ').next().unwrap_or("SECRET"));
            assert_clean_twice(&store, &root);
            let _ = fs::remove_dir_all(root);
        }
    }

    #[test]
    fn put_is_content_addressed_and_read_verifies_hash() {
        let (store, root) = temp_store();
        let content = "head\n".to_string() + &"middle\n".repeat(1000) + "tail\n";
        let meta = store
            .put(&content, "m1", None, Some("Bash"), Some("c1"))
            .expect("put");
        assert_eq!(meta.byte_len, content.len() as u64);
        let read = store.read(&meta.artifact_id, 0, 32).expect("read");
        assert!(read.text.starts_with("head\n"));
        assert_eq!(read.next_offset, 32);
        assert!(store.info(&meta.artifact_id).is_ok());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn duplicate_put_reuses_the_same_artifact() {
        let (store, root) = temp_store();
        let first = store
            .put("same output", "m1", None, Some("Bash"), None)
            .expect("first");
        let second = store
            .put("same output", "m2", Some("m00001"), Some("Bash"), None)
            .expect("second");
        assert_eq!(first.artifact_id, second.artifact_id);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn message_index_resolves_beyond_old_scalar_scan_cap_and_survives_restart() {
        let (store, root) = temp_store();
        let mut last = None;
        for i in 0..1_100 {
            let content = format!("payload-{i}");
            let id = format!("msg-{i}");
            last = Some(
                store
                    .put(&content, &id, None, Some("Bash"), None)
                    .expect("put"),
            );
        }
        let last = last.expect("artifact");
        assert_eq!(
            store
                .resolve_by_message_id("msg-1099")
                .expect("index resolve")
                .expect("artifact")
                .artifact_id,
            last.artifact_id
        );
        drop(store);
        let reopened = ArtifactStore::new(&root);
        assert_eq!(
            reopened
                .resolve_by_message_id("msg-1099")
                .expect("restored index")
                .expect("artifact")
                .artifact_id,
            last.artifact_id
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn duplicate_content_maps_multiple_message_ids_to_one_artifact() {
        let (store, root) = temp_store();
        let first = store
            .put("same output", "m1", None, Some("Bash"), None)
            .expect("first");
        let second = store
            .put("same output", "m2", Some("m00002"), Some("Bash"), None)
            .expect("second");
        assert_eq!(first.artifact_id, second.artifact_id);
        assert_eq!(
            store
                .resolve_by_message_id("m2")
                .expect("m2")
                .expect("artifact")
                .artifact_id,
            first.artifact_id
        );
        assert_eq!(
            store
                .set_source_ref_for_message("m1", "m00001")
                .expect("ref"),
            1
        );
        assert_eq!(
            store
                .info(&first.artifact_id)
                .expect("info")
                .source_ref
                .as_deref(),
            Some("m00001")
        );
        // A rebuild from metadata must recover the duplicate-content alias, not
        // just the first message id that wrote the artifact.
        drop(store);
        let reopened = ArtifactStore::new(&root);
        let index = root.join("artifacts").join(MESSAGE_INDEX_FILE);
        fs::remove_file(&index).expect("remove index for rebuild");
        assert_eq!(
            reopened
                .resolve_by_message_id("m2")
                .expect("rebuild alias")
                .expect("artifact")
                .artifact_id,
            first.artifact_id
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn duplicate_message_id_is_refused_at_lookup() {
        let (store, root) = temp_store();
        let a = store
            .put("first body", "same-message", None, Some("Bash"), None)
            .expect("a");
        let b = store
            .put("second body", "same-message", None, Some("Bash"), None)
            .expect("b");
        assert_ne!(a.artifact_id, b.artifact_id);
        let err = store
            .resolve_by_message_id("same-message")
            .expect_err("ambiguous id must be refused");
        assert!(err.contains("ambiguity"), "{err}");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn single_artifact_limit_is_enforced_before_the_write() {
        let (store, root) = temp_store();
        let oversized = "x".repeat((MAX_ARTIFACT_BYTES + 1) as usize);
        let error = store
            .put(&oversized, "m-too-large", None, Some("Bash"), None)
            .expect_err("must reject oversized artifact");
        assert!(error.contains("artifact too large"), "{error}");
        assert_eq!(store.stats().artifacts, 0);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn session_stores_are_isolated_even_for_identical_content() {
        let (first, first_root) = temp_store();
        let (second, second_root) = temp_store();
        let content = "shared bytes";
        let meta_a = first
            .put(content, "m1", None, Some("Bash"), None)
            .expect("first put");
        let meta_b = second
            .put(content, "m1", None, Some("Bash"), None)
            .expect("second put");
        assert_eq!(meta_a.artifact_id, meta_b.artifact_id);
        assert_ne!(first_root, second_root);
        let _ = fs::remove_dir_all(first_root.clone());
        assert!(
            second
                .read(&meta_b.artifact_id, 0, 64)
                .expect("second read")
                .text
                .contains(content),
            "deleting one session must not affect another"
        );
        let _ = fs::remove_dir_all(second_root);
    }

    #[test]
    fn metadata_survives_index_loss_after_an_interrupted_write() {
        let (store, root) = temp_store();
        let meta = store
            .put(
                "interrupted body",
                "m-interrupted",
                Some("1"),
                Some("Bash"),
                None,
            )
            .expect("put");
        // Simulate the crash window after body+metadata rename but before the
        // index rename. The metadata is the durable recovery source.
        drop(store);
        let index = root.join("artifacts").join(MESSAGE_INDEX_FILE);
        fs::remove_file(&index).expect("remove index");
        let reopened = ArtifactStore::new(&root);
        let recovered = reopened
            .resolve_by_message_id("m-interrupted")
            .expect("rebuild")
            .expect("artifact");
        assert_eq!(recovered.artifact_id, meta.artifact_id);
        assert_eq!(recovered.source_ref.as_deref(), Some("1"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn missing_or_corrupt_index_rebuilds_from_metadata() {
        let (store, root) = temp_store();
        let meta = store
            .put("indexed body", "m-index", None, Some("Bash"), None)
            .expect("put");
        let index = root.join("artifacts").join(MESSAGE_INDEX_FILE);
        fs::remove_file(&index).expect("remove index");
        assert_eq!(
            store
                .resolve_by_message_id("m-index")
                .expect("rebuild missing")
                .expect("artifact")
                .artifact_id,
            meta.artifact_id
        );
        fs::write(&index, b"{not-json").expect("corrupt index");
        assert_eq!(
            store
                .resolve_by_message_id("m-index")
                .expect("rebuild corrupt")
                .expect("artifact")
                .artifact_id,
            meta.artifact_id
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn stale_index_is_rebuilt_and_missing_artifact_is_not_substituted() {
        let (store, root) = temp_store();
        let meta = store
            .put("stale body", "m-stale", None, Some("Bash"), None)
            .expect("put");
        // A second process/store sees the old cache; add an artifact from the
        // first store and then ask the stale view for it.
        let stale = ArtifactStore::new(&root);
        let _ = stale.index_entry_count().expect("load index");
        let other = store
            .put("later body", "m-later", None, Some("Bash"), None)
            .expect("later put");
        assert_eq!(
            stale
                .resolve_by_message_id("m-later")
                .expect("stale rebuild")
                .expect("artifact")
                .artifact_id,
            other.artifact_id
        );
        // Delete metadata after the index points at it: fail, do not return
        // some other artifact.
        fs::remove_file(
            root.join("artifacts")
                .join(format!("{}.json", meta.artifact_id)),
        )
        .expect("remove meta");
        assert!(stale.resolve_by_message_id("m-stale").is_err());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn concurrent_put_and_lookup_keeps_index_consistent() {
        let (store, root) = temp_store();
        let store = Arc::new(store);
        let mut workers = Vec::new();
        for worker in 0..8 {
            let store = store.clone();
            workers.push(std::thread::spawn(move || {
                for i in 0..24 {
                    let id = format!("thread-{worker}-{i}");
                    let content = format!("body-{worker}-{i}");
                    let meta = store
                        .put(&content, &id, None, Some("Bash"), None)
                        .expect("concurrent put");
                    assert_eq!(
                        store
                            .resolve_by_message_id(&id)
                            .expect("concurrent resolve")
                            .expect("artifact")
                            .artifact_id,
                        meta.artifact_id
                    );
                    assert_eq!(
                        store.read(&meta.artifact_id, 0, 64).expect("read").text,
                        content
                    );
                }
            }));
        }
        for worker in workers {
            worker.join().expect("worker");
        }
        assert!(store.index_entry_count().expect("index") >= 8 * 24);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn search_is_bounded_and_reports_offsets() {
        let (store, root) = temp_store();
        let content = "alpha\nbeta\nalpha again\n";
        let meta = store
            .put(content, "m1", None, Some("Bash"), None)
            .expect("put");
        let search = store
            .search(&meta.artifact_id, "alpha", 2, 1024)
            .expect("search");
        assert_eq!(search.hits.len(), 2);
        assert_eq!(search.hits[0].line, 1);
        assert_eq!(search.hits[1].line, 3);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn total_store_quota_is_enforced_at_the_boundary_and_allows_duplicate_content() {
        let (store, root) = temp_store_with_quota(10);
        let first = store
            .put("1234567890", "m1", None, Some("Bash"), None)
            .expect("exactly at quota");
        assert_eq!(store.stats().bytes, 10);
        assert_eq!(store.stats().remaining_bytes, 0);
        let duplicate = store
            .put("1234567890", "m2", None, Some("Bash"), None)
            .expect("duplicate content adds no bytes");
        assert_eq!(first.artifact_id, duplicate.artifact_id);
        let err = store
            .put("x", "m3", None, Some("Bash"), None)
            .expect_err("one byte over quota");
        assert!(err.contains("quota exceeded"), "{err}");
        assert_eq!(store.stats().bytes, 10);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn concurrent_writes_cannot_bypass_the_total_quota() {
        let (store, root) = temp_store_with_quota(64);
        let store = Arc::new(store);
        let mut workers = Vec::new();
        for i in 0..8 {
            let store = Arc::clone(&store);
            workers.push(std::thread::spawn(move || {
                let content = format!("unique-{i}-{}", "x".repeat(8));
                store.put(&content, &format!("m-{i}"), None, Some("Bash"), None)
            }));
        }
        let mut ok = 0;
        let mut rejected = 0;
        for worker in workers {
            match worker.join().expect("worker") {
                Ok(_) => ok += 1,
                Err(err) => {
                    assert!(err.contains("quota exceeded"), "{err}");
                    rejected += 1;
                }
            }
        }
        assert_eq!(ok + rejected, 8);
        assert!(ok >= 2, "some writes should fit");
        assert!(store.stats().bytes <= 64, "quota must hold under races");
        assert_eq!(store.stats().bytes, ok as u64 * 17);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn metadata_rename_failure_removes_the_new_body_and_leaves_no_stub() {
        let (store, root) = temp_store();
        let dir = root.join("artifacts");
        fs::create_dir_all(&dir).expect("artifact dir");
        let content = "body that must not survive a metadata failure";
        let id = artifact_id_for(content);
        // Force the metadata rename target to be a directory. The body has
        // already been renamed into place, so this exercises the cleanup path.
        fs::create_dir_all(dir.join(format!("{id}.json"))).expect("block metadata path");
        let err = store
            .put(content, "m-meta-fail", None, Some("Bash"), None)
            .expect_err("metadata rename must fail");
        assert!(err.contains("meta"), "{err}");
        assert!(!dir.join(format!("{id}.txt")).exists(), "no orphan body");
        assert!(!store.body_path(&id).exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn body_path_collision_with_a_directory_leaves_no_metadata_or_stub() {
        let (store, root) = temp_store();
        let dir = root.join("artifacts");
        fs::create_dir_all(&dir).expect("artifact dir");
        let content = "body path collision";
        let id = artifact_id_for(content);
        fs::create_dir_all(dir.join(format!("{id}.txt"))).expect("block body path");
        let err = store
            .put(content, "m-collision", None, Some("Bash"), None)
            .expect_err("body rename must fail");
        assert!(err.contains("artifact rename"), "{err}");
        assert!(!store.meta_path(&id).exists());
        assert_eq!(store.stats().artifacts, 0);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn cleanup_refuses_an_incomplete_reference_scan_without_deleting_anything() {
        let (store, root) = temp_store();
        let meta = store
            .put("keep me", "m1", None, Some("Bash"), None)
            .expect("put");
        let err = store
            .cleanup(u64::MAX / 2, 0, &ArtifactProtection::default())
            .expect_err("incomplete scan must abort");
        assert!(err.contains("incomplete"), "{err}");
        assert!(store.info(&meta.artifact_id).is_ok());
        let audit = fs::read_to_string(root.join("artifacts").join(AUDIT_FILE))
            .expect("cleanup refusal must be audited");
        assert!(audit.contains("cleanup_refuse"), "{audit}");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn cleanup_protects_live_messages_and_removes_only_unreferenced_expired_artifacts() {
        let (store, root) = temp_store();
        let protected = store
            .put("protected payload", "m-live", None, Some("Bash"), None)
            .expect("protected");
        let expired = store
            .put("expired payload", "m-dead", None, Some("Bash"), None)
            .expect("expired");
        let mut protection = ArtifactProtection::complete();
        protection.protect_message("m-live");
        let now = store.info(&expired.artifact_id).expect("info").created_at + 10_000;
        let report = store.cleanup(now, 1, &protection).expect("cleanup");
        assert!(report.protected_artifacts >= 1);
        assert_eq!(report.expired_removed, 1);
        assert!(report.bytes_reclaimed >= 1);
        assert!(store.info(&protected.artifact_id).is_ok());
        assert!(store.read(&protected.artifact_id, 0, 64).is_ok());
        assert!(!store.body_path(&expired.artifact_id).exists());
        assert!(store.info(&expired.artifact_id).is_err());
        let second = store
            .cleanup(now + 10_000, 1, &protection)
            .expect("idempotent cleanup");
        assert_eq!(second.expired_removed, 0);
        assert_eq!(second.orphan_bodies_removed, 0);
        assert_eq!(second.orphan_metadata_removed, 0);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn cleanup_keeps_corrupt_metadata_and_stale_index_refs_as_uncertain() {
        let (store, root) = temp_store();
        let meta = store
            .put("uncertain payload", "m-uncertain", None, Some("Bash"), None)
            .expect("put");
        let meta_path = store.meta_path(&meta.artifact_id);
        fs::write(&meta_path, b"{ not json").expect("corrupt metadata");
        let report = store
            .cleanup(u64::MAX / 2, 0, &ArtifactProtection::complete())
            .expect("cleanup");
        assert!(report.kept_uncertain >= 1);
        assert!(report.failed == 0);
        assert!(store.body_path(&meta.artifact_id).exists());
        assert!(meta_path.exists());
        let audit = fs::read_to_string(root.join("artifacts").join(AUDIT_FILE))
            .expect("uncertain cleanup must be audited");
        assert!(audit.contains("cleanup_uncertain"), "{audit}");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn cleanup_reclaims_orphan_bodies_and_orphan_metadata() {
        let (store, root) = temp_store();
        let dir = root.join("artifacts");
        fs::create_dir_all(&dir).expect("dir");
        let orphan_body = dir.join("a_000000000000000000000001.txt");
        fs::write(&orphan_body, "orphan").expect("orphan body");
        let meta = store
            .put("metadata without body", "m1", None, Some("Bash"), None)
            .expect("put");
        fs::remove_file(store.body_path(&meta.artifact_id)).expect("remove body");
        let report = store
            .cleanup(u64::MAX / 2, 0, &ArtifactProtection::complete())
            .expect("cleanup");
        assert_eq!(report.orphan_bodies_removed, 1);
        assert_eq!(report.orphan_metadata_removed, 1);
        assert!(!orphan_body.exists());
        assert!(!store.meta_path(&meta.artifact_id).exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn interrupted_cleanup_recovers_and_repeated_cleanup_is_idempotent() {
        let (store, root) = temp_store();
        let meta = store
            .put("interrupted cleanup", "m1", None, Some("Bash"), None)
            .expect("put");
        // Crash window: metadata was removed first but the body removal did not
        // happen. The stale index still points at the body, so the first pass
        // must keep it as uncertain. After an index rebuild the body is proven
        // orphaned and a second pass may reclaim it.
        fs::remove_file(store.meta_path(&meta.artifact_id)).expect("remove metadata");
        drop(store);
        let reopened = ArtifactStore::new(&root);
        let first = reopened
            .cleanup(u64::MAX / 2, 0, &ArtifactProtection::complete())
            .expect("first cleanup keeps stale-index body");
        assert_eq!(first.orphan_bodies_removed, 0);
        assert!(first.kept_uncertain >= 1);
        assert!(reopened.body_path(&meta.artifact_id).exists());
        reopened.rebuild_index().expect("rebuild index");
        let second = reopened
            .cleanup(u64::MAX / 2, 0, &ArtifactProtection::complete())
            .expect("recovery cleanup");
        assert_eq!(second.orphan_bodies_removed, 1);
        assert!(!reopened.body_path(&meta.artifact_id).exists());
        let third = reopened
            .cleanup(u64::MAX / 2, 0, &ArtifactProtection::complete())
            .expect("idempotent");
        assert_eq!(third.orphan_bodies_removed, 0);
        assert_eq!(third.expired_removed, 0);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn failed_index_write_keeps_body_and_metadata_recoverable_after_restart() {
        let (store, root) = temp_store();
        let content = "recoverable body";
        let id = artifact_id_for(content);
        let dir = root.join("artifacts");
        fs::create_dir_all(dir.join(MESSAGE_INDEX_FILE)).expect("block index path");
        let err = store
            .put(content, "m-index-fail", None, Some("Bash"), None)
            .expect_err("index rename must fail");
        assert!(err.contains("index"), "{err}");
        assert!(store.body_path(&id).exists());
        assert!(store.meta_path(&id).exists());
        drop(store);
        fs::remove_dir_all(dir.join(MESSAGE_INDEX_FILE)).expect("unblock index");
        let reopened = ArtifactStore::new(&root);
        let recovered = reopened
            .resolve_by_message_id("m-index-fail")
            .expect("rebuild")
            .expect("artifact");
        assert_eq!(recovered.artifact_id, id);
        let _ = fs::remove_dir_all(root);
    }
}

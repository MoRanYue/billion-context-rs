//! Per-request upstream usage ledger (D-2).
//!
//! The host used to keep only conversation totals.  D1b needs one record per
//! actual upstream model request, so each request appends two JSONL lines under
//! `<sessions>/requests/<sha256(session)>.jsonl`:
//!
//! * `event=start` — identity, time, request hash, usage source, prune snapshot;
//! * `event=finish` — outcome, response hash, usage fields, prune result.
//!
//! A `start` line without a `finish` line means the outcome is unknown (client
//! disconnect, host exit, or a stream that never completed) and must never be
//! read as success.
//!
//! Concurrency: appends are serialised by a process-wide per-path mutex and
//! each line is handed to one `write_all` after the mutex is held, so two
//! concurrent requests of the same conversation cannot interleave line bytes.
//! Cross-process appends to one sessions directory are not coordinated; the
//! supported deployment is one host instance per directory.
//!
//! Hash scope: `request_hash` covers only the HTTP request body bytes the host
//! actually sent upstream (after prepare); `response_hash` covers only the
//! response body bytes actually received (a prefix when the stream broke).
//! Neither is a full wire-level hash — method, URL, headers and framing are not
//! covered, and bodies are not archived.  `*_hash_scope` repeats this in-band.
//!
//! Durability: `append_line` flushes but never fsyncs.  A host crash or power
//! loss can lose the trailing bytes of a line; the truncated line then fails to
//! parse and `audit_ledger` reports `invalid_line` instead of skipping it.
//!
//! Integrity gate for a future D1b runner (read-only; host behaviour unchanged):
//! `audit_ledger` must return `complete()` with no issues before the run counts;
//! otherwise the run is invalid and no further upstream calls may be issued.
//!
//! This module records facts only.  It never decides whether to prune, and it
//! never invents numbers the provider did not report.

use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use axum::http::HeaderMap;
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::prune::PruneReport;
use crate::session_store::now_ms;
use crate::usage::{UsageFields, context_tokens, missing_fields};

pub const SCHEMA: &str = "bili.usage.request.v1";

/// The host has no automatic upstream retry path, so `retry_count` is always 0
/// and no retry parent exists.  A future retry path must allocate a new
/// `request_id` and set `retry_of_request_id`; request headers are never read
/// for retry state.
pub const RETRY_COUNT_BASIS: &str = "host_has_no_retry_path";

/// `request_hash` covers the HTTP request body bytes sent upstream, nothing
/// else: no method, URL, headers or chunk framing.
pub const REQUEST_HASH_SCOPE: &str = "http_body_bytes_after_prepare";

/// `response_hash` covers the HTTP response body bytes received, nothing else.
/// When the stream breaks it is the hash of the received prefix.
pub const RESPONSE_HASH_SCOPE: &str = "received_response_body_bytes";

/// Fixed placeholder written to `upstream_host` when the URL cannot be parsed.
/// The original URL is never stored: it may contain userinfo or other secrets.
pub const UNPARSED_UPSTREAM_HOST: &str = "unparsed_upstream";

const CLIENT_REQUEST_ID_HEADERS: [&str; 2] = ["x-bili-client-request-id", "x-request-id"];
const MAX_IDENTIFIER_LEN: usize = 256;

/// Where the provider numbers came from, decided from the host's own runtime
/// (startup configuration or the actual upstream endpoint) — never from a
/// request header.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum UsageSource {
    Provider,
    Mock,
}

impl UsageSource {
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "provider" => Some(Self::Provider),
            "mock" => Some(Self::Mock),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestStatus {
    Success,
    HttpError,
    TransportError,
    Timeout,
    StreamError,
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

/// Normalized host portion of a URL: no scheme, userinfo, port, path, query or
/// fragment.  `None` means the string was not a URL we can parse; callers that
/// persist this value must use `UNPARSED_UPSTREAM_HOST` rather than the raw URL.
pub fn upstream_host(upstream: &str) -> Option<String> {
    let (scheme, rest) = upstream.split_once("://")?;
    if scheme.is_empty() {
        return None;
    }
    let authority = rest.split(['/', '?', '#']).next()?;
    if authority.is_empty() {
        return None;
    }
    // Userinfo is everything before the last '@' in the authority.  It must
    // never become part of the recorded host, even when it contains a colon.
    let host_port = authority
        .rsplit_once('@')
        .map(|(_, host_port)| host_port)
        .unwrap_or(authority);
    let host = match host_port.strip_prefix('[') {
        Some(after_bracket) => after_bracket.split_once(']')?.0,
        None => host_port.split(':').next()?,
    };
    let host = host.trim();
    if host.is_empty() || host.chars().any(|c| c.is_whitespace() || c.is_control()) {
        return None;
    }
    Some(host.to_ascii_lowercase())
}

pub fn is_loopback_host(host: &str) -> bool {
    matches!(host, "localhost" | "::1" | "127.0.0.1") || host.starts_with("127.")
}

/// Decide `usage_source` from host-owned facts.
///
/// An explicit startup configuration wins.  Otherwise a loopback upstream is
/// recorded as `mock`: a provider cannot live on 127.0.0.1, so the bytes came
/// from something the operator had to run locally.  The known limitation is a
/// real gateway reached through a loopback proxy, which this rule cannot tell
/// apart from a mock; the explicit configuration exists for that case.
pub fn usage_source(explicit: Option<UsageSource>, upstream: &str) -> (UsageSource, &'static str) {
    if let Some(source) = explicit {
        return (source, "explicit_host_config");
    }
    match upstream_host(upstream) {
        Some(host) if is_loopback_host(&host) => (UsageSource::Mock, "loopback_upstream"),
        Some(_) => (UsageSource::Provider, "non_loopback_upstream"),
        None => (
            UsageSource::Provider,
            "unparsed_upstream_host_treated_as_provider",
        ),
    }
}

/// Client-supplied correlation id, kept apart from the host-generated
/// `request_id`.  Invalid or oversized values are dropped instead of trusted.
pub fn client_request_id(headers: &HeaderMap) -> Option<String> {
    for name in CLIENT_REQUEST_ID_HEADERS {
        if let Some(value) = headers.get(name).and_then(|value| value.to_str().ok()) {
            let trimmed = value.trim();
            if !trimmed.is_empty()
                && trimmed.len() <= MAX_IDENTIFIER_LEN
                && !trimmed.chars().any(char::is_control)
            {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

/// Coarse transport class.  The raw error string is deliberately not stored:
/// reqwest messages embed the upstream URL, which may carry credentials in a
/// query string.
pub fn error_class(err: &reqwest::Error) -> &'static str {
    if err.is_timeout() {
        "timeout"
    } else if err.is_connect() {
        "connect"
    } else if err.is_body() {
        "body"
    } else if err.is_decode() {
        "decode"
    } else if err.is_request() {
        "request"
    } else if err.is_redirect() {
        "redirect"
    } else if err.is_status() {
        "status"
    } else {
        "other"
    }
}

/// Prune facts that this request itself produced.  The report comes straight
/// from the request's own `prepare_with_budget` call, so the counters cannot be
/// another request's totals.
///
/// `artifact_refs` evidence boundary: the ids are resolved from this request's
/// own `pruned_ids` plus the artifact metadata's `created_at` falling inside
/// this request's time window.  That association is *not* a strict causal proof
/// when another request of the same conversation runs concurrently and shares
/// the millisecond window: an artifact written by that request can be admitted.
/// `artifact_refs_attribution` therefore names the weaker basis — `prune_report_proven`
/// means "all ids matched inside the window", never "this request definitely
/// created these artifacts".  Unresolved or ambiguous cases stay `partial`/
/// `unavailable` with an empty or partial list.  These refs must not be used as
/// certain per-request cost attribution.
#[derive(Clone, Debug)]
pub struct PruneAttribution {
    pub triggered: bool,
    pub items: u64,
    pub artifact_writes: u64,
    pub artifact_bytes: u64,
    pub artifact_refs: Vec<String>,
    pub artifact_refs_attribution: &'static str,
}

impl PruneAttribution {
    pub fn new(report: &PruneReport, refs: Vec<String>, attribution: &'static str) -> Self {
        Self {
            triggered: report.triggered,
            items: report.pruned_items,
            artifact_writes: report.artifact_writes,
            artifact_bytes: report.artifact_bytes,
            artifact_refs: refs,
            artifact_refs_attribution: attribution,
        }
    }
}

/// Terminal facts for one upstream request.
#[derive(Clone, Debug)]
pub struct Finish {
    pub status: RequestStatus,
    pub http_status: Option<u16>,
    pub error_class: Option<&'static str>,
    pub response_hash: Option<String>,
    pub response_bytes: Option<u64>,
    pub response_complete: bool,
    pub response_body_inspected: bool,
    pub usage: UsageFields,
    pub usage_from_complete_stream: bool,
}

impl Finish {
    /// HTTP status outside 2xx: the body is streamed to the client untouched,
    /// so the host does not inspect or hash it.
    pub fn http_error(status: u16) -> Self {
        Self {
            status: RequestStatus::HttpError,
            http_status: Some(status),
            error_class: None,
            response_hash: None,
            response_bytes: None,
            response_complete: false,
            response_body_inspected: false,
            usage: UsageFields::default(),
            usage_from_complete_stream: false,
        }
    }

    /// The upstream exchange failed before any usable response arrived.
    pub fn transport(err: &reqwest::Error) -> Self {
        let class = error_class(err);
        Self {
            status: if err.is_timeout() {
                RequestStatus::Timeout
            } else {
                RequestStatus::TransportError
            },
            http_status: None,
            error_class: Some(class),
            response_hash: None,
            response_bytes: None,
            response_complete: false,
            response_body_inspected: false,
            usage: UsageFields::default(),
            usage_from_complete_stream: false,
        }
    }

    pub fn success(usage: UsageFields, response_hash: String, response_bytes: u64) -> Self {
        Self {
            status: RequestStatus::Success,
            http_status: Some(200),
            error_class: None,
            response_hash: Some(response_hash),
            response_bytes: Some(response_bytes),
            response_complete: true,
            response_body_inspected: true,
            usage,
            usage_from_complete_stream: true,
        }
    }

    pub fn stream_error(usage: UsageFields, response_hash: String, response_bytes: u64) -> Self {
        Self {
            status: RequestStatus::StreamError,
            http_status: Some(200),
            error_class: Some("stream_interrupted"),
            response_hash: Some(response_hash),
            response_bytes: Some(response_bytes),
            response_complete: false,
            response_body_inspected: true,
            usage,
            usage_from_complete_stream: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RequestRecord {
    path: PathBuf,
    request_id: String,
    client_request_id: Option<String>,
    session_id: String,
    protocol: String,
    model: Option<String>,
    upstream_host: String,
    usage_source: UsageSource,
    usage_source_basis: &'static str,
    started_at: u64,
    request_hash: String,
    request_bytes: u64,
    prepare_ran: bool,
    prune: Option<PruneAttribution>,
}

impl RequestRecord {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        path: PathBuf,
        session_id: &str,
        protocol: &str,
        model: Option<String>,
        upstream: &str,
        usage_source: UsageSource,
        usage_source_basis: &'static str,
        started_at: u64,
        request_hash: String,
        request_bytes: u64,
        prepare_ran: bool,
        prune: Option<PruneAttribution>,
        headers: &HeaderMap,
    ) -> Self {
        Self {
            path,
            request_id: Uuid::new_v4().to_string(),
            client_request_id: client_request_id(headers),
            session_id: session_id.to_string(),
            protocol: protocol.to_string(),
            model,
            upstream_host: upstream_host(upstream)
                .unwrap_or_else(|| UNPARSED_UPSTREAM_HOST.to_string()),
            usage_source,
            usage_source_basis,
            started_at,
            request_hash,
            request_bytes,
            prepare_ran,
            prune,
        }
    }

    #[allow(dead_code)] // read by the ledger tests
    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    #[allow(dead_code)] // read by the ledger tests
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append the `start` line.  Failure is reported to the caller, which logs
    /// it and keeps serving the request.
    pub fn start(&self) -> std::io::Result<()> {
        self.append(None)
    }

    pub fn finish(&self, finish: Finish) -> std::io::Result<()> {
        self.append(Some(finish))
    }

    fn append(&self, finish: Option<Finish>) -> std::io::Result<()> {
        let value = self.line(finish.as_ref());
        let text = serde_json::to_string(&value).map_err(std::io::Error::other)?;
        append_line(&self.path, &text)
    }

    fn line(&self, finish: Option<&Finish>) -> Value {
        let finished_at = finish.map(|_| now_ms());
        let duration_ms = finished_at.map(|end| end.saturating_sub(self.started_at));
        let usage = finish.map(|f| f.usage);
        let (context, context_rule) = match usage {
            Some(fields) => context_tokens(&fields),
            None => (None, None),
        };
        let (status, http_status, error_class, response_hash, response_bytes) = match finish {
            Some(f) => (
                Some(f.status),
                f.http_status,
                f.error_class,
                f.response_hash.clone(),
                f.response_bytes,
            ),
            None => (None, None, None, None, None),
        };
        json!({
            "schema": SCHEMA,
            "event": if finish.is_some() { "finish" } else { "start" },
            "request_id": self.request_id,
            "client_request_id": self.client_request_id,
            "session_id": self.session_id,
            "protocol": self.protocol,
            "model": self.model,
            "upstream_host": self.upstream_host,
            "usage_source": self.usage_source,
            "usage_source_basis": self.usage_source_basis,
            // The host does not retry, and never trusts a request header here.
            "retry_count": 0u64,
            "retry_of_request_id": Value::Null,
            "retry_count_basis": RETRY_COUNT_BASIS,
            "started_at": self.started_at,
            "finished_at": finished_at,
            "duration_ms": duration_ms,
            "request_hash": self.request_hash,
            "request_bytes": self.request_bytes,
            "request_hash_scope": REQUEST_HASH_SCOPE,
            "prepare_ran": self.prepare_ran,
            "status": status,
            "http_status": http_status,
            "error_class": error_class,
            "response_hash": response_hash,
            "response_bytes": response_bytes,
            "response_hash_scope": RESPONSE_HASH_SCOPE,
            "response_complete": finish.map(|f| f.response_complete),
            "response_body_inspected": finish.map(|f| f.response_body_inspected),
            "usage_object_present": usage.map(|f| f.object_present),
            // No usage numbers at all.  Fields below stay null; nothing is
            // substituted with 0 or estimated from the request text.
            "usage_missing": usage.map(|f| f.missing_all()),
            "usage_partial": usage.map(|f| {
                let missing = missing_fields(&f);
                f.object_present && !missing.is_empty() && !f.missing_all()
            }),
            "usage_fields_missing": usage.map(|f| missing_fields(&f)),
            "input_tokens": usage.and_then(|f| f.input),
            "cache_read_tokens": usage.and_then(|f| f.cache_read),
            "cache_creation_tokens": usage.and_then(|f| f.cache_creation),
            "output_tokens": usage.and_then(|f| f.output),
            "context_tokens": context,
            "context_tokens_rule": context_rule,
            "usage_from_complete_stream": finish.map(|f| f.usage_from_complete_stream),
            "prune": self.prune.as_ref().map(|prune| json!({
                "triggered": prune.triggered,
                "items": prune.items,
                "artifact_writes": prune.artifact_writes,
                "artifact_bytes": prune.artifact_bytes,
                "artifact_refs": prune.artifact_refs,
                "artifact_refs_attribution": prune.artifact_refs_attribution,
            })),
        })
    }
}

fn ledger_locks() -> &'static Mutex<HashMap<PathBuf, Arc<Mutex<()>>>> {
    static LOCKS: OnceLock<Mutex<HashMap<PathBuf, Arc<Mutex<()>>>>> = OnceLock::new();
    LOCKS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn append_lock(path: &Path) -> Arc<Mutex<()>> {
    let mut guard = ledger_locks()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    guard
        .entry(path.to_path_buf())
        .or_insert_with(|| Arc::new(Mutex::new(())))
        .clone()
}

fn append_line(path: &Path, line: &str) -> std::io::Result<()> {
    let lock = append_lock(path);
    let _held = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    file.write_all(line.as_bytes())?;
    file.write_all(b"\n")?;
    file.flush()?;
    Ok(())
}

/// Path of one conversation's ledger.  The session id is hashed so a header
/// value cannot escape the directory or collide with a snapshot name.
pub fn ledger_path(requests_dir: &Path, session_id: &str) -> PathBuf {
    let mut hasher = Sha256::new();
    hasher.update(session_id.as_bytes());
    requests_dir.join(format!("{}.jsonl", hex::encode(hasher.finalize())))
}

/// Why one ledger cannot support a trusted experiment run.
#[allow(dead_code)] // constructed by audit_ledger, consumed by a future runner
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LedgerIssue {
    /// The ledger could not be opened (missing, permissions, directory).
    Unreadable,
    /// A line is not valid JSON, carries no `request_id`, or has an unknown
    /// `event`.  A crash-truncated trailing line lands here.
    InvalidLine,
    /// A `finish` line has no preceding `start` line for the same request.
    FinishWithoutStart,
    /// More than one `start` line for one request id.
    DuplicateStart,
    /// More than one `finish` line for one request id.
    DuplicateFinish,
    /// A request has a `start` but no `finish`: outcome unknown, never success.
    MissingFinish,
}

impl LedgerIssue {
    #[allow(dead_code)] // used by a future D1b runner's gate report
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Unreadable => "unreadable",
            Self::InvalidLine => "invalid_line",
            Self::FinishWithoutStart => "finish_without_start",
            Self::DuplicateStart => "duplicate_start",
            Self::DuplicateFinish => "duplicate_finish",
            Self::MissingFinish => "missing_finish",
        }
    }
}

/// Read-only integrity summary of one conversation's ledger.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LedgerAudit {
    pub lines: usize,
    /// Distinct request ids seen in either line kind.
    pub requests: usize,
    /// Requests with both a start and a finish line.
    pub finished: usize,
    /// Raw `start`/`finish` line counts, including malformed duplicates.  A
    /// runner compares these with its own dispatch count; `requests` alone
    /// cannot prove that no extra line was appended.
    pub starts: usize,
    pub finishes: usize,
    pub issues: Vec<LedgerIssue>,
}

impl LedgerAudit {
    /// The minimal gate a future D1b runner must pass before a run counts: at
    /// least one request recorded, every request fully started and finished,
    /// and no structural issue at all.
    #[allow(dead_code)] // used by a future D1b runner's gate, and by tests
    pub fn complete(&self) -> bool {
        self.requests > 0
            && self.issues.is_empty()
            && self.finished == self.requests
            && self.starts == self.requests
            && self.finishes == self.requests
    }
}

/// Audit a ledger without changing it or the host's fail-open behaviour.
///
/// Callers that run an experiment must treat a `false` result as "this run is
/// invalid": stop issuing upstream calls and do not aggregate the data.
#[allow(dead_code)] // gate for a future D1b runner; used by tests
pub fn audit_ledger(path: &Path) -> LedgerAudit {
    let mut audit = LedgerAudit::default();
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(_) => {
            audit.issues.push(LedgerIssue::Unreadable);
            return audit;
        }
    };
    let mut events: HashMap<String, (bool, bool)> = HashMap::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        audit.lines += 1;
        let value: Value = match serde_json::from_str(line) {
            Ok(value) => value,
            Err(_) => {
                audit.issues.push(LedgerIssue::InvalidLine);
                continue;
            }
        };
        let request_id = match value.get("request_id").and_then(Value::as_str) {
            Some(id) if !id.is_empty() => id.to_string(),
            _ => {
                audit.issues.push(LedgerIssue::InvalidLine);
                continue;
            }
        };
        let state = events.entry(request_id).or_insert((false, false));
        match value
            .get("event")
            .and_then(Value::as_str)
            .unwrap_or_default()
        {
            "start" => {
                audit.starts += 1;
                if state.0 {
                    audit.issues.push(LedgerIssue::DuplicateStart);
                }
                state.0 = true;
            }
            "finish" => {
                audit.finishes += 1;
                if !state.0 {
                    audit.issues.push(LedgerIssue::FinishWithoutStart);
                }
                if state.1 {
                    audit.issues.push(LedgerIssue::DuplicateFinish);
                }
                state.1 = true;
            }
            _ => audit.issues.push(LedgerIssue::InvalidLine),
        }
    }
    audit.requests = events.len();
    audit.finished = events
        .values()
        .filter(|(start, finish)| *start && *finish)
        .count();
    for (start, finish) in events.values() {
        if *start && !*finish {
            audit.issues.push(LedgerIssue::MissingFinish);
        }
    }
    audit
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(tag: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("bili-ledger-{tag}-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).expect("temp root");
        root
    }

    fn read_lines(path: &Path) -> Vec<Value> {
        let text = fs::read_to_string(path).unwrap_or_default();
        text.lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str::<Value>(line).expect("ledger line json"))
            .collect()
    }

    fn empty_headers() -> HeaderMap {
        HeaderMap::new()
    }

    fn record(root: &Path, source: UsageSource) -> RequestRecord {
        RequestRecord::new(
            ledger_path(&root.join("requests"), "conv-1"),
            "conv-1",
            "anthropic",
            Some("mock-model".into()),
            "http://127.0.0.1:9/v1/messages",
            source,
            "loopback_upstream",
            1_700_000_000_000,
            sha256_hex(b"request"),
            7,
            true,
            None,
            &empty_headers(),
        )
    }

    #[test]
    fn start_and_finish_share_one_host_generated_request_id() {
        let root = temp_root("ids");
        let record = record(&root, UsageSource::Provider);
        let client_id = record.request_id().to_string();
        record.start().expect("start");
        record
            .finish(Finish::success(
                UsageFields::default(),
                sha256_hex(b"response"),
                8,
            ))
            .expect("finish");
        let lines = read_lines(record.path());
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["event"], "start");
        assert_eq!(lines[1]["event"], "finish");
        for line in &lines {
            assert_eq!(line["request_id"], client_id);
            assert_eq!(line["request_hash"], sha256_hex(b"request"));
            assert_eq!(line["request_bytes"], 7);
        }
        assert_ne!(lines[0]["request_id"], Value::Null);
    }

    #[test]
    fn request_ids_are_unique_per_record() {
        let root = temp_root("unique");
        let first = record(&root, UsageSource::Provider)
            .request_id()
            .to_string();
        let second = record(&root, UsageSource::Provider)
            .request_id()
            .to_string();
        assert_ne!(first, second);
    }

    #[test]
    fn client_request_id_is_kept_apart_and_never_becomes_the_request_id() {
        let root = temp_root("client-id");
        let mut headers = HeaderMap::new();
        headers.insert("x-bili-client-request-id", "client-abc".parse().unwrap());
        headers.insert("x-bili-retry-count", "3".parse().unwrap());
        headers.insert("x-bili-usage-source", "mock".parse().unwrap());
        let path = ledger_path(&root.join("requests"), "conv-1");
        let record = RequestRecord::new(
            path,
            "conv-1",
            "anthropic",
            None,
            "https://api.example.com",
            UsageSource::Provider,
            "non_loopback_upstream",
            1_700_000_000_000,
            sha256_hex(b"req"),
            3,
            true,
            None,
            &headers,
        );
        record.start().expect("start");
        let lines = read_lines(record.path());
        assert_eq!(lines[0]["client_request_id"], "client-abc");
        assert_ne!(lines[0]["request_id"], "client-abc");
        // Retry state comes from the host, not from headers.
        assert_eq!(lines[0]["retry_count"], 0);
        assert_eq!(lines[0]["retry_of_request_id"], Value::Null);
        assert_eq!(lines[0]["retry_count_basis"], RETRY_COUNT_BASIS);
        // usage_source is decided by the host, never by x-bili-usage-source.
        assert_eq!(lines[0]["usage_source"], "provider");
    }

    #[test]
    fn usage_source_uses_host_configuration_then_actual_upstream() {
        for (explicit, upstream, expected, basis) in [
            (
                Some(UsageSource::Mock),
                "https://api.anthropic.com",
                UsageSource::Mock,
                "explicit_host_config",
            ),
            (
                Some(UsageSource::Provider),
                "http://127.0.0.1:8080",
                UsageSource::Provider,
                "explicit_host_config",
            ),
            (
                None,
                "http://127.0.0.1:9999",
                UsageSource::Mock,
                "loopback_upstream",
            ),
            (
                None,
                "http://localhost:9999",
                UsageSource::Mock,
                "loopback_upstream",
            ),
            (
                None,
                "https://api.gameaigc.cn",
                UsageSource::Provider,
                "non_loopback_upstream",
            ),
        ] {
            assert_eq!(usage_source(explicit, upstream), (expected, basis));
        }
    }

    #[test]
    fn upstream_host_is_normalized_and_never_records_userinfo() {
        assert_eq!(
            upstream_host("https://user:secret@API.EXAMPLE.COM:443/v1?q=1#frag"),
            Some("api.example.com".to_string())
        );
        assert_eq!(
            upstream_host("https://user:secret@[::1]:8443/v1"),
            Some("::1".to_string())
        );
        assert_eq!(
            upstream_host("https://api.example.com?q=1"),
            Some("api.example.com".to_string())
        );
        assert_eq!(upstream_host("not a url"), None);
        assert_eq!(upstream_host("https://"), None);
    }

    #[test]
    fn ledger_never_writes_userinfo_or_raw_unparsed_url() {
        let root = temp_root("host-sanitize");

        let path = ledger_path(&root.join("requests"), "userinfo");
        let record = RequestRecord::new(
            path,
            "userinfo",
            "openai",
            Some("model".into()),
            "https://user:secret@API.EXAMPLE.COM:443/v1",
            UsageSource::Provider,
            "non_loopback_upstream",
            1_700_000_000_000,
            sha256_hex(b"request"),
            7,
            true,
            None,
            &empty_headers(),
        );
        record.start().expect("start");
        let lines = read_lines(record.path());
        assert_eq!(lines[0]["upstream_host"], "api.example.com");
        let raw = fs::read_to_string(record.path()).expect("read ledger");
        assert!(
            !raw.contains("secret"),
            "userinfo must not be written: {raw}"
        );
        assert!(!raw.contains("user:secret"), "userinfo must not be written");

        let path = ledger_path(&root.join("requests"), "unparsed");
        let record = RequestRecord::new(
            path,
            "unparsed",
            "openai",
            Some("model".into()),
            "not a url with secret",
            UsageSource::Provider,
            "non_loopback_upstream",
            1_700_000_000_000,
            sha256_hex(b"request"),
            7,
            true,
            None,
            &empty_headers(),
        );
        record.start().expect("start");
        let lines = read_lines(record.path());
        assert_eq!(lines[0]["upstream_host"], UNPARSED_UPSTREAM_HOST);
        let raw = fs::read_to_string(record.path()).expect("read ledger");
        assert!(
            !raw.contains("not a url with secret"),
            "raw URL must not be written"
        );
    }

    #[test]
    fn provider_without_usage_marks_usage_missing_and_keeps_every_field_null() {
        let root = temp_root("missing");
        let record = record(&root, UsageSource::Provider);
        record.start().expect("start");
        record
            .finish(Finish::success(UsageFields::default(), sha256_hex(b"x"), 1))
            .expect("finish");
        let lines = read_lines(record.path());
        let finish = &lines[1];
        assert_eq!(finish["usage_source"], "provider");
        assert_eq!(finish["usage_missing"], true);
        assert_eq!(finish["usage_object_present"], false);
        for field in [
            "input_tokens",
            "cache_read_tokens",
            "cache_creation_tokens",
            "output_tokens",
            "context_tokens",
        ] {
            assert_eq!(finish[field], Value::Null, "{field} must stay null");
        }
        assert_eq!(finish["context_tokens_rule"], Value::Null);
    }

    #[test]
    fn partial_usage_object_is_flagged_and_unknown_numbers_stay_null() {
        let root = temp_root("partial");
        let record = record(&root, UsageSource::Provider);
        record.start().expect("start");
        let usage = UsageFields {
            input: Some(10),
            protocol: Some(crate::usage::UsageProtocol::Anthropic),
            object_present: true,
            ..UsageFields::default()
        };
        record
            .finish(Finish::success(usage, sha256_hex(b"r"), 1))
            .expect("finish");
        let lines = read_lines(record.path());
        let finish = &lines[1];
        assert_eq!(finish["usage_object_present"], true);
        assert_eq!(finish["usage_missing"], false);
        assert_eq!(finish["usage_partial"], true);
        assert_eq!(finish["input_tokens"], 10);
        assert_eq!(finish["cache_read_tokens"], Value::Null);
        assert_eq!(finish["cache_creation_tokens"], Value::Null);
        assert_eq!(finish["output_tokens"], Value::Null);
        assert_eq!(finish["context_tokens"], Value::Null);
        assert_eq!(
            finish["usage_fields_missing"].as_array().map(Vec::len),
            Some(3)
        );
    }

    #[test]
    fn absent_start_line_means_the_outcome_is_unknown() {
        let root = temp_root("interrupted");
        let record = record(&root, UsageSource::Provider);
        record.start().expect("start");
        // No finish: the host exited mid-request.  The ledger must not carry
        // any success marker for it.
        let lines = read_lines(record.path());
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0]["status"], Value::Null);
        assert_eq!(lines[0]["finished_at"], Value::Null);
    }

    #[test]
    fn concurrent_appends_to_one_conversation_stay_line_atomic() {
        let root = temp_root("concurrent");
        let path = ledger_path(&root.join("requests"), "conv-1");
        let mut handles = Vec::new();
        for worker in 0..8 {
            let path = path.clone();
            handles.push(std::thread::spawn(move || {
                for item in 0..20 {
                    let payload = json!({
                        "schema": SCHEMA,
                        "event": "finish",
                        "request_id": format!("{worker}-{item}"),
                        "worker": worker,
                        "item": item,
                        "pad": "x".repeat(512),
                    });
                    append_line(&path, &serde_json::to_string(&payload).unwrap()).unwrap();
                }
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }
        let lines = read_lines(&path);
        assert_eq!(lines.len(), 160);
        for line in lines {
            assert!(line["request_id"].is_string());
            assert_eq!(line["pad"].as_str().unwrap().len(), 512);
        }
    }

    #[test]
    fn http_and_transport_failures_never_claim_success() {
        let root = temp_root("failures");
        let http = record(&root, UsageSource::Mock);
        http.start().expect("start");
        http.finish(Finish::http_error(503)).expect("finish");
        let lines = read_lines(http.path());
        assert_eq!(lines[1]["status"], "http_error");
        assert_eq!(lines[1]["http_status"], 503);
        assert_eq!(lines[1]["response_hash"], Value::Null);
        assert_eq!(lines[1]["response_body_inspected"], false);
        assert_eq!(lines[1]["usage_source"], "mock");
        assert_eq!(lines[1]["usage_missing"], true);
    }

    #[test]
    fn prune_refs_are_only_what_this_request_proved() {
        let root = temp_root("prune");
        let record = RequestRecord::new(
            ledger_path(&root.join("requests"), "conv-1"),
            "conv-1",
            "anthropic",
            None,
            "http://127.0.0.1:9/v1/messages",
            UsageSource::Mock,
            "loopback_upstream",
            1_700_000_000_000,
            sha256_hex(b"req"),
            3,
            true,
            Some(PruneAttribution {
                triggered: true,
                items: 1,
                artifact_writes: 1,
                artifact_bytes: 4096,
                artifact_refs: vec!["a_123".into()],
                artifact_refs_attribution: "prune_report_proven",
            }),
            &empty_headers(),
        );
        record.start().expect("start");
        record
            .finish(Finish::success(UsageFields::default(), sha256_hex(b"r"), 1))
            .expect("finish");
        let lines = read_lines(record.path());
        let prune = &lines[1]["prune"];
        assert_eq!(prune["items"], 1);
        assert_eq!(prune["artifact_refs"][0], "a_123");
        assert_eq!(prune["artifact_refs_attribution"], "prune_report_proven");
    }

    #[test]
    fn hashes_state_their_exact_scope_in_band() {
        let root = temp_root("scope");
        let record = record(&root, UsageSource::Provider);
        record.start().expect("start");
        record
            .finish(Finish::success(
                UsageFields::default(),
                sha256_hex(b"partial"),
                9,
            ))
            .expect("finish");
        let lines = read_lines(record.path());
        assert_eq!(lines[0]["request_hash_scope"], REQUEST_HASH_SCOPE);
        assert_eq!(lines[1]["response_hash_scope"], RESPONSE_HASH_SCOPE);
    }

    #[test]
    fn audit_accepts_one_complete_pair() {
        let root = temp_root("audit-ok");
        let record = record(&root, UsageSource::Mock);
        record.start().expect("start");
        record
            .finish(Finish::success(UsageFields::default(), sha256_hex(b"r"), 1))
            .expect("finish");
        let audit = audit_ledger(record.path());
        assert_eq!(audit.lines, 2);
        assert_eq!(audit.requests, 1);
        assert_eq!(audit.finished, 1);
        assert!(audit.issues.is_empty());
        assert!(audit.complete());
    }

    #[test]
    fn audit_flags_orphans_and_unfinished_requests() {
        let root = temp_root("audit-bad");
        let path = root.join("requests").join("manual.jsonl");
        fs::create_dir_all(path.parent().expect("dir")).expect("requests dir");
        let lines = [
            json!({"event": "start", "request_id": "a"}).to_string(),
            json!({"event": "finish", "request_id": "a"}).to_string(),
            json!({"event": "start", "request_id": "b"}).to_string(),
            json!({"event": "finish", "request_id": "c"}).to_string(),
            json!({"event": "finish", "request_id": "a"}).to_string(),
        ];
        fs::write(&path, lines.join("\n") + "\n").expect("write ledger");
        let audit = audit_ledger(&path);
        assert_eq!(audit.lines, 5);
        assert_eq!(audit.requests, 3);
        assert_eq!(audit.finished, 1);
        assert!(!audit.complete());
        assert!(audit.issues.contains(&LedgerIssue::MissingFinish));
        assert!(audit.issues.contains(&LedgerIssue::FinishWithoutStart));
        assert!(audit.issues.contains(&LedgerIssue::DuplicateFinish));
        assert_eq!(LedgerIssue::MissingFinish.as_str(), "missing_finish");
    }

    #[test]
    fn audit_flags_a_crash_truncated_line_instead_of_skipping_it() {
        let root = temp_root("audit-truncated");
        let path = root.join("requests").join("truncated.jsonl");
        fs::create_dir_all(path.parent().expect("dir")).expect("requests dir");
        // A host crash between write_all and newline can leave this tail.
        let text = format!(
            "{}\n{}\n{{\"event\":\"finish\",\"request_id\":\"a\"",
            json!({"event": "start", "request_id": "a"}),
            json!({"event": "finish", "request_id": "a"})
        );
        fs::write(&path, text).expect("write ledger");
        let audit = audit_ledger(&path);
        assert_eq!(audit.lines, 3);
        assert_eq!(audit.finished, 1);
        assert!(!audit.complete());
        assert!(audit.issues.contains(&LedgerIssue::InvalidLine));
    }

    #[test]
    fn audit_of_a_never_written_ledger_is_not_complete() {
        let root = temp_root("audit-missing");
        let path = ledger_path(&root.join("requests"), "never-ran");
        let audit = audit_ledger(&path);
        assert!(audit.issues.contains(&LedgerIssue::Unreadable));
        assert!(!audit.complete());
    }
}

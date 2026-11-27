//! Upstream usage capture.
//!
//! The host streams responses through untouched, so the only place billing
//! numbers exist is the bytes in flight: Anthropic reports a partial `usage` on
//! `message_start` and the output count on `message_delta`, OpenAI reports one
//! complete object on the final chunk, Google attaches `usageMetadata` to its
//! JSON.  A stream tee parses those lines and records the merged result against
//! the conversation the request belonged to.

use async_stream::try_stream;
use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::session_store::Store;
use crate::usage_ledger::{Finish, RequestRecord};

/// Billing numbers for one upstream response.  All counters are cumulative
/// session totals, so callers add rather than overwrite.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    pub input: u64,
    pub cached: u64,
    pub output: u64,
    /// Total prompt tokens as the provider billed them.  Anthropic reports
    /// cache reads outside `input_tokens`, OpenAI includes them.
    pub context: u32,
}

/// Which upstream protocol produced the numbers.  The input/cache accounting
/// differs per protocol, so the ledger keeps the protocol instead of applying
/// one generic formula.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UsageProtocol {
    Anthropic,
    Openai,
    Google,
}

/// One response's provider numbers with absence preserved as `None`.  A field
/// the provider did not report stays `None`; nothing is substituted with 0.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UsageFields {
    pub input: Option<u64>,
    pub cache_read: Option<u64>,
    pub cache_creation: Option<u64>,
    pub output: Option<u64>,
    pub protocol: Option<UsageProtocol>,
    /// True when the response carried a usage object/event at all, even if it
    /// was empty.  This separates "provider sent nothing" from "provider sent
    /// an object with fields missing".
    pub object_present: bool,
}

impl UsageFields {
    /// No canonical token number at all for this protocol.
    pub fn missing_all(&self) -> bool {
        self.input.is_none()
            && self.cache_read.is_none()
            && self.cache_creation.is_none()
            && self.output.is_none()
    }
}

/// Fields the protocol is expected to report.  Used to tell a partial usage
/// object from a missing one without inventing values.
pub fn missing_fields(fields: &UsageFields) -> Vec<&'static str> {
    let mut missing = Vec::new();
    if fields.input.is_none() {
        missing.push("input_tokens");
    }
    if fields.cache_read.is_none() {
        missing.push("cache_read_tokens");
    }
    if matches!(fields.protocol, Some(UsageProtocol::Anthropic)) && fields.cache_creation.is_none()
    {
        missing.push("cache_creation_tokens");
    }
    if fields.output.is_none() {
        missing.push("output_tokens");
    }
    missing
}

/// Context tokens under a protocol-defined rule only; `None` means the host
/// has no verified rule for the fields present.
pub fn context_tokens(fields: &UsageFields) -> (Option<u64>, Option<&'static str>) {
    match fields.protocol {
        // Anthropic: `input_tokens` excludes cache reads and cache writes, so
        // the billed prompt total is the sum of all three reported parts.
        Some(UsageProtocol::Anthropic) => {
            match (fields.input, fields.cache_read, fields.cache_creation) {
                (Some(input), Some(cache_read), Some(cache_creation)) => (
                    Some(
                        input
                            .saturating_add(cache_read)
                            .saturating_add(cache_creation),
                    ),
                    Some("anthropic_input_plus_cache_read_plus_cache_creation"),
                ),
                _ => (None, None),
            }
        }
        // OpenAI: `prompt_tokens` already includes cached prompt tokens.
        Some(UsageProtocol::Openai) => match fields.input {
            Some(prompt_tokens) => (
                Some(prompt_tokens),
                Some("openai_prompt_tokens_includes_cached"),
            ),
            None => (None, None),
        },
        // Google usageMetadata was not part of this change's verified scope.
        Some(UsageProtocol::Google) => (None, None),
        None => (None, None),
    }
}

/// One response can split its usage across events; keep the parts separately so
/// a later `message_delta` that only carries `output_tokens` does not erase the
/// input count from `message_start`.
#[derive(Default)]
pub struct Accum {
    input: Option<u64>,
    cached: Option<u64>,
    creation: Option<u64>,
    output: Option<u64>,
    model: Option<String>,
    openai: bool,
    anthropic: bool,
    google: bool,
    /// Set when the response carried a usage object/event, even an empty one.
    usage_object_present: bool,
}

impl Accum {
    fn absorb(&mut self, value: &Value) {
        if value.get("usage").is_some() || value.get("usageMetadata").is_some() {
            self.usage_object_present = true;
        }
        if let Some(usage) = value.get("usage") {
            self.absorb_anthropic(usage);
        }
        if let Some(message) = value.get("message") {
            if let Some(model) = message.get("model").and_then(Value::as_str) {
                self.model = Some(model.to_string());
            }
            if let Some(usage) = message.get("usage") {
                self.usage_object_present = true;
                self.absorb_anthropic(usage);
            }
        }
        if let Some(usage) = value.get("usageMetadata") {
            self.absorb_google(usage);
        }
        // OpenAI chat chunks (and a bare final JSON body).
        if let Some(usage) = value.get("usage") {
            self.absorb_openai(usage);
        }
    }

    fn absorb_anthropic(&mut self, usage: &Value) {
        let pick = |key: &str| usage.get(key).and_then(Value::as_u64);
        // `input_tokens` and `output_tokens` are present on both start and
        // delta; only overwrite when the field actually arrived.
        if let Some(value) = pick("input_tokens") {
            self.input = Some(value);
            self.anthropic = true;
        }
        if let Some(value) = pick("cache_read_input_tokens") {
            self.cached = Some(value);
        }
        if let Some(value) = pick("cache_creation_input_tokens") {
            self.creation = Some(value);
        }
        if let Some(value) = pick("output_tokens") {
            self.output = Some(value);
        }
    }

    fn absorb_openai(&mut self, usage: &Value) {
        let pick = |key: &str| usage.get(key).and_then(Value::as_u64);
        if let Some(value) = pick("prompt_tokens") {
            self.input = Some(value);
            self.openai = true;
        }
        if let Some(value) = pick("completion_tokens") {
            self.output = Some(value);
        }
        if let Some(value) = usage
            .get("prompt_tokens_details")
            .and_then(|details| details.get("cached_tokens"))
            .and_then(Value::as_u64)
        {
            self.cached = Some(value);
        }
    }

    fn absorb_google(&mut self, usage: &Value) {
        let pick = |key: &str| usage.get(key).and_then(Value::as_u64);
        if let Some(value) = pick("promptTokenCount") {
            self.input = Some(value);
            self.google = true;
        }
        if let Some(value) = pick("candidatesTokenCount") {
            self.output = Some(value);
        }
        if let Some(value) = pick("cachedContentTokenCount") {
            self.cached = Some(value);
        }
    }

    /// The merged numbers, or `None` when no event carried any.
    pub fn finish(&self) -> Option<Usage> {
        if self.input.is_none() && self.output.is_none() {
            return None;
        }
        let input = self.input.unwrap_or(0);
        let cached = self.cached.unwrap_or(0);
        let creation = self.creation.unwrap_or(0);
        let output = self.output.unwrap_or(0);
        // OpenAI and Google already count cached prompt tokens inside
        // `prompt_tokens`; Anthropic reports them beside `input_tokens`.
        let context = if self.openai || self.google {
            input
        } else {
            input + cached + creation
        };
        Some(Usage {
            input,
            cached,
            output,
            context: context as u32,
        })
    }

    #[allow(
        dead_code,
        reason = "usage model accessor retained for ledger integrations"
    )]
    pub fn model(&self) -> Option<String> {
        self.model.clone()
    }

    /// Per-request view: absence preserved and the protocol identified, so the
    /// ledger can apply the provider's own accounting instead of a generic
    /// formula.
    pub fn fields(&self) -> UsageFields {
        let protocol = if self.anthropic {
            Some(UsageProtocol::Anthropic)
        } else if self.openai {
            Some(UsageProtocol::Openai)
        } else if self.google {
            Some(UsageProtocol::Google)
        } else {
            None
        };
        UsageFields {
            input: self.input,
            cache_read: self.cached,
            cache_creation: self.creation,
            output: self.output,
            protocol,
            object_present: self.usage_object_present,
        }
    }
}

/// Parse a single SSE `data:` payload (or a whole non-streamed body).
pub fn absorb_json(acc: &mut Accum, text: &str) {
    let Ok(value) = serde_json::from_str::<Value>(text) else {
        return;
    };
    acc.absorb(&value);
}

/// Feed complete lines of an SSE stream into the accumulator. Returns true
/// once a terminal frame (`data: [DONE]` or Anthropic `message_stop`) has been
/// seen, so the caller can persist billed usage before the client disconnects.
pub fn absorb_sse_lines(acc: &mut Accum, buffer: &mut String, chunk: &str) -> bool {
    let mut terminal = false;
    buffer.push_str(chunk);
    while let Some(index) = buffer.find('\n') {
        let line = buffer[..index].trim_end_matches('\r').trim();
        let payload = line.strip_prefix("data:").map(str::trim);
        if let Some(payload) = payload {
            if payload == "[DONE]" {
                terminal = true;
            } else if !payload.is_empty() {
                if let Ok(value) = serde_json::from_str::<Value>(payload) {
                    if value.get("type").and_then(Value::as_str) == Some("message_stop") {
                        terminal = true;
                    }
                    acc.absorb(&value);
                }
            }
        }
        buffer.drain(..=index);
    }
    terminal
}

/// Tee an upstream response body through the usage parser.  Bytes are yielded
/// unchanged and in order.  A dropped connection records the numbers that did
/// arrive before the error is propagated, so a truncated response can never be
/// mistaken for a complete one.
pub fn inspect_stream<S, E>(
    stream: S,
    content_type: String,
    store: Store,
    key: String,
    fallback_model: Option<String>,
    ledger: Option<RequestRecord>,
) -> impl Stream<Item = Result<Bytes, E>>
where
    S: Stream<Item = Result<Bytes, E>> + Send + Unpin + 'static,
    E: std::error::Error + Send + Sync + 'static,
{
    try_stream! {
        let mut acc = Accum::default();
        let mut buffer = String::new();
        let mut body = String::new();
        let sse = content_type.contains("text/event-stream");
        let json = content_type.contains("application/json");
        // The hash covers the bytes actually received, in arrival order.  It is
        // an integrity check, not an archive: the host does not persist the
        // raw response body.
        let mut hasher = Sha256::new();
        let mut received: u64 = 0;
        let mut recorded = false;
        let mut stream = stream;
        loop {
            match stream.next().await {
                Some(Ok(chunk)) => {
                    hasher.update(&chunk);
                    received = received.saturating_add(chunk.len() as u64);
                    if sse || json {
                        let text = String::from_utf8_lossy(&chunk);
                        if sse {
                            let terminal = absorb_sse_lines(&mut acc, &mut buffer, &text);
                            if terminal && !recorded {
                                // A terminal frame proves the upstream finished
                                // billing even if the transport closes later.
                                if let Some(usage) = acc.finish() {
                                    store.record_usage(&key, fallback_model.as_deref(), usage);
                                }
                                if let Some(record) = ledger.as_ref() {
                                    let digest = hex::encode(hasher.clone().finalize());
                                    if let Err(write_err) = record.finish(Finish::success(
                                        acc.fields(),
                                        digest,
                                        received,
                                    )) {
                                        eprintln!(
                                            "[billion-context-proxy] usage ledger write failed: {write_err}"
                                        );
                                    }
                                }
                                recorded = true;
                            }
                        } else if body.len() < 8 * 1024 * 1024 {
                            body.push_str(&text);
                        }
                    }
                    yield chunk;
                }
                Some(Err(err)) => {
                    let digest = hex::encode(hasher.clone().finalize());
                    if !recorded {
                        if let Some(record) = ledger.as_ref() {
                            if let Err(write_err) = record
                                .finish(Finish::stream_error(acc.fields(), digest, received))
                            {
                                eprintln!("[billion-context-proxy] usage ledger write failed: {write_err}");
                            }
                        }
                    }
                    Err(err)?;
                }
                None => break,
            }
        }
        if !recorded {
            if !sse && json && !body.is_empty() {
                absorb_json(&mut acc, &body);
            }
            let digest = hex::encode(hasher.finalize());
            let fields = acc.fields();
            if let Some(usage) = acc.finish() {
                store.record_usage(&key, fallback_model.as_deref(), usage);
            }
            if let Some(record) = ledger.as_ref() {
                if let Err(write_err) = record.finish(Finish::success(fields, digest, received)) {
                    eprintln!("[billion-context-proxy] usage ledger write failed: {write_err}");
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(lines: &[&str]) -> Accum {
        let mut acc = Accum::default();
        let mut buffer = String::new();
        for line in lines {
            absorb_sse_lines(&mut acc, &mut buffer, &format!("{line}\n"));
        }
        acc
    }

    fn terminal_after(lines: &[&str]) -> bool {
        let mut acc = Accum::default();
        let mut buffer = String::new();
        let mut terminal = false;
        for line in lines {
            terminal |= absorb_sse_lines(&mut acc, &mut buffer, &format!("{line}\n"));
        }
        terminal
    }

    #[test]
    fn openai_done_frame_is_terminal() {
        assert!(terminal_after(&[
            r#"data: {"choices":[{"delta":{"content":"ok"}}]}"#,
            "data: [DONE]",
        ]));
    }

    #[test]
    fn anthropic_message_stop_is_terminal() {
        assert!(terminal_after(&[
            r#"data: {"type":"message_start","message":{"usage":{"input_tokens":10}}}"#,
            r#"data: {"type":"message_stop"}"#,
        ]));
    }

    #[test]
    fn incomplete_sse_is_not_terminal() {
        assert!(!terminal_after(&[
            r#"data: {"type":"message_start","message":{"usage":{"input_tokens":10}}}"#,
            r#"data: {"type":"message_delta","usage":{"output_tokens":5}}"#,
        ]));
    }

    #[test]
    fn openai_prompt_tokens_include_cached_tokens() {
        let acc = feed(&[
            r#"data: {"choices":[{"delta":{"content":"ok"}}]}"#,
            r#"data: {"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":40}}}"#,
            "data: [DONE]",
        ]);
        let usage = acc.finish().expect("usage");
        assert_eq!(usage.context, 100);
        assert_eq!(usage.input, 100);
        assert_eq!(usage.cached, 40);
        assert_eq!(usage.output, 5);
    }

    #[test]
    fn anthropic_input_tokens_exclude_cache_read() {
        let acc = feed(&[
            r#"data: {"type":"message_start","message":{"usage":{"input_tokens":10,"cache_read_input_tokens":80,"cache_creation_input_tokens":10}}}"#,
            r#"data: {"type":"message_delta","usage":{"output_tokens":5}}"#,
            r#"data: {"type":"message_stop"}"#,
        ]);
        let usage = acc.finish().expect("usage");
        assert_eq!(usage.context, 100);
        assert_eq!(usage.input, 10);
        assert_eq!(usage.cached, 80);
        assert_eq!(usage.output, 5);
    }

    #[test]
    fn partial_anthropic_usage_keeps_missing_fields_null_and_context_null() {
        let acc = feed(&[
            r#"data: {"type":"message_start","message":{"usage":{"input_tokens":10}}}"#,
            r#"data: {"type":"message_stop"}"#,
        ]);
        let fields = acc.fields();
        assert_eq!(fields.input, Some(10));
        assert_eq!(fields.cache_read, None);
        assert_eq!(fields.cache_creation, None);
        assert_eq!(fields.output, None);
        assert_eq!(fields.protocol, Some(UsageProtocol::Anthropic));
        // No verified rule for a partial anthropic object.
        assert_eq!(context_tokens(&fields), (None, None));
        assert!(!fields.missing_all());
    }

    #[test]
    fn context_tokens_follow_the_protocol_rule_per_provider() {
        let anthropic = UsageFields {
            input: Some(10),
            cache_read: Some(80),
            cache_creation: Some(10),
            output: Some(5),
            protocol: Some(UsageProtocol::Anthropic),
            object_present: true,
        };
        assert_eq!(
            context_tokens(&anthropic),
            (
                Some(100),
                Some("anthropic_input_plus_cache_read_plus_cache_creation")
            )
        );
        let openai = UsageFields {
            input: Some(100),
            cache_read: Some(40),
            cache_creation: None,
            output: Some(5),
            protocol: Some(UsageProtocol::Openai),
            object_present: true,
        };
        assert_eq!(
            context_tokens(&openai),
            (Some(100), Some("openai_prompt_tokens_includes_cached"))
        );
        let google = UsageFields {
            input: Some(100),
            cache_read: Some(40),
            cache_creation: None,
            output: Some(5),
            protocol: Some(UsageProtocol::Google),
            object_present: true,
        };
        // Not part of this change's verified scope; must stay unset.
        assert_eq!(context_tokens(&google), (None, None));
    }

    #[test]
    fn empty_usage_object_reports_no_numbers_without_zero_fill() {
        let acc = feed(&[r#"data: {"usage":{}}"#]);
        let fields = acc.fields();
        assert!(fields.object_present);
        assert!(fields.missing_all());
        assert_eq!(fields.input, None);
        assert_eq!(fields.output, None);
        assert_eq!(context_tokens(&fields), (None, None));
    }

    fn ledger_test_record(root: &std::path::Path) -> RequestRecord {
        let requests = root.join("requests");
        let headers = axum::http::HeaderMap::new();
        RequestRecord::new(
            crate::usage_ledger::ledger_path(&requests, "usage-test"),
            "usage-test",
            "anthropic",
            Some("mock-model".into()),
            "http://127.0.0.1:9/v1/messages",
            crate::usage_ledger::UsageSource::Mock,
            "loopback_upstream",
            crate::session_store::now_ms(),
            crate::usage_ledger::sha256_hex(b"request"),
            7,
            true,
            None,
            &headers,
        )
    }

    fn read_ledger(path: &std::path::Path) -> Vec<serde_json::Value> {
        std::fs::read_to_string(path)
            .unwrap_or_default()
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str(line).expect("ledger json"))
            .collect()
    }

    #[tokio::test]
    async fn completed_stream_records_the_hash_of_received_bytes() {
        let root = std::env::temp_dir().join(format!("bili-usage-stream-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).expect("temp root");
        let store = Store::with_dir(root.join("sessions"));
        let record = ledger_test_record(&root);
        record.start().expect("start");
        let chunks = vec![
            Bytes::from_static(
                b"data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":10,\"cache_read_input_tokens\":80,\"cache_creation_input_tokens\":10}}}\n\n",
            ),
            Bytes::from_static(b"data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":5}}\n\n"),
        ];
        let expected: Vec<u8> = chunks.iter().flat_map(|chunk| chunk.to_vec()).collect();
        let tee = inspect_stream(
            futures_util::stream::iter(chunks.into_iter().map(Ok::<_, reqwest::Error>)),
            "text/event-stream".to_string(),
            store,
            "usage-test".to_string(),
            Some("mock-model".to_string()),
            Some(record.clone()),
        );
        let received: Vec<Bytes> = tee.map(|item| item.expect("response byte")).collect().await;
        let received: Vec<u8> = received.iter().flat_map(|chunk| chunk.to_vec()).collect();
        assert_eq!(received, expected);

        let lines = read_ledger(record.path());
        let finish = lines.last().expect("finish line");
        assert_eq!(finish["status"], "success");
        assert_eq!(
            finish["response_hash"],
            crate::usage_ledger::sha256_hex(&expected)
        );
        assert_eq!(finish["response_bytes"], expected.len() as u64);
        assert_eq!(finish["usage_object_present"], true);
        assert_eq!(finish["input_tokens"], 10);
        assert_eq!(finish["cache_read_tokens"], 80);
        assert_eq!(finish["cache_creation_tokens"], 10);
        assert_eq!(finish["output_tokens"], 5);
        assert_eq!(finish["context_tokens"], 100);
    }

    #[derive(Debug)]
    struct TruncatedStream;

    impl std::fmt::Display for TruncatedStream {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "truncated upstream stream")
        }
    }

    impl std::error::Error for TruncatedStream {}

    #[tokio::test]
    async fn dropped_stream_leaves_only_the_start_line() {
        // A client abort or host exit drops the tee before it reaches its end:
        // the ledger must then show a start with no finish, never a success.
        let root =
            std::env::temp_dir().join(format!("bili-usage-stream-drop-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).expect("temp root");
        let store = Store::with_dir(root.join("sessions"));
        let record = ledger_test_record(&root);
        record.start().expect("start");
        let chunk = Bytes::from_static(b"data: {\"type\":\"content_block_delta\"}\n\n");
        let mut tee = Box::pin(inspect_stream(
            futures_util::stream::iter(vec![Ok::<_, TruncatedStream>(chunk)]),
            "text/event-stream".to_string(),
            store,
            "usage-test".to_string(),
            None,
            Some(record.clone()),
        ));
        assert!(tee.next().await.is_some());
        drop(tee);

        let lines = read_ledger(record.path());
        assert_eq!(lines.len(), 1, "only the start line was written");
        assert_eq!(lines[0]["event"], "start");
        assert_eq!(lines[0]["status"], serde_json::Value::Null);
        assert_eq!(lines[0]["finished_at"], serde_json::Value::Null);
    }

    #[tokio::test]
    async fn interrupted_stream_records_partial_hash_then_reports_the_error() {
        let root =
            std::env::temp_dir().join(format!("bili-usage-stream-err-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).expect("temp root");
        let store = Store::with_dir(root.join("sessions"));
        let record = ledger_test_record(&root);
        record.start().expect("start");
        let partial: &[u8] =
            b"data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":10}}}\n\n";
        let items: Vec<Result<Bytes, TruncatedStream>> =
            vec![Ok(Bytes::from_static(partial)), Err(TruncatedStream)];
        let tee = inspect_stream(
            futures_util::stream::iter(items.into_iter()),
            "text/event-stream".to_string(),
            store,
            "usage-test".to_string(),
            None,
            Some(record.clone()),
        );
        let results: Vec<Result<Bytes, TruncatedStream>> = tee.collect().await;
        assert_eq!(results.len(), 2, "both the bytes and the error surface");
        assert!(results[0].is_ok(), "received bytes are passed through");
        assert!(results[1].is_err(), "the interruption is not swallowed");

        let lines = read_ledger(record.path());
        let finish = lines.last().expect("finish line");
        assert_eq!(finish["status"], "stream_error");
        assert_eq!(finish["response_complete"], false);
        assert_eq!(
            finish["response_hash"],
            crate::usage_ledger::sha256_hex(partial)
        );
        assert_eq!(finish["response_bytes"], partial.len() as u64);
        assert_eq!(finish["input_tokens"], 10);
        assert_eq!(finish["output_tokens"], serde_json::Value::Null);
        assert_eq!(finish["usage_from_complete_stream"], false);
    }
}

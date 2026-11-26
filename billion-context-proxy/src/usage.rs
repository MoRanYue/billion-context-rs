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

use crate::session_store::Store;

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
}

impl Accum {
    fn absorb(&mut self, value: &Value) {
        if let Some(usage) = value.get("usage") {
            self.absorb_anthropic(usage);
        }
        if let Some(message) = value.get("message") {
            if let Some(model) = message.get("model").and_then(Value::as_str) {
                self.model = Some(model.to_string());
            }
            if let Some(usage) = message.get("usage") {
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

    pub fn model(&self) -> Option<String> {
        self.model.clone()
    }
}

/// Parse a single SSE `data:` payload (or a whole non-streamed body).
pub fn absorb_json(acc: &mut Accum, text: &str) {
    let Ok(value) = serde_json::from_str::<Value>(text) else {
        return;
    };
    acc.absorb(&value);
}

/// Feed complete lines of an SSE stream into the accumulator.
pub fn absorb_sse_lines(acc: &mut Accum, buffer: &mut String, chunk: &str) {
    buffer.push_str(chunk);
    while let Some(index) = buffer.find('\n') {
        let line = buffer[..index].trim_end_matches('\r').trim();
        let payload = line.strip_prefix("data:").map(str::trim);
        if let Some(payload) = payload {
            if payload != "[DONE]" && !payload.is_empty() {
                absorb_json(acc, payload);
            }
        }
        buffer.drain(..=index);
    }
}

/// Tee an upstream response body through the usage parser.  Bytes are yielded
/// unchanged and in order; recording happens at end of stream (or as soon as a
/// terminal usage event completes the record, so a dropped connection still
/// leaves the numbers behind).
pub fn inspect_stream<S>(
    stream: S,
    content_type: String,
    store: Store,
    key: String,
    fallback_model: Option<String>,
) -> impl Stream<Item = Result<Bytes, reqwest::Error>>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Send + Unpin + 'static,
{
    try_stream! {
        let mut acc = Accum::default();
        let mut buffer = String::new();
        let mut body = String::new();
        let sse = content_type.contains("text/event-stream");
        let json = content_type.contains("application/json");
        let mut stream = stream;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            if sse || json {
                let text = String::from_utf8_lossy(&chunk);
                if sse {
                    absorb_sse_lines(&mut acc, &mut buffer, &text);
                } else if body.len() < 8 * 1024 * 1024 {
                    body.push_str(&text);
                }
            }
            yield chunk;
        }
        if !sse && json && !body.is_empty() {
            absorb_json(&mut acc, &body);
        }
        if let Some(usage) = acc.finish() {
            store.record_usage(&key, fallback_model.as_deref(), usage);
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
}

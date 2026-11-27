//! External summary providers (`providers` + `compress.externalSummary`).
//!
//! The fold chain asked the *conversation model* to author every summary until
//! now. That works, but it spends the most expensive model on the part of the
//! turn that is pure bookkeeping (plan §2.1/§3.2). `compress.externalSummary`
//! lets the host hand that job to a separately configured model: the model's
//! own `summary` degrades to an advisory hint and the fold lands the text this
//! chain produced.
//!
//! Configuration is read from the same `billion-context.json` the Node proxy
//! uses, and only the surface that is needed here is interpreted:
//!
//! * `providers.<name>` — a *named dial recipe*: `baseUrl`, `api`, exactly one
//!   credential (`apiKeyEnv` or `credentialRef`), and a `models` table whose
//!   entries carry `contextWindow` / `outputTokens` / `stream`.
//! * `compress.externalSummary` — `enabled`, `targets` (`"provider/model"`
//!   references), and `budget` (`targetTimeoutMs`, `totalTimeoutMs`,
//!   `maxSummaryBytes`).
//!
//! Semantics that matter for correctness:
//!
//! * The chain is **opt-in**. Without `enabled: true` nothing is called and the
//!   conversation model keeps authoring summaries.
//! * Targets are tried in order; a target that errors, times out, or returns an
//!   unusable summary is skipped and the next one is tried.
//! * An unresolvable reference disables the *whole* chain with one warning,
//!   matching the Node proxy — never a half-configured chain that silently
//!   folds with the conversation model instead.
//! * Credentials come from the recipe's environment variable and, when that
//!   variable is absent, from the harness credential file
//!   (`~/.dsh/.credentials.yaml`, `refs:` section). Both are looked up at call
//!   time, so the key never lands in `billion-context.json`.
//!
//! Fail-open is deliberate: if the chain produces nothing, the host keeps the
//! model-authored summaries and the fold still lands. The Node proxy is
//! stricter (it drops the range instead); the difference is documented in
//! `docs/bili-rust-compaction-plan.md` §2.8.

use futures_util::StreamExt;
use serde_json::Value;
use std::collections::BTreeMap;

/// Dedicated process-environment credential for the external summary chain.
/// It takes precedence over the recipe's `apiKeyEnv`/`credentialRef`; the
/// runner passes the key through this variable only, never through a file.
pub const EXTERNAL_SUMMARY_CREDENTIAL_ENV: &str = "BILI_EXTERNAL_SUMMARY_API_KEY";
use std::path::Path;
use std::time::Duration;

/// The prefix DSH uses for its own compaction calls. The summary request below
/// is a plain chat completion, but keeping the same framing keeps the two
/// lanes visually identical to a model that has seen either one.
pub(crate) const EXTERNAL_SUMMARY_INSTRUCTION: &str = "You are now acting as a compaction engine for this AI coding assistant. Summarize the conversation slice the user gives you: preserve file paths with line numbers, exact commands, error text, decisions with their rationale, thresholds and versions; drop consumed exploration and repeated logs; keep open questions and TODOs; write dense scannable bullets under short thematic headers; output only the summary markdown, with no preamble.";

/// Upper bound on the characters sent to a summarizer in one dispatch. The
/// conversation slice for a single fold is usually far below this; a range that
/// is not gets its middle elided so the head and tail (intent + result) both
/// survive. Configurable through `compress.externalSummary.budget.maxInputChars`
/// so a cheaper model with a small window can cap it down further.
const DEFAULT_MAX_INPUT_CHARS: usize = 48_000;

/// A summary shorter than this is treated as a failed dispatch rather than a
/// fold with an empty block.
const MIN_SUMMARY_CHARS: usize = 8;

/// Hard cap on one provider response. A summary is bounded by the session
/// budget, so anything larger is a broken or hostile endpoint; reject it
/// instead of buffering without limit.
const MAX_PROVIDER_RESPONSE_BYTES: usize = 64 * 1024;

const DEFAULT_CONTEXT_WINDOW: u64 = 128_000;
const DEFAULT_OUTPUT_TOKENS: u64 = 8_192;
const DEFAULT_TARGET_TIMEOUT_MS: u64 = 25_000;
const DEFAULT_TOTAL_TIMEOUT_MS: u64 = 50_000;
const DEFAULT_MAX_SUMMARY_BYTES: usize = 65_536;
/// Per-session ceiling on automatic summary dispatches (fallback + speculation
/// share it). A manual model-requested compress is not capped by it.
const DEFAULT_MAX_CALLS_PER_SESSION: u64 = 64;
/// Independent per-session ceiling for background speculation, so speculation
/// cannot eat the whole automatic budget.
const DEFAULT_MAX_SPECULATION_CALLS: u64 = 16;

/// Knobs a recipe's `models` entry may carry. Defaults mirror the documented
/// Node defaults so a bare `{}` behaves the same on both lanes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelKnobs {
    pub context_window: u64,
    pub output_tokens: u64,
    pub stream: bool,
}

impl Default for ModelKnobs {
    fn default() -> Self {
        Self {
            context_window: DEFAULT_CONTEXT_WINDOW,
            output_tokens: DEFAULT_OUTPUT_TOKENS,
            stream: false,
        }
    }
}

impl ModelKnobs {
    fn from_value(value: &Value) -> Self {
        let mut knobs = ModelKnobs::default();
        if let Some(window) = value.get("contextWindow").and_then(Value::as_u64) {
            knobs.context_window = window.max(1);
        }
        if let Some(output) = value.get("outputTokens").and_then(Value::as_u64) {
            // The documented default is `min(8192, window / 4)`; an explicit
            // value is still clamped to what the window can hold so a typo
            // cannot ask for more output than the model accepts.
            knobs.output_tokens = output.clamp(1, (knobs.context_window / 4).max(1));
        } else {
            knobs.output_tokens = DEFAULT_OUTPUT_TOKENS.min((knobs.context_window / 4).max(1));
        }
        knobs.stream = value
            .get("stream")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        knobs
    }
}

/// A named dial recipe. Only `api: "openai"` is implemented; the other
/// documented protocols are rejected at load time so the chain disables itself
/// instead of posting an OpenAI body to an Anthropic endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Recipe {
    pub name: String,
    pub base_url: String,
    pub api: String,
    pub api_key_env: Option<String>,
    pub credential_ref: Option<String>,
    pub models: BTreeMap<String, ModelKnobs>,
}

/// The chain, already resolved against the recipes. `targets` is ordered.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChainSettings {
    pub enabled: bool,
    pub targets: Vec<String>,
    pub target_timeout_ms: u64,
    pub total_timeout_ms: u64,
    pub max_summary_bytes: usize,
    /// Per-dispatch input cap and per-response output-token cap.
    pub max_input_chars: usize,
    pub max_output_tokens: u64,
    /// Session budgets for automatic dispatches. Zero on the speculation knob
    /// means speculation never dispatches.
    pub max_calls_per_session: u64,
    pub max_speculation_calls: u64,
    pub auto_fold: bool,
    pub auto_fold_target_tokens: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedTarget {
    pub recipe: Recipe,
    pub model: String,
    pub knobs: ModelKnobs,
}

#[derive(Debug, Clone, Default)]
pub struct ExternalSummary {
    pub settings: ChainSettings,
    pub targets: Vec<ResolvedTarget>,
}

impl ExternalSummary {
    /// Enabled *and* at least one usable target. A chain with no targets can
    /// only be a half-written config, so it stays off.
    pub fn is_active(&self) -> bool {
        self.settings.enabled && !self.targets.is_empty()
    }
}

/// One range handed to the summarizer: the label the model saw, the optional
/// topic, and the rendered conversation slice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RangeInput {
    pub label: String,
    pub topic: Option<String>,
    pub text: String,
}

/// What one `summarize` batch cost. `calls` counts dispatch attempts across every
/// range and target (a failed target still burns a call); tokens come from the
/// provider's `usage` block when it sends one, otherwise from a char-based
/// estimate, and `estimated_calls` says how many of those were estimates.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SummaryBatch {
    pub summaries: Vec<String>,
    pub calls: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub estimated_calls: u64,
}

#[derive(Debug, Clone, Copy)]
struct Usage {
    prompt_tokens: u64,
    completion_tokens: u64,
}

/// Parse the external-summary surface out of the config document.
///
/// Returns `None` when the document has no `compress.externalSummary` at all,
/// which is the common case: the feature is opt-in, so its absence must not be
/// reported as a configuration error.
pub fn from_config(document: &Value) -> Option<ExternalSummary> {
    let chain = document
        .get("compress")
        .and_then(|compress| compress.get("externalSummary"))?;
    let settings = ChainSettings {
        enabled: chain
            .get("enabled")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        targets: chain
            .get("targets")
            .and_then(Value::as_array)
            .map(|targets| {
                targets
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default(),
        target_timeout_ms: chain
            .get("budget")
            .and_then(|budget| budget.get("targetTimeoutMs"))
            .and_then(Value::as_u64)
            .unwrap_or(DEFAULT_TARGET_TIMEOUT_MS),
        total_timeout_ms: chain
            .get("budget")
            .and_then(|budget| budget.get("totalTimeoutMs"))
            .and_then(Value::as_u64)
            .unwrap_or(DEFAULT_TOTAL_TIMEOUT_MS),
        max_summary_bytes: chain
            .get("budget")
            .and_then(|budget| budget.get("maxSummaryBytes"))
            .and_then(Value::as_u64)
            .map(|bytes| bytes as usize)
            .unwrap_or(DEFAULT_MAX_SUMMARY_BYTES),
        max_input_chars: chain
            .get("budget")
            .and_then(|budget| budget.get("maxInputChars"))
            .and_then(Value::as_u64)
            .map(|chars| (chars as usize).clamp(1_000, 200_000))
            .unwrap_or(DEFAULT_MAX_INPUT_CHARS),
        max_output_tokens: chain
            .get("budget")
            .and_then(|budget| budget.get("maxOutputTokens"))
            .and_then(Value::as_u64)
            .map(|tokens| tokens.clamp(256, 32_768))
            .unwrap_or(DEFAULT_OUTPUT_TOKENS),
        max_calls_per_session: chain
            .get("budget")
            .and_then(|budget| budget.get("maxCallsPerSession"))
            .and_then(Value::as_u64)
            .map(|calls| calls.clamp(1, 10_000))
            .unwrap_or(DEFAULT_MAX_CALLS_PER_SESSION),
        max_speculation_calls: chain
            .get("budget")
            .and_then(|budget| budget.get("maxSpeculationCalls"))
            .and_then(Value::as_u64)
            .map(|calls| calls.min(10_000))
            .unwrap_or(DEFAULT_MAX_SPECULATION_CALLS),
        auto_fold: chain
            .get("autoFold")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        auto_fold_target_tokens: chain.get("autoFoldTargetTokens").and_then(Value::as_u64),
    };
    let recipes = providers_from_config(document);
    let mut targets = Vec::new();
    let mut unresolved = None;
    for reference in &settings.targets {
        match resolve_target(reference, &recipes) {
            Some(target) => targets.push(target),
            None => {
                unresolved = Some(reference.clone());
                break;
            }
        }
    }
    if let Some(reference) = unresolved {
        // The Node proxy disables the whole chain here; doing anything else
        // would silently fold with the conversation model while the operator
        // believes a cheap summarizer is writing the block.
        log_warn(&format!(
            "compress.externalSummary: {reference} does not resolve to a provider/model recipe - chain disabled"
        ));
        return Some(ExternalSummary {
            settings: ChainSettings {
                enabled: false,
                ..settings
            },
            targets: Vec::new(),
        });
    }
    Some(ExternalSummary { settings, targets })
}

/// Named dial recipes only. URL-keyed routing entries are deliberately ignored:
/// they describe how to forward requests, not where a summarizer lives.
fn providers_from_config(document: &Value) -> BTreeMap<String, Recipe> {
    let mut recipes = BTreeMap::new();
    let Some(providers) = document.get("providers").and_then(Value::as_object) else {
        return recipes;
    };
    for (name, entry) in providers {
        let Some(object) = entry.as_object() else {
            continue;
        };
        let Some(base_url) = object.get("baseUrl").and_then(Value::as_str) else {
            continue;
        };
        let Some(api) = object.get("api").and_then(Value::as_str) else {
            continue;
        };
        let model_table = object.get("models").and_then(Value::as_object);
        let Some(model_table) = model_table else {
            continue;
        };
        let models = model_table
            .iter()
            .map(|(model, knobs)| (model.clone(), ModelKnobs::from_value(knobs)))
            .collect();
        recipes.insert(
            name.clone(),
            Recipe {
                name: name.clone(),
                base_url: base_url.trim_end_matches('/').to_string(),
                api: api.to_string(),
                api_key_env: object
                    .get("apiKeyEnv")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                credential_ref: object
                    .get("credentialRef")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                models,
            },
        );
    }
    recipes
}

/// Resolve one named dial recipe from the already-parsed config document.
/// Used by the standalone D1b runner so its credential binding comes from the
/// provider entry, not from a hardcoded global variable.
pub fn provider_recipe(document: &Value, name: &str) -> Option<Recipe> {
    providers_from_config(document).remove(name)
}

fn resolve_target(reference: &str, recipes: &BTreeMap<String, Recipe>) -> Option<ResolvedTarget> {
    let (provider, model) = reference.split_once('/')?;
    let recipe = recipes.get(provider)?.clone();
    let knobs = recipe.models.get(model)?.clone();
    if recipe.api != "openai" {
        log_warn(&format!(
            "compress.externalSummary: provider {provider} speaks `{}`, only `openai` is implemented in the Rust host - chain disabled",
            recipe.api
        ));
        return None;
    }
    if recipe.api_key_env.is_none() && recipe.credential_ref.is_none() {
        log_warn(&format!(
            "compress.externalSummary: provider {provider} declares no apiKeyEnv/credentialRef - chain disabled"
        ));
        return None;
    }
    Some(ResolvedTarget {
        recipe,
        model: model.to_string(),
        knobs,
    })
}

/// The credential the recipe names, or `None` when neither the environment nor
/// the harness credential file has it. Reading the file here (rather than at
/// startup) is what lets the user rotate a key without restarting the proxy.
pub fn resolve_credential(recipe: &Recipe) -> Option<String> {
    if recipe.credential_ref.as_deref() == Some("external-summary") {
        // The dedicated external-summary provider is env-only: the credential
        // file may still be used by legacy recipes, but never for this one.
        return std::env::var(EXTERNAL_SUMMARY_CREDENTIAL_ENV)
            .ok()
            .filter(|value| !value.trim().is_empty())
            .map(|value| value.trim().to_string());
    }
    if let Some(variable) = &recipe.api_key_env
        && let Some(value) = std::env::var(variable)
            .ok()
            .filter(|v| !v.trim().is_empty())
    {
        return Some(value.trim().to_string());
    }
    let name = recipe
        .api_key_env
        .as_deref()
        .or(recipe.credential_ref.as_deref())?;
    credential_from_file(name)
}

/// Minimal reader for the harness credential file. The `refs:` section stores
/// literal values (`NAME: "sk-..."`); a full YAML parse would be a dependency
/// for one shape, so this walks the lines and takes the first top-level match
/// whose value is long enough to be a key.
fn credential_from_file(name: &str) -> Option<String> {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .ok()?;
    let path = Path::new(&home).join(".dsh").join(".credentials.yaml");
    let text = std::fs::read_to_string(&path).ok()?;
    for line in text.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with('#') {
            continue;
        }
        let Some((key, value)) = trimmed.split_once(':') else {
            continue;
        };
        if key.trim() != name {
            continue;
        }
        let value = value.trim().trim_matches('"').trim_matches('\'');
        if value.is_empty() {
            continue;
        }
        return Some(value.to_string());
    }
    None
}

/// Ask the chain for one summary per input range.
///
/// Returns `None` when every target failed — the caller then keeps the model's
/// own summaries (fail-open) instead of dropping the fold.
pub async fn summarize(external: &ExternalSummary, inputs: &[RangeInput]) -> Option<SummaryBatch> {
    if !external.is_active() || inputs.is_empty() {
        return None;
    }
    let settings = &external.settings;
    let deadline =
        tokio::time::Instant::now() + Duration::from_millis(settings.total_timeout_ms.max(1_000));
    let mut client_targets = Vec::new();
    for target in &external.targets {
        let Some(key) = resolve_credential(&target.recipe) else {
            log_warn(&format!(
                "compress.externalSummary: provider {} has no credential in env or .credentials.yaml - skipped",
                target.recipe.name
            ));
            continue;
        };
        client_targets.push((target.clone(), key));
    }
    if client_targets.is_empty() {
        return None;
    }
    let per_target = Duration::from_millis(external.settings.target_timeout_ms.max(1_000));
    let client = match reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(per_target)
        .build()
    {
        Ok(client) => client,
        Err(err) => {
            log_warn(&format!("compress.externalSummary: no HTTP client: {err}"));
            return None;
        }
    };

    let mut batch = SummaryBatch::default();
    for input in inputs {
        if tokio::time::Instant::now() >= deadline {
            log_warn(
                "compress.externalSummary: total budget exhausted before all ranges were summarized",
            );
            return None;
        }
        let body = summary_request_body(
            input,
            &client_targets[0].0,
            settings.max_input_chars,
            settings.max_output_tokens,
        );
        let prompt_chars = body
            .get("messages")
            .and_then(|messages| messages.get(1))
            .and_then(|message| message.get("content"))
            .and_then(Value::as_str)
            .map(|content| content.chars().count() as u64)
            .unwrap_or(0);
        let mut produced = None;
        for (target, key) in &client_targets {
            batch.calls += 1;
            match dispatch(&client, target, key, &body).await {
                Ok((text, usage)) => {
                    if let Some(usage) = usage {
                        batch.input_tokens += usage.prompt_tokens;
                        batch.output_tokens += usage.completion_tokens;
                    } else {
                        batch.input_tokens += estimate_tokens(prompt_chars);
                        batch.output_tokens += estimate_tokens(text.chars().count() as u64);
                        batch.estimated_calls += 1;
                    }
                    let text = canonical_summary(&text, settings.max_summary_bytes);
                    if text.chars().count() >= MIN_SUMMARY_CHARS {
                        produced = Some(text);
                        break;
                    }
                    log_warn(&format!(
                        "compress.externalSummary: {} returned a summary too short to fold ({}) - next target",
                        target.model,
                        text.chars().count()
                    ));
                }
                Err(err) => {
                    // The dispatch burned a call even though it returned no text;
                    // charge the prompt as an estimate so the budget stays honest.
                    batch.input_tokens += estimate_tokens(prompt_chars);
                    batch.estimated_calls += 1;
                    log_warn(&format!(
                        "compress.externalSummary: {} for {} failed: {err} - next target",
                        target.recipe.name, target.model
                    ));
                }
            }
        }
        batch.summaries.push(produced?);
    }
    Some(batch)
}

/// Coarse char-to-token estimate for providers that omit `usage`. Four chars
/// per token is the conventional English approximation; summaries are mostly
/// English identifiers and prose, so this stays within the right order of
/// magnitude without pretending to be exact.
fn estimate_tokens(chars: u64) -> u64 {
    chars.div_ceil(4)
}

fn summary_request_body(
    input: &RangeInput,
    target: &ResolvedTarget,
    max_input_chars: usize,
    max_output_tokens: u64,
) -> Value {
    let slice = render_slice(&input.text, max_input_chars);
    let mut prompt = format!("Range: {}", input.label);
    if let Some(topic) = &input.topic
        && !topic.trim().is_empty()
    {
        prompt.push_str(&format!("\nTopic hint: {}", topic.trim()));
    }
    prompt.push_str("\n\n<conversation_slice>\n");
    prompt.push_str(&slice);
    prompt.push_str("\n</conversation_slice>");
    serde_json::json!({
        "model": target.model,
        "messages": [
            {"role": "system", "content": EXTERNAL_SUMMARY_INSTRUCTION},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": target.knobs.output_tokens.min(max_output_tokens),
        "temperature": 0.2,
        "stream": target.knobs.stream
    })
}

async fn dispatch(
    client: &reqwest::Client,
    target: &ResolvedTarget,
    key: &str,
    body: &Value,
) -> Result<(String, Option<Usage>), String> {
    let url = format!("{}/chat/completions", target.recipe.base_url);
    let mut request = client
        .post(&url)
        .header("content-type", "application/json")
        .header("authorization", format!("Bearer {key}"))
        .json(body);
    if target.knobs.stream {
        request = request.header("accept", "text/event-stream");
    }
    let response = request.send().await.map_err(|err| err.to_string())?;
    let status = response.status();
    if !status.is_success() {
        // Never echo the provider body: a hostile endpoint can reflect request
        // headers (including Authorization) into its error text, and the error
        // string is logged.
        return Err(format!("HTTP {status}"));
    }
    let text = read_capped_response(response, MAX_PROVIDER_RESPONSE_BYTES).await?;
    if target.knobs.stream {
        return stream_summary(&text).map(|text| (text, None));
    }
    let parsed: Value =
        serde_json::from_str(&text).map_err(|err| format!("invalid JSON: {err}"))?;
    let text =
        completion_text(&parsed).ok_or_else(|| "response carried no assistant text".to_string())?;
    Ok((text, completion_usage(&parsed)))
}

async fn read_capped_response(
    response: reqwest::Response,
    max_bytes: usize,
) -> Result<String, String> {
    let mut body = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|err| err.to_string())?;
        if body.len().saturating_add(chunk.len()) > max_bytes {
            return Err(format!("provider response exceeded {max_bytes} bytes"));
        }
        body.extend_from_slice(&chunk);
    }
    String::from_utf8(body).map_err(|err| format!("provider response is not UTF-8: {err}"))
}

/// Provider-reported token usage, when the response carries it. Some
/// OpenAI-compatible endpoints omit `usage` entirely, and prompts that set
/// `stream: true` never include it; callers then fall back to an estimate.
fn completion_usage(body: &Value) -> Option<Usage> {
    let usage = body.get("usage")?;
    let prompt_tokens = usage
        .get("prompt_tokens")
        .or_else(|| usage.get("input_tokens"))
        .and_then(Value::as_u64)?;
    let completion_tokens = usage
        .get("completion_tokens")
        .or_else(|| usage.get("output_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    Some(Usage {
        prompt_tokens,
        completion_tokens,
    })
}

/// OpenAI chat-completion text: `content` is a string for most providers and an
/// array of typed parts for the newer ones.
fn completion_text(body: &Value) -> Option<String> {
    let message = body
        .get("choices")
        .and_then(|choices| choices.get(0))
        .and_then(|choice| choice.get("message"))?;
    if let Some(text) = message.get("content").and_then(Value::as_str)
        && !text.trim().is_empty()
    {
        return Some(text.to_string());
    }
    message
        .get("content")
        .and_then(Value::as_array)
        .map(|parts| {
            parts
                .iter()
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("")
        })
        .filter(|text| !text.trim().is_empty())
}

fn stream_summary(body: &str) -> Result<String, String> {
    let mut out = String::new();
    for line in body.lines() {
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data.is_empty() || data == "[DONE]" {
            continue;
        }
        let Ok(frame) = serde_json::from_str::<Value>(data) else {
            continue;
        };
        if let Some(delta) = frame
            .get("choices")
            .and_then(|choices| choices.get(0))
            .and_then(|choice| choice.get("delta"))
            .and_then(|delta| delta.get("content"))
            .and_then(Value::as_str)
        {
            out.push_str(delta);
        }
    }
    if out.trim().is_empty() {
        Err("stream carried no content deltas".to_string())
    } else {
        Ok(out)
    }
}

/// Normalize what came back: strip a fenced-markdown wrapper a provider may add
/// and clamp to the configured byte budget on a character boundary.
pub(crate) fn canonical_summary(raw: &str, max_bytes: usize) -> String {
    let mut text = raw.trim().to_string();
    if text.starts_with("```") {
        let after_first_line = text.split_once('\n').map(|(_, rest)| rest.to_string());
        if let Some(inner) = after_first_line {
            let inner = inner.trim_end();
            if let Some(stripped) = inner.strip_suffix("```") {
                text = stripped.trim().to_string();
            }
        }
    }
    if text.len() > max_bytes {
        let mut cut = max_bytes;
        while cut > 0 && !text.is_char_boundary(cut) {
            cut -= 1;
        }
        text.truncate(cut);
    }
    text
}

fn render_slice(text: &str, cap: usize) -> String {
    if text.len() <= cap {
        return text.to_string();
    }
    // Keep the head (intent) and the tail (result) in equal parts; the elided
    // middle is the least likely part to carry anything the summary needs.
    let head = text.floor_char_boundary_approx(cap / 3);
    let tail_start = text.ceil_char_boundary_approx(text.len() - cap / 3);
    format!(
        "{}\n… [middle elided: {} chars] …\n{}",
        &text[..head],
        tail_start - head,
        &text[tail_start..]
    )
}

trait CharBoundary {
    fn floor_char_boundary_approx(&self, index: usize) -> usize;
    fn ceil_char_boundary_approx(&self, index: usize) -> usize;
}

impl CharBoundary for str {
    fn floor_char_boundary_approx(&self, index: usize) -> usize {
        let mut index = index.min(self.len());
        while index > 0 && !self.is_char_boundary(index) {
            index -= 1;
        }
        index
    }

    fn ceil_char_boundary_approx(&self, index: usize) -> usize {
        let mut index = index.min(self.len());
        while index < self.len() && !self.is_char_boundary(index) {
            index += 1;
        }
        index
    }
}

/// The host logs to stderr like the rest of the binary; a dedicated prefix keeps
/// summarizer lines greppable next to `dsh compaction call identified`.
fn log_warn(message: &str) {
    eprintln!("[billion-context-proxy] {message}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn oneagent_config(base_url: &str) -> Value {
        json!({
            "providers": {
                "oneagent": {
                    "baseUrl": base_url,
                    "api": "openai",
                    "apiKeyEnv": "ONEAGENT_MODEL_API_KEY",
                    "models": {
                        "deepseek-v4-1-flash-economy-261017": {
                            "contextWindow": 262144,
                            "outputTokens": 4096
                        }
                    }
                }
            },
            "compress": {
                "externalSummary": {
                    "enabled": true,
                    "targets": ["oneagent/deepseek-v4-1-flash-economy-261017"],
                    "budget": {"targetTimeoutMs": 15000, "totalTimeoutMs": 30000, "maxSummaryBytes": 8192}
                }
            }
        })
    }

    #[test]
    fn parses_named_recipe_and_resolves_the_target() {
        let parsed = from_config(&oneagent_config("https://api.gameaigc.cn/v1")).expect("chain");
        assert!(parsed.is_active());
        assert_eq!(parsed.settings.target_timeout_ms, 15_000);
        assert_eq!(parsed.settings.max_summary_bytes, 8192);
        let target = &parsed.targets[0];
        assert_eq!(target.recipe.base_url, "https://api.gameaigc.cn/v1");
        assert_eq!(target.model, "deepseek-v4-1-flash-economy-261017");
        assert_eq!(target.knobs.output_tokens, 4096);
        assert_eq!(target.knobs.context_window, 262144);
    }

    #[test]
    fn budget_knobs_are_clamped_and_defaulted() {
        let mut config = oneagent_config("https://api.gameaigc.cn/v1");
        config["compress"]["externalSummary"]["budget"] = json!({
            "maxInputChars": 900,
            "maxOutputTokens": 99,
            "maxCallsPerSession": 0,
            "maxSpeculationCalls": 0
        });
        let parsed = from_config(&config).expect("chain");
        assert_eq!(parsed.settings.max_input_chars, 1_000);
        assert_eq!(parsed.settings.max_output_tokens, 256);
        assert_eq!(parsed.settings.max_calls_per_session, 1);
        assert_eq!(parsed.settings.max_speculation_calls, 0);
        assert_eq!(parsed.settings.max_summary_bytes, DEFAULT_MAX_SUMMARY_BYTES);

        // The per-dispatch caps reach the request body: the model's own
        // outputTokens knob is clamped by maxOutputTokens, and the slice is
        // clipped to maxInputChars.
        let long = "x".repeat(5_000);
        let input = RangeInput {
            label: "m1–m9".to_string(),
            topic: None,
            text: long,
        };
        let body = summary_request_body(
            &input,
            &parsed.targets[0],
            parsed.settings.max_input_chars,
            parsed.settings.max_output_tokens,
        );
        assert_eq!(body["max_tokens"], 256);
        let prompt = body["messages"][1]["content"].as_str().expect("prompt");
        assert!(prompt.chars().count() < 1_200, "prompt was not clipped");

        // Defaults: with no budget block at all both session knobs come from
        // the documented defaults.
        let defaults = from_config(&oneagent_config("https://api.gameaigc.cn/v1")).expect("chain");
        assert_eq!(defaults.settings.max_input_chars, DEFAULT_MAX_INPUT_CHARS);
        assert_eq!(defaults.settings.max_output_tokens, DEFAULT_OUTPUT_TOKENS);
        assert_eq!(
            defaults.settings.max_calls_per_session,
            DEFAULT_MAX_CALLS_PER_SESSION
        );
        assert_eq!(
            defaults.settings.max_speculation_calls,
            DEFAULT_MAX_SPECULATION_CALLS
        );
    }

    #[test]
    fn absent_chain_is_not_an_error() {
        assert!(from_config(&json!({"compress": {"maxContextLimit": 0.68}})).is_none());
    }

    #[test]
    fn unresolved_target_disables_the_whole_chain() {
        let mut config = oneagent_config("https://api.gameaigc.cn/v1");
        config["compress"]["externalSummary"]["targets"] = json!(["oneagent/not-a-model"]);
        let parsed = from_config(&config).expect("chain");
        assert!(!parsed.is_active());
        assert!(parsed.targets.is_empty());
    }

    #[test]
    fn unsupported_protocol_disables_the_chain() {
        let mut config = oneagent_config("https://api.gameaigc.cn/v1");
        config["providers"]["oneagent"]["api"] = json!("anthropic");
        let parsed = from_config(&config).expect("chain");
        assert!(!parsed.is_active());
    }

    #[test]
    fn credential_falls_back_to_the_harness_credential_file() {
        let _guard =
            CredentialFileGuard::install("ONEAGENT_MODEL_API_KEY", "sk-test-credential-value");
        let parsed = from_config(&oneagent_config("https://api.gameaigc.cn/v1")).expect("chain");
        let key = resolve_credential(&parsed.targets[0].recipe).expect("credential");
        assert_eq!(key, "sk-test-credential-value");
    }

    #[test]
    fn canonical_summary_strips_a_fenced_wrapper() {
        assert_eq!(
            canonical_summary("```markdown\n- one fact\n```\n", 100),
            "- one fact"
        );
    }

    #[test]
    fn canonical_summary_clamps_to_the_budget() {
        let text = canonical_summary(&"字".repeat(20), 12);
        assert_eq!(text.chars().count(), 4);
    }

    #[tokio::test]
    async fn summarizes_against_an_openai_endpoint() {
        use axum::{Router, routing::post};
        use std::sync::{Arc, Mutex};

        let seen: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
        let probe = seen.clone();
        let app = Router::new().route(
            "/chat/completions",
            post(move |body: axum::Json<Value>| {
                let probe = probe.clone();
                async move {
                    probe.lock().expect("lock").push(body.0.clone());
                    axum::Json(json!({
                        "choices": [{
                            "index": 0,
                            "message": {"role": "assistant", "content": "- fold landed with the cheap model"},
                            "finish_reason": "stop"
                        }],
                        "usage": {"prompt_tokens": 120, "completion_tokens": 30}
                    }))
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let address = listener.local_addr().expect("addr");
        tokio::spawn(async move { axum::serve(listener, app).await.expect("serve") });

        let _guard =
            CredentialFileGuard::install("ONEAGENT_MODEL_API_KEY", "sk-test-credential-value");
        let parsed = from_config(&oneagent_config(&format!("http://{address}"))).expect("chain");
        let inputs = vec![RangeInput {
            label: "m00001–m00003".to_string(),
            topic: None,
            text: "[m00001] user: keep the JSON store\n[m00002] assistant: done".to_string(),
        }];
        let batch = summarize(&parsed, &inputs).await.expect("summary");
        assert_eq!(batch.summaries[0], "- fold landed with the cheap model");
        assert_eq!(batch.calls, 1);
        assert_eq!(batch.input_tokens, 120);
        assert_eq!(batch.output_tokens, 30);
        assert_eq!(batch.estimated_calls, 0);

        let bodies = seen.lock().expect("lock");
        assert_eq!(bodies.len(), 1);
        assert_eq!(bodies[0]["model"], "deepseek-v4-1-flash-economy-261017");
        let user = bodies[0]["messages"][1]["content"]
            .as_str()
            .expect("string");
        assert!(user.contains("m00001–m00003"));
        assert!(user.contains("keep the JSON store"));
    }

    #[tokio::test]
    async fn a_five_hundred_disables_the_dispatch() {
        use axum::{Router, routing::post};
        let app = Router::new().route(
            "/chat/completions",
            post(|| async { (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "boom") }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let address = listener.local_addr().expect("addr");
        tokio::spawn(async move { axum::serve(listener, app).await.expect("serve") });

        let _guard =
            CredentialFileGuard::install("ONEAGENT_MODEL_API_KEY", "sk-test-credential-value");
        let parsed = from_config(&oneagent_config(&format!("http://{address}"))).expect("chain");
        let inputs = vec![RangeInput {
            label: "m1–m2".to_string(),
            topic: None,
            text: "x".to_string(),
        }];
        assert!(summarize(&parsed, &inputs).await.is_none());
    }

    #[tokio::test]
    async fn a_redirect_is_not_followed() {
        use axum::{Router, http::StatusCode, routing::post};
        let app = Router::new().route(
            "/chat/completions",
            post(|| async {
                (
                    StatusCode::FOUND,
                    [(axum::http::header::LOCATION, "/elsewhere")],
                )
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let address = listener.local_addr().expect("addr");
        tokio::spawn(async move { axum::serve(listener, app).await.expect("serve") });

        let _guard =
            CredentialFileGuard::install("ONEAGENT_MODEL_API_KEY", "sk-test-credential-value");
        let parsed = from_config(&oneagent_config(&format!("http://{address}"))).expect("chain");
        let inputs = vec![RangeInput {
            label: "m1–m2".to_string(),
            topic: None,
            text: "x".to_string(),
        }];
        assert!(summarize(&parsed, &inputs).await.is_none());
    }

    #[tokio::test]
    async fn an_oversized_provider_response_is_refused_without_buffering() {
        use axum::{Router, routing::post};
        let app = Router::new().route(
            "/chat/completions",
            post(|| async { "x".repeat(MAX_PROVIDER_RESPONSE_BYTES + 1) }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let address = listener.local_addr().expect("addr");
        tokio::spawn(async move { axum::serve(listener, app).await.expect("serve") });

        let _guard =
            CredentialFileGuard::install("ONEAGENT_MODEL_API_KEY", "sk-test-credential-value");
        let parsed = from_config(&oneagent_config(&format!("http://{address}"))).expect("chain");
        let inputs = vec![RangeInput {
            label: "m1–m2".to_string(),
            topic: None,
            text: "x".to_string(),
        }];
        assert!(summarize(&parsed, &inputs).await.is_none());
    }

    /// Points `USERPROFILE` at a scratch directory holding the credential file,
    /// so the fallback is exercised without touching the real one.
    ///
    /// Holds the crate-wide environment lock for its whole lifetime. The
    /// mutation is `USERPROFILE`, but `d1b_runner`'s tests write
    /// `ONEAGENT_MODEL_API_KEY`; both are process-global, so a module-private
    /// mutex let one test's restore land inside another test's window. The lock
    /// is taken first and released last, which makes the pair atomic.
    struct CredentialFileGuard {
        dir: std::path::PathBuf,
        previous: crate::test_env::Prior,
        _env: std::sync::MutexGuard<'static, ()>,
    }

    impl CredentialFileGuard {
        fn install(name: &str, secret: &str) -> Self {
            // SAFETY: the guard holds the crate-wide environment lock, so no
            // other test can read or write the environment while it lives.
            let env = crate::test_env::lock();
            let dir = std::env::temp_dir().join(format!(
                "bili-summarizer-cred-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0)
            ));
            std::fs::create_dir_all(dir.join(".dsh")).expect("scratch dir");
            std::fs::write(
                dir.join(".dsh").join(".credentials.yaml"),
                format!("version: 1\nrefs:\n  {name}: \"{secret}\"\nrecords: {{}}\n"),
            )
            .expect("credential file");
            let previous = crate::test_env::capture("USERPROFILE");
            unsafe { crate::test_env::set("USERPROFILE", &dir) };
            Self {
                dir,
                previous,
                _env: env,
            }
        }
    }

    impl Drop for CredentialFileGuard {
        fn drop(&mut self) {
            // SAFETY: still holding the environment lock; released after this.
            unsafe { crate::test_env::restore_taken("USERPROFILE", self.previous.take()) };
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }
}

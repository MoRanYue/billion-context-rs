//! Minimal D1b experiment runner.
//!
//! This is deliberately a separate, single-shot path: it does not start the
//! proxy, does not touch the plugin/auto-fallback paths, and never retries.
//! One `execute` call can dispatch at most one upstream request.  It writes the
//! same usage ledger the host writes, then audits that ledger before calling
//! the run complete.
//!
//! Fail-closed rules:
//! * no credential, budget refusal or oversized input -> REFUSED, no dispatch;
//! * unreadable/append-failed ledger -> INVALID, no dispatch (or stop);
//! * transport error, timeout, non-2xx, missing usage, malformed response,
//!   output above the cap, missing finish or any count mismatch -> INVALID.
//!
//! Cost is only derived from tokens when a price for this exact model, on this
//! exact provider, was published by the local `oneagent-pricing` plugin and is
//! still within its staleness window.  Anything else leaves `price_known=false`
//! and the cap status `checked=false`.
//!
//! Monetary cap contract (P4-2T / F15+F16):
//! * configuring `--d1b-cost-cap` implies requiring a usable price — a cap
//!   number alone is not a cap, so a run whose cost cannot be bounded is
//!   refused before the call budget is spent or a socket is opened;
//! * `cost_cap_configured` / `cost_cap_checked` / `cost_cap_passed` /
//!   `cap_enforced` are distinct facts, so a report never claims a check that
//!   did not happen;
//! * `cap_enforced=true` only means the local gate executed the configured
//!   monetary policy.  It does not promise the provider's invoice, does not
//!   survive token-estimation error, and is not a currency-settlement bound.

use crate::pricing::{self, CostBasis, Refusal};
use crate::prune::estimate_tokens;
use crate::summarizer::Recipe;
use crate::usage::UsageFields;
use crate::usage_ledger::{self, Finish, LedgerAudit, RequestRecord, RequestStatus, UsageSource};
use reqwest::header::HeaderMap;
use serde::Serialize;
use serde_json::{Value, json};
use std::path::PathBuf;

#[cfg(test)]
use std::path::Path;
use std::time::Duration;
use tokio::time::timeout;

const DEFAULT_MAX_CALLS: u64 = 1;
const DEFAULT_MAX_INPUT_TOKENS: u64 = 32_000;
const DEFAULT_MAX_OUTPUT_TOKENS: u64 = 4_096;
const DEFAULT_WALL_CLOCK_MS: u64 = 30_000;
const DEFAULT_PROMPT: &str = "Reply with exactly: D1B-OK";

/// The only environment variable D1b may read for the main experiment model.
pub const ONEAGENT_CREDENTIAL_ENV: &str = "ONEAGENT_MODEL_API_KEY";
/// The only environment variable D1b may read for the external-summary provider.
pub const ONEAGENT_SUMMARY_CREDENTIAL_ENV: &str =
    crate::summarizer::EXTERNAL_SUMMARY_CREDENTIAL_ENV;
const EXTERNAL_SUMMARY_CREDENTIAL_REF: &str = "external-summary";

/// Fixed provider-to-credential-environment allowlist for D1b.  Provider config
/// may describe these entries, but it cannot choose an arbitrary variable.
pub fn allowed_credential_env(provider: &str) -> Option<&'static str> {
    match provider {
        "oneagent" => Some(ONEAGENT_CREDENTIAL_ENV),
        "oneagent-summary" => Some(ONEAGENT_SUMMARY_CREDENTIAL_ENV),
        _ => None,
    }
}

/// Validate the configured recipe against the fixed allowlist.  Returning
/// `None` means the provider is unknown, the mapping is mismatched, or a mapped
/// value is empty; the caller must refuse rather than fall back to another
/// credential source.
pub fn provider_binding_from_recipe(provider: &str, recipe: &Recipe) -> Option<&'static str> {
    let expected = allowed_credential_env(provider)?;
    let exact = match provider {
        "oneagent" => {
            recipe.credential_ref.is_none()
                && recipe.api_key_env.as_deref() == Some(ONEAGENT_CREDENTIAL_ENV)
        }
        "oneagent-summary" => {
            recipe.api_key_env.is_none()
                && recipe.credential_ref.as_deref() == Some(EXTERNAL_SUMMARY_CREDENTIAL_REF)
        }
        _ => false,
    };
    exact.then_some(expected)
}

#[derive(Clone, Debug)]
pub struct Limits {
    pub max_calls: u64,
    pub max_input_tokens: u64,
    pub max_output_tokens: u64,
    pub wall_clock_ms: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_calls: DEFAULT_MAX_CALLS,
            max_input_tokens: DEFAULT_MAX_INPUT_TOKENS,
            max_output_tokens: DEFAULT_MAX_OUTPUT_TOKENS,
            wall_clock_ms: DEFAULT_WALL_CLOCK_MS,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RunOptions {
    /// Full chat-completions URL or a base URL; `/v1/chat/completions` is
    /// appended when it is missing.
    pub upstream: String,
    pub model: String,
    pub prompt: String,
    pub session_id: String,
    pub limits: Limits,
    pub cost_cap: Option<f64>,
    pub cost_currency: String,
    /// Refuse the run unless a monetary cap can actually be enforced. Without
    /// this, a priced engine still runs when the price is unknown — but it then
    /// reports `price_known=false` and claims nothing about cost.
    ///
    /// Setting `cost_cap` enables this automatically: see `cap_policy_required`.
    pub require_cost_cap: bool,
    /// Named provider entry used to bind the credential environment variable.
    pub provider: Option<String>,
    /// Environment variable selected from the fixed allowlist; empty means no
    /// provider binding was resolvable and the run must be refused.  This field
    /// is crate-private so a library caller cannot inject an arbitrary name;
    /// use `bind_credential` instead.
    pub(crate) credential_env: String,
    /// Set by the host; tests use a temp directory.
    pub sessions_dir: PathBuf,
}

impl RunOptions {
    pub fn new(
        upstream: impl Into<String>,
        model: impl Into<String>,
        sessions_dir: PathBuf,
    ) -> Self {
        Self {
            upstream: upstream.into(),
            model: model.into(),
            prompt: DEFAULT_PROMPT.to_string(),
            session_id: format!("d1b-{}", uuid::Uuid::new_v4()),
            limits: Limits::default(),
            cost_cap: None,
            cost_currency: "USD".to_string(),
            require_cost_cap: false,
            provider: None,
            credential_env: String::new(),
            sessions_dir,
        }
    }

    /// Bind the provider and credential environment together from the fixed
    /// allowlist.  Unknown providers are rejected without setting a binding.
    pub fn bind_credential(&mut self, provider: &str) -> Result<(), &'static str> {
        self.provider = None;
        self.credential_env.clear();
        let env_name = allowed_credential_env(provider).ok_or("unknown_provider")?;
        self.provider = Some(provider.to_string());
        self.credential_env = env_name.to_string();
        Ok(())
    }

    /// Whether this run's price must be usable before dispatch.
    ///
    /// A configured monetary cap is never a decorative number: switching one on
    /// necessarily switches on the price requirement (F15).  Refusing here keeps
    /// `--d1b-cost-cap` from meaning "a limit that is only enforced when it
    /// happens to be enforceable".
    pub fn cap_policy_required(&self) -> bool {
        self.cost_cap.is_some() || self.require_cost_cap
    }

    /// A cap must be a finite, non-negative amount.  NaN would make every
    /// comparison false (silently refusing everything), an infinity would make
    /// every comparison true (silently disabling the cap), and a negative number
    /// is not an amount anyone can spend.  `0` is allowed: it is a legitimate
    /// "spend nothing" cap that only free runs can pass.
    pub fn invalid_cost_cap(&self) -> Option<f64> {
        match self.cost_cap {
            Some(cap) if !cap.is_finite() || cap < 0.0 => Some(cap),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum RunStatus {
    Complete,
    Refused,
    Invalid,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum CredentialStatus {
    Present,
    Missing,
}

#[derive(Clone, Debug, Serialize)]
pub struct CostReport {
    pub currency: String,
    pub manual_cap: Option<f64>,
    /// The local gate executed the configured monetary policy: it either let the
    /// projection through or refused the run because of the cap.  This says
    /// nothing about the provider's final invoice, about token-estimation error,
    /// or about any currency settlement.
    pub cap_enforced: bool,
    /// Whether this run configured a monetary cap at all.
    pub cost_cap_configured: bool,
    /// Whether `estimated_cost <= cap` was actually compared.  Remains false
    /// whenever no price was usable, so a missing check is never mistaken for a
    /// passed one.
    pub cost_cap_checked: bool,
    /// The comparison result when it happened; `null` when it did not.  A
    /// boolean would collapse "not checked" into "failed", so the JSON keeps the
    /// key present with a null instead of omitting it.
    pub cost_cap_passed: Option<bool>,
    pub price_known: bool,
    /// What the price was based on, or exactly which check refused it.
    pub basis: CostBasis,
    /// Set only when a cost was actually computed from verified rates.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimated_cost: Option<f64>,
    pub note: &'static str,
}

/// The monetary gate's own record of what it did, kept next to the report fields
/// it fills so the four flags cannot drift apart.
#[derive(Clone, Copy, Debug)]
struct CapStatus {
    configured: bool,
    checked: bool,
    passed: Option<bool>,
    enforced: bool,
}

impl CapStatus {
    /// The gate never ran (credential/budget/input refusal): nothing was checked.
    fn idle(opts: &RunOptions) -> Self {
        Self {
            configured: opts.cost_cap.is_some(),
            checked: false,
            passed: None,
            enforced: false,
        }
    }

    /// The gate ran and refused before a comparison could be made (no usable
    /// price, unusable currency, unusable arithmetic).  Only a configured cap
    /// makes this an enforcement of the monetary policy.
    fn refused(opts: &RunOptions) -> Self {
        Self {
            configured: opts.cost_cap.is_some(),
            checked: false,
            passed: None,
            enforced: opts.cost_cap.is_some(),
        }
    }

    /// The gate compared a projected cost against the cap.
    fn compared(opts: &RunOptions, passed: bool) -> Self {
        Self {
            configured: opts.cost_cap.is_some(),
            checked: opts.cost_cap.is_some(),
            passed: Some(passed),
            enforced: opts.cost_cap.is_some(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct RunReport {
    pub run_id: String,
    pub status: RunStatus,
    pub reason: String,
    pub expected_calls: u64,
    pub actual_calls: u64,
    pub provider_calls: u64,
    pub retries: u64,
    pub max_calls: u64,
    pub max_input_tokens: u64,
    pub max_output_tokens: u64,
    pub wall_clock_ms: u64,
    pub input_tokens_estimated: u64,
    pub output_tokens_reported: Option<u64>,
    pub http_status: Option<u16>,
    pub provider_usage_present: bool,
    pub session_id: String,
    pub ledger_path: String,
    pub ledger_lines: usize,
    pub ledger_starts: usize,
    pub ledger_finishes: usize,
    pub audit_requests: usize,
    pub audit_finished: usize,
    pub ledger_complete: bool,
    pub counts_consistent: bool,
    pub invalid_reasons: Vec<String>,
    pub duration_ms: u64,
    pub cost: CostReport,
}

impl RunReport {
    fn refused(reason: &str, opts: &RunOptions, run_id: String, duration_ms: u64) -> Self {
        Self {
            run_id,
            status: RunStatus::Refused,
            reason: reason.to_string(),
            expected_calls: 1,
            actual_calls: 0,
            provider_calls: 0,
            retries: 0,
            max_calls: opts.limits.max_calls,
            max_input_tokens: opts.limits.max_input_tokens,
            max_output_tokens: opts.limits.max_output_tokens,
            wall_clock_ms: opts.limits.wall_clock_ms,
            input_tokens_estimated: 0,
            output_tokens_reported: None,
            http_status: None,
            provider_usage_present: false,
            session_id: opts.session_id.clone(),
            ledger_path: String::new(),
            ledger_lines: 0,
            ledger_starts: 0,
            ledger_finishes: 0,
            audit_requests: 0,
            audit_finished: 0,
            ledger_complete: false,
            counts_consistent: false,
            invalid_reasons: Vec::new(),
            duration_ms,
            cost: cost_report(
                opts,
                CostBasis::unknown(&Refusal::NotEvaluated),
                CapStatus::idle(opts),
            ),
        }
    }
}

fn cost_report(opts: &RunOptions, basis: CostBasis, cap: CapStatus) -> CostReport {
    let price_known = basis.is_known();
    CostReport {
        currency: opts.cost_currency.clone(),
        manual_cap: opts.cost_cap,
        cap_enforced: cap.enforced,
        cost_cap_configured: cap.configured,
        cost_cap_checked: cap.checked,
        cost_cap_passed: cap.passed,
        price_known,
        basis,
        estimated_cost: None,
        note: cap_note(cap, price_known),
    }
}

impl CostReport {
    /// Single place that keeps the four cap flags mutually consistent: they all
    /// come from one `CapStatus`, so `checked=false && passed=true` and
    /// `configured=false && passed=true` are unrepresentable.
    fn set_cap_status(&mut self, cap: CapStatus) {
        self.cap_enforced = cap.enforced;
        self.cost_cap_configured = cap.configured;
        self.cost_cap_checked = cap.checked;
        self.cost_cap_passed = cap.passed;
        self.note = cap_note(cap, self.price_known);
    }
}

/// What the report may say about money for this run.  The note follows the cap
/// status, not the price alone, so "checked and passed" is never confused with
/// "nobody looked".
fn cap_note(cap: CapStatus, price_known: bool) -> &'static str {
    match (cap.configured, cap.checked, cap.passed) {
        (false, _, _) if price_known => {
            "a price was published for this model on this provider and a projection was computed; \
             no monetary cap was configured, so token caps bound this run"
        }
        (false, _, _) => {
            "no verified price for this model on this provider; token caps are not a monetary cap and no cost is claimed"
        }
        (true, true, Some(true)) => {
            "a projected cost was compared against the configured monetary cap and stayed within it; \
             the projection is not the provider's invoice"
        }
        (true, true, Some(false)) => {
            "the projected cost exceeds the configured monetary cap; the run was refused before dispatch"
        }
        (true, _, _) => {
            "a monetary cap was configured but no usable price was available to check it against; \
             the run was refused before dispatch"
        }
    }
}

/// The monetary decision for one projected request, taken before dispatch.
///
/// Returning `Err(reason)` means the run must be refused: a required cap that
/// cannot be computed is not a cap, and proceeding would let the report imply a
/// bound that does not exist.
fn cost_gate(
    opts: &RunOptions,
    input_tokens: u64,
    output_tokens: u64,
    now: u64,
) -> (CostReport, Result<Option<f64>, &'static str>) {
    let resolved = pricing::resolve_quote(&opts.model, &opts.upstream, input_tokens, now);
    let (basis, quote) = match resolved {
        Ok(quote) => (CostBasis::known(&quote), Some(quote)),
        Err(refusal) => (CostBasis::unknown(&refusal), None),
    };

    let Some(quote) = quote else {
        // No verified price. A run that demands a monetary bound cannot go on.
        // Since F15 that demand follows from the cap itself, not only from the
        // explicit `--d1b-require-cost-cap` switch.
        let report = cost_report(opts, basis, CapStatus::refused(opts));
        return (
            report,
            if opts.cap_policy_required() {
                Err("cost_cap_unavailable")
            } else {
                Ok(None)
            },
        );
    };

    // The snapshot keeps the first child group's rates for a model that more
    // than one group lists. Those rates are real, but nothing proves they are
    // what this request is billed at, so they never become this run's cost.
    if quote.group_ambiguous {
        let refusal = Refusal::AmbiguousGroup {
            model: quote.model.clone(),
            group_id: quote.group_id,
        };
        let report = cost_report(opts, CostBasis::unknown(&refusal), CapStatus::refused(opts));
        return (
            report,
            if opts.cap_policy_required() {
                Err("cost_group_ambiguous")
            } else {
                Ok(None)
            },
        );
    }

    // A cap denominated in another currency cannot be compared without a rate.
    if opts.cost_cap.is_some() && pricing::assert_cap_currency(&opts.cost_currency, &quote).is_err()
    {
        let report = cost_report(opts, basis, CapStatus::refused(opts));
        return (report, Err("cost_cap_currency_mismatch"));
    }

    let projected = match quote.cost(input_tokens, 0, output_tokens) {
        Ok(cost) => cost,
        Err(_) => {
            let report = cost_report(opts, basis, CapStatus::refused(opts));
            return (report, Err("cost_arithmetic_invalid"));
        }
    };
    let mut report = cost_report(opts, basis, CapStatus::refused(opts));
    report.estimated_cost = Some(projected.amount);
    report.currency = quote.currency.clone();

    // A cap is inclusive: a projection exactly at the cap passes.  `0` passes
    // only a projection that is also `0`, and the all-zero rate card is already
    // refused upstream (F8), so no "free" model can ride a zero cap.
    let (verdict, status) = match opts.cost_cap {
        Some(cap) if !pricing::within_cap(projected.amount, cap) => {
            (Err("cost_cap_exceeded"), CapStatus::compared(opts, false))
        }
        Some(_) => (Ok(Some(projected.amount)), CapStatus::compared(opts, true)),
        None if opts.cap_policy_required() => (Err("cost_cap_missing"), CapStatus::idle(opts)),
        None => (Ok(Some(projected.amount)), CapStatus::idle(opts)),
    };
    report.set_cap_status(status);
    (report, verdict)
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn chat_url(upstream: &str) -> String {
    let trimmed = upstream.trim().trim_end_matches('/');
    if trimmed.ends_with("/v1/chat/completions") {
        trimmed.to_string()
    } else {
        format!("{trimmed}/v1/chat/completions")
    }
}

fn credential_status_with<F>(env_name: &str, lookup: F) -> CredentialStatus
where
    F: Fn(&str) -> Option<String>,
{
    if env_name.is_empty() {
        return CredentialStatus::Missing;
    }
    match lookup(env_name) {
        Some(value) if !value.trim().is_empty() => CredentialStatus::Present,
        _ => CredentialStatus::Missing,
    }
}

/// Presence check only.  The value is never returned, logged or copied.
pub fn credential_status(env_name: &str) -> CredentialStatus {
    credential_status_with(env_name, |name| std::env::var(name).ok())
}

#[derive(Clone, Copy, Debug, Default)]
struct CallGate {
    max: u64,
    used: u64,
}

impl CallGate {
    fn new(max: u64) -> Self {
        Self { max, used: 0 }
    }

    fn try_reserve(&mut self) -> bool {
        if self.used >= self.max {
            return false;
        }
        self.used = self.used.saturating_add(1);
        true
    }
}

pub(crate) struct Runner {
    opts: RunOptions,
    gate: CallGate,
    client: reqwest::Client,
}

impl Runner {
    pub(crate) fn new(opts: RunOptions) -> Result<Self, String> {
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .no_proxy()
            .timeout(Duration::from_millis(opts.limits.wall_clock_ms.max(1)))
            .build()
            .map_err(|err| format!("failed to build HTTP client: {err}"))?;
        Ok(Self {
            gate: CallGate::new(opts.limits.max_calls),
            opts,
            client,
        })
    }

    pub(crate) async fn execute(&mut self, api_key: Option<String>) -> RunReport {
        let run_id = uuid::Uuid::new_v4().to_string();
        let started = std::time::Instant::now();
        let opts = self.opts.clone();

        // A cap that cannot be read as an amount is a configuration error, not a
        // provider run.  Checked before the credential gate so the operator sees
        // the real problem even when the key is also missing.
        if opts.invalid_cost_cap().is_some() {
            return RunReport::refused(
                "invalid_cost_cap",
                &opts,
                run_id,
                started.elapsed().as_millis() as u64,
            );
        }

        let Some(key) = api_key.filter(|k| !k.trim().is_empty()) else {
            return RunReport::refused(
                "credential_missing",
                &opts,
                run_id,
                started.elapsed().as_millis() as u64,
            );
        };

        if opts.limits.max_calls == 0 {
            return RunReport::refused(
                "call_budget_zero",
                &opts,
                run_id,
                started.elapsed().as_millis() as u64,
            );
        }

        let body = json!({
            "model": opts.model,
            "messages": [{ "role": "user", "content": opts.prompt }],
            "max_tokens": opts.limits.max_output_tokens,
            "stream": false,
        });
        let body_bytes = match serde_json::to_vec(&body) {
            Ok(bytes) => bytes,
            Err(_) => {
                return RunReport::refused(
                    "request_body_unserializable",
                    &opts,
                    run_id,
                    started.elapsed().as_millis() as u64,
                );
            }
        };
        let input_estimate = estimate_tokens(&String::from_utf8_lossy(&body_bytes)) as u64 + 16;
        if input_estimate > opts.limits.max_input_tokens {
            return RunReport::refused(
                "input_token_cap",
                &opts,
                run_id,
                started.elapsed().as_millis() as u64,
            );
        }

        // The monetary gate runs before the call budget is spent and before any
        // socket is opened: a run whose cost cannot be bounded must not happen.
        let (cost, cost_verdict) = cost_gate(
            &opts,
            input_estimate,
            opts.limits.max_output_tokens,
            now_ms(),
        );
        if let Err(reason) = cost_verdict {
            let mut refused =
                RunReport::refused(reason, &opts, run_id, started.elapsed().as_millis() as u64);
            refused.cost = cost;
            refused.input_tokens_estimated = input_estimate;
            return refused;
        }

        if !self.gate.try_reserve() {
            // The price was already resolved and the projection computed; the
            // budget, not the price, is what stopped this run. Rebuilding the
            // report with a fresh "not evaluated" basis would claim we never
            // looked, so the numbers stay attached to the refusal.
            let mut refused = RunReport::refused(
                "call_budget_exhausted",
                &opts,
                run_id,
                started.elapsed().as_millis() as u64,
            );
            refused.cost = cost;
            refused.input_tokens_estimated = input_estimate;
            return refused;
        }

        let mut report = RunReport::refused("", &opts, run_id.clone(), 0);
        report.status = RunStatus::Invalid;
        report.reason = "invalid".to_string();
        report.input_tokens_estimated = input_estimate;
        report.actual_calls = 1;
        report.cost = cost;

        let request_hash = usage_ledger::sha256_hex(&body_bytes);
        let path = usage_ledger::ledger_path(&opts.sessions_dir.join("requests"), &opts.session_id);
        report.ledger_path = path.to_string_lossy().to_string();
        let record = RequestRecord::new(
            path.clone(),
            &opts.session_id,
            "openai",
            Some(opts.model.clone()),
            &opts.upstream,
            UsageSource::Provider,
            "d1b_runner_explicit_provider",
            now_ms(),
            request_hash,
            body_bytes.len() as u64,
            false,
            None,
            &HeaderMap::new(),
        );
        if record.start().is_err() {
            report
                .invalid_reasons
                .push("ledger_start_failed".to_string());
            report.duration_ms = started.elapsed().as_millis() as u64;
            return report;
        }

        let url = chat_url(&opts.upstream);
        let wall = opts.limits.wall_clock_ms.max(1);
        let call = async {
            let response = self
                .client
                .post(&url)
                .bearer_auth(&key)
                .json(&body)
                .send()
                .await
                .map_err(DispatchError::Transport)?;
            let status = response.status().as_u16();
            let text = response.text().await.map_err(DispatchError::Transport)?;
            Ok::<(u16, String), DispatchError>((status, text))
        };
        let dispatched = timeout(Duration::from_millis(wall), call).await;
        report.provider_calls = 1;

        let mut response_text = None;
        let finish = match dispatched {
            Err(_) => {
                report
                    .invalid_reasons
                    .push("wall_clock_timeout".to_string());
                Finish {
                    status: RequestStatus::Timeout,
                    http_status: None,
                    error_class: Some("timeout"),
                    response_hash: None,
                    response_bytes: None,
                    response_complete: false,
                    response_body_inspected: false,
                    usage: UsageFields::default(),
                    usage_from_complete_stream: false,
                }
            }
            Ok(Err(err)) => {
                report
                    .invalid_reasons
                    .push(format!("transport_error:{}", err.class()));
                report.provider_calls = 0;
                Finish::transport(err.reqwest())
            }
            Ok(Ok((status, text))) => {
                report.http_status = Some(status);
                response_text = Some(text.clone());
                if !(200..300).contains(&status) {
                    report.invalid_reasons.push(format!("http_status:{status}"));
                    Finish::http_error(status)
                } else {
                    match parse_usage(&text) {
                        Some(usage) => {
                            report.provider_usage_present = true;
                            report.output_tokens_reported = usage.output;
                            if usage.input.is_none() || usage.output.is_none() {
                                report.invalid_reasons.push("usage_partial".to_string());
                            }
                            Finish::success(
                                usage,
                                usage_ledger::sha256_hex(text.as_bytes()),
                                text.len() as u64,
                            )
                        }
                        None => {
                            report.invalid_reasons.push("usage_missing".to_string());
                            report.provider_usage_present = false;
                            Finish::success(
                                UsageFields::default(),
                                usage_ledger::sha256_hex(text.as_bytes()),
                                text.len() as u64,
                            )
                        }
                    }
                }
            }
        };

        if let Some(usage) = response_text.as_deref().and_then(parse_usage) {
            if let Some(output) = usage.output
                && output > opts.limits.max_output_tokens
            {
                report.invalid_reasons.push(format!(
                    "output_cap_exceeded:{output}>{}",
                    opts.limits.max_output_tokens
                ));
            }
            if let Some(input) = usage.input
                && input > opts.limits.max_input_tokens
            {
                report.invalid_reasons.push(format!(
                    "input_cap_exceeded:{input}>{}",
                    opts.limits.max_input_tokens
                ));
            }
        }

        if record.finish(finish).is_err() {
            report
                .invalid_reasons
                .push("ledger_finish_failed".to_string());
        }

        let audit = usage_ledger::audit_ledger(&path);
        report.ledger_lines = audit.lines;
        report.ledger_starts = audit.starts;
        report.ledger_finishes = audit.finishes;
        report.audit_requests = audit.requests;
        report.audit_finished = audit.finished;
        report.ledger_complete = audit.complete();

        let mut mismatches = count_mismatches(
            report.expected_calls,
            report.actual_calls,
            report.ledger_starts,
            report.ledger_finishes,
            &audit,
        );
        report.counts_consistent = mismatches.is_empty();
        report.invalid_reasons.append(&mut mismatches);

        if report.invalid_reasons.is_empty() {
            report.status = RunStatus::Complete;
            report.reason = "complete".to_string();
        } else {
            report.status = RunStatus::Invalid;
            report.reason = "invalid".to_string();
        }
        report.duration_ms = started.elapsed().as_millis() as u64;
        report
    }
}

fn count_mismatches(
    expected: u64,
    actual: u64,
    starts: usize,
    finishes: usize,
    audit: &LedgerAudit,
) -> Vec<String> {
    let mut problems = Vec::new();
    if actual != expected {
        problems.push(format!("count_mismatch:calls:{actual}!={expected}"));
    }
    if starts as u64 != expected {
        problems.push(format!("count_mismatch:ledger_start:{starts}!={expected}"));
    }
    if finishes as u64 != expected {
        problems.push(format!(
            "count_mismatch:ledger_finish:{finishes}!={expected}"
        ));
    }
    if !audit.issues.is_empty() {
        let issue = audit
            .issues
            .first()
            .map(|i| i.as_str())
            .unwrap_or("invalid");
        problems.push(format!("ledger_invalid:{issue}"));
    }
    if audit.requests != expected as usize {
        problems.push(format!(
            "count_mismatch:audit_requests:{}!={expected}",
            audit.requests
        ));
    }
    if audit.finished != expected as usize {
        problems.push(format!(
            "count_mismatch:audit_finished:{}!={expected}",
            audit.finished
        ));
    }
    if !audit.complete() {
        problems.push("ledger_incomplete".to_string());
    }
    problems
}

enum DispatchError {
    Transport(reqwest::Error),
}

impl DispatchError {
    fn class(&self) -> &'static str {
        match self {
            Self::Transport(err) => usage_ledger::error_class(err),
        }
    }

    fn reqwest(&self) -> &reqwest::Error {
        match self {
            Self::Transport(err) => err,
        }
    }
}

fn parse_usage(text: &str) -> Option<UsageFields> {
    let value: Value = serde_json::from_str(text).ok()?;
    let usage = value.get("usage")?;
    let input = usage.get("prompt_tokens").and_then(Value::as_u64);
    let output = usage.get("completion_tokens").and_then(Value::as_u64);
    let cache_read = usage
        .get("prompt_tokens_details")
        .and_then(|details| details.get("cached_tokens"))
        .and_then(Value::as_u64);
    Some(UsageFields {
        input,
        cache_read,
        cache_creation: None,
        output,
        protocol: Some(crate::usage::UsageProtocol::Openai),
        object_present: true,
    })
}

pub(crate) async fn run_with_key(opts: RunOptions, api_key: Option<String>) -> RunReport {
    match Runner::new(opts.clone()) {
        Ok(mut runner) => runner.execute(api_key).await,
        Err(err) => {
            // A client build failure is an operator error, not a provider run.
            // Keep the same fail-closed shape without a dispatch field.
            let _ = err;
            RunReport {
                run_id: uuid::Uuid::new_v4().to_string(),
                status: RunStatus::Refused,
                reason: "client_build_failed".to_string(),
                expected_calls: 1,
                actual_calls: 0,
                provider_calls: 0,
                retries: 0,
                max_calls: opts.limits.max_calls,
                max_input_tokens: opts.limits.max_input_tokens,
                max_output_tokens: opts.limits.max_output_tokens,
                wall_clock_ms: opts.limits.wall_clock_ms,
                input_tokens_estimated: 0,
                output_tokens_reported: None,
                http_status: None,
                provider_usage_present: false,
                session_id: opts.session_id.clone(),
                ledger_path: String::new(),
                ledger_lines: 0,
                ledger_starts: 0,
                ledger_finishes: 0,
                audit_requests: 0,
                audit_finished: 0,
                ledger_complete: false,
                counts_consistent: false,
                invalid_reasons: Vec::new(),
                duration_ms: 0,
                cost: cost_report(
                    &opts,
                    CostBasis::unknown(&Refusal::NotEvaluated),
                    CapStatus::idle(&opts),
                ),
            }
        }
    }
}

pub async fn run(opts: RunOptions) -> RunReport {
    let Some(provider) = opts.provider.as_deref() else {
        return RunReport::refused(
            "credential_binding_missing",
            &opts,
            uuid::Uuid::new_v4().to_string(),
            0,
        );
    };
    if provider.trim().is_empty() {
        return RunReport::refused(
            "credential_binding_missing",
            &opts,
            uuid::Uuid::new_v4().to_string(),
            0,
        );
    }
    let Some(expected_env) = allowed_credential_env(provider) else {
        return RunReport::refused(
            "unknown_provider",
            &opts,
            uuid::Uuid::new_v4().to_string(),
            0,
        );
    };
    if opts.credential_env.is_empty() {
        return RunReport::refused(
            "credential_binding_missing",
            &opts,
            uuid::Uuid::new_v4().to_string(),
            0,
        );
    }
    if opts.credential_env != expected_env {
        return RunReport::refused(
            "credential_binding_mismatch",
            &opts,
            uuid::Uuid::new_v4().to_string(),
            0,
        );
    }
    let api_key = std::env::var(expected_env)
        .ok()
        .filter(|value| !value.trim().is_empty());
    run_with_key(opts, api_key).await
}

/// Exit code contract for the CLI path.
pub fn exit_code(report: &RunReport) -> i32 {
    match report.status {
        RunStatus::Complete => 0,
        RunStatus::Invalid => 2,
        RunStatus::Refused => 3,
    }
}

pub fn parse_args(args: &[String]) -> Result<RunOptions, String> {
    let mut opts = RunOptions::new("", "", PathBuf::new());
    let mut upstream = None;
    let mut model = None;
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        let value = args.get(i + 1).cloned();
        match arg {
            "--d1b-upstream" => {
                upstream = value;
                i += 1;
            }
            "--d1b-model" => {
                model = value;
                i += 1;
            }
            "--d1b-provider" => {
                if let Some(v) = value {
                    opts.provider = Some(v);
                }
                i += 1;
            }
            "--d1b-prompt" => {
                if let Some(v) = value {
                    opts.prompt = v;
                }
                i += 1;
            }
            "--d1b-session" => {
                if let Some(v) = value {
                    opts.session_id = v;
                }
                i += 1;
            }
            "--d1b-max-calls" => {
                opts.limits.max_calls = value
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(|| "invalid --d1b-max-calls".to_string())?;
                i += 1;
            }
            "--d1b-max-input-tokens" => {
                opts.limits.max_input_tokens = value
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(|| "invalid --d1b-max-input-tokens".to_string())?;
                i += 1;
            }
            "--d1b-max-output-tokens" => {
                opts.limits.max_output_tokens = value
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(|| "invalid --d1b-max-output-tokens".to_string())?;
                i += 1;
            }
            "--d1b-timeout-ms" => {
                opts.limits.wall_clock_ms = value
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(|| "invalid --d1b-timeout-ms".to_string())?;
                i += 1;
            }
            "--d1b-cost-cap" => {
                let parsed = value
                    .and_then(|v| v.parse::<f64>().ok())
                    .ok_or_else(|| "invalid --d1b-cost-cap".to_string())?;
                if !parsed.is_finite() || parsed < 0.0 {
                    return Err("--d1b-cost-cap must be a finite non-negative amount".to_string());
                }
                opts.cost_cap = Some(parsed);
                i += 1;
            }
            "--d1b-cost-currency" => {
                if let Some(v) = value {
                    opts.cost_currency = v;
                }
                i += 1;
            }
            "--d1b-require-cost-cap" => {
                opts.require_cost_cap = true;
            }
            _ => {}
        }
        i += 1;
    }
    let upstream = upstream.ok_or_else(|| "missing --d1b-upstream".to_string())?;
    let model = model.ok_or_else(|| "missing --d1b-model".to_string())?;
    if opts.limits.max_calls == 0 {
        return Err("--d1b-max-calls must be at least 1".to_string());
    }
    if opts.limits.max_input_tokens == 0 {
        return Err("--d1b-max-input-tokens must be at least 1".to_string());
    }
    if opts.limits.max_output_tokens == 0 {
        return Err("--d1b-max-output-tokens must be at least 1".to_string());
    }
    if opts.limits.wall_clock_ms == 0 {
        return Err("--d1b-timeout-ms must be at least 1".to_string());
    }
    opts.upstream = upstream;
    opts.model = model;
    // F15: a configured cap is a configured requirement.  Making this explicit
    // here (rather than only inside the gate) keeps every later reader of
    // `require_cost_cap` — logs, tests, future callers — on the same contract.
    if opts.cost_cap.is_some() {
        opts.require_cost_cap = true;
    }
    Ok(opts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::summarizer::provider_recipe;
    use axum::Router;
    use axum::extract::{Json, State};
    use axum::http::StatusCode;
    use axum::response::{IntoResponse, Response};
    use axum::routing::post;
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU16, AtomicU64, Ordering};

    // The lock lives in `crate::test_env` so `summarizer`'s tests serialize
    // against the same mutex; a module-private one only protected this module.
    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        crate::test_env::lock()
    }

    struct MockState {
        hits: AtomicU64,
        delay_ms: AtomicU64,
        fail_status: AtomicU16,
        last: Mutex<Option<Value>>,
    }

    async fn mock_chat(State(state): State<Arc<MockState>>, Json(body): Json<Value>) -> Response {
        state.hits.fetch_add(1, Ordering::SeqCst);
        *state.last.lock().expect("mock lock") = Some(body);
        let delay = state.delay_ms.load(Ordering::SeqCst);
        if delay > 0 {
            tokio::time::sleep(Duration::from_millis(delay)).await;
        }
        let fail = state.fail_status.load(Ordering::SeqCst);
        if fail != 0 {
            return (
                StatusCode::from_u16(fail).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                "mock failure",
            )
                .into_response();
        }
        Json(json!({
            "choices": [{ "index": 0, "message": { "role": "assistant", "content": "D1B-OK" }, "finish_reason": "stop" }],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 3,
                "prompt_tokens_details": { "cached_tokens": 2 }
            }
        }))
        .into_response()
    }

    async fn start_mock() -> (String, Arc<MockState>) {
        let state = Arc::new(MockState {
            hits: AtomicU64::new(0),
            delay_ms: AtomicU64::new(0),
            fail_status: AtomicU16::new(0),
            last: Mutex::new(None),
        });
        let app = Router::new()
            .route("/v1/chat/completions", post(mock_chat))
            .with_state(state.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind mock");
        let addr = listener.local_addr().expect("mock addr");
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        (format!("http://{addr}"), state)
    }

    fn set_env(name: &str, value: impl AsRef<str>) {
        // Test-only mutation, serialized by the crate-wide environment lock.
        unsafe { std::env::set_var(name, value.as_ref()) };
    }

    fn remove_env(name: &str) {
        unsafe { std::env::remove_var(name) };
    }

    fn temp_root(tag: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("bili-d1b-{tag}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).expect("temp root");
        root
    }

    fn opts(upstream: &str, root: &Path) -> RunOptions {
        let mut options = RunOptions::new(upstream, "test-model", root.to_path_buf());
        options.prompt = "Say D1B-OK".to_string();
        options
    }

    #[test]
    fn credential_presence_is_reported_without_the_value() {
        assert_eq!(
            credential_status_with("ONEAGENT_MODEL_API_KEY", |_| {
                Some("secret-value".to_string())
            }),
            CredentialStatus::Present
        );
        assert_eq!(
            credential_status_with("ONEAGENT_MODEL_API_KEY", |_| None),
            CredentialStatus::Missing
        );
        assert_eq!(
            credential_status_with("ONEAGENT_MODEL_API_KEY", |_| { Some("   ".to_string()) }),
            CredentialStatus::Missing
        );
        assert_eq!(
            credential_status_with("", |_| Some("secret-value".to_string())),
            CredentialStatus::Missing
        );
    }

    #[test]
    fn fixed_allowlist_rejects_unknown_and_mismatched_bindings() {
        assert_eq!(
            allowed_credential_env("oneagent"),
            Some(ONEAGENT_CREDENTIAL_ENV)
        );
        assert_eq!(
            allowed_credential_env("oneagent-summary"),
            Some(ONEAGENT_SUMMARY_CREDENTIAL_ENV)
        );
        assert_eq!(allowed_credential_env("unknown"), None);
        assert_eq!(allowed_credential_env(""), None);

        let oneagent = json!({
            "providers": {
                "oneagent": {
                    "baseUrl": "https://api.example/v1",
                    "api": "openai",
                    "apiKeyEnv": "ONEAGENT_MODEL_API_KEY",
                    "models": {}
                }
            }
        });
        let recipe = provider_recipe(&oneagent, "oneagent").expect("oneagent recipe");
        assert_eq!(
            provider_binding_from_recipe("oneagent", &recipe),
            Some(ONEAGENT_CREDENTIAL_ENV)
        );
        assert_eq!(provider_binding_from_recipe("unknown", &recipe), None);

        let wrong_env = json!({
            "providers": {
                "oneagent": {
                    "baseUrl": "https://api.example/v1",
                    "api": "openai",
                    "apiKeyEnv": "BILI_EXTERNAL_SUMMARY_API_KEY",
                    "models": {}
                }
            }
        });
        let recipe = provider_recipe(&wrong_env, "oneagent").expect("oneagent recipe");
        assert_eq!(provider_binding_from_recipe("oneagent", &recipe), None);

        let empty_env = json!({
            "providers": {
                "oneagent": {
                    "baseUrl": "https://api.example/v1",
                    "api": "openai",
                    "apiKeyEnv": "",
                    "models": {}
                }
            }
        });
        let recipe = provider_recipe(&empty_env, "oneagent").expect("oneagent recipe");
        assert_eq!(provider_binding_from_recipe("oneagent", &recipe), None);

        let summary = json!({
            "providers": {
                "oneagent-summary": {
                    "baseUrl": "https://api.example/v1",
                    "api": "openai",
                    "credentialRef": "external-summary",
                    "models": {}
                }
            }
        });
        let recipe = provider_recipe(&summary, "oneagent-summary").expect("summary recipe");
        assert_eq!(
            provider_binding_from_recipe("oneagent-summary", &recipe),
            Some(ONEAGENT_SUMMARY_CREDENTIAL_ENV)
        );

        let conflicting_summary = json!({
            "providers": {
                "oneagent-summary": {
                    "baseUrl": "https://api.example/v1",
                    "api": "openai",
                    "apiKeyEnv": "ONEAGENT_MODEL_API_KEY",
                    "credentialRef": "external-summary",
                    "models": {}
                }
            }
        });
        let recipe =
            provider_recipe(&conflicting_summary, "oneagent-summary").expect("summary recipe");
        assert_eq!(
            provider_binding_from_recipe("oneagent-summary", &recipe),
            None
        );
    }

    #[test]
    fn bind_credential_sets_only_the_fixed_pair() {
        let root = temp_root("bind");
        let mut options = opts("http://127.0.0.1:9", &root);
        assert!(options.bind_credential("unknown").is_err());
        assert!(options.provider.is_none());
        assert!(options.credential_env.is_empty());

        options
            .bind_credential("oneagent")
            .expect("oneagent binding");
        assert_eq!(options.provider.as_deref(), Some("oneagent"));
        assert_eq!(options.credential_env, ONEAGENT_CREDENTIAL_ENV);
    }

    #[tokio::test]
    async fn run_rejects_unbound_unknown_and_mismatched_credentials_before_dispatch() {
        let root = temp_root("binding-rejects");

        let options = opts("http://127.0.0.1:9", &root);
        let report = run(options).await;
        assert_eq!(report.status, RunStatus::Refused, "{report:?}");
        assert_eq!(report.reason, "credential_binding_missing");
        assert_eq!(report.actual_calls, 0);

        let mut options = opts("http://127.0.0.1:9", &root);
        options.provider = Some("unknown".to_string());
        options.credential_env = ONEAGENT_CREDENTIAL_ENV.to_string();
        let report = run(options).await;
        assert_eq!(report.status, RunStatus::Refused, "{report:?}");
        assert_eq!(report.reason, "unknown_provider");
        assert_eq!(report.actual_calls, 0);

        let mut options = opts("http://127.0.0.1:9", &root);
        options
            .bind_credential("oneagent")
            .expect("oneagent binding");
        options.credential_env = ONEAGENT_SUMMARY_CREDENTIAL_ENV.to_string();
        let report = run(options).await;
        assert_eq!(report.status, RunStatus::Refused, "{report:?}");
        assert_eq!(report.reason, "credential_binding_mismatch");
        assert_eq!(report.actual_calls, 0);
    }

    #[test]
    fn only_external_env_does_not_satisfy_main_credential() {
        let lookup = |name: &str| {
            (name == crate::summarizer::EXTERNAL_SUMMARY_CREDENTIAL_ENV)
                .then(|| "external-value".to_string())
        };
        assert_eq!(
            credential_status_with("ONEAGENT_MODEL_API_KEY", lookup),
            CredentialStatus::Missing
        );
    }

    #[test]
    fn only_oneagent_env_does_not_satisfy_external_summary_credential() {
        let lookup =
            |name: &str| (name == "ONEAGENT_MODEL_API_KEY").then(|| "main-value".to_string());
        assert_eq!(
            credential_status_with(crate::summarizer::EXTERNAL_SUMMARY_CREDENTIAL_ENV, lookup),
            CredentialStatus::Missing
        );
    }

    #[test]
    fn both_credentials_missing_is_missing() {
        assert_eq!(
            credential_status_with("ONEAGENT_MODEL_API_KEY", |_| None),
            CredentialStatus::Missing
        );
    }

    #[tokio::test]
    #[allow(
        clippy::await_holding_lock,
        reason = "serializes tests that mutate process env"
    )]
    async fn oneagent_env_is_used_by_run() {
        let _guard = env_lock();
        let old_main = std::env::var("ONEAGENT_MODEL_API_KEY").ok();
        let old_external = std::env::var(crate::summarizer::EXTERNAL_SUMMARY_CREDENTIAL_ENV).ok();
        set_env("ONEAGENT_MODEL_API_KEY", "test-key");
        remove_env(crate::summarizer::EXTERNAL_SUMMARY_CREDENTIAL_ENV);

        let (upstream, state) = start_mock().await;
        let root = temp_root("provider-env");
        let mut options = opts(&upstream, &root);
        options
            .bind_credential("oneagent")
            .expect("oneagent binding");
        let report = run(options).await;
        assert_eq!(report.status, RunStatus::Complete, "{report:?}");
        assert_eq!(state.hits.load(Ordering::SeqCst), 1);

        match old_main {
            Some(value) => set_env("ONEAGENT_MODEL_API_KEY", value),
            None => remove_env("ONEAGENT_MODEL_API_KEY"),
        }
        match old_external {
            Some(value) => set_env(crate::summarizer::EXTERNAL_SUMMARY_CREDENTIAL_ENV, value),
            None => remove_env(crate::summarizer::EXTERNAL_SUMMARY_CREDENTIAL_ENV),
        }
    }

    #[tokio::test]
    #[allow(
        clippy::await_holding_lock,
        reason = "serializes tests that mutate process env"
    )]
    async fn only_external_env_refuses_main_run() {
        let _guard = env_lock();
        let old_main = std::env::var("ONEAGENT_MODEL_API_KEY").ok();
        let old_external = std::env::var(crate::summarizer::EXTERNAL_SUMMARY_CREDENTIAL_ENV).ok();
        remove_env("ONEAGENT_MODEL_API_KEY");
        set_env(
            crate::summarizer::EXTERNAL_SUMMARY_CREDENTIAL_ENV,
            "external-key",
        );

        let root = temp_root("only-external");
        let mut options = opts("http://127.0.0.1:9", &root);
        options
            .bind_credential("oneagent")
            .expect("oneagent binding");
        let report = run(options).await;
        assert_eq!(report.status, RunStatus::Refused, "{report:?}");
        assert_eq!(report.reason, "credential_missing");
        assert_eq!(report.actual_calls, 0);

        match old_main {
            Some(value) => set_env("ONEAGENT_MODEL_API_KEY", value),
            None => remove_env("ONEAGENT_MODEL_API_KEY"),
        }
        match old_external {
            Some(value) => set_env(crate::summarizer::EXTERNAL_SUMMARY_CREDENTIAL_ENV, value),
            None => remove_env(crate::summarizer::EXTERNAL_SUMMARY_CREDENTIAL_ENV),
        }
    }

    #[tokio::test]
    #[allow(
        clippy::await_holding_lock,
        reason = "serializes tests that mutate process env"
    )]
    async fn only_oneagent_env_refuses_external_summary_run() {
        let _guard = env_lock();
        let old_main = std::env::var("ONEAGENT_MODEL_API_KEY").ok();
        let old_external = std::env::var(crate::summarizer::EXTERNAL_SUMMARY_CREDENTIAL_ENV).ok();
        set_env("ONEAGENT_MODEL_API_KEY", "main-key");
        remove_env(crate::summarizer::EXTERNAL_SUMMARY_CREDENTIAL_ENV);

        let root = temp_root("only-oneagent");
        let mut options = opts("http://127.0.0.1:9", &root);
        options
            .bind_credential("oneagent-summary")
            .expect("summary binding");
        let report = run(options).await;
        assert_eq!(report.status, RunStatus::Refused, "{report:?}");
        assert_eq!(report.reason, "credential_missing");
        assert_eq!(report.actual_calls, 0);

        match old_main {
            Some(value) => set_env("ONEAGENT_MODEL_API_KEY", value),
            None => remove_env("ONEAGENT_MODEL_API_KEY"),
        }
        match old_external {
            Some(value) => set_env(crate::summarizer::EXTERNAL_SUMMARY_CREDENTIAL_ENV, value),
            None => remove_env(crate::summarizer::EXTERNAL_SUMMARY_CREDENTIAL_ENV),
        }
    }

    #[tokio::test]
    #[allow(
        clippy::await_holding_lock,
        reason = "serializes tests that mutate process env"
    )]
    async fn missing_both_envs_refuses_main_run() {
        let _guard = env_lock();
        let old_main = std::env::var("ONEAGENT_MODEL_API_KEY").ok();
        let old_external = std::env::var(crate::summarizer::EXTERNAL_SUMMARY_CREDENTIAL_ENV).ok();
        remove_env("ONEAGENT_MODEL_API_KEY");
        remove_env(crate::summarizer::EXTERNAL_SUMMARY_CREDENTIAL_ENV);

        let root = temp_root("missing-both");
        let mut options = opts("http://127.0.0.1:9", &root);
        options
            .bind_credential("oneagent")
            .expect("oneagent binding");
        let report = run(options).await;
        assert_eq!(report.status, RunStatus::Refused, "{report:?}");
        assert_eq!(report.reason, "credential_missing");
        assert_eq!(report.actual_calls, 0);

        match old_main {
            Some(value) => set_env("ONEAGENT_MODEL_API_KEY", value),
            None => remove_env("ONEAGENT_MODEL_API_KEY"),
        }
        match old_external {
            Some(value) => set_env(crate::summarizer::EXTERNAL_SUMMARY_CREDENTIAL_ENV, value),
            None => remove_env(crate::summarizer::EXTERNAL_SUMMARY_CREDENTIAL_ENV),
        }
    }

    #[tokio::test]
    async fn input_cap_refuses_before_any_upstream_call() {
        let (upstream, state) = start_mock().await;
        let root = temp_root("input-cap");
        let mut options = opts(&upstream, &root);
        options.prompt = "x".repeat(4_000);
        options.limits.max_input_tokens = 20;
        let report = run_with_key(options, Some("test-key".to_string())).await;
        assert_eq!(report.status, RunStatus::Refused);
        assert_eq!(report.reason, "input_token_cap");
        assert_eq!(report.actual_calls, 0);
        assert_eq!(state.hits.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn success_run_is_one_call_and_ledger_is_complete() {
        let (upstream, state) = start_mock().await;
        let root = temp_root("success");
        let mut options = opts(&upstream, &root);
        options.limits.max_output_tokens = 8;
        let report = run_with_key(options, Some("test-key".to_string())).await;
        assert_eq!(report.status, RunStatus::Complete, "{report:?}");
        assert_eq!(report.actual_calls, 1);
        assert_eq!(report.provider_calls, 1);
        assert!(report.provider_usage_present);
        assert_eq!(report.output_tokens_reported, Some(3));
        assert!(report.ledger_complete, "{report:?}");
        assert_eq!(report.ledger_starts, 1);
        assert_eq!(report.ledger_finishes, 1);
        assert_eq!(report.audit_requests, 1);
        assert_eq!(report.audit_finished, 1);
        assert_eq!(state.hits.load(Ordering::SeqCst), 1);
        let last = state.last.lock().expect("lock").clone().expect("body");
        assert_eq!(last["max_tokens"], json!(8));
        assert_eq!(last["stream"], json!(false));
    }

    #[tokio::test]
    async fn a_second_dispatch_is_refused_by_the_call_gate() {
        let (upstream, state) = start_mock().await;
        let root = temp_root("gate");
        let options = opts(&upstream, &root);
        let mut runner = Runner::new(options).expect("runner");
        let first = runner.execute(Some("test-key".to_string())).await;
        assert_eq!(first.status, RunStatus::Complete);
        let second = runner.execute(Some("test-key".to_string())).await;
        assert_eq!(second.status, RunStatus::Refused);
        assert_eq!(second.reason, "call_budget_exhausted");
        assert_eq!(second.actual_calls, 0);
        assert_eq!(state.hits.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn transport_error_is_invalid_and_still_has_a_finish() {
        let root = temp_root("transport");
        let mut options = opts("http://127.0.0.1:9", &root);
        options.limits.wall_clock_ms = 2_000;
        let report = run_with_key(options, Some("test-key".to_string())).await;
        assert_eq!(report.status, RunStatus::Invalid);
        assert_eq!(report.actual_calls, 1);
        assert_eq!(report.ledger_starts, 1);
        assert_eq!(report.ledger_finishes, 1);
        assert!(
            report
                .invalid_reasons
                .iter()
                .any(|reason| reason.starts_with("transport_error")),
            "{report:?}"
        );
    }

    #[test]
    fn missing_finish_and_count_mismatch_are_invalid() {
        let missing_finish = LedgerAudit {
            lines: 1,
            requests: 1,
            finished: 0,
            starts: 1,
            finishes: 0,
            issues: vec![usage_ledger::LedgerIssue::MissingFinish],
        };
        let problems = count_mismatches(1, 1, 1, 0, &missing_finish);
        assert!(!problems.is_empty());
        assert!(problems.iter().any(|p| p.contains("missing_finish")));

        let unreadable = LedgerAudit {
            issues: vec![usage_ledger::LedgerIssue::Unreadable],
            ..LedgerAudit::default()
        };
        assert!(!count_mismatches(1, 0, 0, 0, &unreadable).is_empty());

        let good = LedgerAudit {
            lines: 2,
            requests: 1,
            finished: 1,
            starts: 1,
            finishes: 1,
            issues: Vec::new(),
        };
        assert!(count_mismatches(1, 1, 1, 1, &good).is_empty());
        assert!(!count_mismatches(1, 2, 1, 1, &good).is_empty());
    }

    #[tokio::test]
    async fn http_500_is_invalid_and_counts_as_one_provider_call() {
        let (upstream, state) = start_mock().await;
        state.fail_status.store(500, Ordering::SeqCst);
        let root = temp_root("http-500");
        let options = opts(&upstream, &root);
        let report = run_with_key(options, Some("test-key".to_string())).await;
        assert_eq!(report.status, RunStatus::Invalid, "{report:?}");
        assert_eq!(report.actual_calls, 1);
        assert_eq!(report.provider_calls, 1);
        assert_eq!(report.http_status, Some(500));
        assert_eq!(report.ledger_starts, 1);
        assert_eq!(report.ledger_finishes, 1);
        assert!(
            report
                .invalid_reasons
                .iter()
                .any(|reason| reason == "http_status:500"),
            "{report:?}"
        );
        assert_eq!(state.hits.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn wall_clock_timeout_is_invalid_and_still_has_a_finish() {
        let (upstream, state) = start_mock().await;
        state.delay_ms.store(1_000, Ordering::SeqCst);
        let root = temp_root("timeout");
        let mut options = opts(&upstream, &root);
        options.limits.wall_clock_ms = 50;
        let report = run_with_key(options, Some("test-key".to_string())).await;
        assert_eq!(report.status, RunStatus::Invalid, "{report:?}");
        assert_eq!(report.actual_calls, 1);
        assert_eq!(report.ledger_starts, 1);
        assert_eq!(report.ledger_finishes, 1);
        assert!(
            report
                .invalid_reasons
                .iter()
                .any(|reason| reason.contains("timeout")),
            "{report:?}"
        );
    }

    #[tokio::test]
    async fn a_preexisting_unfinished_ledger_marks_the_run_invalid() {
        let (upstream, state) = start_mock().await;
        let root = temp_root("stale-ledger");
        let session = "stale-session";
        let mut options = opts(&upstream, &root);
        options.session_id = session.to_string();
        let path = usage_ledger::ledger_path(&root.join("requests"), session);
        std::fs::create_dir_all(path.parent().expect("ledger parent")).expect("mkdir");
        std::fs::write(
            &path,
            json!({
                "schema": "bili-usage-v1",
                "event": "start",
                "request_id": "foreign-request",
                "session_id": session
            })
            .to_string()
                + "\n",
        )
        .expect("seed ledger");
        let report = run_with_key(options, Some("test-key".to_string())).await;
        assert_eq!(report.status, RunStatus::Invalid, "{report:?}");
        assert_eq!(report.actual_calls, 1);
        assert_eq!(state.hits.load(Ordering::SeqCst), 1);
        assert!(
            report
                .invalid_reasons
                .iter()
                .any(|reason| reason == "ledger_incomplete"),
            "{report:?}"
        );
        assert!(
            report
                .invalid_reasons
                .iter()
                .any(|reason| reason.contains("count_mismatch")),
            "{report:?}"
        );
    }

    #[tokio::test]
    async fn missing_credential_refuses_before_dispatch() {
        let (upstream, state) = start_mock().await;
        let root = temp_root("credential");
        let options = opts(&upstream, &root);
        let report = run_with_key(options, None).await;
        assert_eq!(report.status, RunStatus::Refused);
        assert_eq!(report.reason, "credential_missing");
        assert_eq!(state.hits.load(Ordering::SeqCst), 0);
    }

    // ---- monetary gate -----------------------------------------------------
    //
    // `cost_gate` is the only place a price becomes a decision. These tests
    // drive it with a published snapshot and check that every refusal happens
    // *before* a socket is opened, because a cap that is checked afterwards
    // bounds nothing.

    const SNAPSHOT_MODEL: &str = "test-model";

    struct SnapshotGuard {
        path: PathBuf,
        _lock: std::sync::MutexGuard<'static, ()>,
    }

    impl SnapshotGuard {
        fn install(contents: &str) -> SnapshotGuard {
            let lock = env_lock();
            let path =
                std::env::temp_dir().join(format!("bili-d1b-price-{}.json", uuid::Uuid::new_v4()));
            std::fs::write(&path, contents).expect("write snapshot");
            set_env(pricing::SNAPSHOT_ENV, path.to_string_lossy());
            SnapshotGuard { path, _lock: lock }
        }

        fn priced(updated_offset_ms: i64) -> SnapshotGuard {
            Self::with_groups(updated_offset_ms, false)
        }

        /// A book where the plugin saw the model in more than one child group:
        /// the rates are kept, but nothing proves they bill this request.
        fn ambiguous(updated_offset_ms: i64) -> SnapshotGuard {
            Self::with_groups(updated_offset_ms, true)
        }

        fn with_groups(updated_offset_ms: i64, ambiguous: bool) -> SnapshotGuard {
            let now = now_ms();
            let updated = if updated_offset_ms >= 0 {
                now.saturating_add(updated_offset_ms as u64)
            } else {
                now.saturating_sub(updated_offset_ms.unsigned_abs())
            };
            let body = json!({
                "unit": "CNY",
                "source": "OneAgent model plaza",
                "updatedAt": updated,
                "models": [{
                    "id": SNAPSHOT_MODEL,
                    "groupId": 111,
                    "short": {
                        "uncached": 0.864,
                        "cacheRead": 0.01728,
                        "cacheWrite": 0.01728,
                        "output": 3.456,
                        "threshold": null
                    },
                    "long": null,
                    "threshold": null,
                    "fallback": ambiguous
                }]
            });
            SnapshotGuard::install(&body.to_string())
        }

        /// Rates that leave one band unpublished: real providers omit bands,
        /// and a cap that needs a band the book does not have cannot be checked.
        fn partial_rates(updated_offset_ms: i64) -> SnapshotGuard {
            let now = now_ms();
            let updated = if updated_offset_ms >= 0 {
                now.saturating_add(updated_offset_ms as u64)
            } else {
                now.saturating_sub(updated_offset_ms.unsigned_abs())
            };
            let body = json!({
                "unit": "CNY",
                "source": "OneAgent model plaza",
                "updatedAt": updated,
                "models": [{
                    "id": SNAPSHOT_MODEL,
                    "groupId": 111,
                    "short": {
                        "uncached": 0.864,
                        "cacheRead": 0.01728,
                        "output": 3.456,
                        "threshold": null
                    },
                    "long": null,
                    "threshold": null,
                    "fallback": false
                }]
            });
            SnapshotGuard::install(&body.to_string())
        }

        /// A card whose four bands are all zero: it must never pass as "free".
        fn zero_rates(updated_offset_ms: i64) -> SnapshotGuard {
            let now = now_ms();
            let updated = if updated_offset_ms >= 0 {
                now.saturating_add(updated_offset_ms as u64)
            } else {
                now.saturating_sub(updated_offset_ms.unsigned_abs())
            };
            let body = json!({
                "unit": "CNY",
                "source": "OneAgent model plaza",
                "updatedAt": updated,
                "models": [{
                    "id": SNAPSHOT_MODEL,
                    "groupId": 111,
                    "short": {
                        "uncached": 0.0,
                        "cacheRead": 0.0,
                        "cacheWrite": 0.0,
                        "output": 0.0,
                        "threshold": null
                    },
                    "long": null,
                    "threshold": null,
                    "fallback": false
                }]
            });
            SnapshotGuard::install(&body.to_string())
        }

        /// Point the adapter at a path that does not exist, so a test asserting
        /// "no price" can never accidentally read the operator's live cache.
        fn absent() -> SnapshotGuard {
            let lock = env_lock();
            let path =
                std::env::temp_dir().join(format!("bili-d1b-absent-{}.json", uuid::Uuid::new_v4()));
            set_env(pricing::SNAPSHOT_ENV, path.to_string_lossy());
            SnapshotGuard { path, _lock: lock }
        }
    }

    impl Drop for SnapshotGuard {
        fn drop(&mut self) {
            unsafe { crate::test_env::restore_taken(pricing::SNAPSHOT_ENV, None) };
            let _ = std::fs::remove_file(&self.path);
        }
    }

    fn priced_opts(root: &Path) -> RunOptions {
        // The origin must be the provider's own, otherwise the book rightly
        // refuses to price a route it does not cover.
        let mut options =
            RunOptions::new(pricing::ONEAGENT_ORIGIN, SNAPSHOT_MODEL, root.to_path_buf());
        options.prompt = "Say D1B-OK".to_string();
        options
    }

    #[test]
    fn a_required_cap_without_a_published_price_refuses_the_run() {
        let _guard = SnapshotGuard::absent();
        let root = temp_root("cost-no-price");
        let mut options = priced_opts(&root);
        options.require_cost_cap = true;
        options.cost_cap = Some(1.0);
        options.cost_currency = "CNY".to_string();

        // No snapshot installed at all; the gate must not invent one.
        let (report, verdict) = cost_gate(&options, 1_000, 100, now_ms());
        assert!(!report.price_known, "{report:?}");
        assert_eq!(report.basis.refusal_code, Some("price_snapshot_missing"));
        assert_eq!(
            verdict.expect_err("required cap cannot be honoured"),
            "cost_cap_unavailable"
        );
    }

    #[test]
    fn a_published_price_yields_a_projection_in_the_books_currency() {
        let _guard = SnapshotGuard::priced(-60_000);
        let root = temp_root("cost-priced");
        let options = priced_opts(&root);

        let (report, verdict) = cost_gate(&options, 1_000_000, 500_000, now_ms());
        assert!(report.price_known, "{report:?}");
        assert_eq!(report.currency, "CNY");
        let projected = verdict.expect("no cap, no refusal").expect("projection");
        // 1M input tokens at 0.864/1M = 0.864; 0.5M output at 3.456/1M = 1.728.
        assert!((projected - 2.592).abs() < 1e-9, "{projected}");
        assert_eq!(report.estimated_cost, Some(projected));
        assert!(!report.cap_enforced, "a projection is not an enforced cap");
    }

    #[test]
    fn a_stale_snapshot_cannot_satisfy_a_required_cap() {
        let _guard = SnapshotGuard::priced(-(pricing::MAX_STALE_MS as i64) - 60_000);
        let root = temp_root("cost-stale");
        let mut options = priced_opts(&root);
        options.require_cost_cap = true;
        options.cost_cap = Some(10.0);
        options.cost_currency = "CNY".to_string();

        let (report, verdict) = cost_gate(&options, 1_000, 100, now_ms());
        assert!(!report.price_known);
        assert_eq!(report.basis.refusal_code, Some("price_stale"));
        assert_eq!(verdict.expect_err("stale price"), "cost_cap_unavailable");
    }

    #[test]
    fn an_ambiguous_group_never_declares_a_cost_and_refuses_a_required_cap() {
        let _guard = SnapshotGuard::ambiguous(-60_000);
        let root = temp_root("cost-ambiguous");
        let mut options = priced_opts(&root);
        options.require_cost_cap = true;
        options.cost_cap = Some(1.0);
        options.cost_currency = "CNY".to_string();

        let (report, verdict) = cost_gate(&options, 1_000_000, 500_000, now_ms());
        assert!(!report.price_known, "{report:?}");
        assert!(
            report.estimated_cost.is_none(),
            "an unproven group may not claim an amount"
        );
        assert_eq!(report.basis.refusal_code, Some("price_group_ambiguous"));
        assert!(report.basis.group_ambiguous);
        assert_eq!(report.basis.group_id, Some(111));
        assert_eq!(
            verdict.expect_err("a required cap cannot be honoured"),
            "cost_group_ambiguous"
        );
    }

    #[test]
    fn an_ambiguous_group_without_a_required_cap_proceeds_claiming_nothing() {
        let _guard = SnapshotGuard::ambiguous(-60_000);
        let root = temp_root("cost-ambiguous-open");
        let options = priced_opts(&root);

        let (report, verdict) = cost_gate(&options, 1_000, 100, now_ms());
        assert_eq!(verdict.expect("no required cap"), None);
        assert!(!report.price_known, "no amount is known");
        assert!(report.estimated_cost.is_none());
        assert!(!report.cap_enforced);
        assert_eq!(report.basis.refusal_code, Some("price_group_ambiguous"));
    }

    #[tokio::test]
    async fn a_budget_refusal_keeps_the_price_verdict_the_gate_resolved() {
        // A closed loopback port: the first attempt spends its call slot and
        // fails in transport, the second attempt is stopped by the budget.
        let _guard = SnapshotGuard::priced(-60_000);
        let root = temp_root("budget-keeps-basis");
        let mut options = opts("http://127.0.0.1:9", &root);
        options.limits.wall_clock_ms = 2_000;
        let mut runner = Runner::new(options).expect("runner");

        let first = runner.execute(Some("test-key".to_string())).await;
        assert_eq!(first.status, RunStatus::Invalid);
        assert_eq!(first.actual_calls, 1);
        // A transport failure is attempted but never answered, so no provider
        // call is recorded; the slot is still spent.
        assert_eq!(first.provider_calls, 0);
        assert_eq!(first.ledger_starts, 1);
        assert_eq!(first.ledger_finishes, 1);

        let second = runner.execute(Some("test-key".to_string())).await;
        assert_eq!(second.status, RunStatus::Refused);
        assert_eq!(second.reason, "call_budget_exhausted");
        assert_eq!(second.actual_calls, 0);
        assert_eq!(second.provider_calls, 0);
        // The price was resolved for the loopback route and refused there; the
        // report must keep that verdict instead of claiming nobody looked.
        assert!(!second.cost.price_known);
        assert_eq!(
            second.cost.basis.refusal_code,
            Some("price_provider_mismatch"),
            "the basis must survive the budget refusal: {second:?}"
        );
        assert!(second.input_tokens_estimated > 0);
    }

    #[test]
    fn a_cap_in_another_currency_is_refused_rather_than_converted() {
        let _guard = SnapshotGuard::priced(-60_000);
        let root = temp_root("cost-currency");
        let mut options = priced_opts(&root);
        options.cost_cap = Some(1.0);
        options.cost_currency = "USD".to_string();

        let (report, verdict) = cost_gate(&options, 100, 10, now_ms());
        assert!(report.price_known, "the price itself is known");
        assert_eq!(report.basis.currency.as_deref(), Some("CNY"));
        assert_eq!(
            verdict.expect_err("currency mismatch"),
            "cost_cap_currency_mismatch"
        );
    }

    #[test]
    fn a_projection_above_the_cap_refuses_and_one_at_the_cap_does_not() {
        let _guard = SnapshotGuard::priced(-60_000);
        let root = temp_root("cost-cap");
        let mut options = priced_opts(&root);
        options.cost_currency = "CNY".to_string();

        // 0.864 per 1M input, so 2M input alone is 1.728 — over a 1.0 cap.
        options.cost_cap = Some(1.0);
        let (over, verdict) = cost_gate(&options, 2_000_000, 0, now_ms());
        assert_eq!(verdict.expect_err("over cap"), "cost_cap_exceeded");
        assert!(over.cost_cap_configured);
        assert!(over.cost_cap_checked, "the comparison did happen");
        assert_eq!(over.cost_cap_passed, Some(false));
        assert!(over.cap_enforced);

        // Exactly at the cap must pass: a cap is inclusive.
        options.cost_cap = Some(0.864);
        let (report, verdict) = cost_gate(&options, 1_000_000, 0, now_ms());
        assert!(report.price_known);
        assert!(verdict.expect("at cap").is_some());
        assert!(report.cost_cap_configured);
        assert!(report.cost_cap_checked);
        assert_eq!(report.cost_cap_passed, Some(true));
        assert!(report.cap_enforced);
    }

    #[test]
    fn a_required_cap_with_no_cap_number_is_refused() {
        let _guard = SnapshotGuard::priced(-60_000);
        let root = temp_root("cost-cap-missing");
        let mut options = priced_opts(&root);
        options.require_cost_cap = true;

        let (report, verdict) = cost_gate(&options, 100, 10, now_ms());
        assert!(report.price_known);
        assert_eq!(verdict.expect_err("no number"), "cost_cap_missing");
        // "Require a cap" without a number configured no cap and compared nothing;
        // the refusal is in `reason`, not dressed up as enforcement.
        assert!(!report.cost_cap_configured);
        assert!(!report.cost_cap_checked);
        assert_eq!(report.cost_cap_passed, None);
        assert!(!report.cap_enforced);
    }

    #[tokio::test]
    async fn a_run_requiring_a_cap_refuses_before_any_upstream_call() {
        // A published price exists, but for another provider; the run must stop
        // before the socket, not after.
        let _guard = SnapshotGuard::priced(-60_000);
        let (upstream, state) = start_mock().await;
        let root = temp_root("cost-refuse-dispatch");
        let mut options = opts(&upstream, &root);
        // The mock upstream is not the priced provider, so no price applies.
        options.require_cost_cap = true;
        options.cost_cap = Some(1.0);
        options.cost_currency = "CNY".to_string();

        let report = run_with_key(options, Some("test-key".to_string())).await;
        assert_eq!(report.status, RunStatus::Refused, "{report:?}");
        assert_eq!(report.reason, "cost_cap_unavailable");
        assert_eq!(report.actual_calls, 0);
        assert_eq!(state.hits.load(Ordering::SeqCst), 0);
        assert_eq!(
            report.cost.basis.refusal_code,
            Some("price_provider_mismatch")
        );
        assert!(!report.cost.price_known);
        assert_eq!(report.cost.estimated_cost, None, "no cost may be claimed");
    }

    #[tokio::test]
    async fn a_run_without_a_required_cap_still_reports_cost_as_unknown() {
        let _guard = SnapshotGuard::priced(-60_000);
        let (upstream, state) = start_mock().await;
        let root = temp_root("cost-unknown-ok");
        let options = opts(&upstream, &root);
        let report = run_with_key(options, Some("test-key".to_string())).await;
        assert_eq!(report.status, RunStatus::Complete, "{report:?}");
        assert_eq!(state.hits.load(Ordering::SeqCst), 1);
        assert!(!report.cost.price_known);
        assert_eq!(report.cost.estimated_cost, None);
        assert_eq!(
            report.cost.basis.refusal_code,
            Some("price_provider_mismatch")
        );
        assert!(report.cost.note.contains("no verified price"), "{report:?}");
        // F16 row: no cap, no check, no enforcement claim — even though a
        // projection was computed.
        assert!(!report.cost.cost_cap_configured);
        assert!(!report.cost.cost_cap_checked);
        assert_eq!(report.cost.cost_cap_passed, None);
        assert!(!report.cost.cap_enforced);
    }

    // ---- F15: a configured cap is a configured requirement ------------------
    //
    // Before this change `--d1b-cost-cap` could coexist with a dispatch when no
    // price was known: the number was reported but never enforced. These tests
    // pin the contract that a cap without a usable price refuses before the
    // socket, and that the four report flags never overclaim.

    /// Local helper: a cap without the explicit `--d1b-require-cost-cap` switch.
    fn capped(mut options: RunOptions, cap: f64) -> RunOptions {
        options.cost_cap = Some(cap);
        options.cost_currency = "CNY".to_string();
        options
    }

    #[test]
    fn a_configured_cap_requires_a_price_even_without_the_explicit_switch() {
        let _guard = SnapshotGuard::absent();
        let root = temp_root("f15-absent");
        let options = capped(priced_opts(&root), 1.0);

        let (report, verdict) = cost_gate(&options, 1_000, 100, now_ms());
        assert!(!report.price_known);
        assert_eq!(report.basis.refusal_code, Some("price_snapshot_missing"));
        assert_eq!(
            verdict.expect_err("a cap implies a price requirement"),
            "cost_cap_unavailable"
        );
        assert!(report.cost_cap_configured);
        assert!(!report.cost_cap_checked, "nothing was compared");
        assert_eq!(report.cost_cap_passed, None);
        assert!(
            report.cap_enforced,
            "the gate refused because a cap was set"
        );
    }

    #[test]
    fn a_configured_cap_refuses_a_model_the_book_does_not_price() {
        let _guard = SnapshotGuard::priced(-60_000);
        let root = temp_root("f15-model");
        let mut options = capped(priced_opts(&root), 1.0);
        options.model = "not-in-the-book".to_string();

        let (report, verdict) = cost_gate(&options, 1_000, 100, now_ms());
        assert_eq!(report.basis.refusal_code, Some("price_model_unknown"));
        assert_eq!(verdict.expect_err("unpriced model"), "cost_cap_unavailable");
        assert!(report.cost_cap_configured);
        assert!(!report.cost_cap_checked);
        assert_eq!(report.cost_cap_passed, None);
    }

    #[test]
    fn a_configured_cap_refuses_on_a_stale_snapshot() {
        let _guard = SnapshotGuard::priced(-(pricing::MAX_STALE_MS as i64) - 60_000);
        let root = temp_root("f15-stale");
        let options = capped(priced_opts(&root), 10.0);

        let (report, verdict) = cost_gate(&options, 1_000, 100, now_ms());
        assert_eq!(report.basis.refusal_code, Some("price_stale"));
        assert_eq!(verdict.expect_err("stale price"), "cost_cap_unavailable");
        assert!(!report.cost_cap_checked);
        assert_eq!(report.cost_cap_passed, None);
    }

    #[test]
    fn a_configured_cap_refuses_a_route_the_book_does_not_price() {
        let _guard = SnapshotGuard::priced(-60_000);
        let root = temp_root("f15-provider");
        // The book prices ONEAGENT_ORIGIN only; this route is another provider.
        let options = capped(opts("http://127.0.0.1:9", &root), 1.0);

        let (report, verdict) = cost_gate(&options, 1_000, 100, now_ms());
        assert_eq!(report.basis.refusal_code, Some("price_provider_mismatch"));
        assert_eq!(
            verdict.expect_err("provider mismatch"),
            "cost_cap_unavailable"
        );
        assert!(report.cost_cap_configured);
        assert!(!report.cost_cap_checked);
        assert_eq!(report.cost_cap_passed, None);
    }

    #[test]
    fn a_configured_cap_refuses_a_group_the_snapshot_cannot_identify() {
        let _guard = SnapshotGuard::ambiguous(-60_000);
        let root = temp_root("f15-ambiguous");
        let options = capped(priced_opts(&root), 1.0);

        let (report, verdict) = cost_gate(&options, 1_000_000, 0, now_ms());
        assert_eq!(report.basis.refusal_code, Some("price_group_ambiguous"));
        assert_eq!(
            verdict.expect_err("ambiguous group"),
            "cost_group_ambiguous"
        );
        assert!(report.cost_cap_configured);
        assert!(!report.cost_cap_checked);
        assert_eq!(report.cost_cap_passed, None);
    }

    #[test]
    fn a_configured_cap_refuses_an_incomplete_rate_card() {
        let _guard = SnapshotGuard::partial_rates(-60_000);
        let root = temp_root("f15-partial");
        let options = capped(priced_opts(&root), 1.0);

        let (report, verdict) = cost_gate(&options, 1_000, 100, now_ms());
        assert_eq!(report.basis.refusal_code, Some("price_rate_missing"));
        assert_eq!(verdict.expect_err("missing band"), "cost_cap_unavailable");
        assert!(!report.cost_cap_checked);
        assert_eq!(report.cost_cap_passed, None);
        // The report must not promise a check it could not perform.
        assert!(report.note.contains("no usable price"), "{report:?}");
    }

    #[test]
    fn a_configured_cap_refuses_an_all_zero_rate_card() {
        let _guard = SnapshotGuard::zero_rates(-60_000);
        let root = temp_root("f15-zero");
        let options = capped(priced_opts(&root), 1.0);

        let (report, verdict) = cost_gate(&options, 1_000, 100, now_ms());
        assert_eq!(report.basis.refusal_code, Some("price_rate_zero"));
        assert_eq!(verdict.expect_err("zero card"), "cost_cap_unavailable");
        assert!(!report.cost_cap_checked);
        assert_eq!(report.cost_cap_passed, None);
    }

    #[test]
    fn a_configured_cap_that_passes_reports_the_comparison() {
        let _guard = SnapshotGuard::priced(-60_000);
        let root = temp_root("f15-pass");
        let options = capped(priced_opts(&root), 5.0);

        let (report, verdict) = cost_gate(&options, 1_000_000, 0, now_ms());
        // 1M input at 0.864/1M = 0.864 <= 5.0.
        let projected = verdict.expect("cap honoured").expect("projection");
        assert!((projected - 0.864).abs() < 1e-9, "{projected}");
        assert!(report.price_known);
        assert!(report.cost_cap_configured);
        assert!(report.cost_cap_checked);
        assert_eq!(report.cost_cap_passed, Some(true));
        assert!(report.cap_enforced);
        assert!(report.note.contains("stayed within it"), "{report:?}");
    }

    #[test]
    fn a_run_without_any_cap_keeps_the_pre_fix_semantics() {
        let _guard = SnapshotGuard::absent();
        let root = temp_root("f15-none");
        let options = priced_opts(&root);

        let (report, verdict) = cost_gate(&options, 1_000, 100, now_ms());
        assert_eq!(verdict.expect("no cap at all"), None);
        assert!(!report.cost_cap_configured);
        assert!(!report.cost_cap_checked);
        assert_eq!(report.cost_cap_passed, None);
        assert!(!report.cap_enforced);
    }

    #[test]
    fn a_non_amount_cap_is_a_configuration_error() {
        // `cost_gate` is private and only reached from `execute`, which refuses
        // before it for these values; the validation itself is what this pins.
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
            let mut options =
                RunOptions::new("https://api.gameaigc.cn", "test-model", PathBuf::new());
            options.cost_cap = Some(bad);
            assert!(
                options.invalid_cost_cap().is_some(),
                "{bad} must be refused"
            );
        }
        for good in [0.0, 0.5, 1.5, 1e6] {
            let mut options =
                RunOptions::new("https://api.gameaigc.cn", "test-model", PathBuf::new());
            options.cost_cap = Some(good);
            assert!(options.invalid_cost_cap().is_none(), "{good} is an amount");
        }
    }

    #[tokio::test]
    async fn an_unreadable_cap_refuses_before_any_dispatch() {
        let (upstream, state) = start_mock().await;
        for (tag, bad) in [("nan", f64::NAN), ("inf", f64::INFINITY), ("neg", -1.0)] {
            let root = temp_root(tag);
            let mut options = opts(&upstream, &root);
            options.cost_cap = Some(bad);
            options.cost_currency = "CNY".to_string();
            options.require_cost_cap = true;
            let report = run_with_key(options, Some("test-key".to_string())).await;
            assert_eq!(report.status, RunStatus::Refused, "{bad}: {report:?}");
            assert_eq!(report.reason, "invalid_cost_cap");
            assert_eq!(report.actual_calls, 0);
            assert_eq!(report.provider_calls, 0);
            assert!(report.cost.estimated_cost.is_none());
        }
        assert_eq!(state.hits.load(Ordering::SeqCst), 0);
    }

    // ---- F16: status split, serialization and CLI contract ------------------

    /// Every status combination the gate can produce must satisfy the same
    /// invariants: nothing checked without a configured cap, nothing passed
    /// without a check, and enforcement only for refusals that a cap drove.
    fn assert_cap_invariants(report: &CostReport) {
        if report.cost_cap_checked {
            assert!(
                report.cost_cap_configured,
                "checked without a cap: {report:?}"
            );
            assert!(
                report.cost_cap_passed.is_some(),
                "checked without a verdict: {report:?}"
            );
        }
        if report.cost_cap_passed.is_some() {
            assert!(
                report.cost_cap_checked,
                "verdict without a check: {report:?}"
            );
        }
        if report.cap_enforced {
            assert!(
                report.cost_cap_configured,
                "enforcement claimed without a cap: {report:?}"
            );
        }
    }

    #[test]
    fn cap_status_combinations_are_serialized_explicitly() {
        let _guard = SnapshotGuard::priced(-60_000);
        let root = temp_root("f16-status");
        let base = priced_opts(&root);
        let checked = capped(base.clone(), 5.0);
        let over = capped(base.clone(), 1.0);
        let open = base.clone();

        let cases: Vec<(&str, RunOptions, u64, u64)> = vec![
            ("checked-passed", checked, 1_000_000, 0),
            ("checked-failed", over, 2_000_000, 0),
            ("unchecked", open, 1_000, 100),
        ];
        for (label, options, input, output) in cases {
            let (report, _) = cost_gate(&options, input, output, now_ms());
            assert_cap_invariants(&report);
            let value = serde_json::to_value(&report).expect("serialize");
            // The key stays present for "no check": a consumer reading a boolean
            // must not mistake an omitted field for a passed value.
            assert!(value.get("cost_cap_passed").is_some(), "{label}: {value}");
            assert_eq!(
                value["cost_cap_configured"],
                json!(report.cost_cap_configured),
                "{label}"
            );
            assert_eq!(value["cost_cap_checked"], json!(report.cost_cap_checked));
            assert_eq!(
                value["cost_cap_passed"],
                match report.cost_cap_passed {
                    Some(v) => json!(v),
                    None => json!(null),
                }
            );
            if !report.cost_cap_checked {
                assert_eq!(value["cost_cap_passed"], json!(null), "{label}");
            }
        }
    }

    fn d1b_argv(extra: &[&str]) -> Vec<String> {
        let mut argv = vec![
            "d1b-run".to_string(),
            "--d1b-upstream".to_string(),
            "https://api.gameaigc.cn".to_string(),
            "--d1b-model".to_string(),
            "test-model".to_string(),
        ];
        argv.extend(extra.iter().map(|arg| arg.to_string()));
        argv
    }

    #[test]
    fn the_cli_cap_implies_the_price_requirement() {
        let options = parse_args(&d1b_argv(&["--d1b-cost-cap", "1.5"])).expect("parse");
        assert_eq!(options.cost_cap, Some(1.5));
        assert!(
            options.require_cost_cap,
            "setting a cap must also set the requirement"
        );
        assert!(options.cap_policy_required());
    }

    #[test]
    fn the_cli_rejects_caps_that_are_not_amounts() {
        for raw in ["NaN", "nan", "inf", "-1", "-0.5"] {
            let err =
                parse_args(&d1b_argv(&["--d1b-cost-cap", raw])).expect_err("must be rejected");
            assert!(err.contains("finite non-negative"), "{raw}: {err}");
        }
        for raw in ["abc", ""] {
            let err =
                parse_args(&d1b_argv(&["--d1b-cost-cap", raw])).expect_err("must be rejected");
            assert!(err.contains("invalid --d1b-cost-cap"), "{raw}: {err}");
        }
    }

    #[test]
    fn a_zero_cli_cap_is_accepted_as_a_spend_nothing_bound() {
        let options = parse_args(&d1b_argv(&["--d1b-cost-cap", "0"])).expect("parse");
        assert_eq!(options.cost_cap, Some(0.0));
        assert!(options.require_cost_cap);
    }
}

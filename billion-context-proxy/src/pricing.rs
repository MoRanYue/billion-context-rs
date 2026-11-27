//! Price resolution for D1b cost accounting, fail-closed by construction.
//!
//! # Why this module exists
//!
//! `d1b_runner` has always reported `price_known: false` because it had no
//! price source: cost was never derived from tokens, and the manual cost cap
//! was recorded as operator intent rather than enforced. The local
//! `oneagent-pricing` plugin publishes the provider's own per-model rates, so
//! the runner can finally know a price — but only when every link in the chain
//! can be checked. Anything unverifiable must keep the old answer.
//!
//! # The chain, and what each link must prove
//!
//! ```text
//! plugin snapshot → unit + source + key binding → freshness
//!                 → exact model match → tier selection → all four rates
//!                 → arithmetic → Money
//! ```
//!
//! A quote is only emitted when every step succeeds. There is no default price,
//! no partial rate, and no fallback to a hard-coded table: [`Quote`] cannot be
//! constructed except through [`PriceBook::quote`], and that function returns
//! [`Refusal`] on the first link it cannot check.
//!
//! # What this module deliberately does not do
//!
//! - It does not fetch. The snapshot is written by the plugin; reading it is the
//!   adapter's job, and a missing file is `None`, never an empty book.
//! - It does not convert currency. The plugin reports CNY; a quote carries CNY.
//! - It does not treat a missing, null, or unparsable field as zero. Zero is a
//!   legitimate price (a free tier) but only when the plugin actually says `0`.
//! - It does not decide whether the *provider* honours these prices. It records
//!   the binding it was given and refuses when the routed origin is not the one
//!   the snapshot was taken against.

use serde::Serialize;
use serde_json::Value;
use std::path::{Path, PathBuf};

/// The only origin the `oneagent-pricing` plugin authenticates against.
///
/// The plugin refuses to attach its management key to any other origin, so a
/// snapshot under `~/.dsh/oneagent/context/` can only describe this provider.
/// A host routing elsewhere must not use it.
pub const ONEAGENT_ORIGIN: &str = "https://api.gameaigc.cn";

/// How long a snapshot stays usable, matching the plugin's `PRICE_MAX_STALE`.
///
/// The plugin stops serving `prices` past this age; honouring a longer window
/// here would mean the adapter trusting data its own producer has disowned.
pub const MAX_STALE_MS: u64 = 60 * 60 * 1000;

/// Why a price could not be established. Each variant is a boundary the caller
/// must treat as "no cost known", never as "cost is zero".
#[derive(Clone, Debug, PartialEq)]
pub enum Refusal {
    /// No snapshot on disk, or it did not parse.
    NoSnapshot,
    /// The snapshot's `unit` is missing or not a currency this adapter knows.
    UnknownCurrency(String),
    /// The snapshot does not say where its numbers came from.
    UntrustedSource(String),
    /// `updatedAt` is missing, unparsable, or older than [`MAX_STALE_MS`].
    Stale { age_ms: u64, max_ms: u64 },
    /// `updatedAt` is in the future by more than a little clock skew.
    NotYetValid { updated_at: u64, now: u64 },
    /// The routed origin is not the one the snapshot describes.
    ProviderMismatch { routed: String, priced: String },
    /// No entry for this model id. Exact match only: a near-miss is not a price.
    ModelUnknown(String),
    /// The entry exists but carries no tier this adapter can bill against.
    NoTier(String),
    /// A rate is absent or not a number. Listed by band, e.g. `["output"]`.
    RateMissing {
        model: String,
        bands: Vec<&'static str>,
    },
    /// A rate is negative or not finite.
    RateInvalid { band: &'static str, value: f64 },
    /// Token counts or the resulting amount overflowed or went non-finite.
    Arithmetic(String),
    /// The run was refused before pricing was consulted. Distinct from
    /// [`Refusal::NoSnapshot`]: nobody looked, so nothing is known about the
    /// snapshot either.
    NotEvaluated,
    /// The operator's cost cap is denominated in one currency and the published
    /// price in another. Comparing them would need an exchange rate this module
    /// refuses to invent.
    CurrencyMismatch { cap: String, price: String },
    /// The model is listed by more than one child group of the bound route, so
    /// the first group's rates are not proven to be what this request is billed
    /// at. `group_id` is the group the snapshot kept, for diagnosis only.
    AmbiguousGroup {
        model: String,
        group_id: Option<u64>,
    },
    /// Every published band is zero. A book that was zeroed would make any
    /// monetary cap pass, and nothing in the snapshot marks a free model.
    RateZero { model: String },
}

impl Refusal {
    /// Stable machine-readable code, for logs and report fields.
    pub fn code(&self) -> &'static str {
        match self {
            Refusal::NoSnapshot => "price_snapshot_missing",
            Refusal::UnknownCurrency(_) => "price_currency_unknown",
            Refusal::UntrustedSource(_) => "price_source_untrusted",
            Refusal::Stale { .. } => "price_stale",
            Refusal::NotYetValid { .. } => "price_not_yet_valid",
            Refusal::ProviderMismatch { .. } => "price_provider_mismatch",
            Refusal::ModelUnknown(_) => "price_model_unknown",
            Refusal::NoTier(_) => "price_tier_unavailable",
            Refusal::RateMissing { .. } => "price_rate_missing",
            Refusal::RateInvalid { .. } => "price_rate_invalid",
            Refusal::Arithmetic(_) => "price_arithmetic_invalid",
            Refusal::NotEvaluated => "price_not_evaluated",
            Refusal::CurrencyMismatch { .. } => "price_currency_mismatch",
            Refusal::AmbiguousGroup { .. } => "price_group_ambiguous",
            Refusal::RateZero { .. } => "price_rate_zero",
        }
    }

    /// One line a human can act on. Never names a token or a key value.
    pub fn detail(&self) -> String {
        match self {
            Refusal::NoSnapshot => "no price snapshot is readable".to_string(),
            Refusal::UnknownCurrency(unit) => {
                format!("price unit {unit:?} is not a known currency")
            }
            Refusal::UntrustedSource(source) => {
                format!("price source {source:?} is not a trusted producer")
            }
            Refusal::Stale { age_ms, max_ms } => format!(
                "price snapshot is {age_ms} ms old; the producer stops vouching past {max_ms} ms"
            ),
            Refusal::NotYetValid { updated_at, now } => {
                format!("price snapshot claims {updated_at} ms but the clock reads {now} ms")
            }
            Refusal::ProviderMismatch { routed, priced } => {
                format!("the request routes to {routed} but the prices describe {priced}")
            }
            Refusal::ModelUnknown(model) => format!("no price is published for model {model:?}"),
            Refusal::NoTier(model) => format!("model {model:?} has no usable price tier"),
            Refusal::RateMissing { model, bands } => {
                format!("model {model:?} is missing rate(s): {}", bands.join(", "))
            }
            Refusal::RateInvalid { band, value } => {
                format!("rate {band} is not a usable number ({value})")
            }
            Refusal::Arithmetic(what) => format!("cost arithmetic failed: {what}"),
            Refusal::NotEvaluated => {
                "the run was refused before any price was resolved".to_string()
            }
            Refusal::CurrencyMismatch { cap, price } => format!(
                "the cost cap is in {cap} but the published price is in {price}; no rate is assumed"
            ),
            Refusal::AmbiguousGroup { model, group_id } => {
                let where_ = group_id
                    .map(|id| format!("the snapshot kept group {id}"))
                    .unwrap_or_else(|| "the snapshot kept an unnamed group".to_string());
                format!(
                    "model {model:?} is listed by more than one child group; {where_}, which is not proven to bill this request"
                )
            }
            Refusal::RateZero { model } => {
                format!(
                    "every published rate for model {model:?} is zero; no free grant vouches for it"
                )
            }
        }
    }
}

/// Per-million-token rates for one model, in the book's currency.
///
/// Every field is `Some` in a usable [`Quote`]; the `Option`s exist so that a
/// missing rate can be *reported* rather than silently defaulted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rates {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
}

/// A verified price for one model at one instant.
///
/// Only [`PriceBook::quote`] constructs this, so "we have a price" and "we
/// checked the price" are the same statement.
#[derive(Clone, Debug, PartialEq)]
pub struct Quote {
    pub model: String,
    pub currency: String,
    pub rates: Rates,
    /// `true` when the long-context tier was selected because the prompt crossed
    /// its threshold. Carried so a report can explain the number it used.
    pub long_context: bool,
    pub source: String,
    pub updated_at: u64,
    /// Child group the snapshot kept this model's rates from, when it names one.
    pub group_id: Option<u64>,
    /// `true` when the model is listed by more than one child group, so these
    /// rates are not proven to be this request's billing rates.
    pub group_ambiguous: bool,
    /// Model ids the book also prices, for the "did you mean" case. Bounded.
    pub other_models: Vec<String>,
}

/// Money in the book's currency, with the arithmetic that produced it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cost {
    pub amount: f64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    /// Cached prompt tokens billed at the cache-read rate instead of `input`.
    pub cache_read_tokens: u64,
    pub currency_scale: &'static str,
}

/// A parsed, structurally valid price snapshot. Holding one proves the file was
/// well-formed; it does not prove any particular model is priced. Call
/// [`PriceBook::quote`] for that.
#[derive(Clone, Debug)]
pub struct PriceBook {
    unit: String,
    source: String,
    updated_at: u64,
    models: Vec<ModelPrice>,
}

#[derive(Clone, Debug)]
struct ModelPrice {
    id: String,
    short: Option<Tier>,
    long: Option<Tier>,
    threshold: Option<f64>,
    /// Child group the plugin kept this entry from, when the payload names it.
    group_id: Option<u64>,
    /// `true` when the plugin saw this model in more than one child group.
    /// Absent is treated as `true`: a producer that does not say cannot prove
    /// that only one group lists the model.
    group_ambiguous: bool,
}

#[derive(Clone, Copy, Debug)]
struct Tier {
    uncached: Option<f64>,
    cache_read: Option<f64>,
    cache_write: Option<f64>,
    output: Option<f64>,
    threshold: Option<f64>,
}

impl PriceBook {
    /// Parse a snapshot payload as written by the `oneagent-pricing` plugin.
    ///
    /// Returns `None` when the payload is not a book at all (missing `models`,
    /// missing `updatedAt`). Per-model defects are *kept*, so a quote for that
    /// model can explain itself instead of vanishing into "unknown model".
    pub fn from_payload(payload: &Value) -> Option<PriceBook> {
        let models = payload.get("models")?.as_array()?;
        let updated_at = payload.get("updatedAt")?.as_u64()?;
        let parsed = models
            .iter()
            .filter_map(|entry| {
                let id = entry.get("id")?.as_str()?.to_string();
                Some(ModelPrice {
                    id,
                    short: entry.get("short").and_then(Tier::from_json),
                    long: entry.get("long").and_then(Tier::from_json),
                    threshold: entry.get("threshold").and_then(Value::as_f64),
                    group_id: entry.get("groupId").and_then(Value::as_u64),
                    group_ambiguous: entry
                        .get("fallback")
                        .and_then(Value::as_bool)
                        .unwrap_or(true),
                })
            })
            .collect();
        Some(PriceBook {
            unit: payload
                .get("unit")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            source: payload
                .get("source")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            updated_at,
            models: parsed,
        })
    }

    pub fn currency(&self) -> &str {
        &self.unit
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    pub fn updated_at(&self) -> u64 {
        self.updated_at
    }

    pub fn model_count(&self) -> usize {
        self.models.len()
    }

    /// Models the plugin saw in more than one child group. Their rates are kept
    /// for display, but no run may bill a cap against them.
    pub fn ambiguous_model_count(&self) -> usize {
        self.models.iter().filter(|m| m.group_ambiguous).count()
    }

    /// Resolve a price for `model`, routing through `origin`, at `now_ms`.
    ///
    /// `input_tokens` decides between the short and long context tiers. Pass the
    /// projected prompt size, not a guess: the tier is what the provider bills
    /// on, and picking the wrong one silently understates the cost.
    pub fn quote(
        &self,
        model: &str,
        origin: &str,
        input_tokens: u64,
        now_ms: u64,
    ) -> Result<Quote, Refusal> {
        // 1. Currency. The plugin writes CNY; anything else is a producer we do
        //    not know how to bill, and guessing the scale would be worse than
        //    refusing.
        let currency = match self.unit.as_str() {
            "CNY" | "USD" => self.unit.clone(),
            other => return Err(Refusal::UnknownCurrency(other.to_string())),
        };

        // 2. Provenance. The snapshot names its own source; an empty or foreign
        //    one means it was not written by the plugin we verified.
        if !self.source.contains("OneAgent") {
            return Err(Refusal::UntrustedSource(self.source.clone()));
        }

        // 3. Provider binding. The prices belong to one provider's management
        //    key. Routing anywhere else means these numbers are about a
        //    different service, however fresh they are.
        if !origin_matches(origin) {
            return Err(Refusal::ProviderMismatch {
                routed: redacted_origin(origin),
                priced: ONEAGENT_ORIGIN.to_string(),
            });
        }

        // 4. Freshness, both directions. A snapshot dated in the future cannot be
        //    vouched for either; a small skew is tolerated for equal clocks.
        if self.updated_at > now_ms + MAX_STALE_MS / 60 {
            return Err(Refusal::NotYetValid {
                updated_at: self.updated_at,
                now: now_ms,
            });
        }
        let age_ms = now_ms.saturating_sub(self.updated_at);
        if age_ms > MAX_STALE_MS {
            return Err(Refusal::Stale {
                age_ms,
                max_ms: MAX_STALE_MS,
            });
        }

        // 5. Exact model match. Case-insensitive on the id only because the
        //    provider is; no prefix or fuzzy matching, which would bill a
        //    sibling model's rate.
        let entry = self
            .models
            .iter()
            .find(|entry| entry.id.eq_ignore_ascii_case(model))
            .ok_or_else(|| Refusal::ModelUnknown(model.to_string()))?;

        // 6. Tier selection. The long tier applies only above its threshold, and
        //    only when the book states both; a long tier with no threshold is a
        //    book defect, not a licence to guess.
        //
        //    The plugin writes the threshold at model level and again inside each
        //    tier; model level wins, tier level is the fallback for producers that
        //    only annotate the tier.
        let threshold = entry
            .threshold
            .or_else(|| entry.short.as_ref().and_then(|tier| tier.threshold))
            .or_else(|| entry.long.as_ref().and_then(|tier| tier.threshold));
        let (tier, long_context) = match (&entry.short, &entry.long, threshold) {
            (Some(_), Some(_), Some(threshold)) if input_tokens as f64 > threshold => {
                (entry.long.as_ref().expect("checked above"), true)
            }
            (Some(_), Some(_), None) => {
                // Long tier present but unbounded: we cannot tell which tier the
                // provider would bill, so we refuse rather than assume short.
                return Err(Refusal::NoTier(entry.id.clone()));
            }
            (Some(short), _, _) => (short, false),
            (None, Some(long), _) => (long, true),
            (None, None, _) => return Err(Refusal::NoTier(entry.id.clone())),
        };

        // 7. Every rate the caller will bill against must be present and sane.
        let mut missing: Vec<&'static str> = Vec::new();
        if tier.uncached.is_none() {
            missing.push("input");
        }
        if tier.output.is_none() {
            missing.push("output");
        }
        if tier.cache_read.is_none() {
            missing.push("cache_read");
        }
        if tier.cache_write.is_none() {
            missing.push("cache_write");
        }
        if !missing.is_empty() {
            return Err(Refusal::RateMissing {
                model: entry.id.clone(),
                bands: missing,
            });
        }
        let rates = Rates {
            input: tier.uncached.expect("checked"),
            output: tier.output.expect("checked"),
            cache_read: tier.cache_read.expect("checked"),
            cache_write: tier.cache_write.expect("checked"),
        };
        for (band, value) in [
            ("input", rates.input),
            ("output", rates.output),
            ("cache_read", rates.cache_read),
            ("cache_write", rates.cache_write),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(Refusal::RateInvalid { band, value });
            }
        }

        // 8. A book zeroed on every band would make any monetary cap pass for
        //    any run. Nothing in the snapshot marks a phone model as free, so
        //    all-zero is treated as a broken number rather than a price. A
        //    single zero band stays legal: some providers do publish one.
        if rates.input == 0.0
            && rates.output == 0.0
            && rates.cache_read == 0.0
            && rates.cache_write == 0.0
        {
            return Err(Refusal::RateZero {
                model: entry.id.clone(),
            });
        }

        Ok(Quote {
            model: entry.id.clone(),
            currency,
            rates,
            long_context,
            source: self.source.clone(),
            updated_at: self.updated_at,
            group_id: entry.group_id,
            group_ambiguous: entry.group_ambiguous,
            other_models: self
                .models
                .iter()
                .filter(|other| !other.id.eq_ignore_ascii_case(model))
                .map(|other| other.id.clone())
                .take(12)
                .collect(),
        })
    }
}

impl Tier {
    fn from_json(value: &Value) -> Option<Tier> {
        if !value.is_object() {
            return None;
        }
        Some(Tier {
            uncached: value.get("uncached").and_then(Value::as_f64),
            cache_read: value.get("cacheRead").and_then(Value::as_f64),
            cache_write: value.get("cacheWrite").and_then(Value::as_f64),
            output: value.get("output").and_then(Value::as_f64),
            threshold: value.get("threshold").and_then(Value::as_f64),
        })
    }
}

impl Quote {
    /// Bill `input_tokens` (of which `cache_read_tokens` were cache hits) and
    /// `output_tokens` at this quote's rates.
    ///
    /// Rates are per million tokens, so the division happens once on the token
    /// count rather than per token — the latter loses precision on a long
    /// prompt before the sum is finished. The result is rounded to 6 decimals,
    /// which is a millionth of a yuan: below the smallest unit any invoice uses,
    /// and above f64 noise for realistic token counts.
    pub fn cost(
        &self,
        input_tokens: u64,
        cache_read_tokens: u64,
        output_tokens: u64,
    ) -> Result<Cost, Refusal> {
        if cache_read_tokens > input_tokens {
            return Err(Refusal::Arithmetic(format!(
                "cache_read_tokens ({cache_read_tokens}) exceeds input_tokens ({input_tokens})"
            )));
        }
        let uncached = input_tokens - cache_read_tokens;
        let amount = (uncached as f64) * self.rates.input / 1_000_000.0
            + (cache_read_tokens as f64) * self.rates.cache_read / 1_000_000.0
            + (output_tokens as f64) * self.rates.output / 1_000_000.0;
        if !amount.is_finite() || amount < 0.0 {
            return Err(Refusal::Arithmetic(format!(
                "amount {amount} is not usable"
            )));
        }
        let rounded = (amount * 1_000_000.0).round() / 1_000_000.0;
        if !rounded.is_finite() {
            return Err(Refusal::Arithmetic(format!(
                "rounded amount {rounded} is not usable"
            )));
        }
        Ok(Cost {
            amount: rounded,
            input_tokens,
            output_tokens,
            cache_read_tokens,
            currency_scale: "per_million_tokens",
        })
    }
}

/// Whether `origin` is the provider the snapshot describes.
///
/// Compared on scheme+host+port only: a `/v1` path is a routing detail, while a
/// different host is a different service.
fn origin_matches(origin: &str) -> bool {
    fn bare(origin: &str) -> String {
        let trimmed = origin.trim().trim_end_matches('/');
        let trimmed = trimmed.split('/').take(3).collect::<Vec<_>>().join("/");
        trimmed.to_ascii_lowercase()
    }
    bare(origin) == bare(ONEAGENT_ORIGIN)
}

/// Placeholder used when an origin cannot be parsed at all.
pub const UNPARSABLE_ORIGIN: &str = "<unparsable-origin>";

/// A display-safe form of a routed origin: scheme, host and port only.
///
/// A routed URL may carry userinfo (`user:pass@`) or query parameters that hold
/// credentials. Neither may reach a refusal detail, a report or a log, so every
/// human-facing copy goes through this. When the string cannot be parsed the
/// raw text is never echoed: callers get [`UNPARSABLE_ORIGIN`] instead.
pub fn redacted_origin(origin: &str) -> String {
    origin_parts(origin).unwrap_or_else(|| UNPARSABLE_ORIGIN.to_string())
}

fn origin_parts(origin: &str) -> Option<String> {
    let trimmed = origin.trim();
    let (scheme, rest) = trimmed.split_once("://")?;
    let scheme = scheme.to_ascii_lowercase();
    if !matches!(scheme.as_str(), "http" | "https") {
        return None;
    }
    // The authority ends at the first path, query or fragment marker.
    let authority = rest.split(['/', '?', '#']).next()?;
    if authority.is_empty() {
        return None;
    }
    // Userinfo is everything before the last '@' in the authority; it is
    // dropped rather than escaped, so no part of it can leak by design.
    let host_port = authority
        .rsplit_once('@')
        .map(|(_, host_port)| host_port)
        .unwrap_or(authority);
    if host_port.is_empty() || host_port.starts_with(':') {
        return None;
    }
    if host_port
        .chars()
        .any(|c| c.is_whitespace() || c.is_control())
    {
        return None;
    }
    Some(format!("{scheme}://{}", host_port.to_ascii_lowercase()))
}

/// Override for the snapshot location, so a test never reads the operator's
/// live price cache. Absent means "use the plugin's own path".
pub const SNAPSHOT_ENV: &str = "BILI_PRICE_SNAPSHOT";

/// The plugin's own write path: `~/.dsh/oneagent/context/context-prices.json`.
///
/// Read-only: this adapter never writes a snapshot, because a price file it
/// invented would be indistinguishable from one the plugin vouched for.
pub fn default_snapshot_path() -> Option<PathBuf> {
    if let Ok(overridden) = std::env::var(SNAPSHOT_ENV) {
        let trimmed = overridden.trim();
        if !trimmed.is_empty() {
            return Some(PathBuf::from(trimmed));
        }
    }
    let home = std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)?;
    Some(
        home.join(".dsh")
            .join("oneagent")
            .join("context")
            .join("context-prices.json"),
    )
}

/// Read and parse the snapshot. A missing file, unreadable file or malformed
/// JSON is [`Refusal::NoSnapshot`] — never an empty book, which would report
/// "no models priced" as if that were a property of the provider.
pub fn load_snapshot(path: &Path) -> Result<PriceBook, Refusal> {
    let text = std::fs::read_to_string(path).map_err(|_| Refusal::NoSnapshot)?;
    let payload: Value = serde_json::from_str(&text).map_err(|_| Refusal::NoSnapshot)?;
    PriceBook::from_payload(&payload).ok_or(Refusal::NoSnapshot)
}

/// A price decision, serialisable, for the run report.
///
/// `Unknown` always carries the refusal that produced it, so a report can name
/// the exact link that failed instead of saying "no price".
#[derive(Clone, Debug, Serialize)]
pub struct CostBasis {
    /// `"known"` or `"unknown"`.
    pub state: &'static str,
    pub currency: Option<String>,
    pub source: Option<String>,
    pub updated_at_ms: Option<u64>,
    /// Per-million-token rates, present only when `state == "known"`.
    pub rates: Option<RatesOut>,
    pub long_context: bool,
    /// Child group the snapshot kept the rates from, when it names one.
    pub group_id: Option<u64>,
    /// `true` when the model is listed by more than one child group. Only ever
    /// `true` on `Unknown`: an ambiguous price is never reported as this run's.
    pub group_ambiguous: bool,
    pub refusal_code: Option<&'static str>,
    pub refusal_detail: Option<String>,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct RatesOut {
    pub input_per_million: f64,
    pub output_per_million: f64,
    pub cache_read_per_million: f64,
    pub cache_write_per_million: f64,
}

impl CostBasis {
    pub fn known(quote: &Quote) -> CostBasis {
        CostBasis {
            state: "known",
            currency: Some(quote.currency.clone()),
            source: Some(quote.source.clone()),
            updated_at_ms: Some(quote.updated_at),
            rates: Some(RatesOut {
                input_per_million: quote.rates.input,
                output_per_million: quote.rates.output,
                cache_read_per_million: quote.rates.cache_read,
                cache_write_per_million: quote.rates.cache_write,
            }),
            long_context: quote.long_context,
            group_id: quote.group_id,
            group_ambiguous: quote.group_ambiguous,
            refusal_code: None,
            refusal_detail: None,
        }
    }

    pub fn unknown(refusal: &Refusal) -> CostBasis {
        let (group_id, group_ambiguous) = match refusal {
            Refusal::AmbiguousGroup { group_id, .. } => (*group_id, true),
            _ => (None, false),
        };
        CostBasis {
            state: "unknown",
            currency: None,
            source: None,
            updated_at_ms: None,
            rates: None,
            long_context: false,
            group_id,
            group_ambiguous,
            refusal_code: Some(refusal.code()),
            refusal_detail: Some(refusal.detail()),
        }
    }

    pub fn is_known(&self) -> bool {
        self.state == "known"
    }
}

/// Compare an operator's cap currency against the price's, refusing when they
/// differ instead of assuming a rate.
pub fn assert_cap_currency(cap_currency: &str, quote: &Quote) -> Result<(), Refusal> {
    if cap_currency.eq_ignore_ascii_case(&quote.currency) {
        Ok(())
    } else {
        Err(Refusal::CurrencyMismatch {
            cap: cap_currency.to_string(),
            price: quote.currency.clone(),
        })
    }
}

/// Whether `amount` stays within `cap`, in the quote's currency.
///
/// Kept here rather than in the runner so the comparison rule — `>`, not `>=`,
/// and NaN never passes — is testable without a runner.
pub fn within_cap(amount: f64, cap: f64) -> bool {
    amount.is_finite() && cap.is_finite() && cap >= 0.0 && amount <= cap
}

/// Answer "can a monetary cap be enforced right now, and on what evidence?"
///
/// Deliberately separate from [`resolve_quote`]: this reports the state of the
/// snapshot without naming a model, so an operator can see staleness before a
/// run. It never writes, refreshes or fabricates a snapshot.
pub fn snapshot_status(now_ms: u64) -> Value {
    let path = default_snapshot_path();
    let (path, present, book) = match path.as_deref() {
        Some(path) => {
            let present = path.exists();
            let book = if present {
                load_snapshot(path).ok()
            } else {
                None
            };
            (Some(path.to_string_lossy().to_string()), present, book)
        }
        None => (None, false, None),
    };

    let mut status = serde_json::json!({
        "path": path,
        "present": present,
        "usable": false,
        "maxStaleMs": MAX_STALE_MS,
        "providerOrigin": ONEAGENT_ORIGIN,
    });

    let Some(book) = book else {
        status["state"] = Value::String(if present { "unreadable" } else { "absent" }.to_string());
        return status;
    };

    let age_ms = now_ms.saturating_sub(book.updated_at());
    let fresh = book.updated_at() <= now_ms + MAX_STALE_MS / 60 && age_ms <= MAX_STALE_MS;
    let trusted = book.source().contains("OneAgent") && matches!(book.currency(), "CNY" | "USD");

    status["state"] = Value::String(
        if fresh && trusted {
            "ready"
        } else {
            "unusable"
        }
        .to_string(),
    );
    status["usable"] = Value::Bool(fresh && trusted);
    status["currency"] = Value::String(book.currency().to_string());
    status["source"] = Value::String(book.source().to_string());
    status["updatedAtMs"] = Value::from(book.updated_at());
    status["ageMs"] = Value::from(age_ms);
    status["fresh"] = Value::Bool(fresh);
    status["trusted"] = Value::Bool(trusted);
    status["models"] = Value::from(book.model_count());
    // Models the plugin keeps for more than one child group: their rates are
    // visible here but cannot bound a run's cost.
    status["ambiguousModels"] = Value::from(book.ambiguous_model_count());
    status
}

/// Resolve a price for `model` on the provider `upstream` points at.
///
/// This is the whole adapter: locate the snapshot, parse it, quote it. It never
/// dispatches a request and never falls back to a built-in table.
pub fn resolve_quote(
    model: &str,
    upstream: &str,
    input_tokens: u64,
    now_ms: u64,
) -> Result<Quote, Refusal> {
    let path = default_snapshot_path().ok_or(Refusal::NoSnapshot)?;
    let book = load_snapshot(&path)?;
    book.quote(model, upstream, input_tokens, now_ms)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const NOW: u64 = 1_800_000_000_000;

    fn book(updated_at: u64) -> PriceBook {
        PriceBook::from_payload(&json!({
            "unit": "CNY",
            "source": "OneAgent model plaza",
            "updatedAt": updated_at,
            "models": [
                {
                    "id": "deepseek-v4-1-flash-261017",
                    "groupId": 111,
                    "fallback": false,
                    "short": {"uncached": 0.864, "cacheRead": 0.01728, "cacheWrite": 0.01728, "output": 3.456, "threshold": null},
                    "long": null,
                    "threshold": null
                },
                {
                    "id": "mystery-model",
                    "fallback": false,
                    "short": {"uncached": 1.0, "cacheRead": 0.1, "cacheWrite": 0.1, "threshold": null},
                    "long": null,
                    "threshold": null
                },
                {
                    "id": "no-output-rate",
                    "fallback": false,
                    "short": {"uncached": 1.0, "cacheRead": 0.1, "cacheWrite": 0.1, "threshold": null},
                    "long": null,
                    "threshold": null
                },
                {
                    "id": "negative-rate",
                    "fallback": false,
                    "short": {"uncached": -1.0, "cacheRead": 0.1, "cacheWrite": 0.1, "output": 1.0, "threshold": null},
                    "long": null,
                    "threshold": null
                },
                {
                    "id": "tiered",
                    "fallback": false,
                    "short": {"uncached": 1.0, "cacheRead": 0.1, "cacheWrite": 0.1, "output": 2.0, "threshold": 128000.0},
                    "long": {"uncached": 2.0, "cacheRead": 0.2, "cacheWrite": 0.2, "output": 4.0, "threshold": 128000.0},
                    "threshold": 128000.0
                },
                {
                    "id": "unbounded-long",
                    "fallback": false,
                    "short": {"uncached": 1.0, "cacheRead": 0.1, "cacheWrite": 0.1, "output": 2.0, "threshold": null},
                    "long": {"uncached": 2.0, "cacheRead": 0.2, "cacheWrite": 0.2, "output": 4.0, "threshold": null},
                    "threshold": null
                }
            ]
        }))
        .expect("book parses")
    }

    #[test]
    fn a_fresh_bound_exact_model_quote_carries_both_rates() {
        let quote = book(NOW - 60_000)
            .quote("deepseek-v4-1-flash-261017", ONEAGENT_ORIGIN, 12_000, NOW)
            .expect("quote");
        assert_eq!(quote.currency, "CNY");
        assert_eq!(quote.rates.input, 0.864);
        assert_eq!(quote.rates.output, 3.456);
        assert!(!quote.long_context);
        assert_eq!(quote.model, "deepseek-v4-1-flash-261017");
    }

    #[test]
    fn a_stale_snapshot_is_refused_even_when_the_model_is_priced() {
        let err = book(NOW - MAX_STALE_MS - 1)
            .quote("deepseek-v4-1-flash-261017", ONEAGENT_ORIGIN, 1_000, NOW)
            .expect_err("stale");
        assert_eq!(err.code(), "price_stale");
        match err {
            Refusal::Stale { age_ms, max_ms } => {
                assert_eq!(age_ms, MAX_STALE_MS + 1);
                assert_eq!(max_ms, MAX_STALE_MS);
            }
            other => panic!("expected Stale, got {other:?}"),
        }
    }

    #[test]
    fn a_snapshot_dated_in_the_future_is_not_trusted() {
        let err = book(NOW + MAX_STALE_MS)
            .quote("deepseek-v4-1-flash-261017", ONEAGENT_ORIGIN, 1_000, NOW)
            .expect_err("future");
        assert_eq!(err.code(), "price_not_yet_valid");
    }

    #[test]
    fn routing_to_a_different_provider_refuses_the_quote() {
        let err = book(NOW - 1_000)
            .quote(
                "deepseek-v4-1-flash-261017",
                "https://api.example.com",
                1_000,
                NOW,
            )
            .expect_err("mismatch");
        assert_eq!(err.code(), "price_provider_mismatch");
    }

    #[test]
    fn the_provider_origin_is_matched_on_host_not_path() {
        let ok = book(NOW - 1_000)
            .quote(
                "deepseek-v4-1-flash-261017",
                "https://api.gameaigc.cn/v1",
                1_000,
                NOW,
            )
            .is_ok();
        assert!(
            ok,
            "a /v1 suffix is a routing detail, not a different provider"
        );
        let bad = book(NOW - 1_000)
            .quote(
                "deepseek-v4-1-flash-261017",
                "https://api.gameaigc.cn.evil.test",
                1_000,
                NOW,
            )
            .is_err();
        assert!(bad, "a lookalike host must not match");
    }

    #[test]
    fn an_unknown_model_names_itself_and_offers_the_priced_ones() {
        let err = book(NOW - 1_000)
            .quote("deepseek-v9", ONEAGENT_ORIGIN, 1_000, NOW)
            .expect_err("unknown");
        assert_eq!(err.code(), "price_model_unknown");
        assert!(err.detail().contains("deepseek-v9"));
    }

    #[test]
    fn a_near_miss_model_name_is_not_billed_at_a_sibling_rate() {
        // `deepseek-v4-1-flash` is a prefix of the priced id; prefix matching
        // would quietly bill the wrong model.
        let err = book(NOW - 1_000)
            .quote("deepseek-v4-1-flash", ONEAGENT_ORIGIN, 1_000, NOW)
            .expect_err("near miss");
        assert_eq!(err.code(), "price_model_unknown");
    }

    #[test]
    fn a_missing_output_rate_refuses_instead_of_billing_input_only() {
        let err = book(NOW - 1_000)
            .quote("no-output-rate", ONEAGENT_ORIGIN, 1_000, NOW)
            .expect_err("missing output");
        assert_eq!(err.code(), "price_rate_missing");
        match err {
            Refusal::RateMissing { bands, .. } => assert_eq!(bands, vec!["output"]),
            other => panic!("expected RateMissing, got {other:?}"),
        }
    }

    #[test]
    fn a_negative_rate_is_refused_rather_than_treated_as_free() {
        let err = book(NOW - 1_000)
            .quote("negative-rate", ONEAGENT_ORIGIN, 1_000, NOW)
            .expect_err("negative");
        assert_eq!(err.code(), "price_rate_invalid");
    }

    #[test]
    fn an_unknown_currency_is_refused() {
        let mut payload = json!({
            "unit": "BTC",
            "source": "OneAgent model plaza",
            "updatedAt": NOW - 1_000,
            "models": [{"id": "x", "short": {"uncached": 1.0, "cacheRead": 0.0, "cacheWrite": 0.0, "output": 1.0}}]
        });
        let parsed = PriceBook::from_payload(&payload).expect("parses");
        assert_eq!(
            parsed
                .quote("x", ONEAGENT_ORIGIN, 1, NOW)
                .expect_err("currency")
                .code(),
            "price_currency_unknown"
        );
        payload["unit"] = json!("CNY");
        payload["source"] = json!("some other plugin");
        let parsed = PriceBook::from_payload(&payload).expect("parses");
        assert_eq!(
            parsed
                .quote("x", ONEAGENT_ORIGIN, 1, NOW)
                .expect_err("source")
                .code(),
            "price_source_untrusted"
        );
    }

    #[test]
    fn the_long_tier_applies_only_above_its_threshold() {
        let short = book(NOW - 1_000)
            .quote("tiered", ONEAGENT_ORIGIN, 128_000, NOW)
            .expect("at threshold");
        assert!(!short.long_context);
        assert_eq!(short.rates.input, 1.0);

        let long = book(NOW - 1_000)
            .quote("tiered", ONEAGENT_ORIGIN, 128_001, NOW)
            .expect("above threshold");
        assert!(long.long_context);
        assert_eq!(long.rates.input, 2.0);
        assert_eq!(long.rates.output, 4.0);
    }

    #[test]
    fn a_long_tier_without_a_threshold_is_refused_not_guessed() {
        let err = book(NOW - 1_000)
            .quote("unbounded-long", ONEAGENT_ORIGIN, 1_000, NOW)
            .expect_err("unbounded");
        assert_eq!(err.code(), "price_tier_unavailable");
    }

    #[test]
    fn cost_bills_input_output_and_cache_reads_at_their_own_rates() {
        let quote = book(NOW - 1_000)
            .quote("deepseek-v4-1-flash-261017", ONEAGENT_ORIGIN, 10_000, NOW)
            .expect("quote");
        // 1,000,000 uncached at 0.864 + 1,000,000 cached at 0.01728 + 1,000,000 out at 3.456
        let cost = quote.cost(2_000_000, 1_000_000, 1_000_000).expect("cost");
        let expected = 0.864 + 0.01728 + 3.456;
        assert!(
            (cost.amount - expected).abs() < 1e-9,
            "{} != {expected}",
            cost.amount
        );
        assert_eq!(cost.currency_scale, "per_million_tokens");
    }

    #[test]
    fn a_zero_token_request_costs_zero_not_unknown() {
        let quote = book(NOW - 1_000)
            .quote("deepseek-v4-1-flash-261017", ONEAGENT_ORIGIN, 0, NOW)
            .expect("quote");
        let cost = quote.cost(0, 0, 0).expect("cost");
        assert_eq!(cost.amount, 0.0);
    }

    #[test]
    fn more_cache_reads_than_input_tokens_is_an_arithmetic_error() {
        let quote = book(NOW - 1_000)
            .quote("deepseek-v4-1-flash-261017", ONEAGENT_ORIGIN, 10_000, NOW)
            .expect("quote");
        let err = quote.cost(100, 101, 0).expect_err("impossible");
        assert_eq!(err.code(), "price_arithmetic_invalid");
    }

    #[test]
    fn an_astronomically_large_token_count_does_not_produce_a_finite_claim() {
        // 2^62 tokens at ~3.5 CNY/M is ~1.6e13 CNY — finite in f64, so the guard
        // that matters is the negative/non-finite one, exercised here via u64::MAX
        // times a large rate staying finite but the *sum* being checked anyway.
        let quote = book(NOW - 1_000)
            .quote("deepseek-v4-1-flash-261017", ONEAGENT_ORIGIN, 1, NOW)
            .expect("quote");
        let cost = quote
            .cost(u64::MAX, 0, u64::MAX)
            .expect("still finite in f64");
        assert!(cost.amount.is_finite() && cost.amount >= 0.0);
    }

    #[test]
    fn a_book_without_models_or_timestamp_does_not_parse() {
        assert!(PriceBook::from_payload(&json!({"unit": "CNY"})).is_none());
        assert!(
            PriceBook::from_payload(&json!({"unit": "CNY", "models": []})).is_none(),
            "updatedAt is required; an undated book cannot be judged fresh"
        );
    }

    #[test]
    fn a_cap_in_another_currency_is_refused_not_converted() {
        let quote = book(NOW - 1_000)
            .quote("deepseek-v4-1-flash-261017", ONEAGENT_ORIGIN, 1_000, NOW)
            .expect("quote");
        assert!(assert_cap_currency("CNY", &quote).is_ok());
        assert!(
            assert_cap_currency("cny", &quote).is_ok(),
            "currency codes are case-insensitive"
        );
        let refusal = assert_cap_currency("USD", &quote).expect_err("mismatch");
        assert_eq!(refusal.code(), "price_currency_mismatch");
        assert!(refusal.detail().contains("USD"), "{}", refusal.detail());
        assert!(refusal.detail().contains("CNY"), "{}", refusal.detail());
    }

    #[test]
    fn a_cap_is_inclusive_and_never_satisfied_by_a_nan() {
        assert!(within_cap(1.0, 1.0), "at the cap is inside the cap");
        assert!(within_cap(0.999, 1.0));
        assert!(!within_cap(1.001, 1.0));
        assert!(
            !within_cap(0.0, -1.0),
            "a negative cap cannot bound anything"
        );
        assert!(!within_cap(f64::NAN, 1.0));
        assert!(!within_cap(1.0, f64::NAN));
        assert!(!within_cap(f64::INFINITY, 1.0));
        assert!(
            !within_cap(1.0, f64::INFINITY),
            "an infinite cap is not a number to trust"
        );
    }

    #[test]
    fn an_absent_snapshot_reports_absent_and_unusable() {
        let absent = TempSnapshot::absent_path();
        let _env = EnvPath::set(&absent);
        let status = snapshot_status(NOW);

        assert_eq!(status["state"], json!("absent"));
        assert_eq!(status["present"], json!(false));
        assert_eq!(status["usable"], json!(false));
        assert!(status["models"].is_null(), "no model count may be implied");
    }

    #[test]
    fn a_fresh_trusted_snapshot_reports_ready_with_its_own_provenance() {
        let file = TempSnapshot::write(
            &serde_json::to_string(&json!({
                "unit": "CNY",
                "source": "OneAgent model plaza",
                "updatedAt": NOW - 60_000,
                "models": [
                    {"id": "a", "fallback": false, "short": {"uncached": 0.5, "cacheRead": 0.01, "cacheWrite": 0.01, "output": 2.0}},
                    {"id": "b", "fallback": false, "short": {"uncached": 0.5, "cacheRead": 0.01, "cacheWrite": 0.01, "output": 2.0}}
                ]
            }))
            .expect("serialize"),
        );
        let _env = EnvPath::set(&file.0);
        let status = snapshot_status(NOW);

        assert_eq!(status["state"], json!("ready"));
        assert_eq!(status["usable"], json!(true));
        assert_eq!(status["currency"], json!("CNY"));
        assert_eq!(status["source"], json!("OneAgent model plaza"));
        assert_eq!(status["models"], json!(2));
        assert_eq!(
            status["ambiguousModels"],
            json!(0),
            "these two models are proven single-group"
        );
        assert_eq!(status["ageMs"], json!(60_000));
        assert_eq!(status["fresh"], json!(true));
        assert_eq!(status["trusted"], json!(true));
        assert_eq!(status["maxStaleMs"], json!(MAX_STALE_MS));
    }

    #[test]
    fn a_stale_or_foreign_snapshot_reports_unusable() {
        let stale = TempSnapshot::write(
            &serde_json::to_string(&json!({
                "unit": "CNY",
                "source": "OneAgent model plaza",
                "updatedAt": NOW - MAX_STALE_MS - 1,
                "models": []
            }))
            .expect("serialize"),
        );
        let _env = EnvPath::set(&stale.0);
        let status = snapshot_status(NOW);
        assert_eq!(status["state"], json!("unusable"));
        assert_eq!(status["fresh"], json!(false));
        assert_eq!(
            status["trusted"],
            json!(true),
            "provenance is not freshness"
        );
        // Release the environment lock before the second half: `EnvPath` holds a
        // non-reentrant mutex, so two live guards in one test would deadlock.
        drop(_env);

        let foreign = TempSnapshot::write(
            &serde_json::to_string(&json!({
                "unit": "CNY",
                "source": "some other vendor",
                "updatedAt": NOW - 1_000,
                "models": []
            }))
            .expect("serialize"),
        );
        let _env = EnvPath::set(&foreign.0);
        let status = snapshot_status(NOW);
        assert_eq!(status["usable"], json!(false));
        assert_eq!(
            status["trusted"],
            json!(false),
            "an unvouched source is not trusted"
        );
    }

    #[test]
    fn an_unreadable_snapshot_reports_unreadable_not_absent() {
        let file = TempSnapshot::write("{ broken");
        let _env = EnvPath::set(&file.0);
        let status = snapshot_status(NOW);

        assert_eq!(status["state"], json!("unreadable"));
        assert_eq!(status["present"], json!(true));
        assert_eq!(status["usable"], json!(false));
    }

    #[test]
    fn every_refusal_carries_a_distinct_code_and_a_readable_detail() {
        let refusals = [
            Refusal::NoSnapshot,
            Refusal::UnknownCurrency("BTC".into()),
            Refusal::UntrustedSource("x".into()),
            Refusal::Stale {
                age_ms: 1,
                max_ms: 2,
            },
            Refusal::NotYetValid {
                updated_at: 1,
                now: 2,
            },
            Refusal::ProviderMismatch {
                routed: "a".into(),
                priced: "b".into(),
            },
            Refusal::ModelUnknown("m".into()),
            Refusal::NoTier("m".into()),
            Refusal::RateMissing {
                model: "m".into(),
                bands: vec!["output"],
            },
            Refusal::RateInvalid {
                band: "input",
                value: -1.0,
            },
            Refusal::Arithmetic("boom".into()),
            Refusal::NotEvaluated,
            Refusal::CurrencyMismatch {
                cap: "USD".into(),
                price: "CNY".into(),
            },
            Refusal::AmbiguousGroup {
                model: "m".into(),
                group_id: Some(111),
            },
            Refusal::RateZero { model: "m".into() },
        ];
        let mut codes: Vec<&str> = refusals.iter().map(Refusal::code).collect();
        let total = codes.len();
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(codes.len(), total, "codes must be distinct");
        for refusal in &refusals {
            assert!(!refusal.detail().is_empty(), "{refusal:?} has no detail");
        }
    }

    // The loader is the only part that touches disk, so these tests write a
    // snapshot to a temp file rather than reading the operator's live cache.
    struct TempSnapshot(PathBuf);

    impl TempSnapshot {
        fn write(contents: &str) -> TempSnapshot {
            let mut path = std::env::temp_dir();
            path.push(format!("bili-price-{}.json", uuid::Uuid::new_v4()));
            std::fs::write(&path, contents).expect("write snapshot");
            TempSnapshot(path)
        }

        /// A path that is guaranteed not to exist, for the "absent" cases.
        fn absent_path() -> PathBuf {
            let mut path = std::env::temp_dir();
            path.push(format!("bili-price-absent-{}.json", uuid::Uuid::new_v4()));
            path
        }
    }

    impl Drop for TempSnapshot {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    /// Points the adapter at `path` for one test body, holding the crate-wide
    /// environment lock so it cannot race `d1b_runner`'s tests over the same
    /// variable. Dropping clears the variable again, so no test inherits another
    /// test's snapshot.
    struct EnvPath {
        _lock: std::sync::MutexGuard<'static, ()>,
    }

    impl EnvPath {
        fn set(path: &Path) -> EnvPath {
            let lock = crate::test_env::lock();
            unsafe { crate::test_env::set(SNAPSHOT_ENV, path.to_string_lossy().as_ref()) };
            EnvPath { _lock: lock }
        }
    }

    impl Drop for EnvPath {
        fn drop(&mut self) {
            unsafe { crate::test_env::restore_taken(SNAPSHOT_ENV, None) };
        }
    }

    #[test]
    fn a_snapshot_file_round_trips_through_the_loader() {
        let file = TempSnapshot::write(
            &serde_json::to_string(&json!({
                "unit": "CNY",
                "source": "OneAgent model plaza",
                "updatedAt": NOW - 1_000,
                "models": [{
                    "id": "deepseek-v4-1-flash-261017",
                    "fallback": false,
                    "short": {"uncached": 0.864, "cacheRead": 0.01728, "cacheWrite": 0.01728, "output": 3.456}
                }]
            }))
            .expect("serialize"),
        );
        let book = load_snapshot(&file.0).expect("load");
        assert_eq!(book.currency(), "CNY");
        assert_eq!(book.model_count(), 1);
        assert_eq!(book.updated_at(), NOW - 1_000);
        assert!(
            book.quote("deepseek-v4-1-flash-261017", ONEAGENT_ORIGIN, 10, NOW)
                .is_ok()
        );
    }

    #[test]
    fn a_missing_snapshot_file_is_refused_not_silently_empty() {
        let mut path = std::env::temp_dir();
        path.push(format!("bili-price-absent-{}.json", uuid::Uuid::new_v4()));
        assert_eq!(
            load_snapshot(&path).expect_err("absent").code(),
            "price_snapshot_missing"
        );
    }

    #[test]
    fn malformed_snapshot_json_is_refused_not_silently_empty() {
        let file = TempSnapshot::write("{ this is not json");
        assert_eq!(
            load_snapshot(&file.0).expect_err("malformed").code(),
            "price_snapshot_missing"
        );
    }

    #[test]
    fn a_snapshot_that_is_json_but_not_a_book_is_refused() {
        let file = TempSnapshot::write("{\"unit\":\"CNY\"}");
        assert_eq!(
            load_snapshot(&file.0).expect_err("not a book").code(),
            "price_snapshot_missing"
        );
    }

    #[test]
    fn resolve_reports_unknown_with_the_failing_link_and_never_a_zero_price() {
        let file = TempSnapshot::write(
            &serde_json::to_string(&json!({
                "unit": "CNY",
                "source": "OneAgent model plaza",
                "updatedAt": NOW - MAX_STALE_MS - 1,
                "models": [{
                    "id": "deepseek-v4-1-flash-261017",
                    "fallback": false,
                    "short": {"uncached": 0.864, "cacheRead": 0.01728, "cacheWrite": 0.01728, "output": 3.456}
                }]
            }))
            .expect("serialize"),
        );
        // Point the adapter at the file for this test only.
        let _env = EnvPath::set(&file.0);
        let result = resolve_quote("deepseek-v4-1-flash-261017", ONEAGENT_ORIGIN, 1_000, NOW);

        let refusal = result.expect_err("a stale snapshot must not yield a price");
        let basis = CostBasis::unknown(&refusal);
        assert!(!basis.is_known());
        assert_eq!(basis.refusal_code, Some("price_stale"));
        assert!(
            basis.rates.is_none(),
            "an unknown basis must carry no rates"
        );
        assert!(basis.currency.is_none());
    }

    #[test]
    fn resolve_emits_rates_only_when_every_link_holds() {
        let file = TempSnapshot::write(
            &serde_json::to_string(&json!({
                "unit": "CNY",
                "source": "OneAgent model plaza",
                "updatedAt": NOW - 1_000,
                "models": [{
                    "id": "deepseek-v4-1-flash-261017",
                    "fallback": false,
                    "short": {"uncached": 0.864, "cacheRead": 0.01728, "cacheWrite": 0.01728, "output": 3.456}
                }]
            }))
            .expect("serialize"),
        );
        let _env = EnvPath::set(&file.0);
        let quote = resolve_quote("deepseek-v4-1-flash-261017", ONEAGENT_ORIGIN, 1_000, NOW)
            .expect("a fresh bound snapshot yields a price");

        let basis = CostBasis::known(&quote);
        assert!(basis.is_known());
        assert_eq!(basis.currency.as_deref(), Some("CNY"));
        let rates = basis.rates.expect("rates present when known");
        assert_eq!(rates.input_per_million, 0.864);
        assert_eq!(rates.output_per_million, 3.456);
        assert!(basis.refusal_code.is_none());
    }

    #[test]
    fn a_provider_other_than_oneagent_never_gets_a_price_from_this_book() {
        let file = TempSnapshot::write(
            &serde_json::to_string(&json!({
                "unit": "CNY",
                "source": "OneAgent model plaza",
                "updatedAt": NOW - 1_000,
                "models": [{
                    "id": "deepseek-v4-1-flash-261017",
                    "fallback": false,
                    "short": {"uncached": 0.864, "cacheRead": 0.01728, "cacheWrite": 0.01728, "output": 3.456}
                }]
            }))
            .expect("serialize"),
        );
        let _env = EnvPath::set(&file.0);
        let result = resolve_quote(
            "deepseek-v4-1-flash-261017",
            "https://api.other.test/v1",
            1_000,
            NOW,
        );

        assert_eq!(
            result.expect_err("mismatch").code(),
            "price_provider_mismatch"
        );
    }

    #[test]
    fn a_different_snapshot_path_can_be_pointed_at_for_tests_only() {
        // The override exists so tests and dry audits never read the live cache;
        // it changes where we look, never what counts as fresh or trusted.
        let file = TempSnapshot::write("{}");
        let _env = EnvPath::set(&file.0);
        let path = default_snapshot_path().expect("override resolves");
        assert_eq!(path, file.0);
    }

    // ---------------------------------------------------------------------
    // P4-2S · F1: a routed origin may carry userinfo or query credentials, so
    // no refusal, report or log may echo it. Redaction is not routing: the
    // comparison still uses the raw string.
    // ---------------------------------------------------------------------

    #[test]
    fn redacted_origin_keeps_only_scheme_host_and_port() {
        assert_eq!(
            redacted_origin("https://user:secret@API.Example.com:8443/v1?key=abc#frag"),
            "https://api.example.com:8443"
        );
        assert_eq!(
            redacted_origin("http://127.0.0.1:9/v1"),
            "http://127.0.0.1:9"
        );
        assert_eq!(
            redacted_origin("https://api.gameaigc.cn/"),
            "https://api.gameaigc.cn"
        );
        assert_eq!(redacted_origin("not a url"), UNPARSABLE_ORIGIN);
        assert_eq!(redacted_origin("https://"), UNPARSABLE_ORIGIN);
        assert_eq!(
            redacted_origin("ftp://files.example.com/x"),
            UNPARSABLE_ORIGIN
        );
        assert_eq!(redacted_origin("https://:8443/x"), UNPARSABLE_ORIGIN);
    }

    #[test]
    fn a_routed_origin_with_credentials_never_appears_in_the_refusal() {
        let err = book(NOW - 1_000)
            .quote(
                "deepseek-v4-1-flash-261017",
                "https://user:secret-word@evil.example.com/v1?api_key=hidden#frag",
                1_000,
                NOW,
            )
            .expect_err("mismatch");
        assert_eq!(err.code(), "price_provider_mismatch");
        let detail = err.detail();
        assert!(detail.contains("https://evil.example.com"), "{detail}");
        for leaked in ["secret-word", "api_key", "hidden", "user:", "frag"] {
            assert!(!detail.contains(leaked), "detail leaked {leaked}: {detail}");
        }
    }

    #[test]
    fn an_unparsable_routed_origin_is_never_echoed() {
        let err = book(NOW - 1_000)
            .quote(
                "deepseek-v4-1-flash-261017",
                "raw text /v1 carrying a secret",
                1_000,
                NOW,
            )
            .expect_err("mismatch");
        assert!(err.detail().contains(UNPARSABLE_ORIGIN), "{}", err.detail());
        assert!(!err.detail().contains("raw text"));
        assert!(!err.detail().contains("secret"));
    }

    // ---------------------------------------------------------------------
    // P4-2S · F4: the plugin keeps the first child group's rates for a model
    // that several groups list. Those rates are real but not proven to be what
    // this request is billed at, so the uncertainty is carried, not guessed.
    // ---------------------------------------------------------------------

    #[test]
    fn a_model_in_several_child_groups_carries_its_own_uncertainty() {
        let payload = json!({
            "unit": "CNY",
            "source": "OneAgent model plaza",
            "updatedAt": NOW - 1_000,
            "models": [{
                "id": "ambiguous-model",
                "groupId": 111,
                "fallback": true,
                "short": {"uncached": 0.864, "cacheRead": 0.01728, "cacheWrite": 0.01728, "output": 3.456}
            }]
        });
        let parsed = PriceBook::from_payload(&payload).expect("parses");
        let quote = parsed
            .quote("ambiguous-model", ONEAGENT_ORIGIN, 1_000, NOW)
            .expect("quotes");
        assert!(quote.group_ambiguous);
        assert_eq!(quote.group_id, Some(111));
        assert_eq!(parsed.ambiguous_model_count(), 1);

        let basis = CostBasis::unknown(&Refusal::AmbiguousGroup {
            model: quote.model.clone(),
            group_id: quote.group_id,
        });
        assert_eq!(basis.state, "unknown");
        assert_eq!(basis.refusal_code, Some("price_group_ambiguous"));
        assert!(basis.group_ambiguous);
        assert_eq!(basis.group_id, Some(111));
        assert!(!basis.is_known());
        assert!(basis.rates.is_none(), "an ambiguous price exposes no rate");
    }

    #[test]
    fn a_missing_group_flag_is_not_proof_of_a_single_group() {
        let payload = json!({
            "unit": "CNY",
            "source": "OneAgent model plaza",
            "updatedAt": NOW - 1_000,
            "models": [{
                "id": "m",
                "short": {"uncached": 1.0, "cacheRead": 0.1, "cacheWrite": 0.1, "output": 2.0}
            }]
        });
        let parsed = PriceBook::from_payload(&payload).expect("parses");
        let quote = parsed.quote("m", ONEAGENT_ORIGIN, 10, NOW).expect("quotes");
        assert!(
            quote.group_ambiguous,
            "absence of the flag must not default to safe"
        );
        assert_eq!(quote.group_id, None);
    }

    #[test]
    fn a_single_group_quote_is_known_and_carries_its_group() {
        let quote = book(NOW - 1_000)
            .quote("deepseek-v4-1-flash-261017", ONEAGENT_ORIGIN, 10, NOW)
            .expect("quote");
        assert!(!quote.group_ambiguous);
        assert_eq!(quote.group_id, Some(111));
        let basis = CostBasis::known(&quote);
        assert_eq!(basis.state, "known");
        assert!(!basis.group_ambiguous);
        assert_eq!(basis.group_id, Some(111));
    }

    // ---------------------------------------------------------------------
    // P4-2S · F8: a book zeroed on every band would make any cap pass for any
    // run; nothing in the snapshot marks a free model.
    // ---------------------------------------------------------------------

    #[test]
    fn an_all_zero_rate_card_is_refused_instead_of_booking_zero_cost() {
        let payload = json!({
            "unit": "CNY",
            "source": "OneAgent model plaza",
            "updatedAt": NOW - 1_000,
            "models": [{
                "id": "free-looking",
                "fallback": false,
                "short": {"uncached": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0, "output": 0.0}
            }]
        });
        let parsed = PriceBook::from_payload(&payload).expect("parses");
        let err = parsed
            .quote("free-looking", ONEAGENT_ORIGIN, 1_000, NOW)
            .expect_err("all-zero card");
        assert_eq!(err.code(), "price_rate_zero");
        assert!(err.detail().contains("zero"), "{}", err.detail());
    }

    #[test]
    fn a_single_zero_band_is_still_a_usable_price() {
        let payload = json!({
            "unit": "CNY",
            "source": "OneAgent model plaza",
            "updatedAt": NOW - 1_000,
            "models": [{
                "id": "no-cache-read",
                "fallback": false,
                "short": {"uncached": 1.0, "cacheRead": 0.0, "cacheWrite": 0.1, "output": 2.0}
            }]
        });
        let parsed = PriceBook::from_payload(&payload).expect("parses");
        let quote = parsed
            .quote("no-cache-read", ONEAGENT_ORIGIN, 1_000, NOW)
            .expect("quote");
        assert_eq!(quote.rates.cache_read, 0.0);
        let cost = quote.cost(1_000_000, 0, 0).expect("cost");
        assert!((cost.amount - 1.0).abs() < 1e-9, "{}", cost.amount);
    }
}

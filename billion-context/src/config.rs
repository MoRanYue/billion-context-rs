//! Configuration (pure data) and the host hooks (code).
//!
//! Splitting these is the main structural change from the TS kernel, where
//! `Config` carries closures and therefore cannot be serialized, shipped, or
//! diffed. Here `Config` round-trips through any serde format and `Policy`
//! carries the code.

use crate::message::Message;

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TierConfig {
    pub enabled: bool,
    pub tier2_trigger: u32,
    pub tier3_trigger: u32,
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NudgeConfig {
    pub max_context_limit_pct: f32,
    pub min_context_limit_pct: f32,
    pub frequency: u32,
    pub iteration_threshold: u32,
    pub growth_ratio: f32,
    pub growth_floor: u32,
    pub growth_cap: u32,
    pub min_growth_floor: u32,
    pub min_growth_ratio: f32,
    pub emergency_threshold_pct: f32,
    pub tier2_growth_multiplier: f32,
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TruncateConfig {
    pub threshold: f32,
    pub terminal_escape_after: u32,
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CompressConfig {
    pub min_compress_range: usize,
    pub max_summary_length: usize,
    pub min_summary_length: usize,
}

/// Off by default in the reference config, and kept that way here: these are
/// experimental passes, not part of the mechanism. The nodes exist so the
/// pipeline order stays faithful, and short-circuit when disabled.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AbsorbConfig {
    pub enabled: bool,
    pub tool_name: String,
    pub min_tool_tokens: u32,
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CrushConfig {
    pub enabled: bool,
    pub min_reduction: f32,
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CcrConfig {
    pub enabled: bool,
    pub tool_name: String,
    pub min_tool_tokens: u32,
    pub max_head_chars: usize,
}

/// Serializable configuration. Everything here could come from a TOML file or
/// a remote policy service.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Config {
    pub model_context_limit: u32,
    pub tiers: TierConfig,
    pub nudge: NudgeConfig,
    pub truncate: TruncateConfig,
    pub compress: CompressConfig,
    pub absorb: AbsorbConfig,
    pub crush: CrushConfig,
    pub ccr: CcrConfig,
    pub promotion_threshold: u32,
    pub preserve_recent_messages: usize,
    pub preserve_recent_tokens: u32,
    pub protected_tools: Vec<String>,
}

impl Config {
    /// Defaults mirror the reference implementation.
    pub fn new(model_context_limit: u32) -> Self {
        Self {
            model_context_limit,
            tiers: TierConfig {
                enabled: true,
                tier2_trigger: 1000,
                tier3_trigger: 2000,
            },
            nudge: NudgeConfig {
                max_context_limit_pct: 0.75,
                min_context_limit_pct: 0.45,
                frequency: 5,
                iteration_threshold: 15,
                growth_ratio: 0.05,
                growth_floor: 50_000,
                growth_cap: 50_000,
                min_growth_floor: 20_000,
                min_growth_ratio: 0.45,
                emergency_threshold_pct: 0.95,
                tier2_growth_multiplier: 1.5,
            },
            truncate: TruncateConfig {
                threshold: 0.95,
                terminal_escape_after: 3,
            },
            compress: CompressConfig {
                min_compress_range: 5_000,
                max_summary_length: 20_000,
                min_summary_length: 50,
            },
            absorb: AbsorbConfig {
                enabled: false,
                tool_name: "absorb".into(),
                min_tool_tokens: 4_000,
            },
            crush: CrushConfig {
                enabled: false,
                min_reduction: 0.1,
            },
            ccr: CcrConfig {
                enabled: false,
                tool_name: "acp_retrieve".into(),
                min_tool_tokens: 4_000,
                max_head_chars: 96,
            },
            promotion_threshold: 5,
            preserve_recent_messages: 3,
            preserve_recent_tokens: 5_000,
            protected_tools: Vec::new(),
        }
    }

    /// Reports every problem at once, typed — a caller can render all of them,
    /// not just the first.
    pub fn validate(&self) -> Result<(), Vec<ConfigError>> {
        let mut errors = Vec::new();
        if self.model_context_limit == 0 {
            errors.push(ConfigError::NonPositiveWindow);
        }
        if self.nudge.min_context_limit_pct > self.nudge.max_context_limit_pct {
            errors.push(ConfigError::NudgeBandsInverted);
        }
        if self.nudge.max_context_limit_pct > self.nudge.emergency_threshold_pct {
            errors.push(ConfigError::NudgeBandAboveEmergency);
        }
        if !(0.0..=1.0).contains(&self.truncate.threshold) || self.truncate.threshold == 0.0 {
            errors.push(ConfigError::TruncateThresholdOutOfRange);
        }
        if self.tiers.tier3_trigger <= self.tiers.tier2_trigger {
            errors.push(ConfigError::TierTriggersNotAscending);
        }
        if self.compress.min_summary_length > self.compress.max_summary_length {
            errors.push(ConfigError::SummaryBoundsInverted);
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ConfigError {
    NonPositiveWindow,
    NudgeBandsInverted,
    NudgeBandAboveEmergency,
    TruncateThresholdOutOfRange,
    TierTriggersNotAscending,
    SummaryBoundsInverted,
}

impl ConfigError {
    pub fn message(self) -> &'static str {
        match self {
            ConfigError::NonPositiveWindow => "model_context_limit must be positive",
            ConfigError::NudgeBandsInverted => "nudge.min > nudge.max",
            ConfigError::NudgeBandAboveEmergency => "nudge.max > emergency threshold",
            ConfigError::TruncateThresholdOutOfRange => "truncate.threshold outside (0, 1]",
            ConfigError::TierTriggersNotAscending => "tier3 trigger must exceed tier2 trigger",
            ConfigError::SummaryBoundsInverted => "min summary length exceeds max",
        }
    }
}

/// The host hooks. Deliberately narrow: four pure predicates. A `dyn` trait
/// object keeps `Engine` non-generic — these run once per message per turn, so
/// the dispatch cost is noise next to the tokenizer call.
pub trait Policy: Send + Sync + 'static {
    fn is_tool_protected(&self, _tool: &str, _input: Option<&str>) -> bool {
        false
    }

    fn is_message_protected(&self, _msg: &Message) -> bool {
        false
    }

    fn is_compress_tool(&self, name: &str) -> bool {
        name == "compress"
    }

    /// Virtual path for messages that are not tool exchanges, so one set of
    /// `protected_tools` patterns can also govern host-injected state.
    fn message_path<'a>(&self, _msg: &'a Message) -> Option<&'a str> {
        None
    }
}

/// Nothing protected beyond the defaults.
pub struct DefaultPolicy;
impl Policy for DefaultPolicy {}

/// A policy that protects by tool name and by explicit message id.
pub struct ListPolicy {
    pub protected_tools: Vec<String>,
    pub protected_messages: Vec<String>,
}

impl Policy for ListPolicy {
    fn is_tool_protected(&self, tool: &str, _input: Option<&str>) -> bool {
        self.protected_tools.iter().any(|p| p == tool)
    }

    fn is_message_protected(&self, msg: &Message) -> bool {
        self.protected_messages.iter().any(|p| p == msg.id.as_str())
    }
}

/// The capability the kernel cannot own. Hosts inject their real tokenizer;
/// the engine only ever estimates ranges, because the host already knows the
/// conversation's size from usage data.
pub trait TokenEstimator: Send + Sync + 'static {
    fn count(&self, text: &str) -> u32;
}

/// Cheap default, calibrated against the reference implementation: CJK
/// characters cost about one token each, everything else about a quarter.
///
/// Counting them together as `chars / 4` underestimates Chinese, Japanese and
/// Korean text by roughly 4x, which silently moves every threshold in the
/// config (nudge floor, tier triggers, the truncate trip point) out from under
/// a user writing in those languages.
///
/// It is still only a heuristic. Measured against the real DeepSeek vocabulary
/// it lands within about ±40% per message, and the error changes sign with the
/// content: Chinese prose costs ~0.64 tokens per character (so this
/// overestimates it by ~47%), source code ~2.7 characters per token (so this
/// underestimates it by ~32%). No single scale factor repairs that. Hosts that
/// need the real number inject their own [`TokenEstimator`] — see
/// [`crate::tokenizer`] for the ready-made one behind the `hf-tokenizer`
/// feature.
pub struct CharsPerToken4;

impl TokenEstimator for CharsPerToken4 {
    fn count(&self, text: &str) -> u32 {
        if text.is_empty() {
            return 0;
        }
        let mut cjk = 0usize;
        let mut units = 0usize;
        for ch in text.chars() {
            let width = ch.len_utf16();
            units += width;
            if is_cjk(ch) {
                cjk += width;
            }
        }
        let rest = units - cjk;
        (cjk + rest.div_ceil(4)).max(1) as u32
    }
}

/// The three ranges the reference implementation treats as one-token-per-char:
/// CJK unified ideographs, kana, and hangul syllables.
fn is_cjk(ch: char) -> bool {
    matches!(
        ch as u32,
        0x4e00..=0x9fff | 0x3040..=0x30ff | 0xac00..=0xd7af
    )
}

/// One token per character — for tests that want arithmetic they can predict.
pub struct CharsAsTokens;

impl TokenEstimator for CharsAsTokens {
    fn count(&self, text: &str) -> u32 {
        text.chars().count().max(1) as u32
    }
}

/// Ergonomic plumbing for the shapes a config-driven host reaches for.
///
/// [`Session::with_estimator`](crate::Session::with_estimator) takes
/// `impl TokenEstimator`, so without these a host holding a
/// `Arc<dyn TokenEstimator>` picked at startup could not pass it in at all.
impl<T: TokenEstimator + ?Sized> TokenEstimator for std::sync::Arc<T> {
    fn count(&self, text: &str) -> u32 {
        (**self).count(text)
    }
}

impl<T: TokenEstimator + ?Sized> TokenEstimator for Box<T> {
    fn count(&self, text: &str) -> u32 {
        (**self).count(text)
    }
}

/// Wraps a closure as a [`TokenEstimator`], for estimators too small to deserve
/// a type.
///
/// A newtype rather than a blanket `impl TokenEstimator for F: Fn(..)`, because
/// `Arc<F>` and `Box<F>` themselves implement `Fn` — a blanket impl would
/// overlap with the two above and fail coherence.
///
/// ```
/// use billion_context::{Config, Session, config::FnEstimator};
///
/// let session = Session::new(Config::new(128_000))
///     .with_estimator(FnEstimator(|text: &str| (text.len() / 4) as u32));
/// # let _ = session;
/// ```
pub struct FnEstimator<F>(pub F);

impl<F> TokenEstimator for FnEstimator<F>
where
    F: Fn(&str) -> u32 + Send + Sync + 'static,
{
    fn count(&self, text: &str) -> u32 {
        (self.0)(text)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid() {
        assert!(Config::new(200_000).validate().is_ok());
    }

    #[test]
    fn validation_collects_every_problem() {
        let mut cfg = Config::new(0);
        cfg.tiers.tier3_trigger = 10;
        cfg.tiers.tier2_trigger = 20;
        cfg.truncate.threshold = 1.5;
        let errors = cfg.validate().unwrap_err();
        assert!(errors.contains(&ConfigError::NonPositiveWindow));
        assert!(errors.contains(&ConfigError::TierTriggersNotAscending));
        assert!(errors.contains(&ConfigError::TruncateThresholdOutOfRange));
        assert_eq!(errors.len(), 3);
    }
}

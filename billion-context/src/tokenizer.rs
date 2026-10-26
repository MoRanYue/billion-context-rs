//! Exact token counting from a HuggingFace `tokenizer.json`.
//!
//! Available with the `hf-tokenizer` feature. The kernel owns no tokenizer —
//! [`TokenEstimator`] is the one capability a host must supply — and the
//! default ([`CharsPerToken4`]) is a character heuristic that is wrong by up to
//! ±40% depending on what the text is made of. This module supplies a real BPE
//! counter for hosts that would rather not write one.
//!
//! ```
//! use billion_context::{Config, Session};
//! use billion_context::tokenizer::HfTokenizer;
//!
//! # fn demo() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//! let tokenizer = HfTokenizer::from_file("assets/tokenizer.json")?;
//! let session = Session::new(Config::new(1_000_000)).with_estimator(tokenizer);
//! # let _ = session;
//! # Ok(())
//! # }
//! ```
//!
//! # Bring your own vocabulary
//!
//! Nothing here is DeepSeek-specific: a `tokenizer.json` is the format
//! HuggingFace `tokenizers` writes and every major model family publishes, so
//! [`HfTokenizer`] counts tokens exactly for whichever model will actually
//! serve the prompt. The encoding step goes through `deepseek-recipe-encoding`,
//! DeepSeek's own prompt-encoding library, which is why it lives behind a
//! feature rather than in the core.
//!
//! DeepSeek publishes one vocabulary per model family, both MIT licensed, under
//! `static/tokenizers/` in <https://github.com/deepseek-ai/deepseek-recipe>:
//!
//! | directory | model |
//! | --- | --- |
//! | `v4/` | DeepSeek-V4-Pro |
//! | `v41/` | DeepSeek-V4-Flash-Vision-Exp |
//!
//! A vocabulary that does not match the serving model is worse than the char
//! heuristic for some content, so pick it per deployment rather than globally.
//!
//! # What this does and does not fix
//!
//! It makes the *text → token* step exact. It does not make the *messages →
//! prompt* step exact, because the kernel counts messages one at a time and
//! sums them: the chat template's role markers, the tool-definition schemas,
//! and the rendered `<acp …>` tags are not part of any message's text. Those
//! are a roughly constant overhead of a few hundred to a few thousand tokens,
//! which is noise in a 1M window and a couple of percent in a 128k one.
//!
//! # One side effect to know about
//!
//! `deepseek-recipe-encoding` enables `serde_json/preserve_order`, and Cargo
//! features are additive across the whole graph: building with this feature
//! swaps `serde_json::Value`'s map from a `BTreeMap` to an `IndexMap` for every
//! crate in the build, including the host's own JSON handling. That is usually
//! harmless and occasionally load-bearing — a host that relies on `Value`
//! reordering keys (or on the smaller map) will see different behaviour. If
//! that matters, skip this feature and implement [`TokenEstimator`] over
//! `tokenizers` directly: for counting text the two paths agree, and
//! `tokenizers` alone does not touch `serde_json`.
//!
//! # Writing your own instead
//!
//! [`TokenEstimator`] is a two-line trait, and `billion-context` accepts it through
//! `impl TokenEstimator`, `Arc<dyn TokenEstimator>`, `Box<dyn TokenEstimator>`,
//! or [`FnEstimator`]. A host counting with a different stack — `tiktoken-rs`,
//! a sentencepiece binding, a tokenizer it already holds for other reasons —
//! should implement the trait directly and skip this feature entirely:
//!
//! ```
//! use billion_context::{Config, Session, TokenEstimator};
//!
//! struct HalfOfTheBytes;
//! impl TokenEstimator for HalfOfTheBytes {
//!     fn count(&self, text: &str) -> u32 {
//!         (text.len() / 2) as u32
//!     }
//! }
//!
//! let session = Session::new(Config::new(128_000)).with_estimator(HalfOfTheBytes);
//! # let _ = session;
//! ```

use std::path::Path;

use deepseek_recipe_encoding::{PromptTokenizer, TokenizerEncoder};
use tokenizers::Tokenizer;

use crate::config::{CharsPerToken4, TokenEstimator};

/// A [`TokenEstimator`] backed by a real BPE vocabulary.
///
/// Holds the tokenizer without its added vocabulary for plain text, plus the
/// added vocabulary for the special tokens a rendered prompt contains. Plain
/// text is all `billion-context` ever counts, so the split costs nothing here and
/// keeps the same handle usable for prompt encoding.
pub struct HfTokenizer {
    encoder: PromptTokenizer,
}

impl HfTokenizer {
    /// Load a `tokenizer.json`, as published by a model vendor or exported by
    /// HuggingFace `tokenizers`.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        let tokenizer =
            Tokenizer::from_file(path.as_ref()).map_err(|e| TokenizerError::Load(e.to_string()))?;
        Ok(Self::from_tokenizer(&tokenizer))
    }

    /// Parse a `tokenizer.json` already in memory — for embedding the
    /// vocabulary in the binary instead of shipping a file.
    pub fn from_bytes(json: &[u8]) -> Result<Self, TokenizerError> {
        let tokenizer =
            Tokenizer::from_bytes(json).map_err(|e| TokenizerError::Load(e.to_string()))?;
        Ok(Self::from_tokenizer(&tokenizer))
    }

    /// Wrap a tokenizer the host already built.
    pub fn from_tokenizer(tokenizer: &Tokenizer) -> Self {
        Self {
            encoder: PromptTokenizer::new(tokenizer),
        }
    }

    /// The token ids for `text`, without any template or special tokens.
    ///
    /// This is the exact counterpart of [`TokenEstimator::count`]; hosts that
    /// also need ids (to compute logprobs, say) can avoid encoding twice.
    pub fn encode_ids(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        self.encoder
            .encode_ids(text, &[])
            .map_err(TokenizerError::Encode)
    }
}

impl TokenEstimator for HfTokenizer {
    fn count(&self, text: &str) -> u32 {
        if text.is_empty() {
            return 0;
        }
        match self.encoder.encode_ids(text, &[]) {
            Ok(ids) => ids.len() as u32,
            // Encoding can only fail on a malformed vocabulary. Falling back
            // to the heuristic keeps a session alive instead of panicking
            // inside a turn; a host that cares should pre-flight with
            // `encode_ids` at startup.
            Err(_) => CharsPerToken4.count(text),
        }
    }
}

/// Why a tokenizer could not be loaded or used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerError {
    /// The `tokenizer.json` was missing, unreadable, or malformed.
    Load(String),
    /// The vocabulary loaded but could not encode a string.
    Encode(String),
}

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Load(e) => write!(f, "failed to load tokenizer: {e}"),
            Self::Encode(e) => write!(f, "failed to encode text: {e}"),
        }
    }
}

impl std::error::Error for TokenizerError {}

//! Differential check of the token estimator against the reference kernel.
//!
//! Reads the corpus and the reference counts that `diff/tok-ts.mts` produced,
//! recomputes them with [`CharsPerToken4`], and writes both side by side.
//! `diff/compare-tok.mjs` then reports any disagreement.
//!
//! ```text
//! node diff/tok-ts.mts diff/tok-ts.json
//! cargo run --example tok_diff -- diff/tok-ts.json diff/tok-rust.json
//! # optionally also check the real vocabulary:
//! cargo run --features hf-tokenizer --example tok_diff -- \
//!     diff/tok-ts.json diff/tok-rust.json assets/tokenizer.json
//! ```
use billion_context::{CharsPerToken4, TokenEstimator};
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (Some(input), Some(output)) = (args.first(), args.get(1)) else {
        eprintln!("usage: tok_diff <in.json> <out.json> [tokenizer.json]");
        return ExitCode::FAILURE;
    };

    let raw = match std::fs::read_to_string(input) {
        Ok(raw) => raw,
        Err(err) => {
            eprintln!("read {input}: {err}");
            return ExitCode::FAILURE;
        }
    };
    let doc: serde_json::Value = match serde_json::from_str(&raw) {
        Ok(doc) => doc,
        Err(err) => {
            eprintln!("parse {input}: {err}");
            return ExitCode::FAILURE;
        }
    };

    let corpus: Vec<&str> = doc["corpus"]
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();
    let reference: Vec<u64> = doc["counts"]
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_u64()).collect())
        .unwrap_or_default();
    if corpus.len() != reference.len() {
        eprintln!(
            "corpus has {} entries but counts has {}",
            corpus.len(),
            reference.len()
        );
        return ExitCode::FAILURE;
    }

    let heuristic = CharsPerToken4;
    let counts: Vec<u64> = corpus
        .iter()
        .map(|t| u64::from(heuristic.count(t)))
        .collect();

    // `mut` only matters with the feature on, where `exact` is spliced in.
    #[allow(unused_mut)]
    let mut doc_out = serde_json::json!({ "corpus": corpus, "counts": counts });

    // With a vocabulary supplied, also record what the real tokenizer says, so
    // the report can show how far the heuristic is from it on this corpus.
    if let Some(path) = args.get(2) {
        #[cfg(feature = "hf-tokenizer")]
        {
            match billion_context::tokenizer::HfTokenizer::from_file(path) {
                Ok(tok) => {
                    let exact: Vec<u64> = corpus
                        .iter()
                        .map(|t| u64::from(tok.count(t)))
                        .collect();
                    doc_out["exact"] = serde_json::json!(exact);
                }
                Err(err) => {
                    eprintln!("load {path}: {err}");
                    return ExitCode::FAILURE;
                }
            }
        }
        #[cfg(not(feature = "hf-tokenizer"))]
        {
            eprintln!("{path} ignored: rebuild with --features hf-tokenizer");
        }
    }

    match std::fs::write(output, format!("{doc_out:#}")) {
        Ok(()) => {
            println!("wrote {output} samples: {}", corpus.len());
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("write {output}: {err}");
            ExitCode::FAILURE
        }
    }
}

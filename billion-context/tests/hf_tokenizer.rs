//! The ready-made exact estimator, and the gap it closes.
//!
//! The vocabulary is not committed (6 MB, model-owned), so every test here
//! returns early when it is absent. Fetch it with
//! `assets/fetch-tokenizer.ps1` or point `ACP_TOKENIZER_JSON` at a copy.
//!
//! The extension points themselves need no vocabulary and are covered in
//! `tests/estimator.rs`.

use std::path::PathBuf;

use billion_context::{
    CharsPerToken4, Config, HfTokenizer, Message, RenderStrategy, Role, Session, TokenEstimator,
    TurnInput,
};

fn vocabulary() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("ACP_TOKENIZER_JSON") {
        return Some(PathBuf::from(path));
    }
    // Workspace root, not the crate: a 6 MB model-owned file is not part of
    // this crate's package.
    let local = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../assets/tokenizer.json");
    local.exists().then_some(local)
}

fn load() -> Option<HfTokenizer> {
    let Some(path) = vocabulary() else {
        eprintln!(
            "skipping: no vocabulary at {} (run assets/fetch-tokenizer.ps1)",
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../assets/tokenizer.json")
                .display()
        );
        return None;
    };
    Some(HfTokenizer::from_file(path).expect("vocabulary should load"))
}

#[test]
fn empty_text_is_zero_tokens() {
    let Some(tok) = load() else { return };
    assert_eq!(tok.count(""), 0);
}

#[test]
fn count_agrees_with_encode_ids() {
    let Some(tok) = load() else { return };
    for text in ["hello", "你好，世界", "fn main() {}", "{\"a\":1}"] {
        assert_eq!(
            tok.count(text) as usize,
            tok.encode_ids(text).unwrap().len(),
            "{text}"
        );
    }
}

/// The reason this exists: the char heuristic is not a small error.
///
/// Measured on the DeepSeek vocabulary, Chinese prose costs about 0.64 tokens
/// per character, not the 1.0 `CharsPerToken4` assumes, while source code costs
/// about 2.7 characters per token, not 4.0. The two errors point in opposite
/// directions, so no single scale factor repairs them.
#[test]
fn chinese_prose_is_overestimated_by_the_heuristic() {
    let Some(tok) = load() else { return };
    let chinese = "上下文压缩内核把较早的轮次折叠成摘要，让模型始终在窗口内工作。\
                   它分配稳定引用，让被宿主替换的块退场，并且只在可回收体量足够时才建议折叠。";

    let exact = tok.count(chinese);
    let heuristic = CharsPerToken4.count(chinese);

    assert!(exact > 0);
    assert!(
        exact < heuristic * 8 / 10,
        "expected the heuristic to overshoot by >20%, got exact={exact} heuristic={heuristic}"
    );
}

#[test]
fn source_code_is_underestimated_by_the_heuristic() {
    let Some(tok) = load() else { return };
    let code = r#"fn default_count_tokens(text: &str) -> u32 {
    let mut cjk = 0u32;
    for ch in text.chars() {
        let c = ch as u32;
        if (0x4E00..=0x9FFF).contains(&c) {
            cjk += 1;
        }
    }
    cjk + (text.encode_utf16().count() as u32 - cjk).div_ceil(4)
}"#;

    let exact = tok.count(code);
    let heuristic = CharsPerToken4.count(code);

    assert!(
        exact > heuristic * 12 / 10,
        "expected the heuristic to undershoot by >20%, got exact={exact} heuristic={heuristic}"
    );
}

#[test]
fn a_session_accepts_it_wherever_the_heuristic_goes() {
    let Some(tok) = load() else { return };
    let mut session = Session::new(Config::new(1_000_000)).with_estimator(tok);

    let messages = vec![
        Message::text("m1", Role::User, "第一个问题：这个内核怎么用？"),
        Message::text("m2", Role::Assistant, "先把 Config 建好，再调用 process_turn。"),
    ];

    let turn = session.process_turn(TurnInput {
        messages: &messages,
        token_count: 64,
        render: RenderStrategy::All,
        now: 1_000,
    });

    assert_eq!(turn.messages.len(), 2);
    assert!(turn.warnings.is_empty(), "{:?}", turn.warnings);
}

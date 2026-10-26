# billion-context-rs

[![crates.io](https://img.shields.io/crates/v/billion-context.svg)](https://crates.io/crates/billion-context)
[![docs.rs](https://docs.rs/billion-context/badge.svg)](https://docs.rs/billion-context)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**English** · [简体中文](README.zh-CN.md)

A Rust rewrite of the core mechanism in [`ranxianglei/billion-context`](https://github.com/ranxianglei/billion-context) — ref assignment, folding, the 3-tier LSM, nudge cadence, cross-turn reconciliation — for embedding in any Rust program.

The original is a TypeScript ACP proxy. Its kernel (`kernel/`, npm package `acp-kernel`) is a framework-agnostic compression engine, and that is what was ported. The proxy, the web console and the host adapters were not.

```
billion-context/        pure engine: message → ref → fold → recover. No IO, no async, no unsafe
billion-context-wire/   protocol layer: Anthropic / OpenAI wire types, lossless to the IR
billion-context-tool/   tool layer: folding, retrieval, search and status as tools a model can call
```

The default build has **zero dependencies** (`serde` and the real tokenizer are opt-in features), contains no `unsafe` (`#![forbid(unsafe_code)]`), and is fully synchronous.

> Want a shorter introduction? See [`billion-context/README.md`](billion-context/README.md) — that is what crates.io shows.

---

## 1. Seven design decisions

These decide the shape of the API. They are not style choices.

### 1.1 `Config` is plain data; behaviour lives in `Policy`

The TypeScript `Config` carries closures (`shouldCompress`, `isToolFoldable`, …), which makes it neither serializable nor transportable across processes. Here it is split in two:

```rust
let config = Config::new(200_000);          // plain data, serde round-trips
let session = Session::new(config).with_policy(MyPolicy); // code
```

Every `Policy` method has a default, so a host overrides only the hooks it actually needs.

### 1.2 `&mut self`, not `Arc<Mutex<Session>>`

Session state is exclusive. In Rust that is a `Send` value owned by one holder rather than a lock inside the type — if you need concurrency, `fork()` it (a few maps and one short vector).

```rust
let snapshot = session.fork();   // independent from here on
```

### 1.3 The engine does not run the loop; the host does

The TypeScript version is `while (round < 10) { call model; apply }` — the engine issues the requests itself. The Rust version turns the loop inside out:

```rust
let mut step = engine.begin(input);
loop {
    let reply = llm.call(&step.messages).await?;      // your async
    match engine.resume(input, &reply.requests) {
        Next::Send(next) => step = next,
        Next::Done(turn) => break turn,
        Next::Exhausted(turn) => break turn,
    }
}
```

The engine knows nothing about async or the network, so there is no `async fn` bleeding into it and no boxed `dyn Future`.

### 1.4 Failure is a value, not an exception

One stale ref in a batch fold must not discard the other nine. So the result is an `Applied` that reports each range:

```rust
pub struct Applied {
    pub blocks: Vec<BlockId>,          // succeeded
    pub errors: Vec<RangeError>,       // failed, each with kind + detail
    pub tokens_compressed: u32,
    pub partial: bool,
}
```

Every `RangeErrorKind` carries a `hint()` meant to be fed straight back to the model. That is part of the self-healing loop, not a log line for humans.

### 1.5 Refs are newtypes, not `u32`

```rust
MsgRef::new(1)          // Some(m00001)
BlockId::new(1)         // Some(b1)
MsgRef::parse("m00003") // parses the model's output too
```

`MsgRef` and `BlockId` are non-zero values in `1..=9_999_999`, and `m00001` / `b1` are their only wire forms. Numbers are never reused — citing a dead ref is legal and yields an `UnknownRef` hint rather than being silently ignored.

### 1.6 The kernel does not touch content storage

Summaries live in `Block`, but **original messages** do not — that is the host's business (a `ContentStore` in the TypeScript version, a `Retrieval` callback here). The kernel only guarantees that "summary → original" can be rebuilt; where you store it is your call: files, SQLite, object storage.

### 1.7 Rendering is separate from logic

`process_turn` only projects and counts. The `<acp tokens="…">m00001</acp>` wrapper is controlled by the `render` strategy and can be switched off entirely if you render yourself. What the kernel guarantees in a `Turn` is each message's id and token snapshot — not the string format.

---

## 2. One complete call

```rust
use billion_context::{Boundary, CharsAsTokens, CompressRequest, Config, MsgRef,
               RenderStrategy, RequestedRange, Session, TurnInput};

let mut session = Session::new(Config::new(200_000)).with_estimator(CharsAsTokens);

// 1. Project: assign refs, work out foldable ranges, decide whether to nudge.
let turn = session.process_turn(TurnInput {
    messages: &history,
    token_count: estimate(&history),
    render: RenderStrategy::All,
    now: now(),
});

// 2. The model sees turn.messages and possibly turn.nudge. It answers
//    "fold m00012–m00030, here is the summary".
let applied = session.apply_compression(CompressRequest {
    ranges: vec![RequestedRange {
        start: Boundary::Message(MsgRef::new(12).unwrap()),
        end:   Boundary::Message(MsgRef::new(30).unwrap()),
        topic: Some("build setup".into()),
        summary: "…".into(),
    }],
    now: now(),
});

// 3. Next turn: the originals are gone, the numbers are unchanged.
let turn = session.process_turn(TurnInput { messages: &history, .. });

// 4. The model wants the original behind a summary (it cited b3).
let hits = session.retrieve(&[MsgRef::new(12).unwrap()]);

// 5. Or undo a fold entirely.
session.decompress(&[BlockId::new(3)]);
```

### Who rewrites the message array: `ViewMode`

A fold means "m00001–m00004 is now block b1", but a block is not a message — somebody has to turn that statement into an array. Both answers are legitimate and they produce different results, so this is an explicit switch rather than hidden behaviour:

```rust
// Default: the kernel hands back a view you can send — covered messages are
// dropped and replaced by one carrier holding the summary.
let session = Session::new(cfg);

// Or: the kernel leaves the array alone and only reports coverage through
// `blocks`; rewriting is the host's job.
let session = Session::new(cfg).with_view_mode(ViewMode::Host);
```

`ViewMode::Host` is the upstream TypeScript model: `turn.messages` still contains the folded originals (just tagged with `<acp …>`).

### The stateless-host trap

A host that resends the whole transcript every turn brings already-folded originals back with **fresh message ids**. The kernel claims them by content fingerprint (the `reconcile` node), maps them to their original ids and prunes them — otherwise compression would achieve nothing. This is pinned by `resent_originals_are_reconciled_not_double_counted` in `lib.rs`.

Fingerprints are registered **at the moment of folding** (`apply_one`), because that is when "which content went into the block" is certain. Indexing live messages would conflate two live messages that happen to have identical content.

Conversely, a host that **never** resends folded originals (because it does surface replacement itself) should turn this off:

```rust
let session = Session::new(cfg).with_reconcile(false);
```

It has nothing to reconcile, and content-based reconciliation risks merging two identical live messages into one. The switch is inert under `ViewMode::Host`.

---

## 3. API at a glance

| Type | Methods | Notes |
|---|---|---|
| `Session` | `new(Config)` / `try_new` | `try_new` validates config, returning `Vec<ConfigError>` |
| | `with_policy` / `with_estimator` / `with_pipeline` | Override hooks, tokenizer, node sequence |
| | `process_turn(TurnInput)` | The only hot path; synchronous, pure |
| | `apply_compression(CompressRequest)` | Per-range success and failure |
| | `retrieve(&[MsgRef])` / `search(&str, n)` / `block(&BlockId)` | Read paths; none of them mutate the session |
| | `decompress(&[BlockId])` | Undo a fold; originals return to the view |
| | `status()` / `state()` / `fork()` | Observation and snapshots |
| `Engine` | `begin` / `resume` / `round` / `with_max_rounds` | Host-driven loop |
| `pipeline` | `default_pipeline()` / `Node` | 13 nodes, replaceable |
| `persist` | `to_json` / `from_json` / `save` / `load` | Versioned snapshot; refs stored in wire form |
| `billion_context_wire` | `Message` / `Block` / `Usage` / `OpenAiUsage` | Unknown fields preserved, thinking not parsed, tool arguments not re-serialized |
| `billion_context_tool` | `definitions` / `parse_*` / `render_*` | The model-facing `acp_compress`, `acp_retrieve`, `acp_decompress`, `acp_search`, `acp_status` |

### Node order (pinned by `pipeline_order_is_pinned`)

```
reconcile-live-ids → assign-refs → sync-blocks → prune → ccr-store
  → absorb-hide → crush → absorb-prompt → filter → hide-compress-calls
  → recommend → nudge-inject → emergency-truncate
```

The first three do bookkeeping, the middle do the absorbing and folding, the last three make decisions, inject and fall back.

`absorb-*`, `crush` and `ccr-store` are **deliberately empty** in this port: they need model calls or content storage, which a pure kernel cannot supply. They are `enabled: false` in the reference config anyway, so default behaviour matches upstream; turn them on once you implement the corresponding hooks.

### Default configuration (value-for-value identical to the TypeScript version)

| Item | Value |
|---|---|
| tier2 / tier3 trigger | 1000 / 2000 tokens |
| nudge window | 45% – 75%, every 5 turns, forced from turn 15 |
| Fold threshold | range ≥ 5000 chars, summary 50 – 20000 chars |
| Recent kept | 3 messages / 5000 tokens |
| Emergency truncation | 95% occupancy, hard cut after 3 consecutive failures |
| Ref range | `m00001` – `m9999999`, `b1` – `b9999999` |

---

## 4. Wiring into an agent loop (`billion-context-tool`)

The model cannot see your `Session`. It sees five tools.

```rust
use billion_context_tool::{definitions, parse_compress, render_compress};

// 1. Register the tools — five JSON schemas, straight to the provider.
let tools = definitions();

// 2. The model called acp_compress.
if call.name == billion_context_tool::COMPRESS {
    match parse_compress(&call.input) {
        Ok(ranges) => {
            let applied = session.apply_compression(CompressRequest {
                ranges, now: now(),
            });
            // Report both outcomes: don't re-fold what worked, retry what
            // failed using its hint.
            tool_result(render_compress(&applied.blocks, &applied.errors))
        }
        // Shape errors (a malformed ref, start after end) are caught locally.
        Err(error) => tool_result(error.to_string()),
    }
}
```

| Tool | Purpose |
| --- | --- |
| `acp_compress` | Fold a span into a summary |
| `acp_retrieve` | Read the original messages behind a ref |
| `acp_decompress` | Read one folded block's summary in full |
| `acp_search` | Find which block covered a topic |
| `acp_status` | How full the context is, and what is worth folding |

`acp_search` and `acp_decompress` are not conveniences — they are what makes folding rational. A model that cannot check what it folded either refuses to fold (the context grows) or folds and then invents the detail it lost. Both read paths go through `&self`, so curiosity cannot re-inflate the context: `session.block(id)` returns a block without expanding it.

Three deliberate choices:

- **The schema is the prompt.** Fold rules, ref format and summary length all live in the `description`, because a model attends to a tool description far more reliably than to a system prompt 40k tokens away.
- **A refusal is guidance, not a 400.** A failed range comes back as "m00099 is not in the current view, pick another", and the model corrects itself next turn; an HTTP error costs a whole turn.
- **Search matches terms, not a phrase.** `borrow constraints` finds a block whose topic says "borrow checker" and whose summary says "constraints"; a phrase search would find neither. `limit` is clamped rather than rejected — a model asking for 1000 wants "more", not a correction.

## 5. Exact token counting (optional)

The kernel does not tokenize. `TokenEstimator` is the one capability a host must inject, and the default, `CharsPerToken4`, is only a heuristic. Its error is large and **its sign flips with the content**:

| Content | Real tokens | Heuristic | Error |
| --- | ---: | ---: | ---: |
| English prose, 241 chars | 45 | 61 | +35.6% |
| Chinese prose, 90 chars | 58 | 85 | +46.6% |
| Mixed Chinese/English, 140 chars | 65 | 59 | −9.2% |
| Rust code, 273 chars | 101 | 69 | −31.7% |
| JSON, 168 chars | 67 | 42 | −37.3% |
| Tool log, 182 chars | 60 | 46 | −23.3% |

Chinese is really about **0.64 token/char** (not 1.0) and code about **2.7 chars/token** (not 4.0). The errors in the two directions do not cancel, so no single scale factor fixes it.

For exact numbers, enable the `hf-tokenizer` feature. It wires `deepseek-recipe-encoding` (DeepSeek's own prompt-encoding crate) onto the same trait without touching a line of the kernel:

```toml
[dependencies]
billion-context = { path = "../billion-context", features = ["hf-tokenizer"] }
```

```rust
use billion_context::tokenizer::HfTokenizer;

// The vocabulary is not shipped with the crate (6 MB, model-owned); fetch the
// one matching the model you deploy.
let tokenizer = HfTokenizer::from_file("assets/tokenizer.json")?;
let session = Session::new(Config::new(1_000_000)).with_estimator(tokenizer);
```

**The vocabulary is yours to pick; nothing is hardcoded.** `tokenizer.json` is the HuggingFace `tokenizers` format, published by every major model family, so `HfTokenizer` works with any of them — DeepSeek is merely the one it was measured against. DeepSeek publishes them per model family under MIT in [`deepseek-recipe`](https://github.com/deepseek-ai/deepseek-recipe) `static/tokenizers/`: `v4/` is DeepSeek-V4-Pro, `v41/` is DeepSeek-V4-Flash-Vision-Exp.

Four things to know:

- **The vocabulary must match the model you actually serve.** The wrong one is worse than the heuristic on some content, so this is a deployment decision and should not be baked into a kernel default.
- **It makes "text → tokens" exact, not "messages → prompt".** The kernel counts each message and sums, while the template's role markers, the tool definition schemas and the rendered `<acp …>` tags belong to no message body. That is a few hundred to a few thousand tokens of fixed overhead: noise in a 1M window, a few percent in 128k.
- **Cost**: `tokenizers` pulls in `hf-hub` / `ureq` / `rustls` / `esaxx-rs`, roughly 22 s of cold compile and several MB of binary. Hence an **optional feature**: the default build adds no dependency at all, and `cargo test` never touches it.
- **There is a global side effect.** `deepseek-recipe-encoding` enables `serde_json/preserve_order`, and Cargo features are additive across the graph — enabling it switches `serde_json::Value`'s map from `BTreeMap` to `IndexMap` for your entire dependency tree. Usually harmless, but code that relies on `Value` reordering keys (or on the smaller map) will behave differently. That is also why `billion-context-wire`'s `value_round_trip_reorders_keys_and_changes_bytes` case prints *skip* rather than *pass* with the feature on: the reordering risk is gone, not fixed. If that bothers you, skip the feature and implement `TokenEstimator` directly on `tokenizers` — the counts agree, and `tokenizers` leaves `serde_json` alone.

Counting is cached: each message is counted **once, when it first gets a ref**, and written into `state.token_snapshot` and never recomputed (recomputing would change the rendered tag bytes and invalidate the upstream prefix cache). So an exact tokenizer only charges you for **new messages** — measured at 865 ns/token, about 3.5 ms for a new 4k-token message.

That cache also means **switching tokenizers is a migration**: `token_snapshot` is persisted with the state, so old messages keep their old counts. To switch mid-flight you must clear the stored `token_snapshot` (or recompute it), or one state will mix two conventions.

### Implementing `TokenEstimator` yourself

`TokenEstimator` is a one-method trait, a dozen lines to implement. If you already count tokens some other way (`tiktoken-rs`, sentencepiece bindings, a tokenizer you hold for other reasons), implement it directly and ignore `hf-tokenizer`:

```rust
use billion_context::{Config, Session, TokenEstimator};

struct TiktokenEstimator { bpe: tiktoken_rs::CoreBPE }

impl TokenEstimator for TiktokenEstimator {
    fn count(&self, text: &str) -> u32 {
        self.bpe.encode_ordinary(text).len() as u32
    }
}

let session = Session::new(Config::new(128_000))
    .with_estimator(TiktokenEstimator { bpe });
```

`with_estimator` takes `impl TokenEstimator`, and the kernel already implements it for these shapes, so you do not need a wrapper:

| You have | Pass |
| --- | --- |
| A concrete type | `with_estimator(MyEstimator)` |
| `Arc<dyn TokenEstimator>` | the same (`impl TokenEstimator for Arc<T>` exists) |
| `Box<dyn TokenEstimator>` | the same |
| A closure | `with_estimator(FnEstimator(\|t: &str\| ...))` |

A closure needs the `FnEstimator` newtype rather than a blanket `impl Fn`: `Arc<F>` and `Box<F>` also implement `Fn`, so a bare blanket impl overlaps the two above and fails coherence.

## 6. Verification

```
cargo test                                        # 52: 31 + 4 + 10 + 5 + 2 doctests
cargo test --features hf-tokenizer                # 59: plus 5 tokenizer cases and 2 doctests
cargo run --example basic                         # a full pass
cargo clippy --all-targets                        # no warnings (both configurations)
```

The five exact-tokenizer cases need a vocabulary; without one they skip and print a note:

```
cd billion-context && ..\assets\fetch-tokenizer.ps1      # fetch the 6 MB vocabulary into assets/
```


The tests cover the mechanism, not coverage: refs are never reused, folding removes originals while numbers survive, resent originals are claimed rather than double-counted, reasoning blocks stay byte-identical, partial failure keeps what succeeded, the round cap terminates a stubborn model, and a mistyped ref inside tool arguments is reported against the specific field.

### Differential testing

Self-tests only prove you agree with yourself. The real oracle is the upstream TypeScript kernel, so there is a differential suite (`diff/`): the same scenario into both, each emitting isomorphic canonical JSON, then a field-by-field diff.

```
cd path/to/billion-context
node --import tsx path/to/billion-context-rs/diff/run-ts.mts \
     path/to/billion-context-rs/diff/scenario-a.json path/to/billion-context-rs/diff/out-ts.json

cd path/to/billion-context-rs
cargo run --offline -q --example diff_harness -- diff/scenario-a.json diff/out-rust.json
cd diff && node compare.mjs out-ts.json out-rust.json 500
```

Currently **0 differences** (5 `processTurn` rounds + 2 `applyCompression` calls + `decompress` + `status`, covering every field of `view` / `refs` / `blocks` / `nudge`). This suite caught 9 defects the self-tests could not, two of them outright logic errors: "block absorption was never implemented" and "folding dropped the first user message". Details in [`diff/REPORT.md`](diff/REPORT.md).

The tokenizer has its own suite, because what it introduces is a **source of numbers** rather than new logic — a divergence does not crash, it quietly shifts every threshold:

```
node diff/tok-ts.mts diff/tok-ts.json                 # corpus + TypeScript reference values
cargo run --example tok_diff -- diff/tok-ts.json diff/tok-rust.json
node diff/compare-tok.mjs diff/tok-ts.json diff/tok-rust.json
```

73 samples (including every CJK block boundary, non-BMP emoji and U+20000, combining marks, and pure whitespace) agree **value for value, 0 mismatches**. One level up, both sides are given the same synthetic counter and the whole scenario is replayed: the injection alone changes 60 fields, and the two sides still show **0 differences** — proving the counter is not only honoured in one place. One trap worth noting: JavaScript's `text.length` counts UTF-16 units, so the Rust side must accumulate `ch.len_utf16()`; using `chars().count()` is off by 2× on text with emoji.

---

## 7. Not done

- **Async adapters**: `Engine` is already host-driven, so wrapping it in `async` is thin glue and does not belong in the kernel.
- **The TypeScript cache ledger** (prefix-cache accounting): that is proxy-layer observability, not compression. `billion-context-wire` keeps only the order-preserving serialization it depends on.
- **Original-message storage**: `retrieve` returns the summary and the span it replaced; the originals are yours, and where they live (files / SQLite / object storage) is your decision, not the kernel's.

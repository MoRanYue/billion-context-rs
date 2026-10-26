# billion-context

Lossless-tiered context compression for LLM agents.

Long conversations outgrow the context window, and the usual fix — truncate the
oldest messages — throws away the part the model is still reasoning about. This
crate compresses instead: a span of messages is replaced by a **summary that
keeps its address**, so the model can still refer to it, and the originals stay
retrievable on demand.

```toml
[dependencies]
billion-context = "0.1"
```

The default build has **zero dependencies**. `serde` and a real BPE tokenizer
are opt-in features.

## The shape of it

Three ideas carry the whole design.

**Refs are stable addresses, not positions.** Every message gets a short ref
(`m00001`) that never changes and is never reused, even after the message is
folded away. Positions shift every turn; addresses don't. That is what lets a
model say "the thing in `m00012`" ten turns later and still be understood.

**Blocks are tiers.** A folded span becomes a block (`b1`) with a tier: tier 1
is the summary itself, tier 2 folds *summaries of summaries*, tier 3 folds
those. Folding is how the transcript gets smaller than a linear summary chain —
the compression ratio improves the further back you go, and the model never
sees a cliff where detail vanishes.

**The view is explicit.** A fold means "`m00001`–`m00004` are now block `b1`",
but a block is not a message, so *someone* has to turn that sentence into a
message array. Both answers are legitimate and they produce different bytes, so
it is a switch rather than a hidden behaviour (see `ViewMode` below).

## Quickstart

```rust
use billion_context::{
    BlockId, Boundary, CharsAsTokens, CompressRequest, Config, MsgRef, RenderStrategy,
    RequestedRange, Session, TurnInput,
};

let mut session = Session::new(Config::new(200_000)).with_estimator(CharsAsTokens);

// 1. Project: assign refs, work out what is foldable, decide whether to nudge.
let turn = session.process_turn(TurnInput {
    messages: &history,
    token_count: estimate(&history),
    render: RenderStrategy::All,
    now: now(),
});

// 2. The model saw turn.messages, maybe got turn.nudge, and answered:
//    "fold m00012-m00030, here is the summary".
let applied = session.apply_compression(CompressRequest {
    ranges: vec![RequestedRange {
        start: Boundary::Message(MsgRef::new(12).unwrap()),
        end: Boundary::Message(MsgRef::new(30).unwrap()),
        topic: Some("build setup".into()),
        summary: "…".into(),
    }],
    now: now(),
});

// 3. Next turn the originals are gone from the view; the refs survive.
let turn = session.process_turn(TurnInput { messages: &history, .. });

// 4. The model cites b1 and wants the detail back.
let hits = session.retrieve(&[MsgRef::new(12).unwrap()]);

// 5. Or undo the fold entirely.
session.decompress(&[BlockId::new(1)]);
```

`process_turn` is the only hot path: synchronous, no I/O, no clock, no RNG. It
takes `&mut self` and returns a value — there is no `Arc<Mutex<Session>>` in the
design, and no async.

## Who rewrites the message array

```rust
// Default: the kernel hands back a view you can send as-is. Covered messages
// are dropped and replaced by one carrier message holding the summary.
let session = Session::new(cfg);

// Or: the kernel leaves the array alone and only reports coverage through
// `blocks`. Rewriting is the host's job.
let session = Session::new(cfg).with_view_mode(ViewMode::Host);
```

`ViewMode::Host` is the upstream model: `turn.messages` still contains the
folded originals, just tagged with `<acp …>` refs.

### The stateless-host trap

A host that resends the whole transcript every turn will send the already-folded
originals back with **fresh message ids**. The kernel recognises them by content
fingerprint and maps them onto their original ids before pruning — otherwise
compression silently does nothing. Fingerprints are only registered at the
moment of folding, because that is when "what went into this block" is known;
indexing live messages would conflate two live messages with identical text.

Hosts that never resend folded originals (they do their own surface replacement)
should turn it off:

```rust
let session = Session::new(cfg).with_reconcile(false);
```

It is a no-op under `ViewMode::Host`.

## API

| Type | Methods |
|---|---|
| `Session` | `new(Config)` / `try_new` (returns `Vec<ConfigError>`), `with_policy` / `with_estimator` / `with_pipeline`, `process_turn`, `apply_compression`, `retrieve`, `search`, `decompress`, `status`, `state`, `fork` |
| `Engine` | `begin` / `resume` / `round` / `with_max_rounds` — the host-driven fold loop |
| `pipeline` | `default_pipeline()`, `Node` — 13 nodes, replaceable |
| `persist` | `to_json` / `from_json` / `save` / `load` — versioned snapshots |

`process_turn` runs the pipeline in a pinned order:

```
reconcile-live-ids → assign-refs → sync-blocks → prune → ccr-store
  → absorb-hide → crush → absorb-prompt → filter → hide-compress-calls
  → recommend → nudge-inject → emergency-truncate
```

The `absorb-*`, `crush` and `ccr-store` nodes are **deliberately empty** here:
they need model calls or a content store, which a pure kernel cannot supply.
They are `enabled: false` in the reference configuration, so default behaviour
matches upstream; a host that implements the hooks can turn them on.

## Counting tokens

The kernel only ever *estimates* ranges — the host already knows the real size
from usage data. The default estimator counts CJK characters as one token each
and everything else as a quarter, which is within about ±40% of a real
vocabulary. When that is not good enough:

```toml
billion-context = { version = "0.1", features = ["hf-tokenizer"] }
```

```rust
use billion_context::tokenizer::HfTokenizer;

let session = Session::new(cfg).with_estimator(HfTokenizer::from_file("tokenizer.json")?);
```

Any HuggingFace `tokenizer.json` works — the vocabulary is a deployment choice,
not a kernel default. `TokenEstimator` is a one-method trait, so a host counting
with `tiktoken-rs` or a tokenizer it already holds should just implement it.

## Provenance

This is a Rust port of the kernel inside
[billion-context](https://github.com/ranxianglei/billion-context) (TypeScript),
and it is verified the hard way: the same scenario is run through both
implementations and the canonical output is diffed field by field. That harness
found nine defects self-tests could not, including "blocks never absorb, so the
tier ladder is decorative" and "the first user message is dropped on fold".

Current state: **0 differences**, 52 tests (59 with `hf-tokenizer`), clippy clean
on both configurations.

## License

MIT.

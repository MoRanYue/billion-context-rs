# Rust kernel vs TypeScript kernel: differential test report

The same `scenario-a.json` goes into both implementations, each emits isomorphic canonical JSON, and the two are diffed field by field.

```
# TypeScript side (runs kernel/src through tsx, does not need dist/)
# Must run from the repository root: tsx lives in the root node_modules.
cd path/to/billion-context
node --import tsx path/to/billion-context-rs/diff/run-ts.mts \
     path/to/billion-context-rs/diff/scenario-a.json \
     path/to/billion-context-rs/diff/out-ts.json

# Rust side
cd path/to/billion-context-rs
cargo run --offline -q --example diff_harness -- diff\scenario-a.json diff\out-rust.json

# Compare
cd diff
node compare.mjs out-ts.json out-rust.json 500
node hist.mjs          # group by field
```

## Current result: 111 → **0 differences**

Every field of the seven steps (5 `processTurn` rounds + 2 `applyCompression` calls + `decompress` + `status`) is byte-identical:
`view`, `refs`, `blocks` (count / tier / compressedTokens / survived / active), `tokensCompressed`,
`modelContextLimit`, and every field of `nudge` (`reason` / `tier` / `maxPending` / `growth` / `ranges`).

---

## How it converged

### 111 → 45: working out who rewrites the message array

The first report called this "TypeScript uses surface replacement, Rust uses a carrier". **Wrong.**
In fact the TypeScript side **also** drops covered messages and inserts a summary carrier on the
step after the fold takes effect (step4, 10 → 8 messages). The real divergence was only at step5:
the host resent the same content under **fresh ids**, and Rust's `reconcile-live-ids` claimed it by
content fingerprint and pruned it again, while TypeScript does no such thing.

So `ViewMode` was added (`billion-context/src/pipeline/mod.rs`):

- `ViewMode::Send` (default): the kernel hands back a view you can send — covered messages are
  dropped and replaced by one carrier message holding the summary.
- `ViewMode::Host`: the kernel leaves the array alone and reports coverage through `blocks`;
  rewriting is the host's job.

The companion `Session::with_reconcile(false)` turns off content-fingerprint reconciliation. A host
that never resends folded originals has nothing to reconcile, and reconciling by content risks
merging two identical live messages into one.

### 45 → 30: `reconcile` was the root cause of those 64 `refs` differences

With it off, `refs[][]` and `refs.length` went green in one step.

### 30 → 23: `compressedTokens` 3599 vs 3555 was **the harness's** fault, not the kernel's

The first report said "TypeScript counts the rendered, tagged text". Reading `tokenize.ts` showed
`countMessageTokens = countTokens(text) + thinkingTokenValue` — body text only. The gap of 44 came
from `run-ts.mts` feeding `applyCompression` the **rendered view** (with `<acp …>` tags) that
`processTurn` returned; a real host passes its own unrendered transcript.

After keeping a separate `lastRaw` on the TypeScript side, `compressedTokens` and `tokensCompressed`
agree. **The Rust counting convention was right all along.**

### 23 → 18: a block must retire together with the messages it covered

`sync-blocks` used to do an unconditional `survived += 1` for every active block. TypeScript's rule
is: if a block's `directMessageIds` are no longer in the current transcript, the host replaced the
surface and the block no longer represents anything → set `active: false` and **do not advance the
survival counter**.

Doing this in `SyncBlocks` fixed `active` (1), `survived` (3) and `view.length` (1) at once.

### 18 → 12: tier is derived, not declared by the caller

The first report blamed "`modelContextLimit` reported as 0, so usage differed". Also **wrong**.
The cause was `CompressRequest.tier` letting the caller declare the level; the harness passed T2
while TypeScript chose T1.

TypeScript's rule (`unit-projection.ts:171`):

```ts
const outputTier = isBlockBoundary ? Math.min(3, targetTier + 1) : 1;
```

Folding messages → T1; folding blocks → one tier above the highest block absorbed, capped at 3.
That is kernel knowledge and should not be outsourced.

So the `tier` field was removed from `CompressRequest` and is derived by `Session::derive_tier`.
This also filled in **block absorption**, which had been missing entirely (`direct_blocks`, absorbed
blocks set inactive, absorbed summary tokens counted into `compressedTokens`) — before that, T2/T3
could never be produced and the three-tier pyramid was decorative.

`Status` gained `model_context_limit`, and the `status` step no longer reports 0.

### 12 → 0: porting the whole second pass of `recommend.ts`

All 12 remaining differences were in `nudge`. This is not "advisory output, close enough": the
protection logic decides *what must not be folded* (otherwise you over-compress and lose
information), and the foldability screen decides *whether a fold would be refused* (otherwise you
lose messages, or loop forever). So it was ported in full, adding `billion-context/src/recommend.rs`:

| Piece | Role |
| --- | --- |
| `protected_refs` | Rule 1 (last N messages) + rule 2 (last N tokens, **rewound from the tail**) + rule 3 (last N user turns) + tool-pair closure |
| `integrity_withdrawals` | INV1 (a reasoning block and its tool_call fold and stay together) / INV2 (call/result are inseparable), to a fixed point |
| `span_owns_foldable` | A span must cover every direct id of each summary carrier inside it, or `evaluateRefold` declares a livelock |
| Segmentation and batching | Break on a user message once a group reaches 3; each batch is checked against `minCompressRange`; a short tail merges into the previous batch and a short head is dropped |
| `screenRecommendedUnits` | Re-project each unit; drop it if fatal or if it produces nothing |
| `pendingByTier` | T1 = chars of the merged effective ranges; T2 = tokens of T1 block summaries; T3 = tokens of T2 block summaries |

Three more subtle contracts:

1. **Growth is measured against "last nudge shown", not the baseline**:
   `ref = lastNudgeShownTokens > 0 ? lastNudgeShownTokens : (lastPerMessageNudgeTokens > 0 ? that : tokenCount)`.
   The low-water anchor must be updated **after** `decideNudge`, and both reset to zero after a
   successful compression.
2. **The two floors in `resolveAdaptiveGrowth` are different fields**:
   `min(growthCap, max(growthFloor, round(limit × growthRatio)))` uses `growthFloor`, while the
   nudge threshold `max(minGrowthFloor, minGrowthRatio × that)` uses `minGrowthFloor`.
3. **`defaultCountTokens` is `cjk + ceil((len_utf16 − cjk) / 4)`**, not `ceil(len/4)`; the CJK test
   is `0x4E00–0x9FFF / 0x3040–0x30FF / 0xAC00–0xD7AF`. On Chinese text the two differ by up to 3×,
   which directly moves when compression starts being suggested.

---

## Tokenizer differential: estimator agreement and injection plumbing

The `hf-tokenizer` feature introduces a **source of numbers**, not new logic. Its divergence does
not crash — it quietly shifts every threshold, which is exactly why it needs its own comparison.

### 1. Value-by-value: the estimator must produce the same number

```
node diff/tok-ts.mts diff/tok-ts.json                 # corpus + TypeScript reference values
cargo run --example tok_diff -- diff/tok-ts.json diff/tok-rust.json
node diff/compare-tok.mjs diff/tok-ts.json diff/tok-rust.json
```

**73 samples**, built on the TypeScript side (both sides count the same strings), deliberately
covering where a hand-written character class drifts most:

- **Every boundary** of the three CJK blocks (`0x303f/0x3040/0x3041`, `0x30fe/0x30ff/0x3100`,
  `0x4dff/0x4e00/0x9ffe/0x9fff/0xa000`, `0xabff/0xac00/0xd7ae/0xd7af/0xd7b0`), as a single
  character, repeated 8×, and embedded in ASCII
- Non-BMP: `😀`, a ZWJ family emoji, `𠀀` (U+20000, **2 UTF-16 units but not in a CJK block**)
- Combining `e` + `U+0301`, pure whitespace, pure newlines, a 1000-character run

**Result: 0 mismatches.** The key point is that `text.length` in JavaScript is **UTF-16 units**, so
the Rust side must accumulate `ch.len_utf16()` rather than `chars().count()` — on text with emoji
that is off by 2×, and "testing yourself against yourself" will never reveal it.

### 2. Injection plumbing: the estimator must reach every counting site

Agreeing on the estimator is not enough; it has to be proven that it reaches everywhere it should
(render tags, nudge, block `compressedTokens`, `status`). Both sides are given the **same** synthetic
counter:

```js
// diff/run-ts.mts, the 4th argument `inject`
const synthetic = (text) => {
  let sum = 0;
  for (const ch of text) sum += ch.codePointAt(0);
  return Math.ceil(text.length / 3) + (sum % 11);
};
createCore({ countTokens: synthetic });
```

```rust
// billion-context/examples/diff_harness.rs, same 4th argument
struct SyntheticCounter;
impl TokenEstimator for SyntheticCounter {
    fn count(&self, text: &str) -> u32 {
        let units: usize = text.chars().map(|c| c.len_utf16()).sum();
        let sum: u32 = text.chars().map(|c| c as u32).sum();
        units.div_ceil(3) as u32 + sum % 11
    }
}
```

**The injection does have an effect**: TypeScript baseline vs injected differs in **60 fields**
(render tags go from `900` to `1.2K`; nudge and status all move). **And Rust vs TypeScript on the
injected run: 0 differences.**

So both paths — the default heuristic and an injected custom counter — are byte-identical, and the
counter is not honoured in only one place.

### 3. Comparison against the real vocabulary (calibration, not a consistency test)

The same command plus `assets/tokenizer.json` shows how far the heuristic sits from the real thing:

```
heuristic vs real vocabulary: total 2027 vs 2784 (-27.2%), over on 9, under on 53
  worst: "     " +100.0%
```

The error flips sign with the content (code and emoji underestimated, pure whitespace
overestimated), confirming that no scale factor fixes it — and that defaulting to the heuristic
while offering a feature for exactness is the right shape.

### 4. What this test does not cover

There is **no** independent tokenizer reference in this environment (no npm `tokenizers`, no Python
`tokenizers`, no `@anthropic-ai/tokenizer`), so `HfTokenizer` itself was not compared id-by-id
against another implementation. What was checked is internal consistency (encode/decode round trip,
`count == ids.len()`, determinism) plus **equivalence at every downstream counting site**. Closing
that loop properly needs a third-party `tokenizers` install.

---

## Conclusion

The differential suite caught **9 defects that self-tests could never find**:

1. Render tags missing the trailing newline
2. Tokens not going through `formatTokens` (`1.5K` instead of `1500`)
3. `type` using `contentType` instead of `toolName`
4. The first user message being dropped after a fold (upstream rejects the request outright)
5. Nudge swallowed when not advising, where TypeScript always returns a decision object
6. Blocks not retiring with their covered messages, leaving ghost carriers
7. **Block absorption never implemented** — T2/T3 unreachable
8. Nudge growth measured from the wrong baseline, so it stayed silent when it should have advised
9. The adaptive threshold using the wrong field, turning 50000 into 45000

Of these, 4 and 7 are outright logic errors that "testing yourself against yourself" cannot expose.

It also corrected 3 **misdiagnoses** in the first report (view model, token convention, tier
attribution), 2 of which turned out to be bugs in the harness itself. That is the value of
differential testing: it calibrates the test tooling along with the code.

Final state: **0 differences** (0 on the default heuristic path and 0 on the injected-counter path),
tokenizer values agree 73/73, `cargo test` reports 52 passed by default and 59 with `hf-tokenizer`,
and `cargo clippy` is clean in both configurations.

# Changelog

## Unreleased

### Fixed

- **A tool exchange could never be offered for folding.** The recommender
  classified a message as compressible only when it had prose to rewrite, and a
  `ToolCall` has none — its arguments are user intent the host executes, so a
  summary may not touch them. The call therefore became a *hole*, which made the
  pairing invariant read both halves of every pair as split apart: it withdrew
  every tool result, and a transcript full of tool calls offered nothing but the
  prose between them. A tool-heavy conversation — the one that most needs
  folding — reported almost nothing compressible.

  A call is now foldable *content* even though it is not foldable *text*. In a
  5-round parallel-tool-call transcript the advertised pool goes from 4225 to
  10472 tokens, and the ranges go from the 2-message prose gaps to the 6-message
  rounds they should be — value-for-value what the TypeScript kernel reports.

- **Every tool call had the same fingerprint**, because the fingerprint fed only
  foldable text and a call has none. `reconcile` matches a resent original by
  fingerprint, so a host that resends its transcript could have a call matched
  against the wrong original — and the pairing it should have preserved would go
  with it. A call now hashes its name, call id and arguments.

- `token_snapshot` recorded `0` tokens for a tool call, so its arguments were
  free in every pool and range total. They are counted now.

- **Coverage keyed on the whole id, but an id may name one piece of a message.**
  A host that splits a provider message into content blocks — the TypeScript
  kernel's `baseIdOf` convention, `base#c0`, `base#r0` — folds *a piece*, and the
  next turn may re-emit a different piece of the same original. Matching ids
  verbatim let that sibling back into the window as an unfolded original, which
  is the exact content the host asked to fold. Coverage (`State::is_covered`),
  the content index, and block presence now all compare `MessageId::base`, so
  every projection of a folded original stays folded.

### Added

- `strip_ref_tag` — remove a leading rendered ref tag. `RenderStrategy::All`
  tags whatever payload a message carries, a tool call's arguments included
  (value-for-value what the TypeScript kernel renders), so a host that parses
  that payload as JSON for a provider strips it here.
- `Content::tool_arguments` — a call's arguments verbatim, for anything that
  sizes or identifies a message.
- `Content::tool_call_id` — the call a piece of content is half of, whether it
  is the call or the result. The identity that keeps a pair together through a
  fold, and what a host needs to check a range it wrote itself.
- README §8 "Tool calls, including parallel ones": one `Message` carries one
  `Content`, so a provider message with N `tool_use` blocks is N core messages;
  pairing still finds the results by call id, and the host merges them back on
  the way out.

## 0.3.0 — 2026-10-27

### Added

- **Media protection.** `Message::with_unrecoverable_payload` /
  `Message::has_unrecoverable_payload` mark a message whose payload a fold would
  destroy: an image behind a remote URL, a provider file reference, a document.
  The recommender treats such a message as a **hole** — no suggested range may
  span it — so folding the text around it can never take the picture with it.

  This was the last correctness gap against the reference kernel, and the only
  one with no recovery path: a summary is text, so once an image block has been
  folded there is nothing behind `decompress` to restore it.

- `billion_context_wire::Block::has_unrecoverable_payload` and
  `billion_context_wire::Message::has_unrecoverable_payload` — the Anthropic
  classifier, so a host does not have to re-derive which media is restorable.
  Inline `base64` and `data:` payloads keep their bytes and stay foldable; a
  remote URL, a file id, or any block type this crate never modelled does not.

### Changed (breaking)

- `billion_context::Message` gained a public `unrecoverable_payload: bool`.
  Struct literals must add it; the constructors (`Message::text`,
  `Message::tool_call`, `Message::tool_result`) set it to `false`.
- `billion_context_wire::Block::Unknown` now carries the block's whole
  `Map<String, Value>` instead of being a unit marker. An unmodelled block used
  to round-trip as `{"type":"unknown"}`, silently dropping an image's payload.

  Caveat: the map is sorted, so key order inside an unknown block is not
  preserved. `RawValue` would keep the bytes verbatim, but it cannot be used
  while `Content` is `#[serde(untagged)]` — serde buffers the value through its
  own `Content`, which has no raw-value support.

## 0.2.0 — 2026-10-26

### Added

- **`acp_decompress`, `acp_search` and `acp_status`** — the tool layer now
  covers the whole model-facing surface of the original, not just folding.
  Previously a model could fold a span and then had no way to look at what it
  had folded; that asymmetry is what makes models either refuse to compress or
  compress and then invent the detail they lost.

  | Tool | Kernel call |
  |---|---|
  | `acp_decompress` | `Session::block` |
  | `acp_search` | `Session::search` |
  | `acp_status` | `Session::status` |

- `Session::block(&BlockId) -> Option<&Block>` — the read-only counterpart to
  `decompress`. Reading a block does not expand it, so a host answering
  "what does this block say" cannot accidentally re-inflate the context.
- `SearchArgs`, `DEFAULT_SEARCH_LIMIT`, `parse_search`, `parse_decompress`,
  `parse_status`, `render_decompression`, `render_search`, `render_status`.

### Changed

- **`search` matches terms, not a phrase.** It previously matched the query as
  one substring, so `borrow constraints` found nothing even when a block's topic
  said "borrow checker" and its summary said "constraints". Terms are now scored
  separately: topic hits weigh 0.15 each (capped 0.45), summary hits 0.04 each
  (capped 0.2), total capped at 1.0.
- `Hit::score` is `f64` rather than `u32` — the score is a weighted blend, and
  an integer count collapsed genuinely different matches into ties.
- `snippet` anchors on the earliest matching term instead of the first term.

## 0.1.1 — 2026-10-26

### Fixed

- **The `serde` feature did not compile.** `billion-context 0.1.0` advertised a
  `serde` feature that failed to build for every downstream crate, for three
  independent reasons:
  - `serde_json` was declared only under `[dev-dependencies]`, so `persist`
    could not see it once the feature was on;
  - `State`, `RefMap` and `RefEntry` carried no `Serialize`/`Deserialize`
    derives, so `persist`'s envelope could not be built;
  - the feature's own test module referred to `State` without importing it.

  The feature now pulls `serde` + `serde_json` itself and all three types
  round-trip. `cargo test --features serde` passes (55 tests).

- `serde_json` stays in `[dev-dependencies]` as well, because the differential
  examples parse JSON regardless of features.

### Added

- CI matrix over all four feature combinations (`default`, `serde`,
  `hf-tokenizer`, `serde,hf-tokenizer`) running both `cargo test` and
  `cargo clippy -- -D warnings`. `cargo publish` only verifies the default
  feature set, which is exactly how this shipped; a matrix is the guard.

### Notes

- No API changes. `billion-context-wire` and `billion-context-tool` are
  republished at 0.1.1 to keep the three crates in lockstep; their code is
  unchanged apart from the dependency requirement.

## 0.1.0 — 2026-10-26

First release.

- `billion-context`: the kernel — refs, blocks, three-tier folding, the nudge
  rhythm, cross-turn reconciliation, and the two-pass fold protocol.
- `billion-context-wire`: Anthropic / OpenAI wire types, losslessly convertible
  to and from the internal representation.
- `billion-context-tool`: fold / retrieve exposed as tools a model can call.

# Changelog

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

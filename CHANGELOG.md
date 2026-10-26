# Changelog

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

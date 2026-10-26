# billion-context-wire

Provider wire types for [`billion-context`](https://crates.io/crates/billion-context).

```toml
[dependencies]
billion-context-wire = "0.1"
```

Anthropic and OpenAI request/response shapes, modelled with one rule: **a round
trip must not change the bytes.** Provider payloads are cached upstream and
billed per prefix, so a serde struct that reorders keys, drops an unknown field,
or re-serialises tool arguments is not a cosmetic problem — it invalidates the
cache on every request.

Concretely:

- **Unknown fields are preserved.** Every struct keeps the leftovers in a
  flattened `extra` map, so a field you have never heard of survives.
- **`thinking` blocks are not parsed.** Their content is opaque and their
  signature is verified byte for byte; rewriting them bricks a session.
- **Tool arguments are not re-serialised.** They arrive as a JSON *string* and
  leave as the same string. Parsing and re-encoding changes key order and float
  formatting, which the provider notices.
- **`Usage` keeps the provider's own convention.** Anthropic reports
  `input_tokens` *excluding* cache reads and writes; OpenAI's
  `prompt_tokens` *includes* them. Normalising either way silently misprices
  every request, so both shapes are kept as they are.

```rust
use billion_context_wire::{Block, Message, Usage};

let msg: Message = serde_json::from_str(&raw)?;
// ... pass it around, append to the transcript ...
assert_eq!(serde_json::to_string(&msg)?, raw); // modulo key order
```

The `value_round_trip_reorders_keys_and_changes_bytes` test exists to make that
caveat visible rather than paper over it: `serde_json`'s map is a `BTreeMap`, so
keys come back sorted. If byte equality matters to you, keep the original text
and use these types for reading only.

## License

MIT.

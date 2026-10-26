# billion-context-tool

Tool-call adapter for [`billion-context`](https://crates.io/crates/billion-context):
the two tools a model needs to compress its own context, and the parsing that
turns a model's arguments into something the kernel accepts.

```toml
[dependencies]
billion-context-tool = "0.1"
```

## Why this is a crate and not three functions in a README

The kernel's job is to *apply* a fold. Deciding to fold, and saying what got
folded, is the model's job — and a model reaches that decision through a tool
call whose schema it has to understand. Two things go wrong if you hand-roll it:

**The model folds the wrong thing.** A span is addressed by its first and last
ref, and refs are not positions. A model that "reconstructs" the range from what
it can see will get it subtly wrong. The schema pins both ends explicitly and
says so in the description.

**Partial failure reads as total failure.** `apply_compression` returns a
per-range verdict: some ranges fold, some are refused, and the model needs to
know which. `render_compress` writes that back as one message with the accepted
blocks named and the refused spans quoted with their reason, so the model does
not conclude the whole call failed and retry blindly.

## Usage

```rust
use billion_context::{CompressRequest, Session};
use billion_context_tool as tool;

// Give these to the model alongside your own tools.
let defs = tool::definitions();           // [acp_compress, acp_retrieve]

// The model called acp_compress. Its arguments are untrusted JSON.
let ranges = tool::parse_compress(&args)?; // -> Vec<RequestedRange>

let applied = session.apply_compression(CompressRequest { ranges, now });

// Tell it what actually happened, in its own vocabulary.
let reply = tool::render_compress(&applied.block_ids, &applied.errors);

// Later it cites a ref and wants the original back.
let refs = tool::parse_retrieve(&args)?;
let hits = session.retrieve(&refs);
let text = tool::render_retrieval(&hits);
```

| Item | Purpose |
|---|---|
| `definitions()` | Both tool schemas, ready for the provider's `tools` field |
| `compress_definition()` / `retrieve_definition()` | One at a time |
| `parse_compress` / `parse_retrieve` | Untrusted arguments → typed requests |
| `render_compress` / `render_retrieval` | Results → one message the model can act on |

`parse_compress` is deliberately strict about malformed refs: it reports which
field was wrong (`"m12x"` is not a ref, and `m00012` is not `m12`), because a
model that gets a precise complaint fixes it, and one that gets "invalid
arguments" usually does not.

## License

MIT.

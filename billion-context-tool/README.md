# billion-context-tool

Tool-call adapter for [`billion-context`](https://crates.io/crates/billion-context):
the five tools a model needs to manage its own context, and the parsing that
turns a model's arguments into something the kernel accepts.

```toml
[dependencies]
billion-context-tool = "0.2"
```

## Why this is a crate and not five functions in a README

The kernel's job is to *apply* a fold. Deciding to fold, and saying what got
folded, is the model's job — and a model reaches that decision through a tool
call whose schema it has to understand. Three things go wrong if you hand-roll it:

**The model folds the wrong thing.** A span is addressed by its first and last
ref, and refs are not positions. A model that "reconstructs" the range from what
it can see will get it subtly wrong. The schema pins both ends explicitly and
says so in the description.

**Partial failure reads as total failure.** `apply_compression` returns a
per-range verdict: some ranges fold, some are refused, and the model needs to
know which. `render_compress` writes that back as one message with the accepted
blocks named and the refused spans quoted with their reason, so the model does
not conclude the whole call failed and retry blindly.

**A model that cannot look back compresses badly.** Folding is only safe when the
model believes it can recover what it folded. Without `acp_search` and
`acp_decompress` it either refuses to fold (context grows) or folds and then
hallucinates the detail it lost.

## The five tools

| Tool | What it is for | Kernel call |
|---|---|---|
| `acp_compress` | Fold a span into a summary | `Session::apply_compression` |
| `acp_retrieve` | Read the original messages behind a ref | `Session::retrieve` |
| `acp_decompress` | Read one folded block's summary in full | `Session::block` |
| `acp_search` | Find which block covered a topic | `Session::search` |
| `acp_status` | See how full the context is and what is worth folding | `Session::status` |

`acp_decompress` and `acp_search` are the read path. Neither mutates the
session — reading a block does not expand it, so a model cannot accidentally
re-inflate the context it just compressed by being curious.

## Usage

```rust
use billion_context::{CompressRequest, Session};
use billion_context_tool as tool;

// Give these to the model alongside your own tools.
let defs = tool::definitions();          // all five schemas

// The model called acp_compress. Its arguments are untrusted JSON.
let ranges = tool::parse_compress(&args)?;        // -> Vec<RequestedRange>
let applied = session.apply_compression(CompressRequest { ranges, now });
let reply = tool::render_compress(&applied.blocks, &applied.errors);

// It wants something it folded earlier.
let query = tool::parse_search(&args)?;           // -> SearchArgs { query, limit }
let hits = session.search(&query.query, query.limit);
let reply = tool::render_search(&hits);

// It picked one and wants the whole summary.
let id = tool::parse_decompress(&args)?;          // -> BlockId
let reply = session
    .block(id)
    .map(tool::render_decompression)
    .unwrap_or_else(|| "No such block.".into());

// It asks how much room is left.
tool::parse_status(&args)?;
let reply = tool::render_status(&session.status(), token_count, &last_ranges);
```

| Item | Purpose |
|---|---|
| `definitions()` | All five schemas, ready for the provider's `tools` field |
| `compress_definition()` … `status_definition()` | One at a time |
| `parse_compress` / `parse_retrieve` / `parse_decompress` / `parse_search` / `parse_status` | Untrusted arguments → typed requests |
| `render_compress` / `render_retrieval` / `render_decompression` / `render_search` / `render_status` | Results → one message the model can act on |

## Two deliberate strictnesses

`parse_compress` reports *which* field was wrong (`"m12x"` is not a ref, and
`m00012` is not `m12`), because a model that gets a precise complaint fixes it,
and one that gets "invalid arguments" usually does not.

`parse_decompress` refuses a message ref. `m00012` belongs to `acp_retrieve`;
accepting it here would let a model believe it had decompressed a block when it
had read one message — and the difference is exactly what it is about to reason
from.

`parse_search` is deliberately *lenient* about `limit`: 1000 becomes 50 rather
than an error, because a model asking for a huge number wants "more", not a
correction. Every term is matched separately against the topic and the summary,
so `borrow constraints` finds a block whose topic says "borrow checker" and
whose summary says "constraints" — a phrase search would find neither.

## License

MIT.

feat: stop hiding the fold receipt, and read what callers actually meant

HideCompressCalls removed every compress exchange but the newest, every turn.
That traded a few hundred bytes against a prefix-cache break at a seam that had
already been paid for, and it deleted two things the model needs: the exchanges
for blocks still in force (the record of which ranges are already folded) and
the exchange of a failed attempt (the only copy of the error, and the only thing
that stops a deterministic model from repeating a no-op fold).

Removal is now tied to the places where the prefix has already moved: a block
that was absorbed or expanded, and orphans at or after the lowest ref a fold
touched. The sweep point belongs to the fold that set it and is consumed on use.

Also:

- CompressRequest::call_id carries the call that caused a fold, so the block can
  remember which exchange produced it (Block::compress_call_id).
- A reversed range is normalised instead of rejected: the caller meant the span
  between the endpoints.
- Synthetic ids (the first-message pin, summary carriers) no longer count as
  range membership, so a fold cannot claim to cover what it left behind.
- ALWAYS_PROTECTED_TOOLS keeps `compress` and `acp_rule` unfoldable whatever the
  host configuration says.

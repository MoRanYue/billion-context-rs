//! Blocks, the ref map, and the whole mutable state of one conversation.

use crate::ids::{BlockId, Boundary, MessageId, MsgRef, Timestamp};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Tier {
    T1 = 1,
    T2 = 2,
    T3 = 3,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Generation {
    Young,
    Old,
}

/// One compression block. `direct` is what the model folded; `effective` is the
/// transitive closure including blocks absorbed from lower tiers, which is what
/// pruning must actually drop.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Block {
    pub id: BlockId,
    pub run: u32,
    pub tier: Tier,
    pub topic: Option<String>,
    pub summary: String,
    pub direct: Vec<MessageId>,
    pub effective: Vec<MessageId>,
    pub direct_blocks: Vec<BlockId>,
    pub compressed_tokens: u32,
    pub created_at: Timestamp,
    pub survived: u32,
    pub generation: Generation,
    pub active: bool,
    pub expanded: bool,
    pub restored_inline: bool,
    /// The refs the caller named, **as named**. Folding `b3..b4` records
    /// `b3`/`b4`, not the messages those blocks happen to cover: the declared
    /// span is what a later citation has to match, and it is what upstream
    /// reports. Use [`Session::message_span_of`] to reach the messages.
    pub span: Option<(Boundary, Boundary)>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RefEntry {
    Assigned(MsgRef),
    /// Hard-protected: never advertised, never foldable by ref.
    Blocked,
}

/// Bidirectional ref map.
///
/// The cursor is stored rather than recomputed. The TS kernel rescans the whole
/// map for `highestUsedIndex` on every turn; keeping it here turns an O(n) pass
/// into O(1) and makes the monotonicity invariant local to `allocate`.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RefMap {
    pub(crate) by_raw: HashMap<MessageId, RefEntry>,
    pub(crate) by_ref: HashMap<MsgRef, MessageId>,
    pub(crate) cursor: u32,
}

impl RefMap {
    pub fn get(&self, id: &MessageId) -> Option<RefEntry> {
        self.by_raw.get(id).copied()
    }

    pub fn ref_of(&self, id: &MessageId) -> Option<MsgRef> {
        match self.by_raw.get(id) {
            Some(RefEntry::Assigned(r)) => Some(*r),
            _ => None,
        }
    }

    pub fn raw_of(&self, r: MsgRef) -> Option<&MessageId> {
        self.by_ref.get(&r)
    }

    pub fn is_blocked(&self, id: &MessageId) -> bool {
        matches!(self.by_raw.get(id), Some(RefEntry::Blocked))
    }

    /// Next number to be handed out. Monotonic by construction.
    pub fn cursor_hint(&self) -> u32 {
        self.cursor
    }

    pub fn len(&self) -> usize {
        self.by_ref.len()
    }

    /// Every `(raw id, assigned ref)` pair, in unspecified order. Blocked ids are
    /// skipped — they own no number. Hosts need this to persist the map; the TS
    /// kernel reads `state.messageRefs.byRaw` directly.
    pub fn iter_assigned(&self) -> impl Iterator<Item = (&MessageId, MsgRef)> + '_ {
        self.by_raw.iter().filter_map(|(id, entry)| match entry {
            RefEntry::Assigned(r) => Some((id, *r)),
            RefEntry::Blocked => None,
        })
    }

    pub fn is_empty(&self) -> bool {
        self.by_ref.is_empty()
    }

    pub(crate) fn mark_blocked(&mut self, id: MessageId) {
        self.by_raw.insert(id, RefEntry::Blocked);
    }

    /// Allocate the next free ref. Never reuses a number — the model may cite
    /// any number it has ever seen, including one whose message has died.
    // The entry API cannot express this: `by_ref` and `by_raw` must both be
    // written, and an `Entry` guard would hold the borrow across the second map.
    #[allow(clippy::map_entry)]
    pub(crate) fn allocate(&mut self, id: &MessageId) -> Option<MsgRef> {
        let mut index = self.cursor.max(MsgRef::MIN);
        while index <= MsgRef::MAX {
            let candidate = MsgRef::new(index)?;
            if !self.by_ref.contains_key(&candidate) {
                self.by_ref.insert(candidate, id.clone());
                self.by_raw.insert(id.clone(), RefEntry::Assigned(candidate));
                self.cursor = index + 1;
                return Some(candidate);
            }
            index += 1;
        }
        None
    }
}

#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Stats {
    pub tokens_compressed: u64,
    pub compression_count: u64,
    pub absorbed_tokens: u64,
    pub stored_count: u64,
    pub retrieval_count: u64,
}

/// A payload the kernel clipped, kept so a resend of the original bytes can be
/// clipped again instead of reinstating the whole thing.
///
/// Keyed by ref, not by message id: a host that resends its transcript renames
/// every message, but the ref survives because it is the kernel's own name for
/// that content.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TruncateLatch {
    pub r: MsgRef,
    /// Exactly the bytes the kernel substituted. A resend that is longer than
    /// this has been re-inflated by the host and must be clipped back to it.
    pub text: String,
}

/// Bounded latch list. Unbounded, it grows by one entry per clipped payload per
/// turn — a slow leak that only shows up on a very long session, which is exactly
/// the session that cannot afford it.
pub const MAX_TRUNCATE_LATCH: usize = 64;

#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NudgeState {
    pub last_shown_tokens: u32,
    pub baseline_tokens: u32,
    pub shown_by_tier: HashMap<u8, u32>,
    pub consecutive_truncations: u32,
    /// Turns in a row where truncation ran and could not free anything. Freed on
    /// its own terms, this number is the difference between "keep trying" and
    /// "stop, the situation is terminal".
    pub terminal_streak: u32,
    pub truncate_latch: Vec<TruncateLatch>,
}

/// The whole mutable state of one conversation: serializable, clonable, and
/// forkable. `Session::fork` is just `state.clone()`.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct State {
    pub blocks: Vec<Block>,
    pub refs: RefMap,
    /// First-render token count per ref, written once and read forever so the
    /// rendered `<acp tokens=...>` tags never churn and bust the prefix cache.
    pub token_snapshot: HashMap<MsgRef, u32>,
    /// Fingerprint of every folded message, so a stateless host that resends the
    /// original under a fresh id is recognised instead of double-counted.
    pub content_index: HashMap<u64, MessageId>,
    pub nudge: NudgeState,
    pub stats: Stats,
    pub dead_refs: HashSet<MsgRef>,
    pub next_block: u32,
    pub next_run: u32,
}

impl State {
    pub fn new() -> Self {
        Self {
            next_block: 1,
            next_run: 1,
            ..Self::default()
        }
    }

    pub fn active_blocks(&self) -> impl Iterator<Item = &Block> {
        self.blocks.iter().filter(|b| b.active)
    }

    pub fn covered_ids(&self) -> HashSet<&MessageId> {
        self.active_blocks()
            .flat_map(|b| b.effective.iter())
            .collect()
    }

    /// Whether an active block already owns this message.
    ///
    /// Decided per *original* message: a host that splits one provider message
    /// into `base#c0`, `base#c1` names projections of the same thing, and a
    /// block that recorded one of them covers them all. Comparing ids verbatim
    /// would let the sibling the host emits next turn walk back into the
    /// transcript as an unfolded original.
    pub fn is_covered(&self, id: &MessageId) -> bool {
        let base = id.base();
        self.active_blocks()
            .any(|b| b.effective.iter().any(|c| c.base() == base))
    }

    pub fn block(&self, id: BlockId) -> Option<&Block> {
        self.blocks.iter().find(|b| b.id == id)
    }

    pub fn allocate_block(&mut self) -> Option<BlockId> {
        let id = BlockId::new(self.next_block)?;
        self.next_block += 1;
        Some(id)
    }

    pub fn allocate_run(&mut self) -> u32 {
        let run = self.next_run;
        self.next_run += 1;
        run
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocation_is_monotonic_and_skips_used() {
        let mut map = RefMap::default();
        let a = MessageId::new("a");
        let b = MessageId::new("b");
        assert_eq!(map.allocate(&a).unwrap().index(), 1);
        assert_eq!(map.allocate(&b).unwrap().index(), 2);
        assert_eq!(map.cursor_hint(), 3);
        // Re-allocating an already-assigned id would be a caller bug; the map
        // would happily hand out a second number, so callers check first.
        assert_eq!(map.ref_of(&a).unwrap().index(), 1);
    }

    #[test]
    fn blocked_ids_consume_no_number() {
        let mut map = RefMap::default();
        let secret = MessageId::new("secret");
        map.mark_blocked(secret.clone());
        assert!(map.is_blocked(&secret));
        assert!(map.ref_of(&secret).is_none());
        assert_eq!(map.cursor_hint(), 0);
        assert_eq!(map.allocate(&MessageId::new("visible")).unwrap().index(), 1);
    }
}

//! Identity: the only identifiers a model ever sees.
//!
//! Refs are allocated once and never reused, which is what makes an `mNNNNN`
//! written into a summary three turns ago still resolvable today. That property
//! is load-bearing for the whole mechanism, so it lives in its own module with
//! its own tests.

use std::fmt;
use std::sync::Arc;

/// Milliseconds since the Unix epoch. Always injected, never read from the
/// clock: the engine must be a deterministic function of its inputs.
pub type Timestamp = u64;

/// A message reference (`m00001`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MsgRef(u32);

impl MsgRef {
    pub const MIN: u32 = 1;
    pub const MAX: u32 = 9_999_999;

    pub fn new(index: u32) -> Option<Self> {
        (Self::MIN..=Self::MAX).contains(&index).then_some(Self(index))
    }

    pub fn index(self) -> u32 {
        self.0
    }

    pub fn parse(text: &str) -> Option<Self> {
        let digits = text.trim().strip_prefix('m')?;
        if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        Self::new(digits.parse().ok()?)
    }
}

impl fmt::Display for MsgRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "m{:05}", self.0)
    }
}

impl fmt::Debug for MsgRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MsgRef({self})")
    }
}

/// A compression block reference (`b1`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId(u32);

impl BlockId {
    pub fn new(n: u32) -> Option<Self> {
        (n >= 1).then_some(Self(n))
    }

    pub fn number(self) -> u32 {
        self.0
    }

    pub fn parse(text: &str) -> Option<Self> {
        let digits = text.trim().strip_prefix('b')?;
        if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        Self::new(digits.parse().ok()?)
    }
}

impl fmt::Display for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "b{}", self.0)
    }
}

impl fmt::Debug for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BlockId({self})")
    }
}

/// Host-native message id. The host owns the id space and must keep it
/// append-only — position-based ids break the moment a message is folded.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct MessageId(Arc<str>);

impl MessageId {
    pub fn new(id: impl AsRef<str>) -> Self {
        Self(Arc::from(id.as_ref()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for MessageId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// A boundary a model may cite: a message or a whole block.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Boundary {
    Message(MsgRef),
    Block(BlockId),
}

impl Boundary {
    /// `m00123` is a message, `b12` is a block. `None` for anything else: a
    /// host parsing a model's citation wants "not a boundary", not an error
    /// value it has to discard.
    pub fn parse(text: &str) -> Option<Self> {
        MsgRef::parse(text)
            .map(Boundary::Message)
            .or_else(|| BlockId::parse(text).map(Boundary::Block))
    }
}

// ---------------------------------------------------------------------------
// Serde: refs serialize in their WIRE form
// ---------------------------------------------------------------------------

// `m00001` / `b1` are the only identifiers a model sees, so they are the only
// sensible on-disk representation too. Storing the integer would invent a
// second id space that has to be kept in sync forever.
#[cfg(feature = "serde")]
mod wire {
    use super::{BlockId, MessageId, MsgRef};
    use serde::de::Error as _;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    macro_rules! wire_id {
        ($ty:ty, $parse:path, $name:literal) => {
            impl Serialize for $ty {
                fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                    s.collect_str(self)
                }
            }
            impl<'de> Deserialize<'de> for $ty {
                fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                    let raw = String::deserialize(d)?;
                    $parse(&raw).ok_or_else(|| {
                        D::Error::custom(format!(concat!("invalid ", $name, ": {}"), raw))
                    })
                }
            }
        };
    }

    wire_id!(MsgRef, MsgRef::parse, "message ref");
    wire_id!(BlockId, BlockId::parse, "block id");

    impl Serialize for MessageId {
        fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
            s.serialize_str(self.as_str())
        }
    }
    impl<'de> Deserialize<'de> for MessageId {
        fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
            Ok(MessageId::new(String::deserialize(d)?))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refs_round_trip_through_their_wire_form() {
        let r = MsgRef::parse("m00007").unwrap();
        assert_eq!(r.index(), 7);
        assert_eq!(r.to_string(), "m00007");
        assert!(MsgRef::parse("m00000").is_none());
        assert!(MsgRef::parse("m10000000").is_none());
        assert!(MsgRef::parse("m7").is_some());
        assert!(MsgRef::parse("00007").is_none());
    }

    #[test]
    fn boundaries_accept_both_shapes() {
        assert_eq!(
            Boundary::parse("m00003"),
            Some(Boundary::Message(MsgRef::new(3).unwrap()))
        );
        assert_eq!(
            Boundary::parse("b12"),
            Some(Boundary::Block(BlockId::new(12).unwrap()))
        );
        assert!(Boundary::parse("x1").is_none());
    }
}

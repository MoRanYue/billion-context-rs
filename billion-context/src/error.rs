//! Compression-request errors.
//!
//! Every kind maps to a stable machine code **and** a message the model reads.
//! The text is part of the protocol: it tells the model what to change on the
//! next attempt, so changing a word here changes behaviour. Treat the strings
//! as a public API and cover them with snapshot tests.

use std::fmt;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RangeErrorKind {
    InvalidRef,
    UnknownRef,
    Consumed,
    TooSmall,
    SummaryTooShort,
    SummaryTooLong,
    WouldSplitTurn,
    ProtectedZone,
    /// The kernel has run out of block ids. Only reachable after millions of
    /// folds, but it must fail loudly rather than silently drop a range.
    Capacity,
}

impl RangeErrorKind {
    pub fn code(self) -> &'static str {
        match self {
            RangeErrorKind::InvalidRef => "invalid_ref",
            RangeErrorKind::UnknownRef => "unknown_ref",
            RangeErrorKind::Consumed => "consumed",
            RangeErrorKind::TooSmall => "too_small",
            RangeErrorKind::SummaryTooShort => "summary_too_short",
            RangeErrorKind::SummaryTooLong => "summary_too_long",
            RangeErrorKind::WouldSplitTurn => "would_split_turn",
            RangeErrorKind::ProtectedZone => "protected_zone",
            RangeErrorKind::Capacity => "capacity",
        }
    }

    /// What the model should do differently. Kept separate from `code` so a
    /// host can localize the prose without touching the machine contract.
    pub fn hint(self) -> &'static str {
        match self {
            RangeErrorKind::InvalidRef => "use the mNNNNN refs shown in the transcript",
            RangeErrorKind::UnknownRef => "that ref is not in this view; pick one that is",
            RangeErrorKind::Consumed => "that content is already folded; fold something else",
            RangeErrorKind::TooSmall => "pick a larger contiguous range",
            RangeErrorKind::SummaryTooShort => "write a summary that can stand in for the range",
            RangeErrorKind::SummaryTooLong => "shorten the summary",
            RangeErrorKind::WouldSplitTurn => "do not cut a tool call away from its result",
            RangeErrorKind::ProtectedZone => "that range is protected; choose another",
            RangeErrorKind::Capacity => "fold fewer, larger ranges",
        }
    }
}

impl fmt::Display for RangeErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.code())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RangeError {
    pub start: String,
    pub end: String,
    pub kind: RangeErrorKind,
    /// Concrete numbers, e.g. `range holds 1200 chars, minimum is 5000`.
    pub detail: String,
}

impl fmt::Display for RangeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}-{} [{}]: {}",
            self.start, self.end, self.kind, self.detail
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_kind_has_a_distinct_code() {
        let all = [
            RangeErrorKind::InvalidRef,
            RangeErrorKind::UnknownRef,
            RangeErrorKind::Consumed,
            RangeErrorKind::TooSmall,
            RangeErrorKind::SummaryTooShort,
            RangeErrorKind::SummaryTooLong,
            RangeErrorKind::WouldSplitTurn,
            RangeErrorKind::ProtectedZone,
            RangeErrorKind::Capacity,
        ];
        let mut codes: Vec<&str> = all.iter().map(|k| k.code()).collect();
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(codes.len(), all.len(), "codes must be unique");
    }
}

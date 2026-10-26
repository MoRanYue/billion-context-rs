//! Protection and recommendation.
//!
//! Ported from `kernel/src/recommend.ts` + `kernel/src/turn-integrity.ts`.
//!
//! Two questions get answered every turn, and getting either wrong is a
//! correctness bug rather than a cosmetic one:
//!
//! * **Protection** — which messages must not be folded? Folding the tail
//!   destroys the conversation the model is working on; folding half a
//!   tool exchange strands an orphan `function_call` that strict providers
//!   reject outright.
//! * **Foldability** — which spans may actually be folded? A range advertised
//!   here that `apply_compression` would refuse is worse than no advice: the
//!   caller submits it, gets an error, and retries forever.
//!
//! Everything here is pure: no state mutation, no I/O.

use std::collections::{HashMap, HashSet};

use crate::config::Config;
use crate::ids::{MessageId, MsgRef};
use crate::message::{Content, Message, Role};
use crate::pipeline::nodes::SUMMARY_HEADER;
use crate::state::State;

// ─── Message classification ───────────────────────────────────────────────────

fn is_reasoning(m: &Message) -> bool {
    matches!(m.content, Content::Reasoning(_))
}

fn is_tool_call(m: &Message) -> bool {
    m.role == Role::Assistant && matches!(m.content, Content::ToolCall { .. })
}

fn is_tool_result(m: &Message) -> bool {
    matches!(m.content, Content::ToolResult { .. })
}

fn tool_call_id(m: &Message) -> Option<&str> {
    match &m.content {
        Content::ToolCall { call_id, .. } | Content::ToolResult { call_id, .. } => Some(call_id),
        _ => None,
    }
}

/// Assistant output that can follow a reasoning run: prose or a call.
fn is_assistant_act(m: &Message) -> bool {
    m.role == Role::Assistant && matches!(m.content, Content::Text(_) | Content::ToolCall { .. })
}

fn is_user(m: &Message) -> bool {
    m.role == Role::User
}

/// A message holding a block's summary in place of its originals.
pub(crate) fn is_summary_carrier(m: &Message) -> bool {
    matches!(&m.content, Content::Text(t) if t.starts_with(SUMMARY_HEADER))
}

// ─── Turn groups ──────────────────────────────────────────────────────────────

/// Atomic turn groups: a reasoning run, the assistant burst that follows it,
/// and every tool-result paired to that burst's calls.
///
/// Providers that echo thinking back reject a rebuilt request whose call
/// survived without its reasoning; every OpenAI-wire provider rejects a result
/// whose call is gone. So a turn folds whole or not at all.
///
/// Grouping is adjacency-based and the groups are disjoint. Messages belonging
/// to no turn (user, system, orphan reasoning, summaries) get no group — they
/// carry no pairing constraint.
pub(crate) fn turn_groups(messages: &[Message]) -> Vec<Vec<MessageId>> {
    let mut result_id_by_call: HashMap<&str, &MessageId> = HashMap::new();
    for m in messages {
        if is_tool_result(m) {
            if let Some(call_id) = tool_call_id(m) {
                result_id_by_call.entry(call_id).or_insert(&m.id);
            }
        }
    }

    let mut grouped: HashSet<MessageId> = HashSet::new();
    let mut groups: Vec<Vec<MessageId>> = Vec::new();

    for i in 0..messages.len() {
        let msg = &messages[i];
        if grouped.contains(&msg.id) || !(is_reasoning(msg) || is_assistant_act(msg)) {
            continue;
        }

        // Walk back to the head of the reasoning run that precedes this burst,
        // or to the head of the burst itself when there is no reasoning.
        let mut reasoning_start = i;
        if is_reasoning(msg) {
            while reasoning_start > 0 && is_reasoning(&messages[reasoning_start - 1]) {
                reasoning_start -= 1;
            }
        } else {
            let mut s = i;
            while s > 0 && is_assistant_act(&messages[s - 1]) {
                s -= 1;
            }
            reasoning_start = s;
            while reasoning_start > 0 && is_reasoning(&messages[reasoning_start - 1]) {
                reasoning_start -= 1;
            }
        }

        let mut burst_start = reasoning_start;
        while burst_start < messages.len() && is_reasoning(&messages[burst_start]) {
            burst_start += 1;
        }
        // Orphan reasoning with no companion burst: no pairing constraint.
        if burst_start >= messages.len() || !is_assistant_act(&messages[burst_start]) {
            continue;
        }

        let mut burst_end = burst_start;
        while burst_end + 1 < messages.len() && is_assistant_act(&messages[burst_end + 1]) {
            burst_end += 1;
        }

        let mut members: HashSet<MessageId> = HashSet::new();
        for m in &messages[reasoning_start..=burst_end] {
            members.insert(m.id.clone());
            if is_tool_call(m) {
                if let Some(call_id) = tool_call_id(m) {
                    if let Some(result_id) = result_id_by_call.get(call_id) {
                        members.insert((*result_id).clone());
                    }
                }
            }
        }
        for id in &members {
            grouped.insert(id.clone());
        }
        groups.push(members.into_iter().collect());
    }
    groups
}

// ─── Fold integrity ───────────────────────────────────────────────────────────

/// The ids a fold must leave visible, given the ids it would remove.
///
/// * **INV1** — a surviving assistant tool-call must keep its reasoning run, so
///   a turn whose reasoning folds while one of its calls stays visible is
///   withdrawn whole.
/// * **INV2** — a call and its result are one exchange: a pair the fold would
///   split is withdrawn together.
///
/// The invariants feed each other (withdrawing a call for INV2 can strand a
/// reasoning run for INV1, and vice versa), so this repeats until nothing
/// changes. The reverse stays allowed: reasoning and text may survive while the
/// call and result fold — that leaves a valid stream.
pub(crate) fn integrity_withdrawals(
    messages: &[Message],
    folded: &HashSet<MessageId>,
) -> HashSet<MessageId> {
    let mut remaining = folded.clone();
    let mut withdrawn: HashSet<MessageId> = HashSet::new();
    let mut handled_turns: HashSet<usize> = HashSet::new();
    let mut handled_pairs: HashSet<String> = HashSet::new();

    let mut reasoning_ids: HashSet<&MessageId> = HashSet::new();
    let mut call_ids: HashSet<&MessageId> = HashSet::new();
    let mut call_id_by_message: HashMap<&MessageId, String> = HashMap::new();
    let mut result_id_by_call: HashMap<String, &MessageId> = HashMap::new();
    for m in messages {
        if is_reasoning(m) {
            reasoning_ids.insert(&m.id);
        }
        if is_tool_call(m) {
            call_ids.insert(&m.id);
            if let Some(call_id) = tool_call_id(m) {
                call_id_by_message.insert(&m.id, call_id.to_string());
            }
        }
        if is_tool_result(m) {
            if let Some(call_id) = tool_call_id(m) {
                result_id_by_call
                    .entry(call_id.to_string())
                    .or_insert(&m.id);
            }
        }
    }

    let groups = turn_groups(messages);
    loop {
        let mut changed = false;

        for (g, group) in groups.iter().enumerate() {
            if handled_turns.contains(&g) {
                continue;
            }
            let fold_has_reasoning = group
                .iter()
                .any(|id| remaining.contains(id) && reasoning_ids.contains(id));
            if !fold_has_reasoning {
                continue;
            }
            let kept_has_call = group
                .iter()
                .any(|id| !remaining.contains(id) && call_ids.contains(id));
            if !kept_has_call {
                continue;
            }
            handled_turns.insert(g);
            for id in group {
                remaining.remove(id);
                withdrawn.insert(id.clone());
            }
            changed = true;
        }

        for m in messages {
            if !call_ids.contains(&m.id) {
                continue;
            }
            let Some(call_id) = call_id_by_message.get(&m.id) else {
                continue;
            };
            if handled_pairs.contains(call_id) {
                continue;
            }
            let Some(result_id) = result_id_by_call.get(call_id) else {
                continue;
            };
            if remaining.contains(&m.id) == remaining.contains(*result_id) {
                continue;
            }
            handled_pairs.insert(call_id.clone());
            remaining.remove(&m.id);
            remaining.remove(*result_id);
            withdrawn.insert(m.id.clone());
            withdrawn.insert((*result_id).clone());
            changed = true;
        }

        if !changed {
            break;
        }
    }

    withdrawn
}

// ─── Protection ───────────────────────────────────────────────────────────────

/// The refs forming the soft-protected zone at the tail of the conversation.
///
/// Three rules, plus a closure over tool pairs:
///
/// 1. the last `preserve_recent_messages` visible messages;
/// 2. the last `preserve_recent_tokens` tokens, expanding backward;
/// 3. the last visible user message — the intent the model is answering.
///
/// Rule 2 is what makes a large conversation foldable at all: with ~900-token
/// messages and a 5 000-token window the zone reaches back six messages, not
/// three. Skipping it recommends folding content the model is still working on.
///
/// Then the tool-pair closure: if either half of a call/result pair is
/// protected, both are. Otherwise the zone can hold the call while the fold
/// takes its result, stranding an orphan call.
pub(crate) fn protected_refs(
    messages: &[Message],
    state: &State,
    cfg: &Config,
) -> HashSet<MsgRef> {
    let preserve_n = cfg.preserve_recent_messages;
    let preserve_tokens = cfg.preserve_recent_tokens;

    let mut result: HashSet<MsgRef> = HashSet::new();
    let mut visible: Vec<(MsgRef, u32)> = Vec::new();

    for msg in messages {
        if is_summary_carrier(msg) || state.is_covered(&msg.id) {
            continue;
        }
        let Some(r) = state.refs.ref_of(&msg.id) else {
            continue;
        };
        if state.refs.is_blocked(&msg.id) {
            continue;
        }
        let tokens = state.token_snapshot.get(&r).copied().unwrap_or(0);
        visible.push((r, tokens));
    }

    // Rule 1: the last N messages.
    if preserve_n > 0 {
        for (r, _) in visible.iter().skip(visible.len().saturating_sub(preserve_n)) {
            result.insert(*r);
        }
    }

    // Rule 2: the last N tokens, expanding backward from the tail.
    if preserve_tokens > 0 {
        let mut accumulated = 0u32;
        for (r, tokens) in visible.iter().rev() {
            if accumulated >= preserve_tokens {
                break;
            }
            result.insert(*r);
            accumulated = accumulated.saturating_add(*tokens);
        }
    }

    // Rule 3: the last visible user message. Coupled to Rule 1 so that
    // `preserve_recent_messages = 0` opts out of both. Scanned over the raw
    // array so user intent is protected even when the message that would have
    // matched it was skipped above.
    if preserve_n > 0 {
        for msg in messages.iter().rev() {
            if !is_user(msg) || is_summary_carrier(msg) || state.is_covered(&msg.id) {
                continue;
            }
            if let Some(r) = state.refs.ref_of(&msg.id) {
                if !state.refs.is_blocked(&msg.id) {
                    result.insert(r);
                }
            }
            break;
        }
    }

    // Close the tool-pair gap. Monotonic: only ever adds refs.
    {
        let mut msg_by_ref: HashMap<MsgRef, &Message> = HashMap::new();
        for m in messages {
            if let Some(r) = state.refs.ref_of(&m.id) {
                if !state.refs.is_blocked(&m.id) {
                    msg_by_ref.insert(r, m);
                }
            }
        }
        let mut call_ref_by_call_id: HashMap<&str, MsgRef> = HashMap::new();
        let mut result_ref_by_call_id: HashMap<&str, MsgRef> = HashMap::new();
        for m in messages {
            let Some(call_id) = tool_call_id(m) else {
                continue;
            };
            if is_summary_carrier(m) || state.is_covered(&m.id) {
                continue;
            }
            let Some(r) = state.refs.ref_of(&m.id) else {
                continue;
            };
            if state.refs.is_blocked(&m.id) {
                continue;
            }
            if is_tool_call(m) {
                call_ref_by_call_id.insert(call_id, r);
            } else if is_tool_result(m) {
                result_ref_by_call_id.insert(call_id, r);
            }
        }
        for r in result.iter().copied().collect::<Vec<_>>() {
            let Some(m) = msg_by_ref.get(&r) else {
                continue;
            };
            let Some(call_id) = tool_call_id(m) else {
                continue;
            };
            if is_tool_call(m) {
                if let Some(other) = result_ref_by_call_id.get(call_id) {
                    result.insert(*other);
                }
            } else if is_tool_result(m) {
                if let Some(other) = call_ref_by_call_id.get(call_id) {
                    result.insert(*other);
                }
            }
        }
    }

    result
}

// ─── Foldability screening ────────────────────────────────────────────────────

/// True when the span still owns at least one message a fold may take.
///
/// A unit that only spans already-covered or blocked refs would be refused by
/// `apply_compression` ("holds no foldable message"), so advertising it starts
/// an advertise/reject loop that never terminates. Screen with the same test
/// the apply side uses.
pub(crate) fn span_owns_foldable(
    state: &State,
    start: MsgRef,
    end: MsgRef,
) -> bool {
    for index in start.index()..=end.index() {
        let Some(r) = MsgRef::new(index) else {
            continue;
        };
        let Some(id) = state.refs.raw_of(r) else {
            continue;
        };
        if state.refs.is_blocked(id) || state.is_covered(id) {
            continue;
        }
        return true;
    }
    false
}


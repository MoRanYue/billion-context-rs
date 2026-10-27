//! Repairing the view so every request it produces is one a provider accepts.
//!
//! Folding removes messages, and removing a message can remove *half* of a pair
//! the wire format requires to be whole: a `tool_result` whose call is gone, a
//! `tool_call` nothing answers, a `reasoning` block whose response was folded
//! away. All three are hard errors upstream, not degradations — OpenAI rejects an
//! unknown `tool_call_id`, Anthropic rejects an orphan `tool_result`, and
//! thinking-mode providers reject `reasoning_content` that lost its response.
//!
//! Boundary widening (see [`crate::session`]) keeps pairs inside one range when
//! the caller asked for a message span. These functions are the backstop for
//! everything widening cannot see: block-boundary ranges, a host that hands over
//! an already-inconsistent transcript, and folds performed on a previous turn.
//! Upstream runs all three after every rebuild, and so does this — a transcript
//! that is only *sometimes* legal is a bug report waiting for a Tuesday.
//!
//! The traversal directions are not stylistic. A call opens its id and a result
//! answers it, so results are checked forward (a call must come first) and calls
//! backward (a result must come later). Checking either one with a plain set
//! membership test ships the wrong half whenever an id appears twice.

use crate::message::{Content, Message, Role};

/// Drop `tool_result` messages whose call is not above them.
pub fn strip_orphaned_tool_results(messages: Vec<Message>) -> Vec<Message> {
    let mut open: std::collections::HashSet<std::sync::Arc<str>> =
        std::collections::HashSet::new();
    let mut keep = Vec::with_capacity(messages.len());
    for message in messages {
        match &message.content {
            Content::ToolCall { call_id, .. } => {
                open.insert(call_id.clone());
                keep.push(message);
            }
            Content::ToolResult { call_id, .. } => {
                if open.contains(call_id) {
                    keep.push(message);
                }
            }
            _ => keep.push(message),
        }
    }
    keep
}

/// Drop `tool_call` messages that no `tool_result` answers below them.
///
/// A compress call is exempt: its result is managed by `hide-compress-calls`, and
/// dropping the call here would make the kernel's own bookkeeping invisible.
pub fn strip_orphaned_tool_calls(
    messages: Vec<Message>,
    is_compress_tool: &dyn Fn(&str) -> bool,
) -> Vec<Message> {
    let mut answered: std::collections::HashSet<std::sync::Arc<str>> =
        std::collections::HashSet::new();
    let mut doomed = vec![false; messages.len()];
    for (i, message) in messages.iter().enumerate().rev() {
        match &message.content {
            Content::ToolResult { call_id, .. } => {
                answered.insert(call_id.clone());
            }
            Content::ToolCall { name, call_id, .. } => {
                if is_compress_tool(name) {
                    continue;
                }
                if !answered.contains(call_id) {
                    doomed[i] = true;
                }
            }
            _ => {}
        }
    }
    messages
        .into_iter()
        .enumerate()
        .filter_map(|(i, m)| (!doomed[i]).then_some(m))
        .collect()
}

/// Drop a `reasoning` run that is not immediately followed by the assistant
/// text or tool call it belongs to.
///
/// Runs are consecutive `reasoning` messages; the companion is the first message
/// after the run. Runs after tool stripping, because dropping an orphaned call can
/// leave its preceding reasoning dangling in turn.
pub fn strip_orphaned_reasoning(messages: Vec<Message>) -> Vec<Message> {
    let mut doomed = vec![false; messages.len()];
    for i in 0..messages.len() {
        if doomed[i] || !matches!(messages[i].content, Content::Reasoning(_)) {
            continue;
        }
        let mut j = i;
        while j + 1 < messages.len() && matches!(messages[j + 1].content, Content::Reasoning(_)) {
            j += 1;
        }
        let answered = matches!(
            messages.get(j + 1),
            Some(Message {
                role: Role::Assistant,
                content: Content::Text(_) | Content::ToolCall { .. },
                ..
            })
        );
        if !answered {
            for slot in doomed.iter_mut().take(j + 1).skip(i) {
                *slot = true;
            }
        }
    }
    messages
        .into_iter()
        .enumerate()
        .filter_map(|(i, m)| (!doomed[i]).then_some(m))
        .collect()
}

/// All three passes, in the order upstream applies them.
pub fn repair(messages: Vec<Message>, is_compress_tool: &dyn Fn(&str) -> bool) -> Vec<Message> {
    strip_orphaned_reasoning(strip_orphaned_tool_calls(
        strip_orphaned_tool_results(messages),
        is_compress_tool,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Message;

    const NEVER: &dyn Fn(&str) -> bool = &|_| false;

    fn call(id: &str, call_id: &str) -> Message {
        Message::tool_call(id, "grep", call_id, "{}")
    }

    fn result(id: &str, call_id: &str) -> Message {
        Message::tool_result(id, "grep", call_id, "ok")
    }

    fn ids(messages: &[Message]) -> Vec<&str> {
        messages.iter().map(|m| m.id.as_str()).collect()
    }

    #[test]
    fn a_result_without_its_call_is_dropped() {
        let out = strip_orphaned_tool_results(vec![
            Message::text("u", Role::User, "hi"),
            result("r", "c1"),
            call("c", "c1"),
        ]);
        assert_eq!(ids(&out), vec!["u", "c"]);
    }

    #[test]
    fn a_later_call_does_not_legitimise_an_earlier_result() {
        // The bug the forward walk exists for: a set-membership check would keep
        // this result because *some* call with that id exists, just not before it.
        let out = strip_orphaned_tool_results(vec![
            result("r", "c1"),
            call("c", "c1"),
            result("r2", "c1"),
        ]);
        assert_eq!(ids(&out), vec!["c", "r2"]);
    }

    #[test]
    fn a_call_nothing_answers_is_dropped() {
        let out = strip_orphaned_tool_calls(
            vec![
                Message::text("u", Role::User, "hi"),
                call("c", "c1"),
                Message::text("a", Role::Assistant, "done"),
            ],
            NEVER,
        );
        assert_eq!(ids(&out), vec!["u", "a"]);
    }

    #[test]
    fn a_compress_call_survives_even_unanswered() {
        let is_compress = |name: &str| name == "compress" || name == "acp_compress";
        for name in ["compress", "acp_compress"] {
            let out = strip_orphaned_tool_calls(
                vec![Message::tool_call("c", name, "c1", "{}")],
                &is_compress,
            );
            assert_eq!(out.len(), 1, "{name} must survive orphan repair");
        }
    }

    #[test]
    fn a_reasoning_run_needs_an_assistant_companion() {
        let out = strip_orphaned_reasoning(vec![
            Message::reasoning("r1", Role::Assistant, "think"),
            Message::text("a", Role::Assistant, "answer"),
            Message::reasoning("r2", Role::Assistant, "dangling"),
            Message::text("u", Role::User, "next"),
        ]);
        assert_eq!(ids(&out), vec!["r1", "a", "u"]);
    }

    #[test]
    fn a_tool_call_counts_as_a_companion() {
        let out = strip_orphaned_reasoning(vec![
            Message::reasoning("r", Role::Assistant, "think"),
            call("c", "c1"),
            result("t", "c1"),
        ]);
        assert_eq!(ids(&out), vec!["r", "c", "t"]);
    }

    #[test]
    fn a_whole_dangling_run_goes_together() {
        let out = strip_orphaned_reasoning(vec![
            Message::reasoning("r1", Role::Assistant, "a"),
            Message::reasoning("r2", Role::Assistant, "b"),
            Message::text("u", Role::User, "next"),
        ]);
        assert_eq!(ids(&out), vec!["u"]);
    }

    #[test]
    fn repair_is_idempotent() {
        let input = || {
            vec![
                Message::text("u", Role::User, "hi"),
                call("c", "c1"),
                Message::reasoning("r", Role::Assistant, "think"),
                Message::text("a", Role::Assistant, "answer"),
            ]
        };
        let once = repair(input(), NEVER);
        let twice = repair(once.clone(), NEVER);
        assert_eq!(ids(&once), ids(&twice));
    }
}

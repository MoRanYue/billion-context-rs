//! Deterministic, host-level reduction of oversized tool results.
//!
//! This module never allocates or changes a kernel ref. It receives the already
//! converted messages, keeps every `Message.id`, and rewrites only the visible
//! `Content::ToolResult` payload. The original bytes live in `ArtifactStore`.

use std::collections::HashSet;

use billion_context::{Content, Message};

use crate::artifact::{ArtifactStore, PREVIEW_POLICY_VERSION};

pub const DEFAULT_PREVIEW_HEAD_CHARS: usize = 4_000;
pub const DEFAULT_PREVIEW_TAIL_CHARS: usize = 4_000;

#[derive(Clone, Debug)]
pub struct PrunePolicy {
    pub enabled: bool,
    pub high_ratio: f32,
    pub low_ratio: f32,
    pub min_tokens: u32,
    pub head_chars: usize,
    pub tail_chars: usize,
    pub cooldown_requests: u64,
    pub growth_ratio: f32,
}

impl Default for PrunePolicy {
    fn default() -> Self {
        Self {
            enabled: false,
            high_ratio: 0.60,
            low_ratio: 0.45,
            min_tokens: 8_000,
            head_chars: DEFAULT_PREVIEW_HEAD_CHARS,
            tail_chars: DEFAULT_PREVIEW_TAIL_CHARS,
            cooldown_requests: 3,
            growth_ratio: 0.05,
        }
    }
}

impl PrunePolicy {
    /// Production defaults. Values remain deliberately conservative; replay
    /// will tune them, not hard-code a provider-specific constant.
    pub fn production() -> Self {
        Self {
            enabled: true,
            ..Self::default()
        }
    }

    pub fn from_env() -> Self {
        let mut policy = Self::production();
        if let Ok(value) = std::env::var("BILI_PRUNE_ENABLED") {
            policy.enabled = !matches!(value.trim(), "0" | "false" | "off");
        }
        if let Some(value) = std::env::var("BILI_PRUNE_HEAD_CHARS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
        {
            policy.head_chars = value.clamp(256, 16_000);
        }
        if let Some(value) = std::env::var("BILI_PRUNE_TAIL_CHARS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
        {
            policy.tail_chars = value.clamp(256, 16_000);
        }
        if let Some(value) = std::env::var("BILI_PRUNE_MIN_TOKENS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
        {
            policy.min_tokens = value.clamp(1_000, 64_000);
        }
        policy
    }
}

#[derive(Clone, Debug, Default)]
pub struct PruneReport {
    pub triggered: bool,
    pub candidates: u64,
    pub pruned_items: u64,
    pub saved_tokens_est: u64,
    pub saved_bytes: u64,
    pub artifact_writes: u64,
    pub artifact_bytes: u64,
    pub pruned_ids: Vec<String>,
    pub first_changed_message_id: Option<String>,
    pub reason: String,
}

pub fn estimate_tokens(text: &str) -> u32 {
    let mut cjk = 0usize;
    let mut other = 0usize;
    for ch in text.chars() {
        let code = ch as u32;
        if (0x4E00..=0x9FFF).contains(&code)
            || (0x3040..=0x30FF).contains(&code)
            || (0xAC00..=0xD7AF).contains(&code)
        {
            cjk += 1;
        } else {
            other += 1;
        }
    }
    (cjk + other.div_ceil(4)) as u32
}

fn is_oversized_tool_result(message: &Message, policy: &PrunePolicy) -> bool {
    if message.unrecoverable_payload {
        return false;
    }
    let Content::ToolResult { name, content, .. } = &message.content else {
        return false;
    };
    if is_protected_tool(name) || content.starts_with("[artifact ") {
        return false;
    }
    estimate_tokens(content) >= policy.min_tokens
}

fn is_protected_tool(name: &str) -> bool {
    matches!(
        name,
        "compress"
            | "acp_compress"
            | "acp_rule"
            | "retrieve"
            | "acp_retrieve"
            | "decompress"
            | "acp_decompress"
            | "artifact_info"
            | "artifact_read"
            | "artifact_search"
            | "artifact.info"
            | "artifact.read"
            | "artifact.search"
    )
}

fn head_tail(content: &str, head: usize, tail: usize) -> (String, String) {
    let chars: Vec<char> = content.chars().collect();
    if chars.len() <= head.saturating_add(tail) {
        return (content.to_string(), String::new());
    }
    let head_text: String = chars.iter().take(head).collect();
    let tail_text: String = chars
        .iter()
        .skip(chars.len().saturating_sub(tail))
        .collect();
    (head_text, tail_text)
}

pub fn render_preview(
    artifact_id: &str,
    source_message_id: &str,
    tool_name: &str,
    byte_len: u64,
    content: &str,
    policy: &PrunePolicy,
) -> String {
    let (head, tail) = head_tail(content, policy.head_chars, policy.tail_chars);
    let tool = if tool_name.is_empty() {
        "tool".to_string()
    } else {
        tool_name.to_string()
    };
    let mut out = String::with_capacity(head.len() + tail.len() + 400);
    out.push_str(&format!(
        "[artifact {artifact_id} · source {source_message_id} · {tool} · {byte_len} bytes · stable preview v{PREVIEW_POLICY_VERSION}]\n"
    ));
    out.push_str(&head);
    out.push_str(&format!(
        "\n\n[artifact {artifact_id}: middle omitted; original is {byte_len} bytes. \
         Use artifact_read(artifact_id=\"{artifact_id}\", offset=..., limit=...) or \
         artifact_search(artifact_id=\"{artifact_id}\", query=...) for the omitted range.]\n\n"
    ));
    out.push_str(&tail);
    out
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TriggerDecision {
    pub triggered: bool,
    pub reason: &'static str,
    pub high_watermark: u32,
    pub growth_tokens: u32,
    pub cooldown_hit: bool,
    /// True only when growth alone lifted the cooldown gate. This event is
    /// deliberately observable rather than folded into `cooldown_hit`.
    pub growth_bypass: bool,
    /// True when the cooldown gate actually stopped a prune.
    pub cooldown_blocked: bool,
}

/// Decide whether one batch should run. `tokens` must be the effective
/// projection budget (prune/stub + fixed request overhead), never the raw
/// request estimate. The previous prune fingerprint is also effective.
pub fn should_trigger_decision(
    tokens: u32,
    context_limit: u32,
    requests: u64,
    last_prune_requests: Option<u64>,
    last_prune_tokens: Option<u32>,
    policy: &PrunePolicy,
) -> TriggerDecision {
    let high = (context_limit as f32 * policy.high_ratio).max(1.0) as u32;
    let growth = (context_limit as f32 * policy.growth_ratio).max(1.0) as u32;
    if !policy.enabled {
        return TriggerDecision {
            triggered: false,
            reason: "disabled",
            high_watermark: high,
            growth_tokens: growth,
            cooldown_hit: false,
            growth_bypass: false,
            cooldown_blocked: false,
        };
    }
    if context_limit == 0 {
        return TriggerDecision {
            triggered: false,
            reason: "zero_context_limit",
            high_watermark: high,
            growth_tokens: growth,
            cooldown_hit: false,
            growth_bypass: false,
            cooldown_blocked: false,
        };
    }
    if tokens < high {
        return TriggerDecision {
            triggered: false,
            reason: "below_high_watermark",
            high_watermark: high,
            growth_tokens: growth,
            cooldown_hit: false,
            growth_bypass: false,
            cooldown_blocked: false,
        };
    }
    let mut growth_bypass = false;
    if let Some(last) = last_prune_requests {
        let cooldown_ok = requests.saturating_sub(last) >= policy.cooldown_requests;
        let grown =
            last_prune_tokens.is_some_and(|previous| tokens.saturating_sub(previous) >= growth);
        if !cooldown_ok && !grown {
            return TriggerDecision {
                triggered: false,
                reason: "cooldown",
                high_watermark: high,
                growth_tokens: growth,
                cooldown_hit: true,
                growth_bypass: false,
                cooldown_blocked: true,
            };
        }
        growth_bypass = !cooldown_ok && grown;
    }
    TriggerDecision {
        triggered: true,
        // Keep `reason=high_watermark` for wire compatibility; the separate
        // `growth_bypass` event states why cooldown did not block this run.
        reason: "high_watermark",
        high_watermark: high,
        growth_tokens: growth,
        cooldown_hit: false,
        growth_bypass,
        cooldown_blocked: false,
    }
}

#[allow(
    dead_code,
    reason = "policy helper retained for hosts that gate pruning themselves"
)]
pub fn should_trigger(
    tokens: u32,
    context_limit: u32,
    requests: u64,
    last_prune_requests: Option<u64>,
    last_prune_tokens: Option<u32>,
    policy: &PrunePolicy,
) -> bool {
    should_trigger_decision(
        tokens,
        context_limit,
        requests,
        last_prune_requests,
        last_prune_tokens,
        policy,
    )
    .triggered
}

/// Shared effective projection estimate over kernel messages, excluding fixed
/// system/tool overhead. This is the message view prune actually rewrites.
/// Re-apply an already stored artifact stub to a resent original. This is
/// request assembly, not a new prune: it writes nothing and allocates no refs,
/// but it is what makes a previous Artifact/Stub decision survive a host that
/// resends the raw transcript every turn.
pub fn apply_existing_stubs(
    messages: &mut [Message],
    store: &ArtifactStore,
    policy: &PrunePolicy,
) -> u64 {
    let mut reapplied = 0u64;
    for message in messages.iter_mut() {
        let Content::ToolResult { name, content, .. } = &message.content else {
            continue;
        };
        if content.starts_with("[artifact ") {
            continue;
        }
        let Some(meta) = store.find_by_message_id(message.id.as_str()) else {
            continue;
        };
        let original = content.to_string();
        let preview = render_preview(
            &meta.artifact_id,
            &meta.source_message_id,
            name,
            meta.byte_len,
            &original,
            policy,
        );
        if preview.len() >= original.len() {
            continue;
        }
        let Content::ToolResult { content, .. } = &mut message.content else {
            continue;
        };
        *content = preview.into();
        reapplied += 1;
    }
    reapplied
}

pub fn estimate_messages_tokens(messages: &[Message]) -> u32 {
    messages
        .iter()
        .map(|message| match &message.content {
            Content::Text(text) | Content::Reasoning(text) => estimate_tokens(text),
            Content::ToolCall { arguments, .. } => estimate_tokens(arguments),
            Content::ToolResult { content, .. } => estimate_tokens(content),
        })
        .fold(0u32, u32::saturating_add)
}

/// Rewrite eligible tool results in place. Returns only successful replacements:
/// an Artifact write failure leaves the original message untouched and never
/// produces a pointer to a missing blob.
#[allow(clippy::too_many_arguments)]
pub fn prune_tool_results(
    messages: &mut [Message],
    tokens: u32,
    context_limit: u32,
    protect_ids: &HashSet<String>,
    preserve_recent_messages: usize,
    preserve_recent_tokens: u32,
    artifacts: Option<&ArtifactStore>,
    policy: &PrunePolicy,
    session_id: &str,
) -> PruneReport {
    let mut report = PruneReport {
        triggered: true,
        reason: "high_watermark".into(),
        ..PruneReport::default()
    };
    let Some(store) = artifacts else {
        report.reason = "no_artifact_store".into();
        return report;
    };
    let low = (context_limit as f32 * policy.low_ratio).max(1.0) as u32;
    let mut protected: HashSet<String> = protect_ids.iter().map(|id| id.to_string()).collect();
    let recent_start = messages.len().saturating_sub(preserve_recent_messages);
    for message in &messages[recent_start..] {
        // A single oversized tool result is the case this phase exists for:
        // keeping it in the tail would make the high-water mark unactionable.
        // Non-tool and small recent messages keep their tail protection.
        if !is_oversized_tool_result(message, policy) {
            protected.insert(message.id.base().to_string());
        }
    }
    let mut tail_tokens = 0u32;
    for message in messages.iter().rev() {
        if tail_tokens >= preserve_recent_tokens {
            break;
        }
        if !is_oversized_tool_result(message, policy) {
            protected.insert(message.id.base().to_string());
        }
        tail_tokens = tail_tokens.saturating_add(match &message.content {
            Content::Text(t) | Content::Reasoning(t) => estimate_tokens(t),
            Content::ToolCall { arguments, .. } => estimate_tokens(arguments),
            Content::ToolResult { content, .. } => estimate_tokens(content),
        });
    }

    let mut candidates: Vec<(usize, u32)> = messages
        .iter()
        .enumerate()
        .filter_map(|(index, message)| {
            if protected.contains(message.id.base()) || message.unrecoverable_payload {
                return None;
            }
            let Content::ToolResult { name, content, .. } = &message.content else {
                return None;
            };
            if is_protected_tool(name) {
                return None;
            }
            if content.starts_with("[artifact ") {
                return None;
            }
            let tokens = estimate_tokens(content);
            (tokens >= policy.min_tokens).then_some((index, tokens))
        })
        .collect();
    report.candidates = candidates.len() as u64;
    if candidates.is_empty() {
        report.reason = "no_candidates".into();
        return report;
    }
    candidates.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let mut saved = 0u32;
    let mut first_index: Option<usize> = None;
    for (index, original_tokens) in candidates {
        if tokens.saturating_sub(saved) <= low {
            break;
        }
        let (source_id, original, tool_name, call_id) = {
            let message = &messages[index];
            let Content::ToolResult {
                name,
                call_id,
                content,
            } = &message.content
            else {
                continue;
            };
            (
                message.id.as_str().to_string(),
                content.to_string(),
                // Artifact metadata is display-only: an absent name and an
                // empty name both render as an unnamed tool here.
                name.to_string(),
                (!call_id.is_empty()).then(|| call_id.to_string()),
            )
        };
        let meta = match store.put_with_session(
            &original,
            &source_id,
            None,
            Some(&tool_name),
            call_id.as_deref(),
            session_id,
        ) {
            Ok(meta) => meta,
            Err(_) => continue,
        };
        let preview = render_preview(
            &meta.artifact_id,
            &source_id,
            &tool_name,
            meta.byte_len,
            &original,
            policy,
        );
        let preview_tokens = estimate_tokens(&preview);
        let preview_len = preview.len();
        if preview_tokens >= original_tokens || preview_len >= original.len() {
            continue;
        }
        let delta = original_tokens.saturating_sub(preview_tokens);
        let Some(message) = messages.get_mut(index) else {
            continue;
        };
        let Content::ToolResult { content, .. } = &mut message.content else {
            continue;
        };
        *content = preview.into();
        saved = saved.saturating_add(delta);
        report.pruned_items += 1;
        report.saved_tokens_est += delta as u64;
        report.saved_bytes += original.len().saturating_sub(preview_len) as u64;
        report.artifact_writes += 1;
        report.artifact_bytes += meta.byte_len;
        report.pruned_ids.push(source_id);
        first_index = Some(first_index.map_or(index, |current| current.min(index)));
    }
    report.saved_tokens_est = (report.saved_tokens_est).min(tokens as u64);
    report.first_changed_message_id =
        first_index.map(|index| messages[index].id.as_str().to_string());
    if report.pruned_items == 0 {
        report.reason = "artifact_write_failed".into();
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact::FaultStage;
    use billion_context::Message;

    #[test]
    fn preview_is_stable_and_has_no_token_estimate() {
        let content = "head ".to_string() + &"middle ".repeat(1000) + "tail";
        let policy = PrunePolicy {
            min_tokens: 1,
            head_chars: 20,
            tail_chars: 20,
            ..PrunePolicy::default()
        };
        let first = render_preview(
            "a_123",
            "m00042",
            "Bash",
            content.len() as u64,
            &content,
            &policy,
        );
        let second = render_preview(
            "a_123",
            "m00042",
            "Bash",
            content.len() as u64,
            &content,
            &policy,
        );
        assert_eq!(first, second);
        assert!(!first.to_lowercase().contains("token"));
        assert!(first.contains("a_123"));
        assert!(first.contains("source m00042"));
    }

    #[test]
    fn prune_preserves_ids_and_the_tool_call_pair() {
        let dir = std::env::temp_dir().join(format!(
            "bili-prune-pair-{}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            std::process::id()
        ));
        let store = ArtifactStore::new(&dir);
        let mut messages = vec![
            Message::tool_call("m1", "Bash", "c1", "{\"command\":\"ls\"}"),
            Message::tool_result("m2", "Bash", "c1", "y".repeat(40_000)),
        ];
        let policy = PrunePolicy {
            enabled: true,
            min_tokens: 1,
            ..PrunePolicy::default()
        };
        let report = prune_tool_results(
            &mut messages,
            20_000,
            20_000,
            &HashSet::new(),
            0,
            0,
            Some(&store),
            &policy,
            "test",
        );
        assert_eq!(report.pruned_items, 1);
        assert_eq!(messages[0].id.as_str(), "m1");
        assert_eq!(messages[1].id.as_str(), "m2");
        assert!(matches!(
            &messages[0].content,
            Content::ToolCall { call_id, .. } if &**call_id == "c1"
        ));
        assert!(matches!(
            &messages[1].content,
            Content::ToolResult { call_id, content, .. }
                if &**call_id == "c1" && content.starts_with("[artifact ")
        ));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn latest_oversized_result_is_prunable_despite_tail_protection() {
        let dir = std::env::temp_dir().join(format!(
            "bili-prune-latest-{}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            std::process::id()
        ));
        let store = ArtifactStore::new(&dir);
        let mut messages = vec![
            Message::text("m1", billion_context::Role::User, "run it"),
            Message::tool_call("m2", "Bash", "c1", "{}"),
            Message::tool_result("m3", "Bash", "c1", "w".repeat(40_000)),
        ];
        let policy = PrunePolicy {
            enabled: true,
            min_tokens: 1,
            ..PrunePolicy::default()
        };
        let report = prune_tool_results(
            &mut messages,
            20_000,
            20_000,
            &HashSet::new(),
            3,
            5_000,
            Some(&store),
            &policy,
            "test",
        );
        assert_eq!(report.pruned_items, 1);
        assert_eq!(messages[2].id.as_str(), "m3");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn an_existing_stub_is_reapplied_to_a_raw_host_resend() {
        let dir = std::env::temp_dir().join(format!(
            "bili-prune-reapply-{}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            std::process::id()
        ));
        let store = ArtifactStore::new(&dir);
        let raw = "reapply payload\n".repeat(2_000);
        let meta = store
            .put(&raw, "m1", Some("1"), Some("Bash"), Some("c1"))
            .expect("artifact store write");
        let policy = PrunePolicy {
            enabled: true,
            min_tokens: 1,
            ..PrunePolicy::default()
        };

        let mut first = vec![Message::tool_result("m1", "Bash", "c1", &raw)];
        assert_eq!(apply_existing_stubs(&mut first, &store, &policy), 1);
        let preview = first[0]
            .content
            .body()
            .expect("tool result body")
            .to_string();
        assert!(preview.starts_with("[artifact "));
        assert!(preview.contains(&meta.artifact_id));

        // The host resends the original bytes. Request assembly must restore the
        // same stub and must not let the raw body back into the projection.
        let mut resend = vec![Message::tool_result("m1", "Bash", "c1", &raw)];
        assert_eq!(apply_existing_stubs(&mut resend, &store, &policy), 1);
        assert_eq!(
            resend[0].content.body(),
            Some(preview.as_str()),
            "a resend must reproduce the stored stub deterministically"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_stored_stub_survives_artifact_store_restart() {
        let dir = std::env::temp_dir().join(format!(
            "bili-prune-restart-{}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            std::process::id()
        ));
        let store = ArtifactStore::new(&dir);
        let raw = "restart payload\n".repeat(2_000);
        let meta = store
            .put(&raw, "m1", Some("1"), Some("Bash"), Some("c1"))
            .expect("artifact store write");
        let policy = PrunePolicy {
            enabled: true,
            min_tokens: 1,
            ..PrunePolicy::default()
        };
        let mut first = vec![Message::tool_result("m1", "Bash", "c1", &raw)];
        assert_eq!(apply_existing_stubs(&mut first, &store, &policy), 1);
        let preview = first[0].content.body().expect("body").to_string();
        assert!(preview.contains(&meta.artifact_id));
        drop(store);

        // Metadata and body must recover after process restart, without the
        // in-memory index or the original request.
        let reopened = ArtifactStore::new(&dir);
        let mut resend = vec![Message::tool_result("m1", "Bash", "c1", &raw)];
        assert_eq!(apply_existing_stubs(&mut resend, &reopened, &policy), 1);
        assert_eq!(resend[0].content.body(), Some(preview.as_str()));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn enospc_artifact_write_leaves_no_stub_and_keeps_the_tool_pair() {
        let root = std::env::temp_dir().join(format!(
            "bili-prune-enospc-{}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            std::process::id()
        ));
        let store = ArtifactStore::new(&root);
        let secret = "ENOSPC-SECRET ".repeat(5_000);
        let policy = PrunePolicy {
            enabled: true,
            min_tokens: 1,
            ..PrunePolicy::default()
        };
        let mut messages = vec![
            Message::tool_call("m0", "Bash", "c1", "{}"),
            Message::tool_result("m1", "Bash", "c1", &secret),
        ];
        store.inject_fault(FaultStage::BodyWrite);
        let report = prune_tool_results(
            &mut messages,
            20_000,
            20_000,
            &HashSet::new(),
            0,
            0,
            Some(&store),
            &policy,
            "s-enospc",
        );
        assert_eq!(report.pruned_items, 0);
        assert_eq!(report.artifact_writes, 0);
        assert_eq!(report.reason, "artifact_write_failed");
        assert_eq!(messages[0].id.to_string(), "m0");
        assert_eq!(messages[1].id.to_string(), "m1");
        assert_eq!(
            messages[1].content.body(),
            Some(secret.as_str()),
            "failed write must leave the raw tool result intact"
        );
        let Content::ToolResult { call_id, .. } = &messages[1].content else {
            panic!("expected tool result");
        };
        assert_eq!(&**call_id, "c1");
        let audit =
            std::fs::read_to_string(root.join("artifacts").join("audit.jsonl")).expect("audit");
        assert!(audit.contains("injected ENOSPC"), "{audit}");
        assert!(!audit.contains("ENOSPC-SECRET"), "{audit}");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn failed_artifact_write_leaves_the_original_untouched() {
        let root = std::env::temp_dir().join(format!(
            "bili-prune-bad-store-{}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            std::process::id()
        ));
        std::fs::write(&root, b"not a directory").expect("seed file");
        let store = ArtifactStore::new(&root);
        let mut messages = vec![Message::tool_result("m1", "Bash", "c1", "z".repeat(40_000))];
        let policy = PrunePolicy {
            enabled: true,
            min_tokens: 1,
            ..PrunePolicy::default()
        };
        let report = prune_tool_results(
            &mut messages,
            20_000,
            20_000,
            &HashSet::new(),
            0,
            0,
            Some(&store),
            &policy,
            "test",
        );
        assert_eq!(report.pruned_items, 0);
        let Content::ToolResult { content, .. } = &messages[0].content else {
            panic!("expected tool result");
        };
        assert!(content.starts_with('z'));
        let _ = std::fs::remove_file(root);
    }

    #[test]
    fn cooldown_requires_growth_or_enough_requests() {
        let policy = PrunePolicy {
            enabled: true,
            cooldown_requests: 3,
            ..PrunePolicy::default()
        };
        assert!(should_trigger(13_000, 20_000, 1, None, None, &policy));
        assert!(!should_trigger(
            13_000,
            20_000,
            2,
            Some(1),
            Some(13_000),
            &policy
        ));
        assert!(should_trigger(
            14_100,
            20_000,
            2,
            Some(1),
            Some(13_000),
            &policy
        ));
        assert!(should_trigger(
            13_000,
            20_000,
            4,
            Some(1),
            Some(13_000),
            &policy
        ));
    }

    #[test]
    fn trigger_events_distinguish_cooldown_and_growth_bypass() {
        let policy = PrunePolicy {
            enabled: true,
            cooldown_requests: 3,
            ..PrunePolicy::default()
        };
        let normal = should_trigger_decision(13_000, 20_000, 1, None, None, &policy);
        assert!(normal.triggered);
        assert!(!normal.growth_bypass && !normal.cooldown_blocked);

        let blocked = should_trigger_decision(13_000, 20_000, 2, Some(1), Some(13_000), &policy);
        assert!(!blocked.triggered);
        assert!(blocked.cooldown_hit && blocked.cooldown_blocked);
        assert_eq!(blocked.reason, "cooldown");

        // Exactly at growth threshold (20_000 * 0.05 = 1_000) bypasses.
        let at_boundary =
            should_trigger_decision(14_000, 20_000, 2, Some(1), Some(13_000), &policy);
        assert!(at_boundary.triggered && at_boundary.growth_bypass);
        let below_boundary =
            should_trigger_decision(13_999, 20_000, 2, Some(1), Some(13_000), &policy);
        assert!(!below_boundary.triggered && below_boundary.cooldown_blocked);

        let expiry = should_trigger_decision(13_000, 20_000, 4, Some(1), Some(13_000), &policy);
        assert!(expiry.triggered && !expiry.growth_bypass);

        // Consecutive growth bursts keep bypassing until the cooldown expires.
        let mut previous = 13_000;
        for request in 2..4 {
            let next = previous + 1_000;
            let decision =
                should_trigger_decision(next, 20_000, request, Some(1), Some(previous), &policy);
            assert!(decision.triggered && decision.growth_bypass);
            previous = next;
        }
    }

    #[test]
    fn prune_changes_content_but_keeps_message_id() {
        let dir = std::env::temp_dir().join(format!(
            "bili-prune-test-{}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis(),
            std::process::id()
        ));
        let store = ArtifactStore::new(&dir);
        let mut messages = vec![Message::tool_result("m1", "Bash", "c1", "x".repeat(40_000))];
        let policy = PrunePolicy {
            enabled: true,
            min_tokens: 1,
            ..PrunePolicy::default()
        };
        let report = prune_tool_results(
            &mut messages,
            20_000,
            20_000,
            &HashSet::new(),
            0,
            0,
            Some(&store),
            &policy,
            "test",
        );
        assert_eq!(report.pruned_items, 1);
        assert_eq!(messages[0].id.as_str(), "m1");
        assert!(matches!(&messages[0].content, Content::ToolResult { .. }));
        let _ = std::fs::remove_dir_all(dir);
    }
}

//! Host-level replay harness for Phase 0 / Phase 1 acceptance.
//!
//! This is deliberately test-only. It drives the real Anthropic conversion,
//! `Store::prepare`, `process_turn`, prune and `from_core` path, but never calls
//! a provider. Quality is therefore measured as deterministic fact retention /
//! artifact recovery, not as an LLM answer score.

use std::collections::BTreeMap;
use std::path::PathBuf;

use billion_context::{Content, Message};
use serde_json::{Value, json};

use crate::artifact::ArtifactMetadata;
use crate::prune::PrunePolicy;
use crate::session_store::Store;

const CONTEXT_LIMIT: u32 = 80_000;

#[derive(Clone, Copy, Debug)]
enum Group {
    Baseline,
    A,
    B,
}

impl Group {
    fn name(self) -> &'static str {
        match self {
            Group::Baseline => "corrected-baseline",
            Group::A => "A-artifact-stub",
            Group::B => "B-hysteresis-batch",
        }
    }
}

fn body(messages: Vec<Value>) -> Value {
    // Keep a real, deterministic fixed prefix in every replay request. The
    // values are fixture-local, but they exercise the same system/tools split
    // as production instead of hiding all overhead behind a fixed 800.
    json!({
        "system": [{"type": "text", "text": "You are a coding agent. Read files, run tests, and report facts exactly."}],
        "tools": [
            {
                "name": "Read",
                "description": "Read a file from the workspace.",
                "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
            },
            {
                "name": "Bash",
                "description": "Run a shell command in the workspace.",
                "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}
            }
        ],
        "messages": messages
    })
}

fn user(text: impl Into<String>) -> Value {
    json!({"role": "user", "content": [{"type": "text", "text": text.into()}]})
}

fn assistant_text(text: impl Into<String>) -> Value {
    json!({"role": "assistant", "content": [{"type": "text", "text": text.into()}]})
}

fn assistant_tool(id: &str, name: &str, input: Value) -> Value {
    json!({
        "role": "assistant",
        "content": [{"type": "tool_use", "id": id, "name": name, "input": input}]
    })
}

fn tool_result(id: &str, content: impl Into<String>) -> Value {
    json!({
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": id, "content": content.into()}]
    })
}

fn huge(tag: &str) -> String {
    format!(
        "FACT_HEAD=ALPHA-HEAD\n{}\nFACT_MID={tag}\n{}\nFACT_TAIL=GAMMA-TAIL\n",
        "h".repeat(120_000),
        "m".repeat(120_000)
    )
}

fn medium() -> String {
    format!(
        "TEST_FAIL=FAIL-1\nFILE_MODIFIED=src/main.rs\nOPEN_PROBLEM=parser edge case\n{}",
        "x".repeat(60_000)
    )
}

fn turn_bodies(fixture: &str) -> Vec<Value> {
    let mut messages = Vec::new();
    let mut out = Vec::new();

    // Turn 1: a file read with a >50k-token output.
    messages.push(user(
        "TASK: implement the parser. CONSTRAINT_NEVER_TOUCH_API. FILE_MODIFIED=src/main.rs",
    ));
    messages.push(assistant_tool(
        "toolu_read",
        "Read",
        json!({"path": "src/main.rs"}),
    ));
    messages.push(tool_result("toolu_read", huge("BETA-MID-1")));
    messages.push(assistant_text("Read complete."));
    out.push(body(messages.clone()));

    // Turn 2: a failing test with a medium output.
    messages.push(user("Run the parser tests."));
    messages.push(assistant_tool(
        "toolu_test1",
        "Bash",
        json!({"command": "cargo test parser"}),
    ));
    messages.push(tool_result("toolu_test1", medium()));
    messages.push(assistant_text("Test failed; patch applied."));
    out.push(body(messages.clone()));

    // Turn 3: a second large output after the patch.
    messages.push(user("Re-run after the patch."));
    messages.push(assistant_tool(
        "toolu_test2",
        "Bash",
        json!({"command": "cargo test parser"}),
    ));
    messages.push(tool_result("toolu_test2", huge("BETA-MID-2")));
    messages.push(assistant_text("Tests pass."));
    out.push(body(messages.clone()));

    // Turn 4: pivot + incomplete tool call. The incomplete call is intentional:
    // it checks that prune does not need a result to preserve the pair.
    messages.push(user(
        "PIVOT_CONSTRAINT=still do not touch API. Summarize final state.",
    ));
    messages.push(assistant_tool(
        "toolu_pivot",
        "Read",
        json!({"path": "src/lib.rs"}),
    ));
    messages.push(assistant_text(
        "FINAL_TEST_PASS=PASS. OPEN_PROBLEM=NONE. FILE_MODIFIED=src/main.rs",
    ));
    out.push(body(messages.clone()));

    // Fixture name only selects which turns are exposed; the fixture content
    // stays paired and deterministic.
    match fixture {
        "long_code" => out,
        "huge_tool" => vec![out[0].clone(), out[2].clone()],
        "pivot_boundary" => vec![out[0].clone(), out[1].clone(), out[3].clone()],
        _ => panic!("unknown fixture {fixture}"),
    }
}

fn policy_a() -> PrunePolicy {
    // A isolates Artifact/Stable Stub. Automatic Hysteresis is OFF; the
    // test-only harness performs one explicit manual prune on turn 1, then
    // never auto-prunes again. This is not a forced-every-turn policy.
    PrunePolicy {
        enabled: false,
        ..PrunePolicy::default()
    }
}

fn policy_b() -> PrunePolicy {
    PrunePolicy::production()
}

fn projection_text(messages: &[billion_context::Message]) -> String {
    let mut out = String::new();
    for message in messages {
        match &message.content {
            Content::Text(text) | Content::Reasoning(text) => out.push_str(text),
            Content::ToolCall {
                name, arguments, ..
            } => {
                out.push_str(name);
                out.push(' ');
                out.push_str(arguments);
            }
            Content::ToolResult { name, content, .. } => {
                out.push_str(name);
                out.push(' ');
                out.push_str(content);
            }
        }
        out.push('\n');
    }
    out
}

fn first_diff(previous: &str, current: &str) -> Option<usize> {
    if previous.is_empty() {
        return None;
    }
    let mut offset = 0usize;
    for (left, right) in previous.bytes().zip(current.bytes()) {
        if left != right {
            return Some(offset);
        }
        offset += 1;
    }
    (offset < current.len()).then_some(offset)
}

fn find_source_id(messages: &[billion_context::Message], tag: &str) -> Option<String> {
    messages.iter().find_map(|message| {
        let Content::ToolResult { content, .. } = &message.content else {
            return None;
        };
        content
            .contains(tag)
            .then(|| message.id.as_str().to_string())
    })
}

struct ArtifactCheck {
    recovery: Option<String>,
    info_ok: bool,
    read_ok: bool,
    search_ok: bool,
    read_cap_ok: bool,
    search_cap_ok: bool,
    source_ref_ok: bool,
}

struct Run {
    store: Store,
    dir: PathBuf,
    previous_projection: Option<String>,
    turns: Vec<Value>,
    total_input: u64,
    total_assembled: u64,
    total_effective: u64,
    total_projection: u64,
    final_visible: BTreeMap<String, bool>,
    artifact_recovery: BTreeMap<String, String>,
    artifact_info_ok: bool,
    artifact_read_ok: bool,
    artifact_search_ok: bool,
    read_limit_ok: bool,
    search_limit_ok: bool,
    source_ref_ok: bool,
    tool_pairs_ok: bool,
    ids_stable: bool,
    stub_has_no_token_word: bool,
    stub_has_source_id: bool,
}

fn artifact_checks(
    store: &Store,
    key: &str,
    converted: &[billion_context::Message],
    tag: &str,
) -> Option<ArtifactCheck> {
    let source_id = find_source_id(converted, tag)?;
    let meta = store.artifact_store()?.find_by_message_id(&source_id)?;
    let mut check = ArtifactCheck {
        recovery: Some(format!("missing:{}", meta.artifact_id)),
        info_ok: false,
        read_ok: false,
        search_ok: false,
        read_cap_ok: true,
        search_cap_ok: true,
        source_ref_ok: meta
            .source_ref
            .as_deref()
            .is_some_and(|r| r.starts_with('m')),
    };
    let (info, info_outcome, _) = store.tool(
        key,
        CONTEXT_LIMIT,
        "artifact_info",
        &json!({"artifact_id": meta.artifact_id}),
    );
    if info_outcome == Some("success") && info.contains(&meta.artifact_id) {
        check.info_ok = true;
        check.recovery = Some(format!("info_only:{}", meta.artifact_id));
    }
    let (search, search_outcome, _) = store.tool(
        key,
        CONTEXT_LIMIT,
        "artifact_search",
        &json!({"artifact_id": meta.artifact_id, "query": tag, "limit": 1, "max_bytes": 512}),
    );
    let Ok(search_json) = serde_json::from_str::<Value>(&search) else {
        return Some(check);
    };
    let Some(offset) = search_json["hits"]
        .as_array()
        .and_then(|hits| hits.first())
        .and_then(|hit| hit["offset"].as_u64())
    else {
        return Some(check);
    };
    if search_outcome == Some("success")
        && search_json["returned_bytes"].as_u64().unwrap_or(9999) <= 512
    {
        check.search_ok = true;
        check.search_cap_ok = true;
    } else {
        check.search_cap_ok = false;
    }
    let (read, read_outcome, _) = store.tool(
        key,
        CONTEXT_LIMIT,
        "artifact_read",
        &json!({"artifact_id": meta.artifact_id, "offset": offset, "limit": 512}),
    );
    if read_outcome == Some("success") {
        check.read_ok = true;
    }
    if let Ok(read_json) = serde_json::from_str::<Value>(&read) {
        let text = read_json["text"].as_str().unwrap_or_default();
        if text.contains(tag) {
            check.recovery = Some(format!("recovered:{}", meta.artifact_id));
        }
        check.read_cap_ok &= text.len() <= 64 * 1024;
    }
    // Ask for far more than the hard cap; the returned text must still be
    // bounded by the implementation.
    let (cap_text, cap_outcome, _) = store.tool(
        key,
        CONTEXT_LIMIT,
        "artifact_read",
        &json!({"artifact_id": meta.artifact_id, "offset": 0, "limit": 10_000_000}),
    );
    if cap_outcome == Some("success") {
        if let Ok(value) = serde_json::from_str::<Value>(&cap_text) {
            check.read_cap_ok &=
                value["text"].as_str().map(str::len).unwrap_or(usize::MAX) <= 64 * 1024;
        }
    }
    check.search_cap_ok &= search_json["returned_bytes"].as_u64().unwrap_or(0) <= 512;
    Some(check)
}

fn check_tool_pairs(messages: &[Message]) -> bool {
    let mut calls = std::collections::HashSet::new();
    let mut results = std::collections::HashSet::new();
    for message in messages {
        match &message.content {
            Content::ToolCall { call_id, .. } => {
                calls.insert(Some(call_id.to_string()));
            }
            Content::ToolResult { call_id, .. } => {
                results.insert(Some(call_id.to_string()));
            }
            _ => {}
        }
    }
    // A result must have a call. An incomplete call without a result is legal.
    results.iter().all(|result| calls.contains(result))
}

fn run_group(fixture: &str, group: Group, run_index: usize) -> Value {
    let dir = std::env::temp_dir().join(format!(
        "bili-replay-{}-{}-{}-{}-{}",
        fixture,
        group.name(),
        run_index,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        std::process::id()
    ));
    let store = match group {
        Group::Baseline => Store::new(),
        Group::A => Store::with_dir_and_policy(dir.clone(), policy_a()),
        Group::B => Store::with_dir_and_policy(dir.clone(), policy_b()),
    };
    let key = format!("replay-{fixture}-{}-{run_index}", group.name());
    let mut run = Run {
        store,
        dir,
        previous_projection: None,
        turns: Vec::new(),
        total_input: 0,
        total_assembled: 0,
        total_effective: 0,
        total_projection: 0,
        final_visible: BTreeMap::new(),
        artifact_recovery: BTreeMap::new(),
        artifact_info_ok: false,
        artifact_read_ok: false,
        artifact_search_ok: false,
        read_limit_ok: true,
        search_limit_ok: true,
        source_ref_ok: true,
        tool_pairs_ok: true,
        ids_stable: true,
        stub_has_no_token_word: true,
        stub_has_source_id: true,
    };

    for (number, body) in turn_bodies(fixture).into_iter().enumerate() {
        let converted = crate::anthropic::to_core(&body).expect("valid test fixture");
        let raw_input = crate::anthropic::estimate_message_tokens(&body) as u32;
        // System prompt + tool definitions + host's fixed 800-token reserve.
        // Public tool definitions are not injected by this replay, so they are
        // intentionally outside this number and documented as such.
        let fixed_overhead =
            (crate::anthropic::estimate_fixed_overhead(&body) as u32).saturating_add(800);
        let prepared = if matches!(group, Group::A) && number == 0 {
            run.store.prepare_manual_prune(
                &key,
                CONTEXT_LIMIT,
                &converted.messages,
                raw_input,
                fixed_overhead,
                None,
            )
        } else {
            run.store.prepare_with_budget(
                &key,
                CONTEXT_LIMIT,
                &converted.messages,
                raw_input,
                fixed_overhead,
                None,
            )
        };
        let rebuilt = crate::anthropic::from_core(&prepared.messages, &converted.sidecar);
        let mut wire = body.clone();
        wire["messages"] = Value::Array(rebuilt);
        let projection_tokens = crate::anthropic::estimate_body_tokens(&wire) as u64;
        let text = projection_text(&prepared.messages);
        let changed = run
            .previous_projection
            .as_deref()
            .and_then(|previous| first_diff(previous, &text));
        let cache_invalidated = changed.map(|offset| text.len().saturating_sub(offset));
        let ids_before: Vec<&str> = converted
            .messages
            .iter()
            .map(|message| message.id.as_str())
            .collect();
        let ids_after: Vec<&str> = prepared
            .messages
            .iter()
            .map(|message| message.id.as_str())
            .collect();
        let missing_id = ids_before.iter().any(|id| !ids_after.contains(id));
        run.ids_stable &= !missing_id;
        run.tool_pairs_ok &= check_tool_pairs(&prepared.messages);
        for message in &prepared.messages {
            if let Content::ToolResult { content, .. } = &message.content {
                if content.starts_with("[artifact ") {
                    run.stub_has_no_token_word &= !content.to_ascii_lowercase().contains("token");
                    // The stub carries the stable source message id available
                    // before ref allocation; metadata later gains the mNNNNN
                    // source_ref without rewriting the stub body.
                    run.stub_has_source_id &= content.contains("source h_");
                }
            }
        }
        for tag in ["BETA-MID-1", "BETA-MID-2"] {
            if let Some(check) = artifact_checks(&run.store, &key, &converted.messages, tag) {
                if let Some(recovery) = check.recovery {
                    run.artifact_recovery.insert(tag.to_string(), recovery);
                }
                run.artifact_info_ok |= check.info_ok;
                run.artifact_read_ok |= check.read_ok;
                run.artifact_search_ok |= check.search_ok;
                run.read_limit_ok &= check.read_cap_ok;
                run.search_limit_ok &= check.search_cap_ok;
                run.source_ref_ok &= check.source_ref_ok;
            }
        }
        run.total_input += raw_input as u64;
        run.total_assembled += prepared.assembled_projection_estimate as u64;
        run.total_effective += prepared.effective_projection_estimate as u64;
        run.total_projection += projection_tokens;
        let trigger_reason = prepared.prune_decision.reason;
        run.turns.push(json!({
            "turn": number + 1,
            "raw_input_estimate": raw_input,
            "input_tokens_est": raw_input,
            "assembled_projection_estimate": prepared.assembled_projection_estimate,
            "effective_projection_estimate": prepared.effective_projection_estimate,
            "effective_input_estimate": prepared.effective_projection_estimate,
            "fixed_overhead_estimate": prepared.fixed_overhead_estimate,
            "provider_input_usage": Value::Null,
            "provider_cached_input_tokens": Value::Null,
            "stub_reapply_items": prepared.stub_reapply_items,
            "projection_tokens_est": projection_tokens,
            "prune_trigger_reason": trigger_reason,
            "cooldown_hit": prepared.prune_decision.cooldown_hit,
            "cooldown_blocked": prepared.prune_decision.cooldown_blocked,
            "growth_bypass_triggered": prepared.prune_decision.growth_bypass,
            "prune_triggered": prepared.prune_triggered,
            "prune_skipped": prepared.prune_skipped,
            "prune_pruned_items": prepared.prune.as_ref().map(|report| report.pruned_items).unwrap_or(0),
            "projection_bytes": text.len(),
            "prune_items": prepared.pruned_ids.len(),
            "prune_saved_tokens_est": run.store.status_json(&key)
                .and_then(|status| status["compaction"]["prune_saved_tokens_est"].as_u64())
                .unwrap_or(0),
            "cache_first_changed_byte": changed,
            "cache_invalidated_est_bytes": cache_invalidated,
            "ids_stable": !missing_id,
            "tool_pairs_ok": check_tool_pairs(&prepared.messages),
        }));
        if number + 1 == turn_bodies(fixture).len() {
            let facts = [
                "CONSTRAINT_NEVER_TOUCH_API",
                "FACT_HEAD",
                "FACT_MID",
                "FACT_TAIL",
                "FILE_MODIFIED",
                "TEST_FAIL",
                "OPEN_PROBLEM",
                "FINAL_TEST_PASS",
                "PIVOT_CONSTRAINT",
            ];
            for fact in facts {
                run.final_visible
                    .insert(fact.to_string(), text.contains(fact));
            }
        }
        run.previous_projection = Some(text);
    }

    let status = run.store.status_json(&key).unwrap_or_else(|| json!({}));
    let visible_count = run
        .final_visible
        .values()
        .filter(|visible| **visible)
        .count();
    let recovery_count = run
        .artifact_recovery
        .values()
        .filter(|value| value.starts_with("recovered:"))
        .count();
    let quality_score = visible_count + recovery_count;
    let result = json!({
        "fixture": fixture,
        "group": group.name(),
        "turns": run.turns,
        "run_index": run_index,
        "total_input_tokens_est": run.total_input,
        "total_assembled_projection_est": run.total_assembled,
        "total_effective_input_est": run.total_effective,
        "total_effective_projection_est": run.total_effective,
        "total_projection_tokens_est": run.total_projection,
        "prune_runs": status["compaction"]["prune_runs"],
        "prune_attempts": status["compaction"]["prune_attempts"],
        "prune_trigger_reason": status["compaction"]["prune_trigger_reason"],
        "cooldown_hits": status["compaction"]["cooldown_hits"],
        "cooldown_blocked": status["compaction"]["cooldownBlocked"],
        "growth_bypass_triggered": status["compaction"]["growthBypassTriggered"],
        "prune_skipped": status["compaction"]["pruneSkipped"],
        "last_prune_tokens": status["lastPruneTokens"],
        "last_change_ref": status["compaction"]["last_change_ref"],
        "prune_items": status["compaction"]["prune_items"],
        "artifact_writes": status["compaction"]["artifact_writes"],
        "artifact_reads": status["compaction"]["artifact_reads"],
        "artifact_read_bytes": status["compaction"]["artifact_read_bytes"],
        "artifact_stats": status["artifacts"],
        "final_visible_facts": run.final_visible,
        "artifact_recovery": run.artifact_recovery,
        "quality_score": quality_score,
        "artifact_info_ok": run.artifact_info_ok,
        "artifact_read_ok": run.artifact_read_ok,
        "artifact_search_ok": run.artifact_search_ok,
        "read_hard_cap_ok": run.read_limit_ok,
        "search_hard_cap_ok": run.search_limit_ok,
        "tool_pairs_ok": run.tool_pairs_ok,
        "ids_stable": run.ids_stable,
        "stub_has_no_token_word": run.stub_has_no_token_word,
        "stub_has_source_id": run.stub_has_source_id,
        "source_ref_ok": run.source_ref_ok,
        "provider_input_usage": null,
        "provider_cached_input_tokens": null,
        "provider_cache_tokens": null,
        "cache_metric_kind": "projection_prefix_proxy_only",
    });
    let _ = std::fs::remove_dir_all(&run.dir);
    result
}

#[test]
fn anthropic_round_trip_keeps_string_tool_result_body() {
    let large_tool_result = format!("TOOL-BODY-{}\n", "y".repeat(8_000));
    let request = body(vec![
        user("inspect the file"),
        assistant_tool("toolu_read", "Read", json!({"path": "src/main.rs"})),
        tool_result("toolu_read", large_tool_result),
    ]);
    let converted = crate::anthropic::to_core(&request).expect("valid test fixture");
    let rebuilt = crate::anthropic::from_core(&converted.messages, &converted.sidecar);
    let wire = json!({"messages": rebuilt});
    assert!(wire.to_string().contains("TOOL-BODY"));
    assert!(
        crate::anthropic::estimate_body_tokens(&wire) > 1_000,
        "tool result must contribute to the request estimate"
    );
}

#[test]
fn phase1_5_replay_baseline_a_b() {
    let fixtures = ["long_code", "huge_tool", "pivot_boundary"];
    let groups = [Group::Baseline, Group::A, Group::B];
    let mut rows = Vec::new();
    for fixture in fixtures {
        for group in groups {
            for run_index in 0..3 {
                let row = run_group(fixture, group, run_index);
                println!("REPLAY_JSON {}", serde_json::to_string(&row).unwrap());
                rows.push(row);
            }
        }
    }

    // Directional replay is deterministic across the three runs.
    for fixture in fixtures {
        for group in groups {
            let first = rows
                .iter()
                .find(|row| {
                    row["fixture"] == fixture
                        && row["group"] == group.name()
                        && row["run_index"] == 0
                })
                .expect("first run");
            for run_index in 1..3 {
                let later = rows
                    .iter()
                    .find(|row| {
                        row["fixture"] == fixture
                            && row["group"] == group.name()
                            && row["run_index"] == run_index
                    })
                    .expect("later run");
                for key in [
                    "total_input_tokens_est",
                    "total_assembled_projection_est",
                    "total_effective_input_est",
                    "total_effective_projection_est",
                    "total_projection_tokens_est",
                    "quality_score",
                    "prune_runs",
                    "prune_items",
                    "cooldown_blocked",
                    "growth_bypass_triggered",
                    "prune_skipped",
                    "artifact_writes",
                ] {
                    assert_eq!(
                        first[key],
                        later[key],
                        "{fixture} {} run {run_index}: {key} differs",
                        group.name()
                    );
                }
            }
        }
    }

    for fixture in fixtures {
        let baseline = rows
            .iter()
            .find(|row| row["fixture"] == fixture && row["group"] == "corrected-baseline")
            .expect("baseline");
        let a = rows
            .iter()
            .find(|row| row["fixture"] == fixture && row["group"] == "A-artifact-stub")
            .expect("A");
        let b = rows
            .iter()
            .find(|row| row["fixture"] == fixture && row["group"] == "B-hysteresis-batch")
            .expect("B");
        for (label, modified) in [("A", a), ("B", b)] {
            assert!(
                modified["quality_score"].as_u64().unwrap_or(0)
                    >= baseline["quality_score"].as_u64().unwrap_or(0),
                "{fixture} {label}: quality below corrected baseline"
            );
            assert_eq!(
                modified["ids_stable"], true,
                "{fixture} {label}: ids drifted"
            );
            assert_eq!(
                modified["tool_pairs_ok"], true,
                "{fixture} {label}: pair broken"
            );
            assert_eq!(
                modified["stub_has_no_token_word"], true,
                "{fixture} {label}: stub token wording"
            );
            assert_eq!(
                modified["stub_has_source_id"], true,
                "{fixture} {label}: stub lacks source id"
            );
            assert_eq!(
                modified["source_ref_ok"], true,
                "{fixture} {label}: artifact source_ref missing"
            );
            assert_eq!(
                modified["read_hard_cap_ok"], true,
                "{fixture} {label}: read cap"
            );
            assert_eq!(
                modified["search_hard_cap_ok"], true,
                "{fixture} {label}: search cap"
            );
        }
        // A is exactly one manual prune and no automatic trigger afterwards.
        assert!(
            a["prune_runs"].as_u64().unwrap_or(0) <= 1,
            "{fixture}: A auto-pruned more than the manual one-shot"
        );
        assert!(
            a["prune_attempts"].as_u64().unwrap_or(0) <= 1,
            "{fixture}: A attempted an automatic prune"
        );
        assert!(
            a["prune_runs"].as_u64().unwrap_or(0) >= 1,
            "{fixture}: A did not execute its manual artifact/stub prune"
        );
        let reapply_sum = |row: &serde_json::Value| -> u64 {
            row["turns"]
                .as_array()
                .map(|turns| {
                    turns
                        .iter()
                        .map(|turn| turn["stub_reapply_items"].as_u64().unwrap_or(0))
                        .sum()
                })
                .unwrap_or(0)
        };
        assert!(
            reapply_sum(a) > 0,
            "{fixture}: A lost its stub on a resent raw transcript"
        );
        assert!(
            reapply_sum(b) > 0,
            "{fixture}: B lost its stub on a resent raw transcript"
        );
        // A's one-shot artifact/stub path must still beat the corrected
        // baseline on both measures; this is not a pruning-by-force result.
        assert!(
            a["total_effective_projection_est"]
                .as_u64()
                .unwrap_or(u64::MAX)
                <= baseline["total_effective_projection_est"]
                    .as_u64()
                    .unwrap_or(0),
            "{fixture}: A effective projection did not beat the baseline"
        );
        assert!(
            a["total_projection_tokens_est"]
                .as_u64()
                .unwrap_or(u64::MAX)
                <= baseline["total_projection_tokens_est"]
                    .as_u64()
                    .unwrap_or(0),
            "{fixture}: A wire projection did not beat the baseline"
        );
        assert!(
            b["prune_runs"].as_u64().unwrap_or(0) > a["prune_runs"].as_u64().unwrap_or(0),
            "{fixture}: B has no additional hysteresis prune over A"
        );
        // B must be data-distinguishable, not a renamed A. At least one
        // fixture must show different effective projection/trigger behavior.
        let a_effective = a["total_effective_projection_est"].as_u64().unwrap_or(0);
        let b_effective = b["total_effective_projection_est"].as_u64().unwrap_or(0);
        let a_runs = a["prune_runs"].as_u64().unwrap_or(0);
        let b_runs = b["prune_runs"].as_u64().unwrap_or(0);
        assert!(
            a_effective != b_effective || a_runs != b_runs || a["turns"] != b["turns"],
            "{fixture}: A and B are indistinguishable"
        );
    }
}

#[test]
fn artifact_write_failure_does_not_create_stub() {
    let root = std::env::temp_dir().join(format!(
        "bili-replay-failure-{}-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        std::process::id()
    ));
    std::fs::create_dir_all(&root).expect("root");
    std::fs::write(root.join("artifacts"), b"not a directory").expect("block artifact dir");
    let store = Store::with_dir_and_policy(root.clone(), policy_b());
    let mut messages = vec![Message::text("u1", billion_context::Role::User, "run")];
    messages.push(Message::tool_result(
        "m1",
        "Bash",
        "c1",
        huge("FAIL-INJECT"),
    ));
    let prepared = store.prepare("failure", CONTEXT_LIMIT, &messages, CONTEXT_LIMIT, None);
    assert!(prepared.pruned_ids.is_empty());
    let Content::ToolResult { content, .. } = &prepared.messages[1].content else {
        panic!("tool result missing");
    };
    assert!(!content.starts_with("[artifact "));
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn missing_artifact_is_refused() {
    let root = std::env::temp_dir().join(format!(
        "bili-replay-missing-{}-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        std::process::id()
    ));
    let store = Store::with_dir_and_policy(root.clone(), policy_b());
    let messages = vec![
        Message::text("u1", billion_context::Role::User, "run"),
        Message::tool_result("m1", "Bash", "c1", huge("MISSING-INJECT")),
    ];
    let prepared = store.prepare("missing", CONTEXT_LIMIT, &messages, CONTEXT_LIMIT, None);
    assert_eq!(prepared.pruned_ids.len(), 1);
    let meta = store
        .artifact_store()
        .expect("store")
        .find_by_message_id("m1")
        .expect("artifact");
    std::fs::remove_file(
        root.join("artifacts")
            .join(format!("{}.txt", meta.artifact_id)),
    )
    .expect("remove body");
    let (text, outcome, _) = store.tool(
        "missing",
        CONTEXT_LIMIT,
        "artifact_read",
        &json!({"artifact_id": meta.artifact_id, "offset": 0, "limit": 64}),
    );
    assert_ne!(outcome, Some("success"));
    assert!(text.contains("unavailable") || text.contains("No such file"));
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn artifact_read_and_search_are_byte_bounded() {
    let root = std::env::temp_dir().join(format!(
        "bili-replay-limits-{}-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        std::process::id()
    ));
    let store = Store::with_dir_and_policy(root.clone(), policy_b());
    let messages = vec![
        Message::text("u1", billion_context::Role::User, "run"),
        Message::tool_result("m1", "Bash", "c1", huge("LIMIT-INJECT")),
    ];
    let prepared = store.prepare("limits", CONTEXT_LIMIT, &messages, CONTEXT_LIMIT, None);
    assert_eq!(prepared.pruned_ids.len(), 1);
    let meta = store
        .artifact_store()
        .expect("store")
        .find_by_message_id("m1")
        .expect("artifact");
    let (read, read_outcome, _) = store.tool(
        "limits",
        CONTEXT_LIMIT,
        "artifact_read",
        &json!({"artifact_id": meta.artifact_id, "offset": 0, "limit": 10_000_000}),
    );
    assert_eq!(read_outcome, Some("success"));
    let read_json: Value = serde_json::from_str(&read).expect("read json");
    assert_eq!(read_json["source_message_id"], "m1");
    assert!(read_json["text"].as_str().expect("text").len() <= 64 * 1024);
    let (search, search_outcome, _) = store.tool(
        "limits",
        CONTEXT_LIMIT,
        "artifact_search",
        &json!({"artifact_id": meta.artifact_id, "query": "LIMIT-INJECT", "limit": 100, "max_bytes": 256}),
    );
    assert_eq!(search_outcome, Some("success"));
    let search_json: Value = serde_json::from_str(&search).expect("search json");
    assert!(search_json["hits"].as_array().expect("hits").len() <= 20);
    assert!(search_json["returned_bytes"].as_u64().expect("bytes") <= 256);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn ordinary_ref_without_artifact_is_unavailable() {
    let store = Store::new();
    let (text, outcome, _) = store.tool(
        "unavailable",
        CONTEXT_LIMIT,
        "artifact_info",
        &json!({"ref": "m99999"}),
    );
    assert_ne!(outcome, Some("success"));
    assert!(text.contains("unavailable") || text.contains("original"));
}

#[test]
fn artifact_hash_corruption_is_refused() {
    let root = std::env::temp_dir().join(format!(
        "bili-replay-corrupt-{}-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        std::process::id()
    ));
    let store = Store::with_dir_and_policy(root.clone(), policy_b());
    let messages = vec![
        Message::text("u1", billion_context::Role::User, "run"),
        Message::tool_result("m1", "Bash", "c1", huge("HASH-INJECT")),
    ];
    let prepared = store.prepare("corrupt", CONTEXT_LIMIT, &messages, CONTEXT_LIMIT, None);
    assert_eq!(prepared.pruned_ids.len(), 1);
    let meta: ArtifactMetadata = store
        .artifact_store()
        .expect("store")
        .find_by_message_id("m1")
        .expect("artifact");
    let body = root
        .join("artifacts")
        .join(format!("{}.txt", meta.artifact_id));
    std::fs::write(&body, b"corrupted").expect("corrupt");
    let (text, outcome, _) = store.tool(
        "corrupt",
        CONTEXT_LIMIT,
        "artifact_read",
        &json!({"artifact_id": meta.artifact_id, "offset": 0, "limit": 64}),
    );
    assert_ne!(outcome, Some("success"));
    assert!(text.contains("hash mismatch"));
    let _ = std::fs::remove_dir_all(root);
}

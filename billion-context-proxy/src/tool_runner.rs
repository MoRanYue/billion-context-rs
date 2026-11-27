//! Fail-closed gate in front of the model-visible tool surface.
//!
//! Two things can go wrong before `Store::tool` ever runs, and both used to be
//! invisible to the model:
//!
//! 1. **A name that no branch implements.** `Store::tool` answers it with
//!    `[unknown tool: x]` + `refused`, but the plugin endpoint always replied
//!    `"ok": true`, so the MCP host treated the call as delivered.
//! 2. **A tool whose backing store is disabled.** The `artifact_*` tools and
//!    `retrieve_raw` need `ArtifactStore`; with `Store::new()` (no directory)
//!    they answered with a prose paragraph and still reported `ok: true`.
//!
//! A model that is told "the artifact store is disabled" inside a successful
//! tool result has no reason to stop; it reads the sentence as the answer and
//! carries on. That is the silent bypass this module removes: the endpoint
//! refuses *before* dispatch, with a non-2xx status and `ok: false`, which is
//! the only shape the host treats as a failed call.
//!
//! The gate is deliberately a pure function of `(name, args, artifacts_available)`
//! so it can be tested offline without a store, a socket, or a model.

use serde_json::Value;

/// Every tool name `public_tools()` in `main.rs` advertises, in order.
///
/// Kept here rather than in `main.rs` so the registry completeness test has one
/// list to compare the advertised surface against. Adding a tool to
/// `public_tools()` without adding it here fails
/// `every_advertised_tool_is_implemented`.
pub const IMPLEMENTED: &[&str] = &[
    "compress",
    "decompress",
    "search_context",
    "artifact_info",
    "artifact_read",
    "artifact_search",
    "retrieve_raw",
    "reset_window",
    "acp_status",
    "acp_cache",
];

/// Every name `Store::tool` has a match arm for, aliases included.
///
/// [`IMPLEMENTED`] is the *advertised* surface; this is the *dispatchable* one,
/// and it is larger because the dispatcher also answers historical spellings
/// (`retrieve`, `search`, `status`, …) that were never advertised. The two must
/// not be confused: a name can be dispatchable without being advertised, but
/// nothing [`canonical`] returns may fall outside this list, or the gate would
/// approve a call that the dispatcher answers with `[unknown tool: …]`.
pub const DISPATCHED: &[&str] = &[
    "compress",
    "decompress",
    "search_context",
    "artifact_info",
    "artifact_read",
    "artifact_search",
    "retrieve_raw",
    "retrieve",
    "reset_window",
    "acp_status",
    "acp_cache",
];

/// Tools that cannot do anything without an `ArtifactStore`.
///
/// `retrieve` is *not* listed: without `mode: "original"` it reads the live
/// session, not an archive. The mode is checked separately in
/// [`requires_artifact_store`].
pub const ARTIFACT_BACKED: &[&str] = &[
    "artifact_info",
    "artifact_read",
    "artifact_search",
    "retrieve_raw",
];

/// Resolve a name (or one of its documented aliases) to the canonical entry in
/// [`IMPLEMENTED`]. `None` means nothing implements it.
pub fn canonical(name: &str) -> Option<&'static str> {
    match name.trim() {
        "compress" | "acp_compress" => Some("compress"),
        "decompress" | "acp_decompress" => Some("decompress"),
        "search_context" | "search" | "acp_search" => Some("search_context"),
        "artifact_info" | "artifact.info" => Some("artifact_info"),
        "artifact_read" | "artifact.read" => Some("artifact_read"),
        "artifact_search" | "artifact.search" => Some("artifact_search"),
        "retrieve_raw" | "acp_retrieve_raw" => Some("retrieve_raw"),
        "retrieve" | "acp_retrieve" => Some("retrieve"),
        "reset_window" | "acp_reset_window" => Some("reset_window"),
        "acp_status" | "status" => Some("acp_status"),
        "acp_cache" | "cache" => Some("acp_cache"),
        _ => None,
    }
}

/// True when `name` needs the artifact store to answer at all.
pub fn requires_artifact_store(name: &str, args: &Value) -> bool {
    if ARTIFACT_BACKED.contains(&name) {
        return true;
    }
    // `retrieve` only reaches the archive when it is explicitly asked for the
    // original; the plain path reads the live session and works without a store.
    name == "retrieve"
        && args
            .get("mode")
            .and_then(Value::as_str)
            .is_some_and(|mode| mode.trim() == "original")
}

/// Why a call was refused before dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    /// The request carried no `tool` field (or an empty one).
    MissingName,
    /// No branch implements the name.
    Unknown(String),
    /// Implemented, advertised, but the backing store is not available.
    Unavailable {
        tool: &'static str,
        reason: &'static str,
    },
}

impl Refusal {
    /// Stable machine-readable code, echoed in the JSON body as `code`.
    pub fn code(&self) -> &'static str {
        match self {
            Refusal::MissingName => "tool_name_missing",
            Refusal::Unknown(_) => "unknown_tool",
            Refusal::Unavailable { .. } => "tool_unavailable",
        }
    }

    /// HTTP status. Both are non-2xx on purpose: the host accepts a call only
    /// when `res.ok && data.ok`, so a 2xx here would restore the silent bypass.
    pub fn status(&self) -> u16 {
        match self {
            // Bad model output: the request itself is wrong.
            Refusal::MissingName | Refusal::Unknown(_) => 400,
            // The tool exists but this host cannot serve it right now.
            Refusal::Unavailable { .. } => 503,
        }
    }

    /// One sentence naming the tool and what the caller must do instead. It
    /// never claims a result was produced.
    pub fn message(&self) -> String {
        match self {
            Refusal::MissingName => {
                "tool refused before dispatch: the request carried no `tool` name, so no tool ran and no result exists".to_string()
            }
            Refusal::Unknown(name) => format!(
                "tool refused before dispatch: `{name}` is not implemented by this host, so no tool ran and no result exists (advertised: {})",
                IMPLEMENTED.join(", ")
            ),
            Refusal::Unavailable { tool, reason } => format!(
                "tool refused before dispatch: `{tool}` needs the {reason}, which this host does not have, so no tool ran and no result exists — do not describe an artifact you could not read; work from the conversation text you already have"
            ),
        }
    }
}

/// The gate. `Err` means: refuse, do not dispatch, do not pretend.
pub fn check(name: &str, args: &Value, artifacts_available: bool) -> Result<&'static str, Refusal> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return Err(Refusal::MissingName);
    }
    let Some(canonical_name) = canonical(trimmed) else {
        return Err(Refusal::Unknown(trimmed.to_string()));
    };
    if !artifacts_available && requires_artifact_store(canonical_name, args) {
        return Err(Refusal::Unavailable {
            tool: canonical_name,
            reason: "artifact store",
        });
    }
    Ok(canonical_name)
}

/// True when the host can actually serve `name` right now.
///
/// Distinct from [`check`]: `check` rejects a *call*, this filters an
/// *advertisement*. A surface that lists a tool the host cannot run teaches the
/// model to call it and then punishes it for the call, so the list shrinks
/// instead.
pub fn available(name: &str, artifacts_available: bool) -> bool {
    if artifacts_available {
        return true;
    }
    match canonical(name) {
        Some(canonical_name) => !ARTIFACT_BACKED.contains(&canonical_name),
        // An unresolvable name is already reported by `unresolved`; leave it in
        // place so the mismatch stays visible rather than silently vanishing.
        None => true,
    }
}

/// Keep only the definitions the host can currently serve.
///
/// Definitions are `public_tools()`-shaped, i.e. `{"name": ...}` objects.
pub fn servable(definitions: &[Value], artifacts_available: bool) -> Vec<Value> {
    definitions
        .iter()
        .filter(|definition| {
            definition
                .get("name")
                .and_then(Value::as_str)
                .is_none_or(|name| available(name, artifacts_available))
        })
        .cloned()
        .collect()
}

/// Names dropped from the advertised surface because this host cannot serve
/// them. Empty on a normal host; non-empty on a storeless one, where it names
/// exactly what the model is not being told about.
pub fn withheld(artifacts_available: bool) -> Vec<String> {
    crate::public_tools()
        .iter()
        .filter_map(|tool| tool.get("name").and_then(Value::as_str))
        .filter(|name| !available(name, artifacts_available))
        .map(str::to_string)
        .collect()
}

/// The wire body for a refused call.
///
/// Shape matters: the MCP host accepts a tool result only when
/// `res.ok && data.ok`, and reads `data.error` when it does not. So every
/// refusal carries `ok: false`, a `code`, and an `error` sentence that states
/// plainly that nothing ran. `result` is deliberately absent — a caller that
/// only looks for `result` finds nothing rather than a paragraph it might quote
/// as if it were tool output.
pub fn refusal_body(refusal: &Refusal, tool: &str, conversation_id: &str) -> Value {
    serde_json::json!({
        "ok": false,
        "code": refusal.code(),
        "tool": tool,
        "conversationId": conversation_id,
        "error": refusal.message(),
    })
}

/// Re-shape the Anthropic-flavoured definitions as OpenAI `function` tools.
///
/// One registry, two wire dialects. The names, descriptions and schemas are the
/// same objects — only the wrapper changes — so a dialect cannot drift into
/// advertising a tool the other one does not have. The manifest used to publish
/// `"openai": []`, which told an OpenAI-protocol reader that this host serves no
/// tools at all.
pub fn openai_tools(definitions: &[Value]) -> Vec<Value> {
    definitions
        .iter()
        .filter_map(|definition| {
            let name = definition.get("name")?.as_str()?;
            let mut function = serde_json::Map::new();
            function.insert("name".to_string(), Value::String(name.to_string()));
            if let Some(description) = definition.get("description").and_then(Value::as_str) {
                function.insert(
                    "description".to_string(),
                    Value::String(description.to_string()),
                );
            }
            let schema = definition
                .get("input_schema")
                .cloned()
                .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}}));
            function.insert("parameters".to_string(), schema);
            Some(serde_json::json!({"type": "function", "function": function}))
        })
        .collect()
}

/// Advertised names that [`canonical`] does not resolve. Empty means the
/// registry and the dispatch table agree. Used by the manifest so a broken
/// surface is visible to the host's operator instead of only to a model.
pub fn unresolved(advertised: &[&str]) -> Vec<String> {
    advertised
        .iter()
        .filter(|name| canonical(name).is_none())
        .map(|name| (*name).to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn advertised() -> Vec<String> {
        crate::public_tools()
            .iter()
            .filter_map(|tool| tool.get("name").and_then(Value::as_str))
            .map(str::to_string)
            .collect()
    }

    #[test]
    fn every_advertised_tool_is_implemented() {
        let names = advertised();
        assert_eq!(names.len(), 10, "advertised surface changed: {names:?}");
        let unresolved = unresolved(&names.iter().map(String::as_str).collect::<Vec<_>>());
        assert!(
            unresolved.is_empty(),
            "advertised but unimplemented: {unresolved:?}"
        );
    }

    #[test]
    fn every_implemented_tool_is_advertised() {
        let names = advertised();
        for tool in IMPLEMENTED {
            assert!(
                names.iter().any(|name| name == tool),
                "{tool} is implemented but never advertised, so no model can call it"
            );
        }
        assert_eq!(
            IMPLEMENTED.len(),
            names.len(),
            "IMPLEMENTED and public_tools() disagree"
        );
    }

    #[test]
    fn implemented_covers_the_dispatch_table() {
        // Every name Store::tool matches on must resolve through the gate,
        // otherwise a reachable branch would be unreachable from the wire.
        for name in [
            "compress",
            "acp_compress",
            "decompress",
            "acp_decompress",
            "search_context",
            "search",
            "acp_search",
            "artifact_info",
            "artifact.info",
            "artifact_read",
            "artifact.read",
            "artifact_search",
            "artifact.search",
            "retrieve_raw",
            "acp_retrieve_raw",
            "retrieve",
            "acp_retrieve",
            "reset_window",
            "acp_reset_window",
            "acp_status",
            "status",
            "acp_cache",
            "cache",
        ] {
            assert!(canonical(name).is_some(), "{name} is dispatched but gated");
        }
    }

    #[test]
    fn a_canonical_name_is_always_dispatchable() {
        // The gate returns `canonical`, and `plugin_tool_handler` now dispatches
        // that value. If a canonical name were missing from the dispatcher, the
        // approved call would land in the `[unknown tool]` fallback — the silent
        // bypass this module exists to remove, wearing the gate's own approval.
        for name in [
            "compress",
            "acp_compress",
            "decompress",
            "acp_decompress",
            "search_context",
            "search",
            "acp_search",
            "artifact_info",
            "artifact.info",
            "artifact_read",
            "artifact.read",
            "artifact_search",
            "artifact.search",
            "retrieve_raw",
            "acp_retrieve_raw",
            "retrieve",
            "acp_retrieve",
            "reset_window",
            "acp_reset_window",
            "acp_status",
            "status",
            "acp_cache",
            "cache",
        ] {
            let canonical_name = canonical(name).expect("gated name resolves");
            assert!(
                DISPATCHED.contains(&canonical_name),
                "{name} resolves to {canonical_name}, which the dispatcher does not implement"
            );
        }
    }

    #[test]
    fn the_advertised_surface_is_a_subset_of_the_dispatchable_one() {
        for name in IMPLEMENTED {
            assert!(
                DISPATCHED.contains(name),
                "{name} is advertised but the dispatcher cannot serve it"
            );
        }
    }

    #[test]
    fn a_missing_name_is_refused_rather_than_dispatched_as_empty() {
        let refusal = check("", &json!({}), true).expect_err("empty name must refuse");
        assert_eq!(refusal, Refusal::MissingName);
        assert_eq!(refusal.code(), "tool_name_missing");
        assert_eq!(refusal.status(), 400);
        assert!(refusal.message().contains("no result exists"));
        assert_eq!(
            check("   ", &json!({}), true).expect_err("blank name must refuse"),
            Refusal::MissingName
        );
    }

    #[test]
    fn an_unknown_tool_names_itself_and_the_advertised_surface() {
        let refusal = check("artifact_write", &json!({}), true).expect_err("unknown must refuse");
        assert_eq!(refusal.code(), "unknown_tool");
        assert_eq!(refusal.status(), 400);
        let message = refusal.message();
        assert!(message.contains("artifact_write"), "{message}");
        assert!(message.contains("artifact_read"), "{message}");
        assert!(message.contains("no result exists"), "{message}");
    }

    #[test]
    fn artifact_tools_refuse_when_the_store_is_disabled() {
        for tool in ARTIFACT_BACKED {
            let refusal = check(tool, &json!({"artifact_id": "a1"}), false)
                .expect_err("a storeless host cannot serve an artifact tool");
            assert_eq!(refusal.code(), "tool_unavailable", "{tool}");
            assert_eq!(refusal.status(), 503, "{tool}");
            let message = refusal.message();
            assert!(message.contains(tool), "{message}");
            assert!(message.contains("artifact store"), "{message}");
            assert!(message.contains("no result exists"), "{message}");
        }
    }

    #[test]
    fn artifact_tools_still_pass_when_the_store_exists() {
        for tool in ARTIFACT_BACKED {
            assert_eq!(
                check(tool, &json!({"artifact_id": "a1"}), true),
                Ok(*tool),
                "{tool}"
            );
        }
    }

    #[test]
    fn plain_retrieve_works_without_a_store_but_mode_original_does_not() {
        assert_eq!(
            check("retrieve", &json!({"refs": ["m1"]}), false),
            Ok("retrieve")
        );
        let refusal = check(
            "retrieve",
            &json!({"refs": ["m1"], "mode": "original"}),
            false,
        )
        .expect_err("mode original reaches the archive");
        assert_eq!(refusal.code(), "tool_unavailable");
        assert_eq!(
            check(
                "retrieve",
                &json!({"refs": ["m1"], "mode": "original"}),
                true
            ),
            Ok("retrieve")
        );
    }

    #[test]
    fn non_artifact_tools_never_depend_on_the_store() {
        // Canonical names only: `check` answers with the canonical spelling, so
        // a raw alias here would compare `Ok("acp_cache")` against `Ok("cache")`.
        for tool in [
            "compress",
            "decompress",
            "search_context",
            "reset_window",
            "acp_status",
            "acp_cache",
        ] {
            assert_eq!(check(tool, &json!({}), false), Ok(tool), "{tool}");
        }
        // The aliases must land on the same availability decision as their
        // canonical spelling, store or no store.
        assert_eq!(check("cache", &json!({}), false), Ok("acp_cache"));
        assert_eq!(check("status", &json!({}), false), Ok("acp_status"));
        assert_eq!(check("search", &json!({}), false), Ok("search_context"));
    }

    #[test]
    fn aliases_resolve_to_their_canonical_names() {
        assert_eq!(canonical("acp_compress"), Some("compress"));
        assert_eq!(canonical("artifact.info"), Some("artifact_info"));
        assert_eq!(canonical("artifact.read"), Some("artifact_read"));
        assert_eq!(canonical("artifact.search"), Some("artifact_search"));
        assert_eq!(canonical("acp_cache"), Some("acp_cache"));
        assert_eq!(canonical("  compress  "), Some("compress"));
        // The aliased name must reach the same availability decision.
        assert!(check("artifact.info", &json!({}), false).is_err());
        assert!(check("  artifact.info  ", &json!({}), true).is_ok());
    }

    #[test]
    fn an_unresolved_advertised_name_is_reported_not_ignored() {
        assert!(unresolved(&["compress", "artifact_read"]).is_empty());
        assert_eq!(
            unresolved(&["compress", "artifact_write"]),
            vec!["artifact_write".to_string()]
        );
    }

    #[test]
    fn a_storeless_host_advertises_exactly_the_tools_it_can_run() {
        let all = crate::public_tools();
        let advertised = servable(&all, false);
        let names: Vec<&str> = advertised
            .iter()
            .filter_map(|tool| tool.get("name").and_then(Value::as_str))
            .collect();
        assert_eq!(names.len(), IMPLEMENTED.len() - ARTIFACT_BACKED.len());
        for withheld_name in ARTIFACT_BACKED {
            assert!(
                !names.contains(withheld_name),
                "{withheld_name} was advertised without a store"
            );
            assert!(
                check(withheld_name, &json!({"artifact_id": "a"}), false).is_err(),
                "{withheld_name} was advertised but the gate refuses it — surface and gate disagree"
            );
        }
        for kept in &names {
            assert!(
                check(kept, &json!({}), false).is_ok(),
                "{kept} was advertised but the gate refuses it"
            );
        }
    }

    #[test]
    fn every_servable_definition_passes_the_gate_it_is_registered_for() {
        // Whatever the store state, the advertised surface and the pre-dispatch
        // gate must make the same call for every tool on it.
        for artifacts_available in [true, false] {
            for definition in servable(&crate::public_tools(), artifacts_available) {
                let name = definition["name"].as_str().expect("named tool");
                let args = if name == "retrieve" {
                    json!({"mode": "original", "refs": ["m1"]})
                } else {
                    json!({"artifact_id": "art"})
                };
                assert!(
                    check(name, &args, artifacts_available).is_ok(),
                    "{name} advertised with artifacts_available={artifacts_available} but refused"
                );
            }
        }
    }

    #[test]
    fn withheld_names_exactly_the_store_dependent_tools() {
        assert!(withheld(true).is_empty(), "a full host withholds nothing");
        let dropped = withheld(false);
        assert_eq!(dropped.len(), ARTIFACT_BACKED.len());
        for name in ARTIFACT_BACKED {
            assert!(dropped.iter().any(|dropped| dropped == name), "{name}");
        }
        // A tool the store does not back must never be withheld.
        assert!(!dropped.iter().any(|name| name == "compress"));
        assert!(!dropped.iter().any(|name| name == "reset_window"));
    }

    #[test]
    fn the_openai_dialect_advertises_exactly_the_anthropic_surface() {
        let definitions = servable(&crate::public_tools(), true);
        let converted = openai_tools(&definitions);
        assert_eq!(
            converted.len(),
            definitions.len(),
            "a dialect must not lose a tool"
        );

        for (source, target) in definitions.iter().zip(converted.iter()) {
            let source_name = source["name"].as_str().expect("name");
            let target_name = target["function"]["name"].as_str().expect("function name");
            assert_eq!(source_name, target_name);
            assert_eq!(target["type"], json!("function"));
            assert_eq!(target["function"]["description"], source["description"]);
            // The schema travels unchanged: a re-shaped schema is a second
            // contract, and two contracts drift.
            assert_eq!(target["function"]["parameters"], source["input_schema"]);
            // No leftover Anthropic keys on the OpenAI entry.
            assert!(
                target.get("name").is_none(),
                "flat `name` would shadow the function"
            );
            assert!(target.get("input_schema").is_none());
        }
    }

    #[test]
    fn a_storeless_host_also_never_advertises_a_withheld_tool_in_either_dialect() {
        let definitions = servable(&crate::public_tools(), false);
        let openai = openai_tools(&definitions);
        assert_eq!(openai.len(), definitions.len());
        for name in withheld(false) {
            assert!(
                !openai
                    .iter()
                    .any(|entry| entry["function"]["name"] == json!(name)),
                "{name} leaked into the OpenAI surface"
            );
        }
    }

    #[test]
    fn the_refusal_body_cannot_be_read_as_a_result() {
        let refusal = Refusal::Unavailable {
            tool: "artifact_read",
            reason: "artifact store",
        };
        let body = refusal_body(&refusal, "artifact_read", "conv-1");
        assert_eq!(body["ok"], json!(false), "ok must not be truthy");
        assert_eq!(body["code"], json!("tool_unavailable"));
        assert_eq!(body["tool"], json!("artifact_read"));
        assert_eq!(body["conversationId"], json!("conv-1"));
        assert!(
            body.get("result").is_none(),
            "a refusal must not carry a `result` field a caller could quote"
        );
        assert!(
            body["error"].as_str().is_some_and(|text| !text.is_empty()),
            "the host reads `error` when ok is false"
        );
    }

    #[test]
    fn every_refusal_shape_is_rejected_by_the_host_acceptance_rule() {
        // The host accepts a call only when `res.ok && data.ok`, so both halves
        // must be false for each refusal kind.
        for refusal in [
            Refusal::MissingName,
            Refusal::Unknown("nope".to_string()),
            Refusal::Unavailable {
                tool: "artifact_search",
                reason: "artifact store",
            },
        ] {
            let body = refusal_body(&refusal, "nope", "conv");
            assert_eq!(body["ok"], json!(false), "{}", refusal.code());
            assert!(
                !(200..300).contains(&refusal.status()),
                "{} returned a 2xx status",
                refusal.code()
            );
        }
    }
}

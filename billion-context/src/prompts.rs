//! The compression rule texts the model is shown, verbatim from the reference
//! kernel's `compression-rules.ts`.
//!
//! These are not documentation — they are **prompt payload**, and their wording
//! was tuned over months of production use. A summary that loses paths,
//! signatures, or the rationale behind a decision cannot be decompressed back
//! into usable context, so every clause here is load-bearing.
//!
//! Overriding any of them is therefore an explicit act:
//! [`Prompts::resolve`] refuses unless the caller passes
//! [`RiskAcknowledged`]. Hosts that only want the source language preserved can
//! ask for that alone — [`Prompts::with_language_preservation`] is additive and
//! needs no acknowledgement, exactly as upstream's `languagePreservation` opt-in
//! behaves.
//!
//! Three amendments are folded into the canonical text and must survive any
//! re-transcription (they were incidents, not preferences):
//!
//! - write-side quote fidelity — a fabricated verbatim user quote stored as a
//!   live "CURRENT TASK" caused loop relapses;
//! - open-objectives carry-forward at every tier — a multi-block merge dropped a
//!   still-open user pivot;
//! - unicode written directly — copying `\uXXXX` escapes out of tool output
//!   makes summaries unreadable to the model that later consumes them.

/// Opt-in language-preservation clause. Not part of the canonical set: appending
/// it by default would change every prompt byte and shift the provider prefix
/// cache on every kernel bump.
pub const LANGUAGE_PRESERVATION_RULE: &str = "Preserve the source conversation's primary language. Do not translate a monolingual conversation without a user request. For mixed-language source, preserve each segment's language. Keep code, commands, identifiers, and quoted text verbatim when the tier's fidelity rules retain them.";

pub const COMPRESS_PHILOSOPHY: &str = "Compression Philosophy:
- All compression serves the primary task, but be frugal.
- Context capacity is precious. Save context by compressing consumed outputs, not by avoiding tools.
- Compress by need, not by percentage.
- Work from summaries, not raw tool outputs. All listed ranges (user prompts, tool outputs, code, logs, exploration, intermediate steps) should be compressed to summary format — the ONLY exceptions are protected content, content the current step is actively using, or critical content you cannot reconstruct.";

pub const HOW_TO_COMPRESS_RULES: &str = "HOW TO COMPRESS

When you call `compress`, the summary you write becomes the only record of the replaced conversation. Make it self-contained and complete: every user request, experiment purpose, and work task in the range must be accurately captured. A later reader (or you, after decompressing) should be able to continue the task WITHOUT needing the original. The summary records the PAST as of this block's creation: label recorded task state as history (\"TASK AS OF THIS BLOCK: ...\") — never as a live instruction, so a later reader treats it as settled context, not something to re-execute. Write plain text with real unicode characters; never copy \\uXXXX escape sequences or JSON-escaped fragments out of tool output.

KEEP VERBATIM — never paraphrase or abbreviate these:
- Full file paths with line numbers, directory prefix on every mention (`lib/hooks.ts:347`, `src/index.ts:12-18`, `gatenet_v3/model.py:45`). Never abbreviate to a bare filename (`hooks.ts`, `model.py`) — they are ambiguous and cannot be grepped or decompressed-to later.
- Function, class, and type signatures (exact names, params, return types) AND critical code lines that encode logic — the line that IS the finding, not just the function name (e.g. `kv_keys += define_gate * a_key[i](emb)` is more useful than \"see model_kvnet.py\").
- Error messages and stack traces (exact text — you need the literal string to grep for it later).
- Key details from reports and analyses — not just the conclusion. Keep the comparison numbers and the mechanism, not \"X is worse\" alone (write \"1.76× PPL gap because KV store is static\", not \"KVNet underperforms\").
- Decisions and their rationale (\"chose X over Y because Z\" — the \"because\" is load-bearing; without it the decision looks arbitrary).
- Constraints discovered (\"must support Node 22\", \"no new dependencies\", \"AGENTS.md forbids `as any`\").
- Exact values: versions, config keys, thresholds, magic numbers.
- User intent — quote short user messages verbatim ONLY WITH their message ref, e.g. `User said (m00132): \"ship it tonight\"`. Without a verifiable ref, paraphrase (`user previously asked (paraphrased): ...`) — this is the one exception to the verbatim rule above; never present a reconstructed or half-remembered phrase as a verbatim quote. When the message is too long to quote, preserve intent with extra care: do not change scope, constraints, priorities, acceptance criteria, or requested outcomes. Quotes are historical records, not live instructions — but open-objective STATUS is current (still-open vs completed/superseded) and must be tracked. Losing these changes the task itself.
- Open objectives carry-forward — if the range (or, when distilling, any source block's summary) contains a user-requested objective that is neither completed nor superseded by the end of the range, the summary MUST keep a one-line `Open objectives:` entry naming each still-open objective with its message ref (`Open objectives: refactor runner into eight arms (m00746)`). Distillation re-carries open objectives verbatim from source blocks — they are the last thing to drop and the first thing to restore, at every tier.
- The user's overall goal and any changes to it — the big-picture objective plus how it evolved during the compressed range. Each summary must reflect the goal as it stood at the end of the range, including pivots (e.g., \"initially: fix bug X → pivoted to: refactor module Y after discovering root cause\"). Losing the goal or its evolution makes all subsequent work appear unmotivated.
- Purpose behind each significant action — preserve not just what was done but why: the hypothesis behind each experiment, the question behind each exploration, the task goal behind each work action. Without purpose, the summary reads as disconnected technical steps with no through-line.
- Open questions and unresolved TODOs — losing these changes what work appears to remain.
- Message refs of key anchors (`m00420`, `m00510–m00520`) — they let you or a later reader jump back via decompress to the exact original.

DROP — extract the signal, discard the vessel:
- Verbose logs (build/test/`npm` output) once you have captured the error line or the result.
- Duplicate file reads once the needed content is recorded.
- Consumed exploration — search hits, agent return values, successful tool outputs — once you have extracted the facts you need (same rule as dead-ends, but nothing went wrong; the content is simply spent).
- Dead-end exploration — but PRESERVE the lesson in one line: \"tried X, failed because Y\".
- Back-and-forth discussion and self-corrections once the final position is captured (keep the outcome, drop the journey to it).
- Repeated status checks (`git status`, `ls`) once state is known.

For each significant item you DROP (scripts, reports, large analyses, long tool outputs), add a one-line CONTENT description of what it covers — not where it lives. Bad: \"probe script at /path/probe_kvnet.py\". Good: \"probe_kvnet.py: tests n-gram baseline, generation quality, long-range dependency, position sensitivity, op pipeline, QUERY attention.\" This lets a later decompress target the right block by relevance, not by guessing locations.

PRIORITY — when the summary must be compact, preserve in this order:
1. User's overall goal, goal evolution, intent, and hard constraints (losing these changes the task).
2. Decisions and rationale.
3. Exact technical artifacts: paths, signatures, errors, values.
4. Conclusions and key findings.
5. Lessons learned: what failed and why.

Write dense, scannable bullets — not narrative prose. If the range spans distinct concerns (request → findings → decision), group bullets under short thematic headers so a reader can scan to the part they need. Every line must earn its place. Do not mimic the style of existing summaries in context; follow these rules.";

pub const TIER2_DISTILL_RULES: &str = "TIER 2 COMPRESSION — DISTILLATION

You are compressing historical summaries (not raw conversation). These summaries have already captured the details. Your job is to DISTILL them: extract only what matters for future work, discard the process.

KEEP — these are the only things that survive distillation:
- Decisions and their rationale (\"chose X over Y because Z\" — the \"because\" is load-bearing).
- Final outcomes: version numbers shipped, PR numbers merged/closed, bugs fixed or deferred.
- Key lessons: what failed and why (\"tried X, failed because Y\"). These prevent repeating mistakes.
- Critical constraints discovered (\"must support Node 22\", \"AGENTS.md forbids as any\").
- Design decisions with architectural impact (\"chose compress-as-anchor over synthetic messages because prefix cache\").
- User quotes and task state only as attributed history: keep the source ref with any user quote; never carry an UNVERIFIED tier-1 \"CURRENT TASK\" claim forward as a live directive — relabel it \"TASK AS OF THIS BLOCK\". A ref-backed objective that no later source marks completed or superseded is not unverified: it carries in the `Open objectives:` entry (next bullet), not as a directive.
- Open objectives — if ANY source block's summary names a user-requested objective that no later source block marks completed or superseded, the distilled summary MUST keep a one-line `Open objectives:` entry re-carrying each still-open objective verbatim with its original ref. They are the last thing to drop and the first thing to restore.
- Whether content is OBSOLETE or SUPERSEDED — mark with one line: \"[SUPERSEDED by PR #NNN]\" or \"[OBSOLETE: deleted in vX.Y.Z]\". Do NOT keep the obsolete content's details — just the marker and reason.
- Function/class/type names and module paths that are the SUBJECT of the work — e.g., \"fixed filterCompressedRanges in prune.ts\", \"added SessionStateRegistry in state.ts\". Not exact line numbers or full signatures — just enough to LOCATE the code without searching.
- Exploration findings: if a block was exploratory with no decision, keep the CONCLUSION in one line (\"explored X, not viable because Y\"). Do not keep the exploration process.

DROP — these were useful during the work but are no longer needed:
- Exact line numbers, diffs, verbose function signatures, full code listings.
- Build/deploy process details, test execution steps.
- Review process details (who reviewed, what rounds, test counts).
- Verbose logs, command output, intermediate debugging steps.

FORMAT:
- Start each distilled block with a source header line:
  `Source: bN+bM+... (XK→YK tok, Zx). [original topic]`
  Example: `Source: b5+b7 (56K+44K→268 tok, 375x). [Tool-result recap + publish]`
- 3-5 bullet points per source block, each a self-contained fact.
- Dense, scannable — no narrative prose.
- Start with the outcome, not the process: \"v1.13.0 shipped (7 PRs bundled)\" not \"implemented 7 PRs then reviewed then merged\".
- Cross-block synthesis: if multiple source blocks cover the same topic (same PR, same feature, same bug), MERGE them into a single group of bullets. Do not repeat the same fact from different blocks — keep it once under the most relevant source header.

SIZE TARGET: 50-150 tokens per source block (excluding the header). If you can't fit it in 150 tokens, you're keeping too much process. If a block has nothing worth keeping (pure noise), output just the header followed by \"[no actionable content].\"";

pub const TIER3_CONDENSE_RULES: &str = "TIER 3 COMPRESSION — ULTRA-CONDENSATION

You are compressing distilled summaries (Tier 2) into ultra-condensed facts (Tier 3). The distilled summaries already contain only decisions and outcomes. Your job is to reduce them to bare factual references.

PRIORITY — when a source block has more facts than the size target allows, keep in this order:
1. Shipped outcomes (versions released, PRs merged) — these are permanent record.
2. Open work — PRs/issues still pending AND still-open user-requested objectives (re-carry any source block's `Open objectives:` entries verbatim); these may need follow-up.
3. Key decisions with architectural impact (\"chose X over Y because Z\").
4. Critical constraints (\"must support Node 22\").
Drop everything else. Tier 3 is a lookup index, not a knowledge base.

FORMAT:
- Start with a source header line:
  `Source: bN+bM+... (XK→YK tok, Zx). [original topic]`
- Output 1-3 facts per source block. Each fact is a single line: subject + outcome.
- No explanations, no rationale, no process — just the fact.
- Format: \"[PR/Issue/Version] — [outcome in ≤8 words]\"
- Merge related facts from different source blocks if they concern the same topic.

EXAMPLES:
- \"v1.13.0 shipped — quality gate + GC fix (7 PRs)\"
- \"PR #196 merged — preserve-first-user (supersedes #169)\"
- \"Bug 1214 fixed — compress consumed all user messages\"
- \"Objective (m00746) — eight-arm runner refactor, still open\"
- \"Chose compress-as-anchor — prefix cache benefit over synthetic injection\"
- \"Constraint: AGENTS.md forbids as any — never suppress types\"

DROP:
- Multi-sentence context. If a fact needs >1 sentence, it's too detailed for Tier 3.
- Lessons learned (\"tried X, failed because Y\") — drop UNLESS the failure is likely to recur and the block is <30 days old.
- Design rationale details — keep the decision, drop the \"because\" unless it's a critical constraint.
- Anything marked [OBSOLETE] or [SUPERSEDED] — drop entirely, note \"[N blocks obsolete]\" in the summary.

SIZE TARGET: 30-60 tokens per source block (including header). For a batch of N source blocks, total output ≈ N × 40 tokens. If a source block has only one trivial fact, output just the header + one line.";

/// The four load-bearing rule texts. One struct instead of four parameters keeps
/// the call sites honest: a host that customises one rule cannot silently keep
/// the kernel default for the other three and assume they agree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prompts {
    pub compress_philosophy: String,
    pub how_to_compress_rules: String,
    pub tier2_distill_rules: String,
    pub tier3_condense_rules: String,
}

impl Default for Prompts {
    fn default() -> Self {
        Self::canonical()
    }
}

/// Proof that the caller was told overriding the rules can degrade summaries.
///
/// A `bool` parameter would be enough for a program that always passes the same
/// literal; a distinct type makes the risk visible in every signature that can
/// accept an override, and makes "just make it configurable" impossible to do by
/// accident.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RiskAcknowledged(());

impl RiskAcknowledged {
    /// State, at the call site, that you accept degraded summary quality.
    pub const fn yes() -> Self {
        Self(())
    }
}

impl Prompts {
    /// The kernel's canonical rules, built at runtime because the texts are long
    /// enough that a const-fn `String` would need an allocation strategy nobody
    /// wants to maintain.
    pub fn canonical() -> Self {
        Self {
            compress_philosophy: COMPRESS_PHILOSOPHY.to_string(),
            how_to_compress_rules: HOW_TO_COMPRESS_RULES.to_string(),
            tier2_distill_rules: TIER2_DISTILL_RULES.to_string(),
            tier3_condense_rules: TIER3_CONDENSE_RULES.to_string(),
        }
    }

    /// Append the language-preservation clause to all four texts.
    ///
    /// Additive only — it cannot remove or replace load-bearing wording, which is
    /// why it does not require [`RiskAcknowledged`]. `philosophy` gets it as a
    /// bullet (it is a bulleted list); the rule texts get it as their own
    /// paragraph.
    pub fn with_language_preservation(mut self) -> Self {
        self.compress_philosophy
            .push_str(&format!("\n- {LANGUAGE_PRESERVATION_RULE}"));
        for rule in [
            &mut self.how_to_compress_rules,
            &mut self.tier2_distill_rules,
            &mut self.tier3_condense_rules,
        ] {
            rule.push_str("\n\n");
            rule.push_str(LANGUAGE_PRESERVATION_RULE);
        }
        self
    }

    /// Overlay host text onto the canonical rules.
    ///
    /// Empty overrides are dropped rather than written: a blank rule would remove
    /// the guidance entirely, which is the one outcome nobody wants from a
    /// "customise the wording" API.
    pub fn resolve(
        overrides: PromptsOverrides,
        acknowledged: Option<RiskAcknowledged>,
    ) -> Result<Self, PromptError> {
        let mut prompts = Self::canonical();
        let mut touched: Vec<&'static str> = Vec::new();

        if let Some(v) = overrides.compress_philosophy.filter(|s| !s.trim().is_empty()) {
            prompts.compress_philosophy = v;
            touched.push("compress_philosophy");
        }
        if let Some(v) = overrides
            .how_to_compress_rules
            .filter(|s| !s.trim().is_empty())
        {
            prompts.how_to_compress_rules = v;
            touched.push("how_to_compress_rules");
        }
        if let Some(v) = overrides.tier2_distill_rules.filter(|s| !s.trim().is_empty()) {
            prompts.tier2_distill_rules = v;
            touched.push("tier2_distill_rules");
        }
        if let Some(v) = overrides
            .tier3_condense_rules
            .filter(|s| !s.trim().is_empty())
        {
            prompts.tier3_condense_rules = v;
            touched.push("tier3_condense_rules");
        }

        if !touched.is_empty() && acknowledged.is_none() {
            return Err(PromptError { fields: touched });
        }
        Ok(prompts)
    }
}

/// Partial override of the rule texts. Every field is optional so a host can
/// replace exactly one rule; `None` keeps the kernel's text.
#[derive(Clone, Debug, Default)]
pub struct PromptsOverrides {
    pub compress_philosophy: Option<String>,
    pub how_to_compress_rules: Option<String>,
    pub tier2_distill_rules: Option<String>,
    pub tier3_condense_rules: Option<String>,
}

impl PromptsOverrides {
    pub fn new() -> Self {
        Self::default()
    }
}

/// The caller tried to override a quality-critical rule without acknowledging the
/// risk. Same refusal upstream makes by throwing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PromptError {
    pub fields: Vec<&'static str>,
}

impl std::fmt::Display for PromptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "overriding compression rules requires an explicit acknowledgement of the quality risk; \
             overridden fields: {}. These rules are tuned over months of production use: summaries \
             that lose paths, signatures, or decisions cannot be decompressed back into usable \
             context.",
            self.fields.join(", ")
        )
    }
}

impl std::error::Error for PromptError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_rules_carry_the_amendments() {
        // Three incidents, each of which must survive transcription.
        assert!(HOW_TO_COMPRESS_RULES.contains("TASK AS OF THIS BLOCK"));
        assert!(HOW_TO_COMPRESS_RULES.contains("Open objectives:"));
        assert!(HOW_TO_COMPRESS_RULES.contains("never copy \\uXXXX escape sequences"));
        // The tier texts must re-carry objectives too, or distillation drops them.
        assert!(TIER2_DISTILL_RULES.contains("Open objectives:"));
        assert!(TIER3_CONDENSE_RULES.contains("Open objectives:"));
    }

    #[test]
    fn philosophy_is_a_bulleted_block() {
        assert!(COMPRESS_PHILOSOPHY.starts_with("Compression Philosophy:\n- "));
        assert!(COMPRESS_PHILOSOPHY.contains("Compress by need, not by percentage."));
    }

    #[test]
    fn overriding_without_acknowledgement_is_refused() {
        let overrides = PromptsOverrides {
            how_to_compress_rules: Some("just wing it".into()),
            ..Default::default()
        };
        let err = Prompts::resolve(overrides, None).unwrap_err();
        assert_eq!(err.fields, vec!["how_to_compress_rules"]);
        assert!(err.to_string().contains("quality risk"));
    }

    #[test]
    fn acknowledged_override_lands() {
        let overrides = PromptsOverrides {
            compress_philosophy: Some("Be brief.".into()),
            ..Default::default()
        };
        let prompts = Prompts::resolve(overrides, Some(RiskAcknowledged::yes())).unwrap();
        assert_eq!(prompts.compress_philosophy, "Be brief.");
        // Untouched rules keep the canonical text.
        assert_eq!(prompts.how_to_compress_rules, HOW_TO_COMPRESS_RULES);
    }

    #[test]
    fn blank_overrides_are_ignored_not_written() {
        let overrides = PromptsOverrides {
            tier2_distill_rules: Some("   ".into()),
            ..Default::default()
        };
        let prompts = Prompts::resolve(overrides, None).expect("a blank override is not an override");
        assert_eq!(prompts.tier2_distill_rules, TIER2_DISTILL_RULES);
    }

    #[test]
    fn language_preservation_is_additive() {
        let plain = Prompts::canonical();
        let tagged = Prompts::canonical().with_language_preservation();
        assert!(tagged.compress_philosophy.starts_with(&plain.compress_philosophy));
        assert!(tagged
            .compress_philosophy
            .contains(LANGUAGE_PRESERVATION_RULE));
        assert!(tagged
            .how_to_compress_rules
            .contains(LANGUAGE_PRESERVATION_RULE));
        // The canonical set stays byte-stable: that is the whole point of the opt-in.
        assert!(!plain.compress_philosophy.contains(LANGUAGE_PRESERVATION_RULE));
    }
}

import fs from "node:fs";
import {
  createCore,
  defaultConfig,
  createInitialState,
} from "file:///D:/GITHUB/billion-context/kernel/src/index.ts";

type M = {
  id: string;
  role: string;
  contentType: string;
  text?: string;
  toolName?: string;
  toolCallId?: string;
};

const scenarioPath = process.argv[2];
const outPath = process.argv[3];
const sc = JSON.parse(fs.readFileSync(scenarioPath, "utf8"));

// Optional 4th argument "inject" swaps in a synthetic counter, mirroring the
// Rust side's `with_estimator`. Same arithmetic on both sides: identical output
// means the estimator reaches every counting site the same way.
const inject = process.argv[4] === "inject";
const synthetic = (text: string): number => {
  let sum = 0;
  for (const ch of text) sum += ch.codePointAt(0)!;
  return Math.ceil(text.length / 3) + (sum % 11);
};

const core = createCore(inject ? { countTokens: synthetic } : undefined);
const cfg = defaultConfig(sc.contextLimit);
// Optional per-scenario overrides. The Rust harness reads the same object, so a
// tier scenario pins identical arbitration on both sides.
{
  const c = sc.config ?? {};
  if (c.tiers) Object.assign(cfg.tiers, c.tiers);
  if (c.nudge) {
    const { tierGrowthTokens, ...rest } = c.nudge;
    Object.assign(cfg.nudge, rest);
    if (tierGrowthTokens) cfg.nudge.tierGrowthTokens = { ...tierGrowthTokens };
  }
  if (c.preserveRecentMessages !== undefined) {
    cfg.preserveRecentMessages = c.preserveRecentMessages;
  }
  if (c.preserveRecentTokens !== undefined) {
    cfg.preserveRecentTokens = c.preserveRecentTokens;
  }
}
let state = createInitialState();
let lastView: M[] = [];
// The host's own transcript, unrendered. `applyCompression` must see this, not
// the rendered view: the rendered view carries the `<acp …>` tags, and counting
// those as compressed content inflates the block by the tag cost.
let lastRaw: M[] = [];

const expand = (m: any): M => {
  const text = m.repeat ? String(m.text).repeat(m.repeat) : m.text;
  const msg: M = { id: m.id, role: m.role, contentType: m.kind, text };
  if (m.toolName !== undefined) msg.toolName = m.toolName;
  if (m.toolCallId !== undefined) msg.toolCallId = m.toolCallId;
  return msg;
};

const normBlock = (b: any) => ({
  blockId: b.blockId,
  tier: b.tier,
  topic: b.topic ?? null,
  summary: b.summary,
  active: b.active,
  startRef: b.startRef ?? null,
  endRef: b.endRef ?? null,
  compressedTokens: b.compressedTokens,
  survived: b.survivedCount,
  generation: b.generation,
});

const normRefs = (st: any) =>
  Object.entries(st.messageRefs.byRaw)
    .map(([k, v]) => [k, v as string])
    .sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));

const steps: any[] = [];

for (const st of sc.steps) {
  if (st.op === "turn") {
    const messages = st.messages.map(expand);
    const res = core.processTurn({
      messages: messages as never,
      state,
      config: cfg,
      tokenCount: st.tokenCount,
    });
    state = res.state;
    lastView = res.messages as M[];
    lastRaw = messages as M[];
    steps.push({
      op: "turn",
      view: res.messages.map((m: M) => ({
        id: m.id,
        role: m.role,
        kind: m.contentType,
        text: m.text ?? null,
      })),
      refs: normRefs(state),
      blocks: state.blocks.map(normBlock),
      nudge: res.nudge
        ? {
            inject: res.nudge.shouldInject,
            reason: res.nudge.reason,
            tier: res.nudge.tier ?? null,
            ranges: res.nudge.compressibleRanges.map((r: any) => ({
              startRef: r.startRef,
              endRef: r.endRef,
              count: r.count,
              tokens: r.tokens,
            })),
          }
        : null,
      truncationSkipped: res.truncationSkipped ?? null,
      terminal: res.terminalEscape ? res.terminalEscape.message : null,
    });
  } else if (st.op === "apply") {
    const res = core.applyCompression({
      ranges: st.ranges,
      messages: lastRaw as never,
      state,
      config: cfg,
    });
    state = res.state;
    steps.push({
      op: "apply",
      blocksCreated: res.result.blocksCreated,
      tokensCompressed: res.result.tokensCompressed,
      errors: res.result.errors,
      warnings: res.result.warnings,
      blocks: state.blocks.map(normBlock),
      refs: normRefs(state),
    });
  } else if (st.op === "decompress") {
    const res: any = core.decompress(st.blocks[0], state);
    steps.push({
      op: "decompress",
      block: res ? normBlock(res) : null,
      blocks: state.blocks.map(normBlock),
      refs: normRefs(state),
    });
  } else if (st.op === "search") {
    const res: any[] = core.search(st.query, state);
    steps.push({
      op: "search",
      hits: res.map((b) => ({ blockId: b.blockId, tier: b.tier, topic: b.topic ?? null })),
    });
  } else if (st.op === "status") {
    const res: any = core.status(state, st.tokenCount ?? 0, cfg);
    steps.push({
      op: "status",
      totalBlocks: res.totalBlocks,
      activeBlocks: res.activeBlocks,
      tokensCompressed: res.tokensCompressed,
      contextUsage: res.contextUsage,
      modelContextLimit: res.modelContextLimit,
    });
  }
}

fs.writeFileSync(outPath, JSON.stringify({ steps }, null, 2));
console.log("wrote", outPath, "steps:", steps.length);

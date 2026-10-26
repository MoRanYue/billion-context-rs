// Compares the reference kernel's token counts against the Rust estimator.
//
//   node diff/compare-tok.mjs diff/tok-ts.json diff/tok-rust.json
//
// Exit code is 1 when any sample disagrees, so this is usable as a check.
import fs from "node:fs";

const [, , tsPath, rustPath] = process.argv;
if (!tsPath || !rustPath) {
  console.error("usage: compare-tok.mjs <ts.json> <rust.json>");
  process.exit(2);
}

const ts = JSON.parse(fs.readFileSync(tsPath, "utf8"));
const rust = JSON.parse(fs.readFileSync(rustPath, "utf8"));

const show = (s) => {
  const esc = JSON.stringify(s);
  return esc.length > 46 ? esc.slice(0, 43) + '…"' : esc;
};

let diffs = 0;
const n = Math.min(ts.counts.length, rust.counts.length);
if (ts.counts.length !== rust.counts.length) {
  console.log(
    `length mismatch: ts ${ts.counts.length} vs rust ${rust.counts.length}`,
  );
  diffs++;
}
for (let i = 0; i < n; i++) {
  const a = ts.counts[i];
  const b = rust.counts[i];
  if (a !== b) {
    diffs++;
    console.log(`  #${i} ${show(ts.corpus[i])}: ts ${a} vs rust ${b}`);
  }
}

console.log(`samples: ${n}  mismatches: ${diffs}`);

// With a real vocabulary in play, show how far the heuristic is from it — the
// number that decides whether a host should opt into `hf-tokenizer`.
if (Array.isArray(rust.exact) && rust.exact.length === n) {
  let worst = { pct: 0, i: -1 };
  let sumTs = 0;
  let sumExact = 0;
  let over = 0;
  let under = 0;
  for (let i = 0; i < n; i++) {
    const exact = rust.exact[i];
    const heur = ts.counts[i];
    sumTs += heur;
    sumExact += exact;
    if (exact === 0) continue;
    const pct = ((heur - exact) / exact) * 100;
    if (Math.abs(pct) > Math.abs(worst.pct)) worst = { pct, i };
    if (pct > 0) over++;
    else if (pct < 0) under++;
  }
  const agg = ((sumTs - sumExact) / sumExact) * 100;
  console.log(
    `heuristic vs real vocabulary: total ${sumTs} vs ${sumExact} ` +
      `(${agg.toFixed(1)}%), over on ${over}, under on ${under}`,
  );
  if (worst.i >= 0) {
    console.log(
      `  worst: #${worst.i} ${show(ts.corpus[worst.i])} ` +
        `${worst.pct > 0 ? "+" : ""}${worst.pct.toFixed(1)}%`,
    );
  }
}

process.exit(diffs === 0 ? 0 : 1);

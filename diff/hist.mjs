// Group compare.mjs differences by the field path they land on, so a large
// diff count collapses into the handful of decisions that actually differ.
import { execFileSync } from "node:child_process";

const [a = "out-ts.json", b = "out-rust.json"] = process.argv.slice(2);
const raw = execFileSync(process.execPath, ["compare.mjs", a, b, "500"], {
  encoding: "utf8",
});

const lines = raw.split("\n").filter((l) => l.includes(" vs "));
const hist = new Map();
for (const line of lines) {
  const path = line.trim().split(":")[0] ?? "?";
  // $.steps[2].nudge.ranges[1].count -> steps[].nudge.ranges[].count
  const key = path
    .replace(/^\$\./, "")
    .replace(/\[\d+\]/g, "[]")
    .replace(/^steps\[\]\./, "steps[].");
  hist.set(key, (hist.get(key) ?? 0) + 1);
}

console.log(`total ${lines.length}`);
for (const [k, n] of [...hist].sort((x, y) => y[1] - x[1])) {
  console.log(String(n).padStart(4), k);
}

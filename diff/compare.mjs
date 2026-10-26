import fs from "node:fs";

const [, , pathA, pathB, limitArg] = process.argv;
const a = JSON.parse(fs.readFileSync(pathA, "utf8"));
const b = JSON.parse(fs.readFileSync(pathB, "utf8"));
const limit = Number(limitArg ?? 40);

const diffs = [];
const push = (m) => diffs.push(m);

function walk(x, y, path) {
  if (x === undefined && y === undefined) return;
  if (x === undefined) return push(`${path}: missing on A, B=${JSON.stringify(y).slice(0, 90)}`);
  if (y === undefined) return push(`${path}: missing on B, A=${JSON.stringify(x).slice(0, 90)}`);
  if (x === null || y === null) {
    if (x !== y) push(`${path}: ${JSON.stringify(x)} vs ${JSON.stringify(y)}`);
    return;
  }
  if (typeof x !== typeof y) return push(`${path}: type ${typeof x} vs ${typeof y}`);
  if (typeof x !== "object") {
    if (x !== y) push(`${path}: ${JSON.stringify(x).slice(0, 90)} vs ${JSON.stringify(y).slice(0, 90)}`);
    return;
  }
  const ax = Array.isArray(x);
  const ay = Array.isArray(y);
  if (ax !== ay) return push(`${path}: array vs object`);
  if (ax) {
    if (x.length !== y.length) push(`${path}.length: ${x.length} vs ${y.length}`);
    for (let i = 0; i < Math.min(x.length, y.length); i++) walk(x[i], y[i], `${path}[${i}]`);
    return;
  }
  for (const k of new Set([...Object.keys(x), ...Object.keys(y)])) walk(x[k], y[k], `${path}.${k}`);
}

walk(a, b, "$");

console.log(`diffs: ${diffs.length}`);
for (const d of diffs.slice(0, limit)) console.log("  " + d);
if (diffs.length > limit) console.log(`  ... ${diffs.length - limit} more`);

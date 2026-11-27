// Acceptance probe: the installed billion-context DSH native shim must plan
// "attach" (not "spawn") when BILLION_CONTEXT_PROXY points at the Rust host,
// and verifyAttachAndRecover must land on that origin.
//
// Usage: set BILI_SHIM_ROOT, then: node shim-attach.mjs http://127.0.0.1:18116
import path from "node:path";
import { pathToFileURL } from "node:url";

const origin = process.argv[2];
const shimRoot = process.env.BILI_SHIM_ROOT;
if (!shimRoot) {
  console.error("set BILI_SHIM_ROOT to the installed billion-context package directory");
  process.exit(2);
}
const {
  planNativeDsh,
  verifyAttachAndRecover,
  _registerStateForTest,
} = await import(pathToFileURL(path.join(shimRoot, "dist/agent/dsh-native.js")).href);
if (!origin) {
  console.error("usage: node shim-attach.mjs <rust-origin>");
  process.exit(2);
}
process.env.BILLION_CONTEXT_PROXY = origin;

const plan = planNativeDsh(process.env);
const attached = await verifyAttachAndRecover(plan.attachOrigin);
const manifest = await fetch(`${origin}/__bili/plugin/manifest`).then((r) => r.json());
const status = await fetch(`${origin}/__bili/status`).then((r) => r.json());
const result = {
  plan,
  attached,
  manifestProtocol: manifest.protocolVersion,
  manifestTools: manifest.toolNames,
  proxyVersion: status.version,
  register: _registerStateForTest(),
  ok: plan.mode === "attach" && attached === origin && status.version === "0.1.217",
};
console.log(JSON.stringify(result, null, 2));
process.exitCode = result.ok ? 0 : 1;

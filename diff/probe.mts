import { createCore, defaultConfig, createInitialState } from "file:///D:/GITHUB/billion-context/kernel/src/index.ts";

const core = createCore();
const cfg = defaultConfig(100_000);

console.log("core keys:", Object.keys(core).sort().join(","));

const messages = [
  { id: "a", role: "user", contentType: "text", text: "hello " + "x".repeat(40) },
  { id: "b", role: "assistant", contentType: "text", text: "world " + "y".repeat(40) },
  { id: "c", role: "user", contentType: "text", text: "again " + "z".repeat(40) },
];

const state = createInitialState();

const res = core.processTurn({
  messages: messages as never,
  state,
  config: cfg,
  tokenCount: 1234,
});

console.log("result keys:", Object.keys(res).sort().join(","));
console.log(JSON.stringify(res, null, 2).slice(0, 4000));

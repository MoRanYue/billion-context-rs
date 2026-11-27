//! billion-context-proxy — Rust replacement for the billion-context Node proxy host.
//!
//! Accepts model traffic rewritten to `/bili/<absolute-url>`, converts OpenAI
//! Chat Completions / Anthropic Messages into the billion-context kernel,
//! returns the folded projection, and relays streaming SSE bodies unchanged.
//! Publishes the plugin manifest/tool/status surface plus a lightweight admin
//! panel, and writes the `proxy-origin` / `instances/<id>.json` markers in the
//! Node layout so a native launcher can discover a live instance.

#![recursion_limit = "256"]

use std::{
    collections::{HashMap, HashSet},
    io::{self, Write},
    net::{IpAddr, SocketAddr},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use axum::{
    Router,
    body::{Body, Bytes},
    extract::{ConnectInfo, OriginalUri, Query, Request, State},
    http::{HeaderMap, HeaderValue, Method, StatusCode, header},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{any, get, post},
};
use serde::Serialize;
use serde_json::json;

mod anthropic;
mod artifact;
mod build_id;
mod d1b_runner;
mod dsh_compaction;
mod openai;
mod pricing;
mod prune;
#[cfg(test)]
mod replay;
mod session_report;
mod session_store;
mod summarizer;
mod summary_carry;
#[cfg(test)]
mod test_env;
mod tool_runner;
mod usage;
mod usage_ledger;

use tokio::{net::TcpListener, sync::watch, time};
use uuid::Uuid;

use serde_json::Value;

/// The package version. Kept as a local alias so the many `json!` literals below
/// stay terse; the single source of truth is `build_id::VERSION`.
const VERSION: &str = build_id::VERSION;

const PANEL_HTML: &str = r#"<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>billion-context · rust host</title>
<style>
:root{color-scheme:dark;--bg:#0b0e14;--panel:#141a24;--line:#232c3a;--text:#dbe4f0;--dim:#8291a6;--accent:#6ea8fe;--good:#5dd39e;--warn:#f4c95d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 "Segoe UI",system-ui,sans-serif}
header{display:flex;align-items:center;justify-content:space-between;padding:18px 26px;border-bottom:1px solid var(--line);background:linear-gradient(180deg,#131a26,#0b0e14)}
h1{margin:0;font-size:19px;letter-spacing:.2px}.logo{color:var(--accent);font-size:22px}.ver{color:var(--dim);font-weight:400;font-size:12px;margin-left:10px}
#clock{color:var(--dim);font-variant-numeric:tabular-nums}
main{padding:22px 26px;max-width:1500px;margin:0 auto}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-bottom:20px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.card .k{color:var(--dim);font-size:12px;text-transform:uppercase;letter-spacing:.08em}
.card .v{font-size:24px;font-weight:600;margin-top:4px;font-variant-numeric:tabular-nums}
.grid{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(360px,.75fr);gap:16px}
section{background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden}
section h2{margin:0;padding:13px 16px;font-size:14px;border-bottom:1px solid var(--line);color:var(--dim);font-weight:600;display:flex;justify-content:space-between}
#sessions{max-height:calc(100vh - 330px);overflow:auto}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th,td{padding:9px 12px;text-align:left;border-bottom:1px solid #18202c;white-space:nowrap}
th{color:var(--dim);font-weight:500;font-size:12px;position:sticky;top:0;background:#111823}
tr.session{cursor:pointer}tr.session:hover{background:#1a2332}tr.session.sel{background:#1d2b44;box-shadow:inset 3px 0 var(--accent)}
.id{max-width:340px;overflow:hidden;text-overflow:ellipsis;display:inline-block;vertical-align:bottom}
.tag{display:inline-block;padding:1px 8px;border-radius:99px;border:1px solid var(--line);color:var(--dim);font-size:11px}
.bar{height:6px;background:#1b2430;border-radius:99px;overflow:hidden;width:80px;display:inline-block;vertical-align:middle;margin-right:8px}
.bar i{display:block;height:100%;background:var(--accent)}.bar.hot i{background:var(--warn)}
#detail{padding:14px 16px;max-height:calc(100vh - 330px);overflow:auto}
.meta{display:grid;grid-template-columns:1fr 1fr;gap:6px 14px;margin-bottom:12px}
.meta div span{color:var(--dim);font-size:12px;display:block}
pre{white-space:pre-wrap;word-break:break-word;background:#0e1420;border:1px solid var(--line);border-radius:8px;padding:10px;font-size:12px;color:#c7d3e3;max-height:260px;overflow:auto}
.block{border:1px solid var(--line);border-radius:8px;padding:8px 10px;margin-bottom:8px;background:#101724}
.block .top{display:flex;justify-content:space-between;color:var(--dim);font-size:12px}
.block .topic{color:var(--text);font-weight:600;margin:2px 0 6px}
.range{font-family:Consolas,monospace;font-size:12px;color:#9fb3cc;padding:2px 0}
.empty{color:var(--dim);padding:16px}
</style></head><body>
<header><h1><span class="logo">∞</span> billion-context <span class="ver" id="ver">rust host</span></h1><div id="clock"></div></header>
<main>
<div class="cards" id="cards"></div>
<div class="grid">
<section><h2>Conversations <span class="tag" id="count">0</span></h2><div id="sessions"><div class="empty">loading…</div></div></section>
<section><h2>Session detail</h2><div id="detail"><div class="empty">Select a conversation.</div></div></section>
</div>
</main>
<script>
const ORIGIN="__ORIGIN__";
let selected=null;
const fmt=n=>typeof n==="number"?n.toLocaleString("en-US"):(n??"—");
const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
function ago(ms){if(!ms)return "—";const d=Math.max(0,Date.now()-ms);if(d<6e4)return Math.floor(d/1e3)+"s ago";if(d<36e5)return Math.floor(d/6e4)+"m ago";if(d<864e5)return Math.floor(d/36e5)+"h ago";return Math.floor(d/864e5)+"d ago"}
async function jget(path){const r=await fetch(ORIGIN+path);if(!r.ok)throw new Error(await r.text());return r.json()}
function cards(list){
  const req=list.reduce((a,s)=>a+(s.requests||0),0);
  const tin=list.reduce((a,s)=>a+(s.inputTokens||0),0);
  const tc=list.reduce((a,s)=>a+(s.cachedTokens||0),0);
  const blk=list.reduce((a,s)=>a+(s.blocks||0),0);
  const hit=tin?Math.round(tc*100/(tin+tc)):0;
  document.getElementById("cards").innerHTML=[
    ["Conversations",list.length],["Requests",req],["Input tokens",tin],["Cached tokens",tc],["Cache hit",hit+"%"],["Blocks",blk]
  ].map(([k,v])=>`<div class="card"><div class="k">${k}</div><div class="v">${esc(fmt(v))}</div></div>`).join("");
}
function rows(list){
  const html=list.map(s=>{
    const pct=s.contextWindow?Math.min(999,Math.round((s.contextTokens||0)*100/s.contextWindow)):0;
    return `<tr class="session${s.id===selected?" sel":""}" data-id="${esc(s.id)}">
      <td><span class="id">${esc(s.id)}</span></td>
      <td><span class="tag">${esc(s.protocol||"anthropic")}</span></td>
      <td>${esc(s.lastModel||"—")}</td>
      <td><span class="bar${pct>=80?" hot":""}"><i style="width:${Math.min(100,pct)}%"></i></span>${pct}%</td>
      <td>${esc(fmt(s.contextTokens))}</td>
      <td>${esc(fmt(s.blocks))}</td>
      <td>${esc(ago(s.lastSeen))}</td></tr>`;
  }).join("");
  return html||'<div class="empty">No conversations yet.</div>';
}
async function load(){
  try{
    const stats=await jget("/__bili/stats");const list=stats.sessions||[];
    cards(list);document.getElementById("count").textContent=list.length;
    document.getElementById("sessions").innerHTML=`<table><thead><tr><th>Conversation</th><th>Wire</th><th>Model</th><th>Context</th><th>Tokens</th><th>Blocks</th><th>Last seen</th></tr></thead><tbody>${rows(list)}</tbody></table>`;
    document.querySelectorAll("tr.session").forEach(tr=>tr.onclick=()=>{selected=tr.dataset.id;load();detail(selected)});
    if(!selected&&list.length){selected=list[0].id;detail(selected);}
    if(selected)detail(selected,true);
  }catch(e){document.getElementById("sessions").innerHTML='<div class="empty">'+esc(e.message)+'</div>';}
}
async function detail(id,quiet){
  try{
    const s=await jget("/__bili/plugin/status?conversationId="+encodeURIComponent(id));
    const ranges=(s.compressibleRanges||[]).map(r=>`<div class="range">${esc(r.start)}–${esc(r.end)} · ${r.count} msgs · ${fmt(r.tokens)} tok · ${fmt(r.chars)} chars</div>`).join("")||'<div class="empty">none</div>';
    const blocks=(s.blocks||[]).map(b=>`<div class="block"><div class="top"><span>${esc(b.id)} · tier ${b.tier}</span><span>${b.active?"active":"inactive"} · ${fmt(b.tokens)} tok</span></div><div class="topic">${esc(b.topic||"(untitled)")}</div><div class="range">direct ${b.directCount} / effective ${b.effectiveCount}${b.span?" · "+esc(b.span[0])+"–"+esc(b.span[1]):""}</div></div>`).join("")||'<div class="empty">No blocks.</div>';
    const pct=s.contextWindow?Math.round((s.contextTokens||0)*100/s.contextWindow):0;
    document.getElementById("detail").innerHTML=`
      <div class="meta">
        <div><span>conversation</span>${esc(s.conversationId)}</div>
        <div><span>model</span>${esc(s.model||"—")}</div>
        <div><span>context</span>${pct}% · ${fmt(s.contextTokens)} / ${fmt(s.contextWindow)} (${esc(s.contextTokensSource)})</div>
        <div><span>sent-view</span>${fmt(s.tokens)} tokens</div>
        <div><span>requests</span>${fmt(s.requests)}</div>
        <div><span>billing</span>in ${fmt(s.inputTokens)} · cached ${fmt(s.cachedTokens)} · out ${fmt(s.outputTokens)}</div>
        <div><span>report</span><a href="/__bili/session-report?session=${encodeURIComponent(s.conversationId)}&amp;format=md">Markdown</a> · <a href="/__bili/session-report?session=${encodeURIComponent(s.conversationId)}&amp;format=svg">SVG</a></div>
        <div><span>blocks</span>${s.blockCount} total / ${s.activeBlocks} active · ${fmt(s.tokensCompressed)} tok folded</div>
        <div><span>nudge</span>${esc(s.panel&&s.panel.includes("Nudge: armed")?"armed":"idle")}</div>
      </div>
      <h2 style="margin:4px 0 8px;color:var(--dim);font-size:13px">Blocks</h2>${blocks}
      <h2 style="margin:14px 0 8px;color:var(--dim);font-size:13px">Compressible ranges</h2>${ranges}
      <h2 style="margin:14px 0 8px;color:var(--dim);font-size:13px">Host panel</h2><pre>${esc(s.panel||"(no panel text)")}</pre>`;
  }catch(e){if(!quiet)document.getElementById("detail").innerHTML='<div class="empty">'+esc(e.message)+'</div>';}
}
document.getElementById("clock").textContent=new Date().toLocaleTimeString();
load();setInterval(load,5000);
</script></body></html>"#;
const DEFAULT_PORT: u16 = 18_787;

struct AppState {
    client: reqwest::Client,
    origin: String,
    host: String,
    port: u16,
    instance_id: String,
    started_at: u64,
    passthrough: bool,
    debug: bool,
    parent_pid: Option<u32>,
    watchers: Mutex<HashSet<u32>>,
    shutdown_tx: watch::Sender<bool>,
    upstream: Mutex<Option<String>>,
    upstream_failures: Mutex<HashMap<String, UpstreamFailure>>,
    sessions: session_store::Store,
    /// Host-owned answer to "is this upstream a mock?".  `None` means the host
    /// derives it from the actual upstream endpoint.  A request header never
    /// changes this (D-2 requirement).
    usage_source: Option<usage_ledger::UsageSource>,
    /// P1 auto fallback: fold on the host's own initiative when the model
    /// ignores the nudge (or the window is nearly full). Opt-in because it
    /// bills a summarizer-chain call per attempt.
    auto_fallback: bool,
    /// Phase 3.1 speculative folds: pre-distill old T1 blocks while the window
    /// sits in the 45–75% band. Opt-in for the same reason: it bills a chain
    /// call, and its candidate can be discarded when the blocks move.
    speculation: bool,
    /// Per-instance token written by the launcher. Admin mutations require it
    /// when present; absent means standalone development mode (still guarded by
    /// the loopback Host and Origin checks).
    launch_token: Option<String>,
    /// Exact non-loopback panel origins allowed to call `/__bili/*`. Loopback
    /// origins are always accepted so the DSH web view keeps working.
    admin_origins: Vec<String>,
    /// Hosts the forward proxy may dial. Entries support exact names and
    /// `*.suffix` wildcards; literal IPs are refused unless explicitly allowed.
    upstream_allowlist: Vec<String>,
    allow_private_upstreams: bool,
}

#[derive(Clone)]
struct UpstreamFailure {
    count: u64,
    first_seen: u64,
    last_seen: u64,
    code: String,
}

#[derive(Debug, Clone)]
struct Options {
    host: String,
    port: u16,
    passthrough: bool,
    debug: bool,
    strict_port: bool,
    allow_remote_bind: bool,
    cleanup_artifacts: bool,
    usage_source: Option<usage_ledger::UsageSource>,
    /// Present only for the explicit `d1b-run` subcommand; the normal proxy
    /// path never constructs a runner.
    d1b: Option<d1b_runner::RunOptions>,
}

fn env_u16(name: &str) -> Option<u16> {
    std::env::var(name).ok().and_then(|v| v.trim().parse().ok())
}

fn env_str(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

fn opts_from_args() -> Options {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!(
            "billion-context-proxy {} — usage: billion-context-proxy [start] [--host H] [--port N] [--passthrough] [--debug] [--cleanup-artifacts] [--usage-source provider|mock]\n  d1b-run --d1b-upstream URL --d1b-model M [--d1b-provider NAME] [--d1b-prompt TEXT] [--d1b-max-calls N] [--d1b-max-input-tokens N] [--d1b-max-output-tokens N] [--d1b-timeout-ms N] [--d1b-cost-cap N] [--d1b-cost-currency CODE] [--d1b-require-cost-cap]\n  (--d1b-cost-cap implies a strict price check: without a usable price the run is refused)",
            build_id::BuildInfo::current().summary()
        );
        std::process::exit(0);
    }
    if args.iter().any(|a| a == "--version" || a == "-V") {
        print!("{}", build_id::BuildInfo::current().cli_text());
        std::process::exit(0);
    }
    let mut host = env_str("BILI_HOST").unwrap_or_else(|| "127.0.0.1".into());
    let mut port = env_u16("BILI_PORT").unwrap_or(DEFAULT_PORT);
    let mut passthrough = env_str("BILI_PASSTHROUGH").as_deref() == Some("1");
    let mut debug = env_str("BILI_DEBUG").as_deref() == Some("1");
    let mut strict_port = env_str("BILI_STRICT_PORT").as_deref() == Some("1");
    let mut allow_remote_bind = env_str("BILI_ALLOW_REMOTE_BIND").as_deref() == Some("1");
    let mut cleanup_artifacts = false;
    // Host-owned usage provenance: set once at startup, never per request.
    let mut usage_source =
        env_str("BILI_USAGE_SOURCE").and_then(|raw| usage_ledger::UsageSource::parse(&raw));
    let mut i = 0;
    let d1b_mode = args.iter().any(|arg| arg == "d1b-run");
    while i < args.len() {
        match args[i].as_str() {
            "start" | "d1b-run" => {}
            "--host" => {
                if let Some(v) = args.get(i + 1) {
                    host = v.clone();
                    i += 1;
                }
            }
            "--port" => {
                if let Some(v) = args.get(i + 1).and_then(|v| v.parse().ok()) {
                    port = v;
                    i += 1;
                }
            }
            "--passthrough" => passthrough = true,
            "--debug" => debug = true,
            "--strict-port" => strict_port = true,
            "--allow-remote-bind" => allow_remote_bind = true,
            "--cleanup-artifacts" => cleanup_artifacts = true,
            "--usage-source" => {
                if let Some(v) = args.get(i + 1) {
                    usage_source = usage_ledger::UsageSource::parse(v);
                    if usage_source.is_none() {
                        eprintln!(
                            "[billion-context-proxy] ignoring invalid --usage-source value: {v}"
                        );
                    }
                    i += 1;
                }
            }
            other if d1b_mode && other.starts_with("--d1b-") => {
                i += 1;
            }
            other => {
                // Unknown flags belong to the Node CLI surface we do not emulate
                // yet; ignore rather than fail so a hand-rolled spawn still boots.
                if !d1b_mode {
                    eprintln!("[billion-context-proxy] ignoring unrecognized argument: {other}");
                }
            }
        }
        i += 1;
    }
    let d1b = if args.iter().any(|arg| arg == "d1b-run") {
        match d1b_runner::parse_args(&args) {
            Ok(options) => Some(options),
            Err(err) => {
                eprintln!("[billion-context-proxy] d1b-run: {err}");
                std::process::exit(2);
            }
        }
    } else {
        None
    };
    Options {
        host,
        port,
        passthrough,
        debug,
        strict_port,
        allow_remote_bind,
        cleanup_artifacts,
        usage_source,
        d1b,
    }
}

fn home_dir() -> PathBuf {
    env_str("USERPROFILE")
        .or_else(|| env_str("HOME"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

fn state_dir() -> PathBuf {
    if let Some(v) = env_str("XDG_STATE_HOME") {
        return Path::new(&v).join("billion-context");
    }
    home_dir()
        .join(".local")
        .join("state")
        .join("billion-context")
}

fn config_path() -> PathBuf {
    if let Some(v) = env_str("BILI_CONFIG_FILE") {
        return PathBuf::from(v);
    }
    let base = env_str("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| home_dir().join(".config"));
    base.join("billion-context").join("billion-context.json")
}

/// Resolve the D1b credential binding from the fixed allowlist and verify the
/// configured provider entry matches it exactly. No credential file is
/// consulted and no arbitrary `apiKeyEnv` name is accepted.
fn d1b_provider_credential_env(provider: &str) -> Option<String> {
    let text = std::fs::read_to_string(config_path()).ok()?;
    let document: Value = serde_json::from_str(&text).ok()?;
    let recipe = summarizer::provider_recipe(&document, provider)?;
    d1b_runner::provider_binding_from_recipe(provider, &recipe).map(str::to_string)
}

fn atomic_write_text(path: &Path, text: &str) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_file_name(format!(
        "{}.{}.{}.tmp",
        path.file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("config"),
        std::process::id(),
        Uuid::new_v4()
    ));
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(text.as_bytes())?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn data_dir() -> PathBuf {
    if let Some(v) = env_str("XDG_DATA_HOME") {
        return Path::new(&v).join("billion-context");
    }
    home_dir()
        .join(".local")
        .join("share")
        .join("billion-context")
}

fn sessions_dir() -> PathBuf {
    env_str("BILI_SESSIONS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| data_dir().join("sessions"))
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn atomic_write_json(path: &Path, value: &impl Serialize) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_file_name(format!(
        "{}.{}.{}.tmp",
        path.file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("marker"),
        std::process::id(),
        Uuid::new_v4()
    ));
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(serde_json::to_string(value)?.as_bytes())?;
        f.write_all(b"\n")?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ProxyInstanceFile {
    origin: String,
    instance_id: String,
    pid: u32,
    started_at: u64,
    host: String,
    port: u16,
    passthrough: bool,
    mitm_domains: Vec<String>,
    model_windows: HashMap<String, u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_max_outputs: Option<HashMap<String, u64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    launch_token: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    lane: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    code_fingerprint: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct RegistryEntry {
    instance_id: String,
    pid: u32,
    port: u16,
    origin: String,
    started_at: u64,
    host: String,
    passthrough: bool,
    mitm_domains: Vec<String>,
    model_windows: HashMap<String, u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_max_outputs: Option<HashMap<String, u64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    lane: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    code_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sessions_dir: Option<String>,
}

fn parse_json_env_map(name: &str) -> HashMap<String, u64> {
    env_str(name)
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
        .and_then(|v| v.as_object().cloned())
        .map(|m| {
            m.into_iter()
                .filter_map(|(k, v)| v.as_u64().map(|n| (k, n)))
                .collect()
        })
        .unwrap_or_default()
}

fn mitm_domains_from_env() -> Vec<String> {
    env_str("BILI_MITM_DOMAINS")
        .map(|raw| {
            raw.split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

const DEFAULT_UPSTREAM_HOSTS: &[&str] = &[
    "api.anthropic.com",
    "api.openai.com",
    "generativelanguage.googleapis.com",
    "openrouter.ai",
    "api.deepseek.com",
    "api.moonshot.cn",
    "api.minimax.chat",
    "api.x.ai",
    "api.groq.com",
    "api.mistral.ai",
    "api.together.xyz",
    "api.cohere.ai",
];

fn upstream_allowlist_from_env(mitm_domains: &[String]) -> Vec<String> {
    let mut hosts: Vec<String> = DEFAULT_UPSTREAM_HOSTS
        .iter()
        .map(|host| host.to_ascii_lowercase())
        .collect();
    let mut extras: Vec<String> = mitm_domains.to_vec();
    if let Some(raw) = env_str("BILI_UPSTREAM_ALLOWLIST") {
        extras.push(raw);
    }
    for extra in &extras {
        for host in extra.split(',') {
            let host = host.trim().to_ascii_lowercase();
            if !host.is_empty() && !hosts.contains(&host) {
                hosts.push(host);
            }
        }
    }
    hosts
}

/// Only http(s) targets whose host is explicitly listed may be dialed. A
/// literal IP is accepted only when the operator opted in
/// (`BILI_ALLOW_PRIVATE_UPSTREAM=1`) or wrote that exact IP into the allowlist
/// (local mock upstreams).
fn upstream_target_allowed(target: &str, allowlist: &[String], allow_ip: bool) -> bool {
    let Ok(url) = reqwest::Url::parse(target) else {
        return false;
    };
    if !matches!(url.scheme(), "http" | "https")
        || !url.username().is_empty()
        || url.password().is_some()
    {
        return false;
    }
    let Some(host) = url.host_str() else {
        return false;
    };
    let host = host
        .trim_matches(|c| c == '[' || c == ']')
        .to_ascii_lowercase();
    if host.parse::<IpAddr>().is_ok() {
        return allow_ip || allowlist.iter().any(|entry| entry == &host);
    }
    allowlist.iter().any(|entry| {
        if let Some(suffix) = entry.strip_prefix("*.") {
            host.ends_with(&format!(".{suffix}"))
        } else {
            host == entry.as_str()
        }
    })
}

fn host_is_loopback(host: &str) -> bool {
    let host = host.trim().trim_start_matches('[').trim_end_matches(']');
    host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<IpAddr>()
            .map(|ip| ip.is_loopback())
            .unwrap_or(false)
}

/// A TCP peer is local when its address is loopback. An IPv4 client accepted
/// on a dual-stack IPv6 listener arrives as `::ffff:127.0.0.1`, which
/// `Ipv6Addr::is_loopback` alone does not recognize.
fn peer_is_loopback(peer: SocketAddr) -> bool {
    match peer.ip() {
        IpAddr::V4(ip) => ip.is_loopback(),
        IpAddr::V6(ip) => {
            ip.is_loopback() || ip.to_ipv4_mapped().is_some_and(|v4| v4.is_loopback())
        }
    }
}

fn host_header_is_loopback(headers: &HeaderMap) -> bool {
    let Some(raw) = headers
        .get(header::HOST)
        .and_then(|value| value.to_str().ok())
    else {
        return false;
    };
    let host = if let Some(rest) = raw.strip_prefix('[') {
        rest.split(']').next().unwrap_or("")
    } else {
        raw.split(':').next().unwrap_or("")
    };
    host_is_loopback(host)
}

fn origin_host(origin: &str) -> Option<String> {
    let url = reqwest::Url::parse(origin).ok()?;
    if !matches!(url.scheme(), "http" | "https") {
        return None;
    }
    url.host_str().map(|host| {
        host.trim_matches(|c| c == '[' || c == ']')
            .to_ascii_lowercase()
    })
}

fn admin_origins_from_env(origin: &str, port: u16) -> Vec<String> {
    let mut origins = vec![
        origin.to_string(),
        format!("http://127.0.0.1:{port}"),
        format!("http://localhost:{port}"),
        format!("http://[::1]:{port}"),
    ];
    if let Some(raw) = env_str("BILI_ADMIN_ORIGINS") {
        for item in raw.split(',') {
            let item = item.trim().trim_end_matches('/');
            if !item.is_empty() && !origins.iter().any(|known| known == item) {
                origins.push(item.to_string());
            }
        }
    }
    origins
}

/// A browser may read `/__bili/*` only from a loopback origin or an explicitly
/// configured panel origin. `Origin: null` and public web origins are denied.
fn admin_origin_allowed(origin: &str, state: &AppState) -> bool {
    state.admin_origins.iter().any(|allowed| allowed == origin)
        || origin_host(origin).is_some_and(|host| host_is_loopback(&host))
}

fn launch_token_ok(headers: &HeaderMap, expected: Option<&str>) -> bool {
    let Some(expected) = expected else {
        return true;
    };
    let presented = headers
        .get("x-bili-launch-token")
        .and_then(|value| value.to_str().ok())
        .or_else(|| {
            headers
                .get(header::AUTHORIZATION)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.strip_prefix("Bearer "))
        });
    presented.is_some_and(|token| token == expected)
}

fn admin_mutation(method: &Method, path: &str) -> bool {
    if matches!(*method, Method::GET | Method::HEAD | Method::OPTIONS) {
        return false;
    }
    // Internal liveness lanes (plugin tools and the owner watcher heartbeat)
    // are not the browser admin surface; they stay reachable so a session
    // launched with a token keeps compressing and stays watched.
    path != "/__bili/plugin/tool" && path != "/__bili/watcher"
}

fn add_cors_headers(response: &mut Response, origin: Option<&str>) {
    response
        .headers_mut()
        .insert(header::VARY, HeaderValue::from_static("Origin"));
    if let Some(origin) = origin {
        if let Ok(value) = HeaderValue::from_str(origin) {
            response
                .headers_mut()
                .insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, value);
        }
    }
    response.headers_mut().insert(
        header::ACCESS_CONTROL_ALLOW_METHODS,
        HeaderValue::from_static("GET, POST, PUT, DELETE, OPTIONS"),
    );
    response.headers_mut().insert(
        header::ACCESS_CONTROL_ALLOW_HEADERS,
        HeaderValue::from_static(
            "content-type, authorization, x-bili-launch-token, x-bili-plugin, \
             x-bili-plugin-conversation, x-bili-plugin-model, \
             x-bili-plugin-context-window, x-bili-plugin-reset",
        ),
    );
}

/// Bind with the Node ladder: strict-port refuses to drift, otherwise +1 on
/// EADDRINUSE so a stuck predecessor cannot block a fresh session.
async fn bind_with_ladder(
    host: &str,
    port: u16,
    strict: bool,
) -> Result<(TcpListener, u16), String> {
    let mut attempt = 0u16;
    loop {
        let candidate = if port == 0 {
            0
        } else {
            port.saturating_add(attempt)
        };
        match TcpListener::bind((host, candidate)).await {
            Ok(listener) => {
                let bound = listener
                    .local_addr()
                    .map(|a: SocketAddr| a.port())
                    .unwrap_or(candidate);
                return Ok((listener, bound));
            }
            Err(err) => {
                let in_use = err.to_string().to_ascii_lowercase().contains("in use")
                    || err
                        .to_string()
                        .to_ascii_lowercase()
                        .contains("address already");
                if !in_use || strict || port == 0 {
                    return Err(format!("bind {host}:{candidate} failed: {err}"));
                }
                attempt = attempt.saturating_add(1);
                if attempt > 20 {
                    return Err(format!(
                        "bind {host}:{port} failed after 21 attempts: {err}"
                    ));
                }
            }
        }
    }
}

async fn forward_handler(
    State(state): State<Arc<AppState>>,
    method: Method,
    OriginalUri(uri): OriginalUri,
    headers: HeaderMap,
    body: Body,
) -> Response {
    let raw = uri
        .path_and_query()
        .map(|pq| pq.as_str().to_string())
        .unwrap_or_else(|| uri.path().to_string());
    let Some(rest) = raw.strip_prefix("/bili/") else {
        return error_response(StatusCode::NOT_FOUND, "not a /bili/ request");
    };
    let target = resolve_target(rest);
    let Some(target) = target else {
        return error_response(
            StatusCode::BAD_REQUEST,
            "missing absolute upstream URL after /bili/",
        );
    };
    if !upstream_target_allowed(
        &target,
        &state.upstream_allowlist,
        state.allow_private_upstreams,
    ) {
        return error_response(
            StatusCode::FORBIDDEN,
            "upstream host is not allowlisted; set BILI_UPSTREAM_ALLOWLIST",
        );
    }
    if let Ok(mut upstream) = state.upstream.lock() {
        *upstream = Some(upstream_base(&target));
    }
    // Start of this upstream attempt, including prepare/prune.  The ledger uses
    // this window to prove which artifacts this request itself wrote.
    let request_started_at = now_ms();
    if state.debug {
        eprintln!(
            "[billion-context-proxy] {} {} -> {}",
            method.as_str(),
            raw,
            pricing::redacted_origin(&target)
        );
    }

    let model_wire = target_has_model_wire(&target);
    let mut forward = HeaderMap::new();
    for (name, value) in headers.iter() {
        let n = name.as_str().to_ascii_lowercase();
        if !request_header_is_forwardable(&n, model_wire) {
            continue;
        }
        forward.insert(name, value.clone());
    }

    let mut request = state.client.request(method.clone(), &target);
    request = request.headers(forward);

    let mut usage_meta: Option<(String, Option<String>)> = None;
    let mut prepare_trace: Option<PrepareTrace> = None;
    let mut sent_digest: Option<(String, u64)> = None;

    // Buffer the request body so the upstream sees an exact Content-Length
    // instead of a streamed chunked body (the Node host relays with a known
    // length, and some gateways reject chunked model requests).  Responses
    // stay fully streamed.
    if !matches!(method, Method::GET | Method::HEAD) {
        const MAX_REQUEST_BODY: usize = 64 * 1024 * 1024;
        let payload = match axum::body::to_bytes(body, MAX_REQUEST_BODY).await {
            Ok(bytes) => bytes,
            Err(err) => {
                return error_response(
                    StatusCode::PAYLOAD_TOO_LARGE,
                    &format!("request body too large or unreadable: {err}"),
                );
            }
        };
        // A DSH native compaction call is the host's summarizer asking for a
        // summary, not the user talking. Answering it from the bili snapshot
        // here keeps it off every model route — a real route answers the
        // instruction with no text block and the host aborts compaction
        // (`summarization produced no text summary content`).
        if !state.passthrough
            && let Some(answer) = dsh_compaction_answer(&state, &target, &headers, &payload).await
        {
            return answer;
        }
        let payload = if state.passthrough {
            payload
        } else {
            // P1: a host-initiated fold must land before the projection is
            // built, otherwise the turn that runs out of room never sees it.
            if state.auto_fallback
                && target_has_model_wire(&target)
                && let Some(note) = maybe_auto_fallback(&state, &headers, &payload).await
            {
                eprintln!("[billion-context-proxy] {note}");
            }
            let (prepared, trace) = maybe_prepare(&state, &target, &headers, payload);
            prepare_trace = trace;
            if target_has_model_wire(&target) {
                if let Ok(body) = serde_json::from_slice::<Value>(&prepared) {
                    let model = body
                        .get("model")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                        .or_else(|| {
                            headers
                                .get("x-bili-plugin-model")
                                .and_then(|value| value.to_str().ok())
                                .map(str::to_string)
                        });
                    usage_meta = Some((conversation_key(&headers, &body), model));
                }
            }
            prepared
        };
        if state.speculation && target_has_model_wire(&target) {
            maybe_background_speculation(state.clone(), headers.clone(), payload.clone()).await;
        }
        sent_digest = Some((usage_ledger::sha256_hex(&payload), payload.len() as u64));
        request = request
            .header(reqwest::header::CONTENT_LENGTH, payload.len())
            .body(payload);
    }

    // One ledger record per upstream model attempt, written before the send so
    // a transport failure still has a row.  A start without a finish means the
    // outcome is unknown and must never be counted as success.
    let mut ledger_record: Option<usage_ledger::RequestRecord> = None;
    if let (Some((key, model)), Some((hash, bytes))) = (usage_meta.as_ref(), sent_digest.as_ref()) {
        if let Some(requests_dir) = state.sessions.requests_dir() {
            let protocol = prepare_trace
                .as_ref()
                .map(|trace| trace.protocol)
                .unwrap_or(if target.contains("/chat/completions") {
                    "openai"
                } else {
                    "anthropic"
                });
            let upstream = upstream_base(&target);
            let (source, basis) = usage_ledger::usage_source(state.usage_source, &upstream);
            let prune = prepare_trace.as_ref().and_then(|trace| {
                trace.prune.as_ref().map(|report| {
                    let (refs, attribution) = state.sessions.resolve_prune_artifacts(
                        report,
                        request_started_at,
                        now_ms(),
                    );
                    usage_ledger::PruneAttribution::new(report, refs, attribution)
                })
            });
            let record = usage_ledger::RequestRecord::new(
                usage_ledger::ledger_path(&requests_dir, key),
                key,
                protocol,
                model.clone(),
                &upstream,
                source,
                basis,
                request_started_at,
                hash.clone(),
                *bytes,
                prepare_trace.is_some(),
                prune,
                &headers,
            );
            if let Err(err) = record.start() {
                eprintln!("[billion-context-proxy] usage ledger start write failed: {err}");
            }
            ledger_record = Some(record);
        }
    }

    let upstream = match request.send().await {
        Ok(resp) => resp,
        Err(err) => {
            // The reqwest message embeds the full URL; only the class and the
            // redacted origin are safe to record.
            eprintln!(
                "[billion-context-proxy] upstream error for {}: {}",
                pricing::redacted_origin(&target),
                usage_ledger::error_class(&err)
            );
            record_upstream_failure(&state, &target, "TRANSPORT");
            if let Some(record) = ledger_record.as_ref() {
                if let Err(write_err) = record.finish(usage_ledger::Finish::transport(&err)) {
                    eprintln!(
                        "[billion-context-proxy] usage ledger finish write failed: {write_err}"
                    );
                }
            }
            return error_response(
                StatusCode::BAD_GATEWAY,
                &format!("upstream relay failed: {}", usage_ledger::error_class(&err)),
            );
        }
    };

    let status = upstream.status();
    if status == StatusCode::TOO_MANY_REQUESTS
        || status == StatusCode::UNAUTHORIZED
        || status == StatusCode::FORBIDDEN
        || status.is_server_error()
    {
        record_upstream_failure(&state, &target, &format!("HTTP {}", status.as_u16()));
    } else if status.is_success() {
        clear_upstream_failure(&state, &target);
    }
    if !status.is_success() {
        // Not a model answer: the body is forwarded untouched (and therefore
        // not hashed), so the ledger states exactly that.
        if let Some(record) = ledger_record.as_ref() {
            if let Err(write_err) = record.finish(usage_ledger::Finish::http_error(status.as_u16()))
            {
                eprintln!("[billion-context-proxy] usage ledger finish write failed: {write_err}");
            }
        }
    }
    // The tee replaces the upstream body with a stream, so a forwarded
    // Content-Length would let the client-side framing declare the response
    // complete before the tee's final bookkeeping ran.  Dropping the length
    // makes hyper frame the identical bytes as chunked and poll the stream to
    // its end, which is what records usage and the finish line.
    let tee = usage_meta.is_some();
    let mut response = Response::builder().status(status);
    if let Some(builder) = response.headers_mut() {
        for (name, value) in upstream.headers().iter() {
            let n = name.as_str().to_ascii_lowercase();
            if matches!(
                n.as_str(),
                "connection"
                    | "transfer-encoding"
                    | "keep-alive"
                    | "upgrade"
                    | "proxy-authenticate"
            ) {
                continue;
            }
            if tee && n == "content-length" {
                continue;
            }
            // Node's passthrough resolves relative redirects against the
            // proxy origin; mirror that so a follow-up lands on the same
            // reverse-proxy path instead of the upstream host directly.
            if n == "location" {
                if let Ok(loc) = value.to_str() {
                    if loc.starts_with('/') {
                        let absolute = format!("{}{}", state.origin, loc);
                        if let Ok(v) = axum::http::HeaderValue::from_str(&absolute) {
                            builder.insert(name, v);
                            continue;
                        }
                    }
                }
            }
            builder.insert(name, value.clone());
        }
    }
    let content_type = upstream
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .to_string();
    let body = if let Some((key, model)) = usage_meta {
        Body::from_stream(usage::inspect_stream(
            upstream.bytes_stream(),
            content_type,
            state.sessions.clone(),
            key,
            model,
            ledger_record.take(),
        ))
    } else {
        Body::from_stream(upstream.bytes_stream())
    };
    match response.body(body) {
        Ok(resp) => resp,
        Err(err) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("response build failed: {err}"),
        ),
    }
}

/// Answers a DSH native compaction call from the bili snapshot instead of
/// forwarding it (Node: `dsh compaction call identified (…) — ANSWERED LOCALLY
/// with the bili snapshot (#3733 authored checkpoint)`).
///
/// `None` means "not a compaction call": the request proceeds to prepare and
/// the upstream exactly as before. When the call is recognized but bili holds
/// no active block, there is nothing to answer with; forwarding would let the
/// host's checkpoint durably shadow the raw history, so the call is refused
/// unless `dsh.authoredCompactionRefuse` is turned off for inspection.
async fn dsh_compaction_answer(
    state: &Arc<AppState>,
    target: &str,
    headers: &HeaderMap,
    payload: &Bytes,
) -> Option<Response> {
    let protocol = dsh_compaction::protocol_of(target)?;
    let parsed = serde_json::from_slice::<Value>(payload).ok();
    let declared = dsh_compaction::declared_compaction_call(headers);
    let marked = dsh_compaction::body_marks_compaction(parsed.as_ref(), payload);
    if !declared && !marked {
        return None;
    }

    let config = config_path();
    let key = parsed
        .as_ref()
        .map(|body| conversation_key(headers, body))
        .unwrap_or_else(|| conversation_key(headers, &Value::Null));
    let stream = parsed
        .as_ref()
        .and_then(|body| body.get("stream"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let messages = parsed
        .as_ref()
        .and_then(|body| body.get("messages"))
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or(0);
    let identified = if declared {
        "declared purpose=compaction (#6880)"
    } else {
        "final user message = COMPACTION_INSTRUCTION"
    };

    let checkpoint = dsh_compaction::authored_compaction_enabled(&config)
        .then(|| state.sessions.authored_checkpoint(&key))
        .flatten();
    let mut checkpoint = checkpoint;

    // Fold before refusing. An empty block set is only *unanswerable*, not
    // unfixable: the session usually still holds foldable ranges, and the
    // caller cannot spend a compress call on them (bili's own breaker forbids
    // it once armed). Refusing here is that deadlock (#7762). So the host
    // authors the fold itself, with the same chain the fallback path uses,
    // and then answers from the checkpoint that fold produced.
    if checkpoint.is_none()
        && dsh_compaction::authored_compaction_enabled(&config)
        && dsh_compaction::refuse_without_checkpoint(&config)
        && dsh_compaction::preflight_fold_enabled(&config)
        && state.sessions.has_foldable_ranges(&key)
    {
        let limit = state
            .sessions
            .context_limit_of(&key)
            .unwrap_or_else(|| context_limit(headers));
        if let Some(note) = run_planned_fold(state, &key, limit, "compaction-preflight").await {
            eprintln!(
                "[billion-context-proxy] dsh compaction call identified ({identified}, {messages} msgs, \
                 conv={key}) - {note}"
            );
        }
        checkpoint = state.sessions.authored_checkpoint(&key);
    }

    if let Some(checkpoint) = checkpoint
        && let Some(answer) = dsh_compaction::build_answer(protocol, &checkpoint.text, stream)
    {
        eprintln!(
            "[billion-context-proxy] dsh compaction call identified ({identified}, {messages} msgs, conv={key}) \
             - ANSWERED LOCALLY with the bili snapshot (#3733 authored checkpoint, blocks={}): \
             upstream not contacted",
            checkpoint.blocks.join("+")
        );
        return Some(
            Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, answer.content_type)
                .header("x-bili-gate", "dsh-compaction-authored")
                .body(Body::from(answer.body))
                .unwrap_or_else(|err| {
                    error_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        &format!("local compaction answer build failed: {err}"),
                    )
                }),
        );
    }

    if !dsh_compaction::refuse_without_checkpoint(&config) {
        eprintln!(
            "[billion-context-proxy] dsh compaction call identified ({identified}, {messages} msgs, conv={key}) \
             - no active block, forwarded (dsh.authoredCompactionRefuse=false)"
        );
        return None;
    }
    eprintln!(
        "[billion-context-proxy] dsh compaction call identified ({identified}, {messages} msgs, conv={key}) \
         - REFUSED, not forwarded: bili owns compression on this lane (#1729); an active block is \
         required before bili can answer with its snapshot"
    );
    Some(
        Response::builder()
            .status(StatusCode::UNPROCESSABLE_ENTITY)
            .header(header::CONTENT_TYPE, "application/json")
            .header("x-bili-gate", "dsh-compaction-refused")
            .body(Body::from(dsh_compaction::refusal_body(protocol)))
            .unwrap_or_else(|err| {
                error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("refusal body build failed: {err}"),
                )
            }),
    )
}

/// What the host actually did while preparing this request, so the ledger can
/// report request-scoped prune facts instead of session totals.
struct PrepareTrace {
    protocol: &'static str,
    /// `Some` only when a kernel prepare ran; the report then describes this
    /// request's own prune attempt.
    prune: Option<crate::prune::PruneReport>,
}

/// Rewrite an Anthropic Messages body through the kernel. Failure to parse or
/// to recognize the wire leaves the request untouched (fail-open: a broken
/// rewrite must never cost the user a turn).
fn maybe_prepare(
    state: &AppState,
    target: &str,
    headers: &HeaderMap,
    payload: Bytes,
) -> (Bytes, Option<PrepareTrace>) {
    let protocol = if target.contains("/chat/completions") {
        "openai"
    } else {
        "anthropic"
    };
    if state.passthrough || !target_has_model_wire(target) {
        return (payload, None);
    }
    let Ok(mut body) = serde_json::from_slice::<Value>(&payload) else {
        return (
            payload,
            Some(PrepareTrace {
                protocol,
                prune: None,
            }),
        );
    };
    if body.get("messages").and_then(Value::as_array).is_none() {
        return (
            payload,
            Some(PrepareTrace {
                protocol,
                prune: None,
            }),
        );
    }
    let key = conversation_key(headers, &body);
    let plugin_mode = headers.contains_key("x-bili-plugin");
    let model = headers
        .get("x-bili-plugin-model")
        .and_then(|value| value.to_str().ok())
        .map(str::to_string)
        .or_else(|| {
            body.get("model")
                .and_then(Value::as_str)
                .map(str::to_string)
        });
    // Phase 3.2: header (plugin wire) > `modelPolicies` > default 200k.
    let context_limit = context_limit_for(headers, model.as_deref());
    let openai_wire = target.contains("/chat/completions");
    let artifacts_available = state.sessions.artifact_store().is_some();
    if openai_wire {
        let raw_input = openai::estimate_message_tokens(&body) as u32;
        let mut fixed_overhead =
            (openai::estimate_fixed_overhead(&body) as u32).saturating_add(800);
        if !plugin_mode {
            fixed_overhead =
                fixed_overhead.saturating_add(advertised_tools_estimate(artifacts_available));
        }
        let converted = openai::to_core(&body);
        let prepared = state.sessions.prepare_with_budget(
            &key,
            context_limit,
            &converted.messages,
            raw_input,
            fixed_overhead,
            model.clone(),
        );
        // Phase 3.2: the routed model's policy also owns the compress knobs.
        if let Some(policy) = policy_for(model.as_deref()) {
            state
                .sessions
                .apply_model_policy(&key, context_limit, &policy);
        }
        state
            .sessions
            .set_meta(&key, "openai", &upstream_base(target));
        let rebuilt = openai::from_core(&prepared.messages, &converted.sidecar);
        body["messages"] = Value::Array(rebuilt);
        openai::inject_standing_instructions(&mut body);
        if let Some(note) = prepared.nudge.as_deref() {
            openai::inject_user_note(&mut body, note);
        }
        if !plugin_mode {
            let defs = advertised_tools(artifacts_available);
            openai::inject_tools(&mut body, &defs);
        }
        if state.debug {
            eprintln!(
                "[billion-context-proxy] prepared openai {key}: {} messages, raw {} + fixed {} = effective {}",
                body["messages"].as_array().map(Vec::len).unwrap_or(0),
                raw_input,
                fixed_overhead,
                prepared.effective_projection_estimate
            );
        }
        let payload = match serde_json::to_vec(&body) {
            Ok(bytes) => Bytes::from(bytes),
            Err(_) => payload,
        };
        return (
            payload,
            Some(PrepareTrace {
                protocol: "openai",
                prune: prepared.prune,
            }),
        );
    }
    let raw_input = anthropic::estimate_message_tokens(&body) as u32;
    let mut fixed_overhead = (anthropic::estimate_fixed_overhead(&body) as u32).saturating_add(800);
    if !plugin_mode {
        fixed_overhead =
            fixed_overhead.saturating_add(advertised_tools_estimate(artifacts_available));
    }
    let mut converted = match anthropic::to_core(&body) {
        Ok(converted) => converted,
        Err(err) => {
            // The adapter rejects shapes Node also rejects. Keep the existing
            // fail-open contract: forward the original wire instead of
            // synthesizing an empty conversation.
            if state.debug {
                eprintln!("[billion-context-proxy] anthropic wire rejected by adapter: {err}");
            }
            return (
                payload,
                Some(PrepareTrace {
                    protocol,
                    prune: None,
                }),
            );
        }
    };
    let routed_model = model.clone();
    let prepared = state.sessions.prepare_with_budget(
        &key,
        context_limit,
        &converted.messages,
        raw_input,
        fixed_overhead,
        model,
    );
    // Phase 3.2: apply the routed model's compress knobs to this conversation.
    if let Some(policy) = policy_for(routed_model.as_deref()) {
        state
            .sessions
            .apply_model_policy(&key, context_limit, &policy);
    }
    // A pruned tool result must be rendered from its stable preview, not from
    // the original raw block kept for image/error round-trips.
    for id in &prepared.pruned_ids {
        converted.sidecar.raw_blocks.remove(id);
    }
    state
        .sessions
        .set_meta(&key, "anthropic", &upstream_base(target));
    let mut rebuilt = anthropic::from_core(&prepared.messages, &converted.sidecar);
    anthropic::attach_cache_breakpoint(&mut rebuilt);
    body["messages"] = Value::Array(rebuilt);
    anthropic::inject_standing_instructions(&mut body);
    if let Some(note) = prepared.nudge.as_deref() {
        anthropic::inject_user_note(&mut body, note);
    }
    // The plugin lane serves tool definitions through the manifest and lets the
    // host inject them; only the direct wire lane needs the proxy to append them.
    if !plugin_mode {
        let defs = advertised_tools(artifacts_available);
        anthropic::inject_tools(&mut body, &defs);
    }
    if state.debug {
        eprintln!(
            "[billion-context-proxy] prepared {key}: {} -> {} messages, raw {} + fixed {} = effective {}",
            body["messages"].as_array().map(Vec::len).unwrap_or(0),
            rebuilt_len(&body),
            raw_input,
            fixed_overhead,
            prepared.effective_projection_estimate
        );
    }
    let payload = match serde_json::to_vec(&body) {
        Ok(bytes) => Bytes::from(bytes),
        Err(_) => payload,
    };
    (
        payload,
        Some(PrepareTrace {
            protocol: "anthropic",
            prune: prepared.prune,
        }),
    )
}

fn target_has_model_wire(target: &str) -> bool {
    target.contains("/v1/messages") || target.contains("/chat/completions")
}

/// Whether a client request header may be copied onto the upstream request.
///
/// `accept-encoding` is dropped for model wires: the usage tee parses the
/// response bytes it forwards, and a compressed body (gzip/brotli/zstd) would
/// hide the `usage` object from the parser, silently zeroing the billing
/// counters. The upstream then answers with identity encoding, which every
/// client accepts; the cost is bandwidth on this loopback hop only.
fn request_header_is_forwardable(name: &str, model_wire: bool) -> bool {
    if matches!(
        name,
        "host" | "connection" | "content-length" | "transfer-encoding" | "expect" | "upgrade"
    ) {
        return false;
    }
    !(model_wire && name == "accept-encoding")
}

/// Origin kept for host-owned metadata: scheme, host and port only.
///
/// The routed target may carry userinfo or query credentials; neither may reach
/// `/health`, the upstream failure table, a ledger field or a log, so the raw
/// prefix is never retained. Routing itself still uses the original target.
fn upstream_base(target: &str) -> String {
    pricing::redacted_origin(target)
}

fn rebuilt_len(body: &Value) -> usize {
    body.get("messages")
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or(0)
}

fn conversation_key(headers: &HeaderMap, body: &Value) -> String {
    for name in [
        "x-bili-plugin-conversation",
        "x-conversation-id",
        "x-session-id",
        "session-id",
    ] {
        if let Some(value) = headers.get(name).and_then(|value| value.to_str().ok()) {
            let value = value.trim();
            if !value.is_empty() && value.len() <= 256 {
                return value.to_string();
            }
        }
    }
    body.get("metadata")
        .and_then(|meta| meta.get("user_id"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| "wire-default".to_string())
}

/// Phase 3.2: read `modelPolicies` from billion-context.json for this model.
/// Only consulted for fields the wire does not carry itself, so the plugin
/// path (which knows the real window) is never overridden by a stale table.
fn policy_for(model: Option<&str>) -> Option<session_store::ModelPolicy> {
    let model = model?;
    let text = std::fs::read_to_string(config_path()).ok()?;
    let policy = session_store::model_policy(&text, model);
    (policy.context_window.is_some()
        || policy.min_compress_range.is_some()
        || policy.max_summary_length.is_some())
    .then_some(policy)
}

fn context_limit_for(headers: &HeaderMap, model: Option<&str>) -> u32 {
    headers
        .get("x-bili-plugin-context-window")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.trim().parse::<u32>().ok())
        .filter(|value| (1024..=2_000_000).contains(value))
        .or_else(|| policy_for(model).and_then(|policy| policy.context_window))
        .unwrap_or(200_000)
}

fn context_limit(headers: &HeaderMap) -> u32 {
    context_limit_for(headers, None)
}

/// The tool surface this host can actually serve right now.
///
/// `public_tools()` is the full registry; a host without an artifact store
/// cannot answer the four artifact-backed tools, so it must not advertise them.
/// Advertising a tool the dispatcher will refuse teaches the model to call it
/// and then fails the call — the surface and the gate have to agree.
fn advertised_tools(artifacts_available: bool) -> Vec<Value> {
    tool_runner::servable(&public_tools(), artifacts_available)
}

/// Token cost of the surface actually injected. The estimate has to track
/// [`advertised_tools`], otherwise a host without an artifact store reserves
/// room for tool definitions it never sends.
fn advertised_tools_estimate(artifacts_available: bool) -> u32 {
    serde_json::to_string(&advertised_tools(artifacts_available))
        .map(|text| anthropic::estimate_tokens(&text) as u32)
        .unwrap_or(0)
}

/// Public tool surface: the Node proxy's names and schemas, shared by the DSH
/// manifest and the non-plugin tool injection.
fn public_tools() -> Vec<Value> {
    let tools = json!([
        {
            "name": "compress",
            "description": "Replace consumed conversation ranges with self-contained summaries using mNNNNN or bN refs; batch multiple ranges into ONE call — pass `content` as ONE string holding every block, each block starting with its 'm00150–m00220 optional topic' header line followed by that block's summary markdown (a JSON array of {startId,endId,summary,topic?} entries is also accepted).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "ONE string holding ALL ranges: each block starts with its 'm00150–m00220 optional topic' header line, followed by that block's summary markdown. Escape it for JSON (quotes as \\\" and newlines as \\n); a JSON-encoded array of {startId,endId,summary,topic?} entries is also accepted."
                    },
                    "summaryMaxChars": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 250000,
                        "description": "Optional per-call ceiling for each summary in characters, overriding the session's configured maxSummaryLength for this call only. Use it when a summary is legitimately longer than the default (the refusal text names the ceiling it hit); the limit is restored afterwards."
                    }
                }
            }
        },
        {
            "name": "decompress",
            "description": "Restore compressed content by block id (b5) or message ref; block mode writes to a file by default, inline: true returns small content inline.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "blockId": {"type": "string", "description": "Block ID to decompress (e.g. b5)"},
                    "toFile": {"type": "string", "description": "Optional: write content to file instead of context"},
                    "full": {"type": "boolean", "description": "Restore all the way to original messages"},
                    "imageRef": {"type": "string", "description": "Optional mNNNNN message ref whose IMAGE payload you want restored. Pass \"list\" to enumerate every image currently restorable."}
                },
                "required": ["blockId"]
            }
        },
        {
            "name": "search_context",
            "description": "Search compressed summaries and historical messages by keyword; returns refs, sizes, previews.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "number", "description": "Max results (default 5)"}
                },
                "required": ["query"]
            }
        },
        {
            "name": "artifact_info",
            "description": "Inspect metadata for an artifact-backed tool output. Pass artifact_id or a stable message ref. Does not return the full body.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "artifact_id": {"type": "string"},
                    "ref": {"type": "string", "description": "Optional mNNNNN ref whose artifact should be resolved"}
                }
            }
        },
        {
            "name": "artifact_read",
            "description": "Read a bounded byte range from an artifact-backed tool output. Pass artifact_id or ref; offset/limit are byte-oriented and hard-capped. Original bytes are only available when the host stored them.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "artifact_id": {"type": "string"},
                    "ref": {"type": "string"},
                    "offset": {"type": "number", "description": "Byte offset, default 0"},
                    "limit": {"type": "number", "description": "Max bytes, default 8192, hard cap 64 KiB"}
                }
            }
        },
        {
            "name": "artifact_search",
            "description": "Search one artifact-backed tool output by literal keyword. Results are bounded in count and returned bytes; the full artifact is never injected automatically.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "artifact_id": {"type": "string"},
                    "ref": {"type": "string"},
                    "query": {"type": "string"},
                    "limit": {"type": "number", "description": "Max hits, default 5, cap 20"},
                    "max_bytes": {"type": "number", "description": "Max returned text bytes, default 4096, cap 32 KiB"}
                },
                "required": ["query"]
            }
        },
        {
            "name": "retrieve_raw",
            "description": "Return the archived ORIGINAL of a folded range, explicitly and bounded (never injected automatically). Address it by `refs` (mNNNNN message refs) or by `block` (bN block id listed in acp_status foldArchives). If no artifact holds the span the answer is OriginalUnavailable; nothing is pretended. `retrieve` with mode=\"original\" reaches the same path.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "refs": {"type": "array", "items": {"type": "string"}, "description": "Message refs (mNNNNN) whose folded original should be read"},
                    "block": {"type": "string", "description": "Folded block id (bN) from acp_status foldArchives"},
                    "offset": {"type": "number", "description": "Byte offset per ref, default 0"},
                    "limit": {"type": "number", "description": "Max bytes per ref, default 8192, hard cap 64 KiB"}
                }
            }
        },
        {
            "name": "reset_window",
            "description": "Explicit last-resort fresh-window reset (never automatic): drop this conversation's folded blocks and start a new window seeded with a task brief plus the last few messages. Archived originals stay reachable by artifact id and the reset is recorded in acp_status freshWindow. This is not compression.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Why the reset is needed (recorded in the receipt)"},
                    "brief": {"type": "string", "description": "Task brief to carry into the new window; defaults to the last user message"},
                    "keepRecent": {"type": "number", "description": "Tail messages kept verbatim, default 6, cap 32"}
                }
            }
        },
        {
            "name": "acp_status",
            "description": "Context usage overview, compressible ranges, block drilldown.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "scope": {"type": "string", "enum": ["uncompressed", "compressed"]},
                    "view": {"type": "string", "enum": ["messages", "ranges"]},
                    "tool": {"type": "string", "description": "With view:messages, only that tool's results"},
                    "sort": {"type": "string", "enum": ["size", "time", "tool", "age"]},
                    "limit": {"type": "number", "description": "Max entries to list"},
                    "archiveDetail": {"type": "boolean", "description": "List archived blocks per id instead of grouped by reason"},
                    "blockId": {"type": "string"}
                }
            }
        },
        {
            "name": "acp_cache",
            "description": "Inspect the prompt-cache ledger: cached tokens, hits, and which prefix is being reused.",
            "input_schema": {"type": "object", "properties": {}}
        }
    ]);
    tools.as_array().cloned().unwrap_or_default()
}

async fn manifest_handler(State(state): State<Arc<AppState>>) -> Response {
    // Advertise only what this host can serve. The host registers whatever is
    // listed here and then calls it; listing an artifact tool on a host with no
    // artifact store produces a guaranteed failed call for a tool the model was
    // explicitly told exists.
    let artifacts_available = state.sessions.artifact_store().is_some();
    let tools = Value::Array(advertised_tools(artifacts_available));
    let names: Vec<&str> = tools
        .as_array()
        .map(|list| {
            list.iter()
                .filter_map(|tool| tool.get("name").and_then(Value::as_str))
                .collect()
        })
        .unwrap_or_default();
    // A name in the full registry that the dispatcher cannot resolve is a
    // broken surface, not a cosmetic bug: the host registers it, the model calls
    // it, and nothing runs. Checked against the full registry so filtering for
    // availability can never hide the mismatch.
    let registry = public_tools();
    let registry_names: Vec<&str> = registry
        .iter()
        .filter_map(|tool| tool.get("name").and_then(Value::as_str))
        .collect();
    let unresolved = tool_runner::unresolved(&registry_names);
    // Same servable surface, OpenAI dialect. The bridge in this profile reads
    // `tools.anthropic` (it serves MCP `tools/list` from it), so this half is
    // inert today — but publishing `[]` claimed the host serves no tools to any
    // OpenAI-protocol reader, which was simply false. `responses` stays empty on
    // purpose: no consumer reads it and its item shape is unverified here, so
    // inventing one would be worse than saying nothing.
    let openai_tools =
        tool_runner::openai_tools(tools.as_array().map(Vec::as_slice).unwrap_or_default());
    axum::Json(json!({
        "ok": true,
        "protocolVersion": 1,
        "proxy": state.origin,
        "version": VERSION,
        "commit": "rust",
        "build": build_id::BuildInfo::current().json(),
        "toolNames": names,
        "tools": { "anthropic": tools, "openai": openai_tools, "responses": [] },
        "registry": {
            "advertised": names.len(),
            "implemented": tool_runner::IMPLEMENTED.len(),
            // Advertised is what the host registers; dispatchable is every name
            // `Store::tool` answers, aliases included. They are reported apart
            // because a canonical name missing from the dispatchable set would
            // pass the gate and then fall into `[unknown tool]`.
            "dispatchable": tool_runner::DISPATCHED.len(),
            "openaiAdvertised": openai_tools.len(),
            "unresolved": unresolved,
            "artifactsAvailable": artifacts_available,
            "withheld": tool_runner::withheld(artifacts_available),
        },
        "headers": {
            "agent": "x-bili-plugin",
            "conversation": "x-bili-plugin-conversation",
            "contextWindow": "x-bili-plugin-context-window",
            "maxOutput": "x-bili-plugin-max-output",
            "model": "x-bili-plugin-model",
            "instructionsMutable": "x-bili-plugin-instructions-mutable",
            "compression": "x-bili-plugin-compression",
        },
        "toolEndpoint": "/__bili/plugin/tool",
        "statusEndpoint": "/__bili/plugin/status",
        "runtimeInfoEndpoint": "/__bili/plugin/runtime-info",
    }))
    .into_response()
}

async fn plugin_tool_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let parsed: Value = match serde_json::from_slice(&body) {
        Ok(value) => value,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                axum::Json(json!({"ok": false, "error": "invalid JSON body"})),
            )
                .into_response();
        }
    };
    let tool = parsed.get("tool").and_then(Value::as_str).unwrap_or("");
    let key = parsed
        .get("conversationId")
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| conversation_key(&headers, &parsed));
    // Tool calls arrive without `x-bili-plugin-context-window`; falling back to
    // the bare default would rebuild the whole session inside `with_entry`
    // (refs/folds/token counters cleared), which makes every ref the nudge just
    // offered an `unknown_ref`. Prefer the limit this conversation was last
    // prepared with; only fall back to the header/default for an unseen key.
    let limit = state
        .sessions
        .context_limit_of(&key)
        .unwrap_or_else(|| context_limit(&headers));
    let args = parsed.get("args").cloned().unwrap_or_else(|| json!({}));
    // Fail closed before any dispatch: an unimplemented name and a tool whose
    // backing store is missing both used to come back as `ok: true` carrying a
    // prose apology, which the host accepted as a delivered result. The gate
    // answers with a non-2xx status and `ok: false` instead, so the call reads
    // as failed at the only place the model can see it.
    let artifacts_available = state.sessions.artifact_store().is_some();
    let canonical = match tool_runner::check(tool, &args, artifacts_available) {
        Ok(canonical) => canonical,
        Err(refusal) => {
            eprintln!(
                "[billion-context-proxy] plugin tool refused before dispatch ({key}): {} ({})",
                refusal.code(),
                tool
            );
            let status = StatusCode::from_u16(refusal.status()).unwrap_or(StatusCode::BAD_REQUEST);
            return (
                status,
                axum::Json(tool_runner::refusal_body(&refusal, tool, &key)),
            )
                .into_response();
        }
    };
    // Dispatch the canonical name, not the raw one. The gate resolved an alias
    // to the branch that implements it; forwarding the raw spelling would let
    // the gate and the dispatcher disagree on names the alias table covers but
    // the match arms do not, so a call the gate approved would land in the
    // `[unknown tool]` fallback and report `refused`.
    let args = external_summary_args(&state, &key, canonical, args).await;
    let (text, outcome, blocks) = state.sessions.tool(&key, limit, canonical, &args);
    let mut reply = json!({
        "ok": true,
        "tool": tool,
        "canonicalTool": canonical,
        "conversationId": key,
        // Every dispatch branch states its own outcome. Reporting "success" for
        // an absent one would turn a dispatcher bug into a delivered result.
        // `failure` is the only honest default: the host marks a call failed on
        // `failure`/`refused` and ignores any outcome word it does not know.
        "outcome": outcome.unwrap_or("failure"),
        "result": text,
    });
    if let Some(blocks) = blocks {
        reply["blocksCreated"] = json!(blocks);
    }
    axum::Json(reply).into_response()
}

/// Optional external-summary preflight for a `compress` tool call
/// (`compress.externalSummary`, plan §2.1).
///
/// Returns the args unchanged whenever the chain is off, the tool is something
/// else, or the summarizer produced nothing. Fail-open is deliberate: keeping
/// the model's own summary keeps a fold landable when the cheap model is down,
/// which is the difference from the Node proxy documented in §2.8.
async fn external_summary_args(
    state: &Arc<AppState>,
    key: &str,
    tool_name: &str,
    args: Value,
) -> Value {
    if tool_name != "compress" && tool_name != "acp_compress" {
        return args;
    }
    let Some(chain) = external_summary_chain() else {
        return args;
    };
    let inputs = state.sessions.range_inputs(key, &args);
    if inputs.is_empty() {
        return args;
    }
    match summarizer::summarize(&chain, &inputs).await {
        Some(batch) => {
            let first = &chain.targets[0];
            eprintln!(
                "[billion-context-proxy] compress.externalSummary: authored {} range summary(ies) with {}/{} ({} provider call(s), {} estimated)",
                batch.summaries.len(),
                first.recipe.name,
                first.model,
                batch.calls,
                batch.estimated_calls
            );
            state.sessions.record_summary_usage(key, false, &batch);
            session_store::apply_external_summaries(&args, &batch.summaries)
        }
        None => {
            eprintln!(
                "[billion-context-proxy] compress.externalSummary: chain produced no summary, folding with the model-authored text"
            );
            args
        }
    }
}

fn external_summary_chain() -> Option<summarizer::ExternalSummary> {
    let text = std::fs::read_to_string(config_path()).ok()?;
    let document: Value = serde_json::from_str(&text).ok()?;
    summarizer::from_config(&document).filter(summarizer::ExternalSummary::is_active)
}

/// P1 auto fallback is opt-in: `BILI_AUTO_FALLBACK=1` (or true/on/yes) turns it
/// on regardless of config, `0`/anything else in the env turns it off; without
/// the env var the config key `compress.autoFallback.enabled` decides, default
/// false. Resolved once at startup so a per-request config read cannot flip it
/// mid-conversation.
fn auto_fallback_enabled() -> bool {
    if let Some(value) = env_str("BILI_AUTO_FALLBACK") {
        let value = value.trim().to_ascii_lowercase();
        if !value.is_empty() {
            return matches!(value.as_str(), "1" | "true" | "on" | "yes");
        }
    }
    let Some(text) = std::fs::read_to_string(config_path()).ok() else {
        return false;
    };
    let Ok(document) = serde_json::from_str::<Value>(&text) else {
        return false;
    };
    document
        .get("compress")
        .and_then(|compress| compress.get("autoFallback"))
        .and_then(|fallback| fallback.get("enabled"))
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

/// Phase 3.1 speculation is opt-in: `BILI_SPECULATION=1` (or true/on/yes), else
/// the config key `compress.speculation.enabled`, default false. Resolved once
/// at startup like the fallback switch.
fn speculation_enabled() -> bool {
    if let Some(value) = env_str("BILI_SPECULATION") {
        let value = value.trim().to_ascii_lowercase();
        if !value.is_empty() {
            return matches!(value.as_str(), "1" | "true" | "on" | "yes");
        }
    }
    let Some(text) = std::fs::read_to_string(config_path()).ok() else {
        return false;
    };
    let Ok(document) = serde_json::from_str::<Value>(&text) else {
        return false;
    };
    document
        .get("compress")
        .and_then(|compress| compress.get("speculation"))
        .and_then(|speculation| speculation.get("enabled"))
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

/// P1: when the model ignored the nudge twice, or the projection crossed the
/// pressure line, the host folds the oldest foldable range itself through the
/// configured summarizer chain. Runs before the request is projected so the
/// turn that triggered it already shrinks. Every path ends with
/// `apply_fallback` or `abort_fallback`: a dead chain must not leave a range
/// stuck in flight.
async fn maybe_auto_fallback(
    state: &Arc<AppState>,
    headers: &HeaderMap,
    payload: &axum::body::Bytes,
) -> Option<String> {
    let body: Value = serde_json::from_slice(payload).ok()?;
    let key = conversation_key(headers, &body);
    let model = headers
        .get("x-bili-plugin-model")
        .and_then(|value| value.to_str().ok())
        .map(str::to_string)
        .or_else(|| {
            body.get("model")
                .and_then(Value::as_str)
                .map(str::to_string)
        });
    let limit = state
        .sessions
        .context_limit_of(&key)
        .unwrap_or_else(|| context_limit_for(headers, model.as_deref()));
    run_planned_fold(state, &key, limit, "autoFallback").await
}

/// Land one host-initiated fold: plan → chain → inputs → budget → summarize →
/// `apply_fallback`.
///
/// Shared by the pre-request fallback and the compaction path's
/// fold-before-refuse: both need a summary the model never authored, and both
/// must leave every exit path with a landed fold or an abort. `source` prefixes
/// the diagnostics so the two callers stay distinguishable in the log.
async fn run_planned_fold(
    state: &Arc<AppState>,
    key: &str,
    limit: u32,
    source: &str,
) -> Option<String> {
    let plan = state.sessions.fallback_plan(key)?;
    let reason = plan.reason;
    let range_count = plan.ranges.len();
    let Some(chain) = external_summary_chain() else {
        state.sessions.abort_fallback(key, "no_chain");
        return Some(format!(
            "{source} skipped ({reason}): no summarizer chain is configured"
        ));
    };
    let inputs = state.sessions.fallback_inputs(key, &plan);
    if inputs.is_empty() {
        state.sessions.abort_fallback(key, "no_input");
        return Some(format!(
            "{source} skipped ({reason}): no range text is available to summarize"
        ));
    }
    if let Err(refusal) = state.sessions.reserve_summary_call(
        key,
        false,
        chain.settings.max_calls_per_session,
        chain.settings.max_speculation_calls,
    ) {
        state.sessions.abort_fallback(key, "budget");
        return Some(format!(
            "{source} skipped ({reason}): {refusal}; the projection was left untouched"
        ));
    }
    let Some(batch) = summarizer::summarize(&chain, &inputs).await else {
        state.sessions.abort_fallback(key, "chain_failed");
        return Some(format!(
            "{source} failed ({reason}): the summarizer chain returned nothing"
        ));
    };
    if batch.summaries.is_empty() {
        state.sessions.abort_fallback(key, "empty_summary");
        return Some(format!(
            "{source} failed ({reason}): the summarizer chain returned no text"
        ));
    }
    state.sessions.record_summary_usage(key, false, &batch);
    let summaries = batch.summaries;
    let landing = state
        .sessions
        .apply_fallback(key, limit, plan, &summaries, now_ms())
        .unwrap_or_else(|| "(no attempt was in flight)".to_string());
    Some(format!(
        "{source} {reason} over {range_count} range(s): {landing}"
    ))
}

/// Phase 3.1: open a candidate in the 45–75% band and distill the old T1
/// blocks off the request path. The candidate carries signatures; when the
/// chain answers, `apply_speculation` refuses to land anything whose blocks
/// moved in the meantime. A dead chain releases the marker so later turns can
/// try again.
async fn maybe_background_speculation(
    state: Arc<AppState>,
    headers: HeaderMap,
    payload: axum::body::Bytes,
) {
    let Ok(body) = serde_json::from_slice::<Value>(&payload) else {
        return;
    };
    let key = conversation_key(&headers, &body);
    let model = headers
        .get("x-bili-plugin-model")
        .and_then(|value| value.to_str().ok())
        .map(str::to_string)
        .or_else(|| {
            body.get("model")
                .and_then(Value::as_str)
                .map(str::to_string)
        });
    let limit = state
        .sessions
        .context_limit_of(&key)
        .unwrap_or_else(|| context_limit_for(&headers, model.as_deref()));
    let Some(plan) = state.sessions.speculation_plan(&key) else {
        return;
    };
    let Some(chain) = external_summary_chain() else {
        state.sessions.abort_speculation(&key, "no_chain");
        return;
    };
    let inputs: Vec<summarizer::RangeInput> = plan
        .blocks
        .iter()
        .zip(plan.inputs.iter())
        .map(|(block, text)| summarizer::RangeInput {
            label: block.clone(),
            topic: None,
            text: text.clone(),
        })
        .collect();
    if inputs.is_empty() {
        state.sessions.abort_speculation(&key, "no_input");
        return;
    }
    if let Err(refusal) = state.sessions.reserve_summary_call(
        &key,
        true,
        chain.settings.max_calls_per_session,
        chain.settings.max_speculation_calls,
    ) {
        state.sessions.abort_speculation(&key, "budget");
        eprintln!("[billion-context-proxy] speculation skipped: {refusal}");
        return;
    }
    tokio::spawn(async move {
        let Some(batch) = summarizer::summarize(&chain, &inputs).await else {
            state.sessions.abort_speculation(&key, "chain_failed");
            return;
        };
        state.sessions.record_summary_usage(&key, true, &batch);
        let note = state
            .sessions
            .apply_speculation(&key, limit, plan, &batch.summaries, now_ms())
            .unwrap_or_else(|| "(the candidate was already resolved)".to_string());
        eprintln!("[billion-context-proxy] {note}");
    });
}

async fn plugin_status_handler(
    State(state): State<Arc<AppState>>,
    Query(query): Query<HashMap<String, String>>,
) -> Response {
    if let Some(conversation) = query
        .get("conversationId")
        .filter(|value| !value.trim().is_empty())
    {
        match state.sessions.status_json(conversation) {
            Some(status) => return axum::Json(with_summary_budget_limits(status)).into_response(),
            None if query.get("fallback").map(String::as_str) != Some("latest") => {
                return (
                    StatusCode::NOT_FOUND,
                    axum::Json(json!({"ok": false, "error": "unknown plugin conversation"})),
                )
                    .into_response();
            }
            None => {}
        }
    }
    axum::Json(with_summary_budget_limits(
        state.sessions.latest_status_json().unwrap_or_else(
            || json!({"ok": true, "phase": "pre-first-request", "blocks": [], "viaFallback": true}),
        ),
    ))
    .into_response()
}

/// Task 2: attach the live chain budget to the per-session status so the panel
/// shows the ceiling next to the spend. The limits are read from the same config
/// the dispatch path uses, so the two can never disagree.
fn with_summary_budget_limits(mut status: Value) -> Value {
    if let Some(chain) = external_summary_chain()
        && let Some(budget) = status.get_mut("summaryBudget")
    {
        budget["limits"] = json!({
            "maxCallsPerSession": chain.settings.max_calls_per_session,
            "maxSpeculationCalls": chain.settings.max_speculation_calls,
            "maxInputChars": chain.settings.max_input_chars,
            "maxOutputTokens": chain.settings.max_output_tokens,
            "maxSummaryBytes": chain.settings.max_summary_bytes,
            "totalTimeoutMs": chain.settings.total_timeout_ms,
            "enabled": chain.is_active(),
        });
    }
    status
}

async fn runtime_info_handler(State(state): State<Arc<AppState>>) -> Response {
    axum::Json(json!({
        "ok": true,
        "runtime": "rust",
        "version": VERSION,
        "commit": "rust",
        "build": build_id::BuildInfo::current().json(),
        "host": state.host,
        "port": state.port,
        "origin": state.origin,
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "stateDir": state_dir().to_string_lossy(),
        "uptimeMs": now_ms().saturating_sub(state.started_at),
        "protocolVersions": [1],
        "capabilities": ["compress", "decompress", "search_context", "status"],
    }))
    .into_response()
}

async fn global_status_handler(State(state): State<Arc<AppState>>) -> Response {
    let stats = state.sessions.stats_json();
    let sessions = stats
        .get("sessions")
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or(0);
    axum::Json(json!({
        "version": VERSION,
        "commit": "rust",
        "build": build_id::BuildInfo::current().json(),
        "origin": state.origin,
        "diskVersion": VERSION,
        "stale": false,
        "lastInstall": null,
        "updateReady": null,
        "advisory": null,
        "updatePolicy": {"source": "rust", "hold": true, "pinned": "rust", "blocked": []},
        "inFlight": 0,
        "splitSessions": [],
        "conflicts": {"sessions": 0, "events": 0, "active": 0, "historical": 0, "latest": []},
        "storage": state.sessions.storage_json(),
        "snapshotShadow": {"enabled": false, "watching": 0, "errors": 0},
        "attribution": {},
        "sessions": sessions,
        "uptimeMs": now_ms().saturating_sub(state.started_at),
    }))
    .into_response()
}

async fn stats_handler(State(state): State<Arc<AppState>>) -> Response {
    axum::Json(state.sessions.stats_json()).into_response()
}

async fn overview_handler(State(state): State<Arc<AppState>>) -> Response {
    axum::Json(state.sessions.overview_json()).into_response()
}

async fn pricing_handler(State(state): State<Arc<AppState>>) -> Response {
    let path = config_path();
    let mut value = state.sessions.pricing_json(&path.to_string_lossy());
    // The plugin's published snapshot, read-only. `price: null` above means "no
    // operator override", not "free"; this block says whether a real published
    // price is available and how old it is.
    value["snapshot"] = pricing::snapshot_status(now_ms());
    if let Ok(raw) = std::fs::read_to_string(&path) {
        if let Ok(config) = serde_json::from_str::<Value>(&raw) {
            if let Some(prices) = config
                .get("billing")
                .and_then(|billing| billing.get("prices"))
                .and_then(Value::as_object)
            {
                if let Some(models) = value.get_mut("models").and_then(Value::as_array_mut) {
                    for model in models {
                        let name = model.get("model").and_then(Value::as_str).unwrap_or("");
                        if let Some(price) = prices.get(name) {
                            model["price"] = price.clone();
                            model["override"] = price.clone();
                        }
                    }
                }
                value["overrides"] = Value::Object(prices.clone());
            }
        }
    }
    axum::Json(value).into_response()
}

fn config_key_is_secret(key: &str) -> bool {
    let key = key.to_ascii_lowercase();
    let compact: String = key.chars().filter(|c| c.is_ascii_alphanumeric()).collect();
    // Match secret-shaped names, not every name containing the substring:
    // `maxTokens` and `prompt_tokens` are limits/counts, not credentials.
    let words: Vec<&str> = key
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|word| !word.is_empty())
        .collect();
    words.iter().any(|word| {
        matches!(
            *word,
            "secret" | "password" | "authorization" | "credential"
        )
    }) || words.last() == Some(&"token")
        || compact.ends_with("apikey")
        || compact.ends_with("secret")
        || compact.ends_with("password")
        || compact.ends_with("token")
        || compact.ends_with("authorization")
        || compact.ends_with("credential")
        || compact.contains("secretkey")
        || compact.contains("passwordhash")
}

/// Replace every credential-looking value before the admin view serializes it.
/// The config file itself is never rewritten here; saving a redacted document
/// is refused by `config_put_handler` so an edit cannot silently erase secrets.
fn redact_config(value: &mut Value) -> bool {
    match value {
        Value::Object(map) => {
            let mut redacted = false;
            for (key, child) in map.iter_mut() {
                if config_key_is_secret(key) && !child.is_null() {
                    *child = Value::String("***REDACTED***".to_string());
                    redacted = true;
                } else {
                    redacted |= redact_config(child);
                }
            }
            redacted
        }
        Value::Array(items) => {
            let mut redacted = false;
            for child in items.iter_mut() {
                redacted |= redact_config(child);
            }
            redacted
        }
        _ => false,
    }
}

async fn config_get_handler() -> Response {
    let path = config_path();
    let exists = path.exists();
    let raw = std::fs::read_to_string(&path).unwrap_or_default();
    let mut parsed = serde_json::from_str::<Value>(&raw).ok();
    let redacted = parsed.as_mut().is_some_and(redact_config);
    let redacted_raw = parsed
        .as_ref()
        .and_then(|value| serde_json::to_string_pretty(value).ok())
        .unwrap_or_default();
    let parse_error = if exists && parsed.is_none() {
        Some("config file is not valid JSON".to_string())
    } else {
        None
    };
    axum::Json(json!({
        "ok": true,
        "path": path.to_string_lossy(),
        "exists": exists,
        // Never hand the unredacted file to an admin reader: a browser page
        // could otherwise exfiltrate provider keys just by fetching this route.
        "raw": redacted_raw,
        "parsed": parsed,
        "redacted": redacted,
        "parseError": parse_error,
    }))
    .into_response()
}

async fn config_put_handler(body: Body) -> Response {
    const MAX_BODY: usize = 2 * 1024 * 1024;
    let bytes = match axum::body::to_bytes(body, MAX_BODY).await {
        Ok(bytes) => bytes,
        Err(err) => {
            return error_response(
                StatusCode::PAYLOAD_TOO_LARGE,
                &format!("config body too large or unreadable: {err}"),
            );
        }
    };
    let request: Value = match serde_json::from_slice(&bytes) {
        Ok(value) => value,
        Err(err) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                &format!("expected JSON object: {err}"),
            );
        }
    };
    if !request.is_object() {
        return error_response(StatusCode::BAD_REQUEST, "expected JSON object");
    }
    if request.to_string().contains("***REDACTED***") {
        return error_response(
            StatusCode::BAD_REQUEST,
            "refusing to save a redacted config; edit the file on disk or re-enter the missing secrets",
        );
    }
    let path = config_path();
    let next = if let Some(file) = request.get("file") {
        let Some(file) = file.as_str() else {
            return error_response(StatusCode::BAD_REQUEST, "file must be a JSON string");
        };
        if file.contains("***REDACTED***") {
            return error_response(
                StatusCode::BAD_REQUEST,
                "refusing to save a redacted config; edit the file on disk or re-enter the missing secrets",
            );
        }
        match serde_json::from_str::<Value>(file) {
            Ok(value) if value.is_object() => value,
            Ok(_) => {
                return error_response(StatusCode::BAD_REQUEST, "file must contain a JSON object");
            }
            Err(err) => {
                return error_response(
                    StatusCode::BAD_REQUEST,
                    &format!("file is not valid JSON: {err}"),
                );
            }
        }
    } else if let Some(dsh) = request.get("dsh").and_then(Value::as_object) {
        let current = std::fs::read_to_string(&path).ok();
        let mut root = match current.as_deref() {
            Some(raw) => match serde_json::from_str::<Value>(raw) {
                Ok(value) if value.is_object() => value,
                Ok(_) => {
                    return error_response(
                        StatusCode::CONFLICT,
                        "existing config is not an object",
                    );
                }
                Err(err) => {
                    return error_response(
                        StatusCode::CONFLICT,
                        &format!("existing config has invalid JSON; fix it by hand first: {err}"),
                    );
                }
            },
            None => json!({}),
        };
        if let Some(object) = root.as_object_mut() {
            object.insert("dsh".into(), Value::Object(dsh.clone()));
        }
        root
    } else {
        return error_response(
            StatusCode::BAD_REQUEST,
            "expected the full config under file, or a dsh merge object",
        );
    };
    let text = serde_json::to_string_pretty(&next).unwrap_or_else(|_| "{}".into());
    if let Err(err) = atomic_write_text(&path, &format!("{text}\n")) {
        return error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("cannot write {}: {err}", path.to_string_lossy()),
        );
    }
    axum::Json(json!({"ok": true, "path": path.to_string_lossy()})).into_response()
}

async fn notices_handler() -> Response {
    axum::Json(json!({"ok": true, "notices": []})).into_response()
}

async fn notice_dismiss_handler() -> Response {
    axum::Json(json!({"ok": true})).into_response()
}

fn update_state_json() -> Value {
    json!({
        "running": VERSION,
        "disk": VERSION,
        "build": build_id::BuildInfo::current().json(),
        "pendingRestart": false,
        "updateAvailable": false,
        "critical": false,
        "checking": false,
        "installing": false,
        "lastCheckAt": null,
        "notices": [],
        "policy": {"source": "rust", "hold": true, "pinned": "rust", "blocked": []},
    })
}

async fn update_state_handler() -> Response {
    axum::Json(update_state_json()).into_response()
}

async fn update_check_handler() -> Response {
    let mut state = update_state_json();
    state["checkedAt"] = json!(now_ms());
    axum::Json(state).into_response()
}

/// Guard the admin surface: loopback peer and loopback Host only,
/// loopback/allowlisted browser origins, and a launcher-provided token for
/// state-changing routes. CORS is emitted from the same allowlist instead of
/// the previous wildcard. The peer check matters when the proxy itself is
/// bound to a non-loopback address: a remote client can otherwise spoof
/// `Host: 127.0.0.1`.
async fn admin_guard(
    State(state): State<Arc<AppState>>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    req: Request,
    next: Next,
) -> Response {
    if !req.uri().path().starts_with("/__bili") {
        return next.run(req).await;
    }
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let origin = req
        .headers()
        .get(header::ORIGIN)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string);
    let origin_allowed = origin
        .as_deref()
        .map(|value| admin_origin_allowed(value, &state))
        .unwrap_or(true);
    let mut response = if !peer_is_loopback(peer) {
        (
            StatusCode::FORBIDDEN,
            axum::Json(json!({"ok": false, "error": "admin endpoint requires a loopback peer"})),
        )
            .into_response()
    } else if !host_header_is_loopback(req.headers()) {
        (
            StatusCode::FORBIDDEN,
            axum::Json(json!({"ok": false, "error": "admin endpoint requires a loopback Host"})),
        )
            .into_response()
    } else if origin.is_some() && !origin_allowed {
        (
            StatusCode::FORBIDDEN,
            axum::Json(json!({"ok": false, "error": "admin origin is not allowlisted"})),
        )
            .into_response()
    } else if method == Method::OPTIONS {
        StatusCode::NO_CONTENT.into_response()
    } else if admin_mutation(&method, &path)
        && !launch_token_ok(req.headers(), state.launch_token.as_deref())
    {
        (
            StatusCode::UNAUTHORIZED,
            axum::Json(json!({"ok": false, "error": "missing or invalid launch token"})),
        )
            .into_response()
    } else {
        next.run(req).await
    };
    add_cors_headers(
        &mut response,
        if origin_allowed {
            origin.as_deref()
        } else {
            None
        },
    );
    response
}

async fn panel_handler(State(state): State<Arc<AppState>>) -> Response {
    let html = PANEL_HTML.replace("__ORIGIN__", &state.origin);
    (
        StatusCode::OK,
        [("content-type", "text/html; charset=utf-8")],
        html,
    )
        .into_response()
}

async fn upstreams_handler(State(state): State<Arc<AppState>>) -> Response {
    let mut rows: Vec<_> = state
        .upstream_failures
        .lock()
        .map(|failures| {
            failures
                .iter()
                .map(|(host, failure)| {
                    json!({
                        "host": host,
                        "port": null,
                        "kind": "primary",
                        "count": failure.count,
                        "code": failure.code,
                        "firstSeen": failure.first_seen,
                        "lastSeen": failure.last_seen,
                        "circuit": null,
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    rows.sort_by(|a, b| a["host"].as_str().cmp(&b["host"].as_str()));
    axum::Json(json!({"ok": true, "upstreams": rows})).into_response()
}

fn current_upstream(state: &AppState) -> Option<String> {
    state
        .upstream
        .lock()
        .ok()
        .and_then(|upstream| upstream.clone())
}

fn record_upstream_failure(state: &Arc<AppState>, target: &str, code: &str) {
    let host = upstream_base(target);
    if let Ok(mut failures) = state.upstream_failures.lock() {
        let now = now_ms();
        let entry = failures.entry(host).or_insert(UpstreamFailure {
            count: 0,
            first_seen: now,
            last_seen: now,
            code: code.to_string(),
        });
        entry.count = entry.count.saturating_add(1);
        entry.last_seen = now;
        entry.code = code.to_string();
    }
}

fn clear_upstream_failure(state: &Arc<AppState>, target: &str) {
    let host = upstream_base(target);
    if let Ok(mut failures) = state.upstream_failures.lock() {
        failures.remove(&host);
    }
}

fn resolve_target(rest: &str) -> Option<String> {
    let candidate = rest.trim_start();
    if candidate.starts_with("http://") || candidate.starts_with("https://") {
        return Some(candidate.to_string());
    }
    let decoded = percent_encoding::percent_decode_str(candidate)
        .decode_utf8()
        .ok()?;
    let decoded = decoded.trim_start();
    if decoded.starts_with("http://") || decoded.starts_with("https://") {
        Some(decoded.to_string())
    } else {
        None
    }
}

fn error_response(status: StatusCode, message: &str) -> Response {
    (status, message.to_string()).into_response()
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct HealthWatchdog {
    armed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    parent_pid: Option<u32>,
    watchers: Vec<u32>,
}

async fn health_handler(State(state): State<Arc<AppState>>) -> Response {
    let watchers: Vec<u32> = state
        .watchers
        .lock()
        .map(|w| w.iter().copied().collect())
        .unwrap_or_default();
    let upstream = current_upstream(&state);
    let payload = json!({
        "ok": true,
        "upstream": upstream,
        "version": VERSION,
        "build": build_id::BuildInfo::current().json(),
        "instanceId": state.instance_id,
        "pid": std::process::id(),
        "startedAt": state.started_at,
        "blindTunnels": {"total": 0, "byHost": {}},
        "watchdog": HealthWatchdog {
            armed: state.parent_pid.is_some(),
            parent_pid: state.parent_pid,
            watchers,
        }
    });
    (
        StatusCode::OK,
        [("content-type", "application/json")],
        payload.to_string(),
    )
        .into_response()
}

async fn watcher_handler(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    let parsed = serde_json::from_slice::<serde_json::Value>(&body).ok();
    let pid = parsed
        .as_ref()
        .and_then(|v| v.get("pid"))
        .and_then(|v| v.as_u64())
        .map(|v| v as u32);
    match pid {
        Some(pid) if pid > 1 && pid != std::process::id() => {
            if state.parent_pid.is_none() {
                (StatusCode::CONFLICT, [("content-type", "application/json")],
                 json!({"ok": false, "error": "watchdog not armed (no parent pid) — this proxy does not take watchers"}).to_string())
                    .into_response()
            } else {
                if let Ok(mut set) = state.watchers.lock() {
                    set.insert(pid);
                }
                (StatusCode::OK, [("content-type", "application/json")],
                 json!({"ok": true, "watchers": state.watchers.lock().map(|w| w.len()).unwrap_or(0)}).to_string())
                    .into_response()
            }
        }
        _ => (
            StatusCode::BAD_REQUEST,
            [("content-type", "application/json")],
            json!({"ok": false, "error": "expected { pid: <integer>, hb?: true }"}).to_string(),
        )
            .into_response(),
    }
}

async fn shutdown_handler(State(state): State<Arc<AppState>>) -> Response {
    let _ = state.shutdown_tx.send(true);
    (
        StatusCode::ACCEPTED,
        [("content-type", "application/json")],
        json!({"ok": true}).to_string(),
    )
        .into_response()
}

#[cfg(windows)]
fn pid_alive(pid: u32) -> bool {
    use windows_sys::Win32::System::Threading::{
        GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
    };
    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if handle.is_null() {
            return false;
        }
        let mut code: u32 = 0;
        let ok = GetExitCodeProcess(handle, &mut code) != 0;
        let _ = windows_sys::Win32::Foundation::CloseHandle(handle);
        ok && code == 259 // STILL_ACTIVE
    }
}

#[cfg(not(windows))]
fn pid_alive(pid: u32) -> bool {
    Path::new(&format!("/proc/{pid}")).exists()
}

#[tokio::main]
async fn main() {
    let opts = opts_from_args();
    if !opts.allow_remote_bind && !host_is_loopback(&opts.host) {
        eprintln!(
            "[billion-context-proxy] refusing non-loopback bind {}; pass --allow-remote-bind or BILI_ALLOW_REMOTE_BIND=1",
            opts.host
        );
        std::process::exit(2);
    }
    if opts.allow_remote_bind && !host_is_loopback(&opts.host) {
        eprintln!(
            "[billion-context-proxy] WARNING: listening on non-loopback {}; admin routes accept loopback peers only and upstream targets are allowlisted",
            opts.host
        );
    }
    let credential =
        d1b_runner::credential_status(crate::summarizer::EXTERNAL_SUMMARY_CREDENTIAL_ENV);
    eprintln!(
        "[billion-context-proxy] external summary credential ({}) : {}",
        crate::summarizer::EXTERNAL_SUMMARY_CREDENTIAL_ENV,
        match credential {
            d1b_runner::CredentialStatus::Present => "present",
            d1b_runner::CredentialStatus::Missing => "missing",
        }
    );
    if let Some(mut run) = opts.d1b.clone() {
        let provider = run
            .provider
            .clone()
            .unwrap_or_else(|| "oneagent".to_string());
        let env_name = d1b_provider_credential_env(&provider);
        if env_name.is_some() {
            // `d1b_provider_credential_env` already checked the fixed
            // allowlist; this method sets both provider and env together.
            run.bind_credential(&provider).ok();
        }
        let credential = d1b_runner::credential_status(env_name.as_deref().unwrap_or(""));
        eprintln!(
            "[billion-context-proxy] d1b credential ({provider} / {}) : {}",
            env_name.as_deref().unwrap_or("<unbound>"),
            match credential {
                d1b_runner::CredentialStatus::Present => "present",
                d1b_runner::CredentialStatus::Missing => "missing",
            }
        );
        run.sessions_dir = sessions_dir();
        let report = d1b_runner::run(run).await;
        println!(
            "{}",
            serde_json::to_string_pretty(&report)
                .unwrap_or_else(|_| "{\"error\":\"report serialize\"}".into())
        );
        std::process::exit(d1b_runner::exit_code(&report));
    }
    if opts.cleanup_artifacts {
        let store = session_store::Store::with_dir(sessions_dir().join("rust-sessions"));
        match store.cleanup_artifacts_default() {
            Ok(report) => {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&report)
                        .unwrap_or_else(|_| "{\"error\":\"report serialize\"}".into())
                );
                std::process::exit(0);
            }
            Err(err) => {
                eprintln!("[billion-context-proxy] artifact cleanup refused: {err}");
                std::process::exit(4);
            }
        }
    }
    let (listener, bound_port) =
        match bind_with_ladder(&opts.host, opts.port, opts.strict_port).await {
            Ok(v) => v,
            Err(err) => {
                eprintln!("[billion-context-proxy] {err}");
                std::process::exit(3);
            }
        };
    let display_host = if opts.host == "0.0.0.0" {
        "127.0.0.1".to_string()
    } else {
        opts.host.clone()
    };
    let origin = format!("http://{display_host}:{bound_port}");
    let instance_id = Uuid::new_v4().to_string();
    let started_at = now_ms();
    let parent_pid = env_str("BILI_PARENT_PID").and_then(|v| v.parse::<u32>().ok());
    let lane = env_str("BILI_LAUNCHER_LANE");
    let launch_token = env_str("BILI_LAUNCH_TOKEN");
    let code_fingerprint = env_str("BILI_CODE_FINGERPRINT");
    let model_windows = parse_json_env_map("BILI_LAUNCHER_MODEL_WINDOWS");
    let model_max_outputs = {
        let m = parse_json_env_map("BILI_LAUNCHER_MODEL_MAX_OUTPUTS");
        if m.is_empty() { None } else { Some(m) }
    };
    let mitm_domains = mitm_domains_from_env();
    let upstream_allowlist = upstream_allowlist_from_env(&mitm_domains);
    let allow_private_upstreams = env_str("BILI_ALLOW_PRIVATE_UPSTREAM").as_deref() == Some("1");
    let admin_origins = admin_origins_from_env(&origin, bound_port);
    let sessions_dir = sessions_dir();

    let client = match reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .no_proxy()
        .build()
    {
        Ok(c) => c,
        Err(err) => {
            eprintln!("[billion-context-proxy] failed to build HTTP client: {err}");
            std::process::exit(1);
        }
    };

    let (shutdown_tx, mut shutdown_rx) = watch::channel(false);
    let initial_watchers: HashSet<u32> = parent_pid.into_iter().collect();
    let state = Arc::new(AppState {
        client,
        origin: origin.clone(),
        host: display_host.clone(),
        port: bound_port,
        instance_id: instance_id.clone(),
        started_at,
        passthrough: opts.passthrough,
        debug: opts.debug,
        parent_pid,
        watchers: Mutex::new(initial_watchers),
        shutdown_tx: shutdown_tx.clone(),
        sessions: session_store::Store::with_dir(sessions_dir.join("rust-sessions")),
        upstream: Mutex::new(env_str("BILI_UPSTREAM")),
        upstream_failures: Mutex::new(HashMap::new()),
        usage_source: opts.usage_source,
        auto_fallback: auto_fallback_enabled(),
        speculation: speculation_enabled(),
        launch_token: launch_token.clone(),
        admin_origins,
        upstream_allowlist,
        allow_private_upstreams,
    });

    let instance = ProxyInstanceFile {
        origin: origin.clone(),
        instance_id: instance_id.clone(),
        pid: std::process::id(),
        started_at,
        host: display_host.clone(),
        port: bound_port,
        passthrough: opts.passthrough,
        mitm_domains: mitm_domains.clone(),
        model_windows: model_windows.clone(),
        model_max_outputs: model_max_outputs.clone(),
        launch_token: launch_token.clone(),
        lane: lane.clone(),
        code_fingerprint: code_fingerprint.clone(),
    };
    let registry = RegistryEntry {
        instance_id: instance_id.clone(),
        pid: std::process::id(),
        port: bound_port,
        origin: origin.clone(),
        started_at,
        host: display_host.clone(),
        passthrough: opts.passthrough,
        mitm_domains,
        model_windows,
        model_max_outputs,
        lane,
        code_fingerprint,
        sessions_dir: Some(sessions_dir.to_string_lossy().to_string()),
    };
    if let Err(err) = atomic_write_json(&state_dir().join("proxy-origin"), &instance) {
        eprintln!("[billion-context-proxy] warning: cannot write proxy-origin: {err}");
    }
    let registry_path = state_dir()
        .join("instances")
        .join(format!("{instance_id}.json"));
    if let Err(err) = atomic_write_json(&registry_path, &registry) {
        eprintln!("[billion-context-proxy] warning: cannot write registry marker: {err}");
    }

    // Parent watchdog (#964): die with the host that spawned us instead of
    // leaking a detached proxy over the same sessions directory.
    if let Some(parent) = parent_pid {
        let tx = shutdown_tx.clone();
        tokio::spawn(async move {
            let mut tick = time::interval(Duration::from_secs(2));
            loop {
                tick.tick().await;
                if !pid_alive(parent) {
                    eprintln!("[billion-context-proxy] parent pid {parent} is gone; shutting down");
                    let _ = tx.send(true);
                    break;
                }
            }
        });
    }

    let app = Router::new()
        .route("/bili", any(forward_handler))
        .route("/bili/{*rest}", any(forward_handler))
        .route("/__bili/health", get(health_handler))
        .route("/__bili/watcher", post(watcher_handler))
        .route("/__bili/shutdown", post(shutdown_handler))
        .route("/__bili/plugin/manifest", get(manifest_handler))
        .route("/__bili/plugin/tool", post(plugin_tool_handler))
        .route("/__bili/plugin/status", get(plugin_status_handler))
        .route("/__bili/plugin/runtime-info", get(runtime_info_handler))
        .route("/__bili/status", get(global_status_handler))
        .route("/__bili/stats", get(stats_handler))
        .route("/__bili/overview", get(overview_handler))
        .route("/__bili/pricing", get(pricing_handler))
        .route(
            "/__bili/config",
            get(config_get_handler).put(config_put_handler),
        )
        .route("/__bili/notices", get(notices_handler))
        .route("/__bili/notices/dismiss", post(notice_dismiss_handler))
        .route("/__bili/update/state", get(update_state_handler))
        .route("/__bili/update/check", post(update_check_handler))
        .route("/__bili/session-report", get(session_report::handler))
        .route("/__bili/", get(panel_handler))
        .route("/__bili", get(panel_handler))
        .route("/__bili/upstreams", get(upstreams_handler))
        .layer(middleware::from_fn_with_state(state.clone(), admin_guard))
        .with_state(state.clone());

    eprintln!(
        "[billion-context-proxy] listening on {origin} (passthrough={})",
        opts.passthrough
    );
    let shutdown_signal = async move {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = shutdown_rx.changed() => {}
        }
    };
    if let Err(err) = axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal)
    .await
    {
        eprintln!("[billion-context-proxy] serve error: {err}");
    }

    // Cleanup markers only when they still point at us.
    if let Ok(raw) = std::fs::read_to_string(state_dir().join("proxy-origin")) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) {
            let ours = v.get("instanceId").and_then(|x| x.as_str()) == Some(instance_id.as_str());
            if ours {
                let _ = std::fs::remove_file(state_dir().join("proxy-origin"));
            }
        }
    }
    let _ = std::fs::remove_file(&registry_path);
    eprintln!("[billion-context-proxy] exited");
}

#[cfg(test)]
mod guard_tests {
    use super::*;

    #[test]
    fn wildcard_allowlist_requires_a_dot_boundary() {
        let allowlist = vec!["*.example.com".to_string()];
        assert!(upstream_target_allowed(
            "https://api.example.com/v1",
            &allowlist,
            false
        ));
        assert!(!upstream_target_allowed(
            "https://evil-example.com/v1",
            &allowlist,
            false
        ));
        assert!(!upstream_target_allowed(
            "https://example.com/v1",
            &allowlist,
            false
        ));
    }

    #[test]
    fn model_wires_force_identity_encoding() {
        assert!(!request_header_is_forwardable("accept-encoding", true));
        assert!(request_header_is_forwardable("accept-encoding", false));
        assert!(request_header_is_forwardable("authorization", true));
        assert!(!request_header_is_forwardable("host", false));
        assert!(!request_header_is_forwardable("content-length", true));
    }

    #[test]
    fn loopback_peers_include_ipv4_mapped_ipv6() {
        assert!(peer_is_loopback(SocketAddr::from(([127, 0, 0, 1], 7000))));
        assert!(peer_is_loopback(SocketAddr::from((
            [0, 0, 0, 0, 0, 0, 0, 1],
            7000
        ))));
        assert!(peer_is_loopback(SocketAddr::from((
            [0, 0, 0, 0, 0, 0xffff, 0x7f00, 0x0001],
            7000
        ))));
        assert!(!peer_is_loopback(SocketAddr::from((
            [192, 168, 15, 201],
            7000
        ))));
    }
}

//! billion-context-proxy — Rust replacement for the billion-context Node proxy host.
//!
//! Accepts model traffic rewritten to `/bili/<absolute-url>`, converts OpenAI
//! Chat Completions / Anthropic Messages into the billion-context kernel,
//! returns the folded projection, and relays streaming SSE bodies unchanged.
//! Publishes the plugin manifest/tool/status surface plus a lightweight admin
//! panel, and writes the `proxy-origin` / `instances/<id>.json` markers in the
//! Node layout so a native launcher can discover a live instance.

// Some helpers mirror the Node host surface but are not wired into every
// endpoint yet; keep them available for parity work without failing `-D warnings`.
#![allow(dead_code)]

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
    extract::{OriginalUri, Query, Request, State},
    http::{HeaderMap, HeaderValue, Method, StatusCode, header},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{any, get, post},
};
use serde::Serialize;
use serde_json::json;

mod anthropic;
mod openai;
mod session_report;
mod session_store;
mod usage;

use tokio::{net::TcpListener, sync::watch, time};
use uuid::Uuid;

use serde_json::Value;

const VERSION: &str = env!("CARGO_PKG_VERSION");
const HEALTH_PATH: &str = "/__bili/health";

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
    /// Per-instance token written by the launcher.  Admin mutations require it
    /// when present; absent means standalone development mode (still guarded by
    /// the loopback Host and Origin checks).
    launch_token: Option<String>,
    /// Exact non-loopback panel origins allowed to call `/__bili/*`.  Loopback
    /// origins are always accepted so the DSH web view keeps working.
    admin_origins: Vec<String>,
    /// Hosts the forward proxy may dial.  Entries support exact names and
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
            "billion-context-proxy {} — usage: billion-context-proxy [start] [--host H] [--port N] [--passthrough] [--debug] [--allow-remote-bind]",
            VERSION
        );
        std::process::exit(0);
    }
    if args.iter().any(|a| a == "--version" || a == "-V") {
        println!("{VERSION}");
        std::process::exit(0);
    }
    let mut host = env_str("BILI_HOST").unwrap_or_else(|| "127.0.0.1".into());
    let mut port = env_u16("BILI_PORT").unwrap_or(DEFAULT_PORT);
    let mut passthrough = env_str("BILI_PASSTHROUGH").as_deref() == Some("1");
    let mut debug = env_str("BILI_DEBUG").as_deref() == Some("1");
    let mut strict_port = env_str("BILI_STRICT_PORT").as_deref() == Some("1");
    let mut allow_remote_bind = env_str("BILI_ALLOW_REMOTE_BIND").as_deref() == Some("1");
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "start" => {}
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
            other => {
                // Unknown flags belong to the Node CLI surface we do not emulate
                // yet; ignore rather than fail so a hand-rolled spawn still boots.
                eprintln!("[billion-context-proxy] ignoring unrecognized argument: {other}");
            }
        }
        i += 1;
    }
    Options {
        host,
        port,
        passthrough,
        debug,
        strict_port,
        allow_remote_bind,
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

/// Only http(s) targets whose host is explicitly listed may be dialed.  A
/// literal IP is accepted only when the operator opted in (`BILI_ALLOW_PRIVATE_UPSTREAM=1`)
/// or wrote that exact IP into the allowlist (local mock upstreams).
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
            host.ends_with(suffix) && host.len() > suffix.len()
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
/// configured panel origin.  `Origin: null` and public web origins are denied.
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
            "upstream target is not in the configured allowlist",
        );
    }
    if let Ok(mut upstream) = state.upstream.lock() {
        *upstream = Some(upstream_base(&target));
    }
    if state.debug {
        eprintln!(
            "[billion-context-proxy] {} {} -> {}",
            method.as_str(),
            raw,
            target
        );
    }

    let mut forward = HeaderMap::new();
    for (name, value) in headers.iter() {
        let n = name.as_str().to_ascii_lowercase();
        if matches!(
            n.as_str(),
            "host" | "connection" | "content-length" | "transfer-encoding" | "expect" | "upgrade"
        ) {
            continue;
        }
        forward.insert(name, value.clone());
    }

    let mut request = state.client.request(method.clone(), &target);
    request = request.headers(forward);

    let mut usage_meta: Option<(String, Option<String>)> = None;

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
        let payload = if state.passthrough {
            payload
        } else {
            let prepared = maybe_prepare(&state, &target, &headers, payload);
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
        request = request
            .header(reqwest::header::CONTENT_LENGTH, payload.len())
            .body(payload);
    }

    let upstream = match request.send().await {
        Ok(resp) => resp,
        Err(err) => {
            eprintln!("[billion-context-proxy] upstream error for {target}: {err}");
            record_upstream_failure(&state, &target, "TRANSPORT");
            return error_response(
                StatusCode::BAD_GATEWAY,
                &format!("upstream relay failed: {err}"),
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

/// Rewrite an Anthropic Messages body through the kernel. Failure to parse or
/// to recognize the wire leaves the request untouched (fail-open: a broken
/// rewrite must never cost the user a turn).
fn maybe_prepare(state: &AppState, target: &str, headers: &HeaderMap, payload: Bytes) -> Bytes {
    if state.passthrough || !target_has_model_wire(target) {
        return payload;
    }
    let Ok(mut body) = serde_json::from_slice::<Value>(&payload) else {
        return payload;
    };
    if body.get("messages").and_then(Value::as_array).is_none() {
        return payload;
    }
    let key = conversation_key(headers, &body);
    let context_limit = context_limit(headers);
    let plugin_mode = headers.contains_key("x-bili-plugin");
    let reset_window =
        headers.contains_key("x-bili-plugin-reset") || headers.contains_key("x-bili-context-reset");
    let model = headers
        .get("x-bili-plugin-model")
        .and_then(|value| value.to_str().ok())
        .map(str::to_string)
        .or_else(|| {
            body.get("model")
                .and_then(Value::as_str)
                .map(str::to_string)
        });
    let openai_wire = target.contains("/chat/completions");
    if openai_wire {
        let token_count = (openai::estimate_body_tokens(&body) as u32).saturating_add(800);
        let converted = openai::to_core(&body);
        let prepared = state.sessions.prepare_with_reset(
            &key,
            context_limit,
            &converted.messages,
            token_count,
            model.clone(),
            reset_window,
        );
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
            openai::inject_tools(&mut body, &public_tools());
        }
        if state.debug {
            eprintln!(
                "[billion-context-proxy] prepared openai {key}: {} messages, tokens {}",
                body["messages"].as_array().map(Vec::len).unwrap_or(0),
                token_count
            );
        }
        return match serde_json::to_vec(&body) {
            Ok(bytes) => Bytes::from(bytes),
            Err(_) => payload,
        };
    }
    let token_count = (anthropic::estimate_body_tokens(&body) as u32).saturating_add(800);
    let converted = anthropic::to_core(&body);
    let prepared =
        state
            .sessions
            .prepare(&key, context_limit, &converted.messages, token_count, model);
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
        let defs = public_tools();
        anthropic::inject_tools(&mut body, &defs);
    }
    if state.debug {
        eprintln!(
            "[billion-context-proxy] prepared {key}: {} -> {} messages, tokens {}",
            body["messages"].as_array().map(Vec::len).unwrap_or(0),
            rebuilt_len(&body),
            token_count
        );
    }
    match serde_json::to_vec(&body) {
        Ok(bytes) => Bytes::from(bytes),
        Err(_) => payload,
    }
}

fn target_has_model_wire(target: &str) -> bool {
    target.contains("/v1/messages") || target.contains("/chat/completions")
}

fn upstream_base(target: &str) -> String {
    let scheme_end = target.find("://").map(|index| index + 3);
    let rest_start = scheme_end.unwrap_or(0);
    let host_end = target[rest_start..]
        .find('/')
        .map(|index| rest_start + index)
        .unwrap_or(target.len());
    target[..host_end].to_string()
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

fn context_limit(headers: &HeaderMap) -> u32 {
    headers
        .get("x-bili-plugin-context-window")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.trim().parse::<u32>().ok())
        .filter(|value| (1024..=2_000_000).contains(value))
        .unwrap_or(200_000)
}

/// Public tool surface: the Node proxy's names and schemas, shared by the DSH
/// manifest and the non-plugin tool injection.
fn public_tools() -> Vec<Value> {
    let tools = json!([
        {
            "name": "compress",
            "description": "Replace consumed conversation ranges with self-contained summaries using mNNNNN or bN refs; batch multiple ranges into ONE call (a single string may hold every range).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "ONE JSON-escaped string holding ALL ranges — one block per range, first line 'm00150–m00220 optional topic', remaining lines the summary markdown."
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
    let tools = Value::Array(public_tools());
    let names: Vec<&str> = tools
        .as_array()
        .map(|list| {
            list.iter()
                .filter_map(|tool| tool.get("name").and_then(Value::as_str))
                .collect()
        })
        .unwrap_or_default();
    axum::Json(json!({
        "ok": true,
        "protocolVersion": 1,
        "proxy": state.origin,
        "version": VERSION,
        "commit": "rust",
        "toolNames": names,
        "tools": { "anthropic": tools, "openai": [], "responses": [] },
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
    let limit = context_limit(&headers);
    let args = parsed.get("args").cloned().unwrap_or_else(|| json!({}));
    let (text, outcome, blocks) = state.sessions.tool(&key, limit, tool, &args);
    let mut reply = json!({
        "ok": true,
        "tool": tool,
        "conversationId": key,
        "outcome": outcome.unwrap_or("success"),
        "result": text,
    });
    if let Some(blocks) = blocks {
        reply["blocksCreated"] = json!(blocks);
    }
    axum::Json(reply).into_response()
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
            Some(status) => return axum::Json(status).into_response(),
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
    axum::Json(state.sessions.latest_status_json().unwrap_or_else(
        || json!({"ok": true, "phase": "pre-first-request", "blocks": [], "viaFallback": true}),
    ))
    .into_response()
}

async fn runtime_info_handler(State(state): State<Arc<AppState>>) -> Response {
    axum::Json(json!({
        "ok": true,
        "runtime": "rust",
        "version": VERSION,
        "commit": "rust",
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
    // Match secret-shaped names, not every name containing the substring: `maxTokens`
    // and `prompt_tokens` are limits/counts, not credentials.
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

/// Guard the admin surface: loopback Host only, loopback/allowlisted browser
/// origins, and a launcher-provided token for state-changing routes.  CORS is
/// emitted from the same allowlist instead of the previous wildcard.
async fn admin_guard(State(state): State<Arc<AppState>>, req: Request, next: Next) -> Response {
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
    let mut response = if !host_header_is_loopback(req.headers()) {
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
            "[billion-context-proxy] WARNING: listening on non-loopback {}; admin routes stay loopback-only and upstream targets are allowlisted",
            opts.host
        );
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
        launch_token: launch_token.clone(),
        admin_origins: admin_origins_from_env(&origin, bound_port),
        upstream_allowlist: upstream_allowlist_from_env(&mitm_domains),
        allow_private_upstreams: env_str("BILI_ALLOW_PRIVATE_UPSTREAM").as_deref() == Some("1"),
    });
    if launch_token.is_none() {
        eprintln!(
            "[billion-context-proxy] warning: BILI_LAUNCH_TOKEN is unset; admin mutations are guarded only by loopback Host/Origin"
        );
    }

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
    if let Err(err) = axum::serve(listener, app)
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
mod tests {
    use super::*;

    fn state() -> AppState {
        let (shutdown_tx, _) = watch::channel(false);
        AppState {
            client: reqwest::Client::new(),
            origin: "http://127.0.0.1:18787".into(),
            host: "127.0.0.1".into(),
            port: 18787,
            instance_id: "test".into(),
            started_at: 0,
            passthrough: false,
            debug: false,
            parent_pid: None,
            watchers: Mutex::new(HashSet::new()),
            shutdown_tx,
            upstream: Mutex::new(None),
            upstream_failures: Mutex::new(HashMap::new()),
            sessions: session_store::Store::new(),
            launch_token: Some("secret".into()),
            admin_origins: vec!["http://127.0.0.1:18787".into()],
            upstream_allowlist: vec!["api.openai.com".into(), "*.example.com".into()],
            allow_private_upstreams: false,
        }
    }

    #[test]
    fn upstream_allowlist_is_exact_or_wildcard() {
        let allow = vec!["api.openai.com".to_string(), "*.example.com".to_string()];
        assert!(upstream_target_allowed(
            "https://api.openai.com/v1/messages",
            &allow,
            false
        ));
        assert!(upstream_target_allowed(
            "https://a.example.com/v1",
            &allow,
            false
        ));
        assert!(!upstream_target_allowed(
            "https://example.com/v1",
            &allow,
            false
        ));
        assert!(!upstream_target_allowed(
            "https://api.openai.com.evil.test/v1",
            &allow,
            false
        ));
        assert!(!upstream_target_allowed(
            "http://127.0.0.1:18080/v1",
            &allow,
            false
        ));
        assert!(!upstream_target_allowed(
            "file:///etc/passwd",
            &allow,
            false
        ));
        assert!(!upstream_target_allowed(
            "https://user:pass@api.openai.com/v1",
            &allow,
            false
        ));
    }

    #[test]
    fn private_upstreams_require_explicit_opt_in() {
        assert!(upstream_target_allowed(
            "http://127.0.0.1:18080/v1",
            &[],
            true
        ));
        assert!(upstream_target_allowed(
            "http://127.0.0.1:18080/v1",
            &["127.0.0.1".into()],
            false
        ));
    }

    #[test]
    fn host_and_origin_loopback_detection() {
        let mut headers = HeaderMap::new();
        headers.insert(header::HOST, HeaderValue::from_static("127.0.0.1:18787"));
        assert!(host_header_is_loopback(&headers));
        headers.insert(header::HOST, HeaderValue::from_static("[::1]:18787"));
        assert!(host_header_is_loopback(&headers));
        headers.insert(header::HOST, HeaderValue::from_static("evil.test"));
        assert!(!host_header_is_loopback(&headers));
        let state = state();
        assert!(admin_origin_allowed("http://127.0.0.1:54528", &state));
        assert!(admin_origin_allowed("http://[::1]:54528", &state));
        assert!(!admin_origin_allowed("http://evil.test", &state));
        assert!(!admin_origin_allowed("null", &state));
    }

    #[test]
    fn launch_token_only_required_for_mutations() {
        let mut headers = HeaderMap::new();
        assert!(!launch_token_ok(&headers, Some("secret")));
        headers.insert("x-bili-launch-token", HeaderValue::from_static("wrong"));
        assert!(!launch_token_ok(&headers, Some("secret")));
        headers.insert("x-bili-launch-token", HeaderValue::from_static("secret"));
        assert!(launch_token_ok(&headers, Some("secret")));
        headers.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("Bearer secret"),
        );
        assert!(launch_token_ok(&headers, Some("secret")));
        assert!(launch_token_ok(&headers, None));
        assert!(!admin_mutation(&Method::GET, "/__bili/config"));
        assert!(admin_mutation(&Method::PUT, "/__bili/config"));
        assert!(!admin_mutation(&Method::POST, "/__bili/plugin/tool"));
        assert!(!admin_mutation(&Method::POST, "/__bili/watcher"));
        assert!(admin_mutation(&Method::POST, "/__bili/shutdown"));
    }

    #[test]
    fn config_redaction_hides_credentials_recursively() {
        let mut config = json!({
            "providers": {
                "openai": {"apiKey": "sk-live", "baseURL": "https://api.openai.com"},
                "anthropic": {"api_key": "sk-ant", "headers": {"Authorization": "Bearer x"}}
            },
            "model": "gpt"
        });
        assert!(redact_config(&mut config));
        assert_eq!(config["providers"]["openai"]["apiKey"], "***REDACTED***");
        assert_eq!(
            config["providers"]["anthropic"]["api_key"],
            "***REDACTED***"
        );
        assert_eq!(
            config["providers"]["anthropic"]["headers"]["Authorization"],
            "***REDACTED***"
        );
        assert_eq!(config["model"], "gpt");
        assert!(!config_key_is_secret("baseURL"));
        assert!(!config_key_is_secret("maxTokens"));
        assert!(!config_key_is_secret("max_tokens"));
        assert!(!config_key_is_secret("prompt_tokens"));
        assert!(!config_key_is_secret("token_count"));
        assert!(!config_key_is_secret("tokenBudget"));
        assert!(config_key_is_secret("access_token"));
        assert!(config_key_is_secret("clientSecret"));
        assert!(config_key_is_secret("secretKey"));
    }
}

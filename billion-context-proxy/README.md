# billion-context-proxy (Rust)

billion-context 宿主/代理的 Rust 实现。把 OpenAI Chat Completions 与 Anthropic
Messages 的请求接入同一 workspace 的 `billion-context` 内核，按折后投影上行，并在
本地提供插件 manifest/tool/status、stats 和轻量管理面板。

## 版本说明

本仓库涉及三个互不换算的版本号：

| 版本 | 载体 | 含义 |
| --- | --- | --- |
| `0.1.216` | npm 包 `billion-context`（TypeScript 项目） | 上游 Node 端发行版本，与 Rust 侧无关 |
| `0.6.0` | 本仓库 `[workspace.package]`，由 `billion-context` 等 crate 继承 | Rust 移植的 workspace 版本 |
| `0.1.217` | `[package] billion-context-proxy` | **本地内部发布流水号**，未发布到任何 registry |

`billion-context-proxy 0.1.217` 不代表上游已发布过 0.1.217 —— npm 的
`billion-context` 停在 `0.1.216`，crates.io 上也没有 `billion-context-proxy`。
它只用于标识本机构建的二进制身份。

## 构建与运行

```powershell
cargo build -p billion-context-proxy --release
cargo test -p billion-context-proxy
target\release\billion-context-proxy.exe start --port 18113 --debug
```

默认 `127.0.0.1:18787`；`--passthrough` 关闭压缩，`--strict-port` 端口占用时退出。
状态目录为 `~/.local/state/billion-context`（可用 `XDG_STATE_HOME` 覆盖），与 Node
版同级但不读写其数据库。

## 安全边界

- 默认只绑定回环地址；绑定非回环地址必须显式 `--allow-remote-bind`（或
  `BILI_ALLOW_REMOTE_BIND=1`），启动时会打印警告。
- `/__bili/*` 管理面要求回环 `Host` 和回环 peer；即使绑定非回环地址，远程连接也
  不能访问管理面。浏览器 `Origin` 只接受回环来源或 `BILI_ADMIN_ORIGINS`（逗号
  分隔），CORS 不再回 `*`。
- 设置 `BILI_LAUNCH_TOKEN` 时，管理写操作（非 GET/HEAD/OPTIONS）必须携带
  `x-bili-launch-token` 或 `Authorization: Bearer <token>`；插件工具车道与
  watcher 心跳豁免，避免压缩调用和生命周期心跳被令牌卡住。Node 启动器会注入
  该变量，面板或中继需要把令牌转发到宿主。
- `GET /__bili/config` 会递归脱敏 `apiKey`、`token`、`secret`、`password` 等字段；
  `PUT` 拒绝任何包含 `***REDACTED***` 的请求（包括 `dsh` 局部合并），避免把脱敏值
  写回磁盘。改配置请直接编辑配置文件，或重新填写缺失的密钥。
- `/bili/<absolute-url>` 只允许默认 provider 域名、`BILI_MITM_DOMAINS` 和
  `BILI_UPSTREAM_ALLOWLIST`；`*.example.com` 表示子域通配。本地 mock 上游需把
  精确 IP/host 加入 allowlist，或设置 `BILI_ALLOW_PRIVATE_UPSTREAM=1`。

## 模块

| 文件 | 职责 |
| --- | --- |
| `src/main.rs` | CLI、axum 路由、`/bili/` 转发、插件与管理端点、HTML 面板 |
| `src/anthropic.rs` | Anthropic wire ↔ 内核消息、工具注入、cache 断点、ref 标签 |
| `src/openai.rs` | OpenAI wire ↔ 内核消息、system 指令注入、工具注入 |
| `src/session_store.rs` | 会话表、prepare/tool/status、range 预检、面板文本 |
| `src/summarizer.rs` | `compress.externalSummary` 外部摘要链（Provider 配方、凭据、切片、调用与规范化） |
| `src/dsh_compaction.rs` | DSH 原生 compaction 调用识别、bili 快照本地应答、无 block 时 422 拒绝 |
| `src/usage.rs` | SSE/JSON usage 旁路采集（按协议解释 cached/context） |
| `tests/mock-upstream.js` | 本地 mock 上游（18110），含 Anthropic/OpenAI/SSE |

## 接入 DSH

已安装的 `billion-context` DSH 原生插件支持两种接法；都要在 **启动 DSH 之前** 设置环境变量。

推荐 attach（不让插件再 spawn 一个代理进程）：

```powershell
$env:BILLION_CONTEXT_PROXY = "http://127.0.0.1:18187"
.\billion-context-proxy.exe start --port 18187
```

或让插件的 spawn 路径直接使用 Rust（插件把 `BILLION_CONTEXT_NODE` 当作 node
运行时，并以 `<exe> <proxy-script> start --host … --port …` 启动；Rust 会忽略
前面的脚本参数）：

```powershell
$env:BILLION_CONTEXT_NODE = "C:\path\to\billion-context-proxy.exe"
```

验收脚本：`tests/shim-attach.ps1`（attach 计划与 `verifyAttachAndRecover`）、
`tests/shim-spawn.ps1`（spawn argv 与 Rust 兼容）。两者都要求先设置
`BILI_SHIM_ROOT` 指向已安装的 `billion-context` 包目录。

## 端点

- `POST /bili/<absolute-url>`：转发；模型请求会先经过 prepare/投影。
- `GET /__bili/`、`GET /__bili/status`、`GET /__bili/stats`
- `GET /__bili/pricing`（仅返回会话模型，未内置价格表）
- `GET /__bili/session-report?session=<id>&format=md|svg`
- `GET /__bili/plugin/manifest`、`POST /__bili/plugin/tool`、
  `GET /__bili/plugin/status`、`GET /__bili/plugin/runtime-info`

## 已知限制

- 会话快照在 `rust-sessions/<sha256(key)>.json`，重启后恢复块、refs 和计数；
  Node 的 SQLite 历史不导入，视图对账（`last_view`）不持久化。
- 非插件直连 lane 不会自动拦截模型返回的 `compress` 工具调用；压缩走插件
  `/__bili/plugin/tool` 或宿主注入的 nudge/tool 面。
- 管理 SPA、图片缩放、更新/通知/配置等 Node 端点未实现；
  `pricing` 仅返回会话模型，价格注册表未内置。

## 外部摘要预检

DSH auto-compaction 的摘要请求（`x-bili-plugin(-agent): dsh` +
`x-bili-plugin-purpose: compaction`，或最后一条 user 文本以
`You are now acting as a compaction engine…` 开头）在 `/bili/` 转发入口被识别，直接用
`<bili-snapshot v1 …>` 快照本地应答（200 + `x-bili-gate: dsh-compaction-authored`），
wire 形状跟随请求的 `stream` 字段；没有 active block 时返回 422
（`x-bili-gate: dsh-compaction-refused`）而不是让 checkpoint 遮蔽原文历史。
开关：`BILI_DSH_AUTHORED_COMPACTION`（≠`0` 开）否则配置 `dsh.authoredCompaction`（默认开）；
诊断用 `BILI_DSH_COMPACTION_REFUSE=0` / `dsh.authoredCompactionRefuse=false` 可恢复转发。

## 外部摘要链（compress.externalSummary）

与 Node 版共用同一份 `~/.config/billion-context/billion-context.json`（本机也可用
`BILI_CONFIG_FILE` 指定）：`providers.<name>` 声明 `baseUrl`、`api`（本实现只支持
`openai`）、恰好一个凭据 `apiKeyEnv` 或 `credentialRef`、以及 `models.<model>` 的
`contextWindow`/`outputTokens`/`stream`；`compress.externalSummary` 打开外链
（`enabled`、`targets: ["provider/model"]`、`budget.totalTimeoutMs/targetTimeoutMs/maxSummaryBytes/maxInputChars/maxOutputTokens/maxCallsPerSession/maxSpeculationCalls`）。
预算只约束两条自动路径（Auto Fallback 与后台推测，默认都关闭）：单次输入/输出上限、会
话级调用上限、推测独立子预算；预算耗尽时自动调用停止并释放状态，但手动 `compress` 与
普通对话照常工作，拒绝不会生成 Fold。完整键表与默认值以
`src/summarizer.rs` 的配置解析为准。

`/__bili/plugin/tool` 的 `compress`/`acp_compress` 在折叠前先按 range 切片调用链上模型写摘要：
成功则覆盖模型自己给的 `summary`（随后仍过 P0.2 占位/拒答闸门，并照旧追加 P0.1
`[Host-carried user intent]`）；链未启用、解析失败或全部 target 失败时保留模型摘要并打印
`chain produced no summary`，折叠照常（fail-open）。成功日志为
`[billion-context-proxy] compress.externalSummary: authored N range summary(ies) with provider/model`。

凭据解析顺序：进程 env → `~/.dsh/.credentials.yaml` 的 `refs:` 同名键。目标无法解析、
`api` 不支持或没有凭据时**整链禁用并警告**，不会静默回退主模型摘要。

# billion-context Rust host {{VERSION}} (Windows x64)

这是 billion-context 代理宿主的 Rust 实现，作为现有 Node 宿主的逐步替代品。
二进制为独立单文件 `billion-context-proxy.exe`，不依赖 Node.js；压缩内核、wire 转换和
HTTP 宿主全部在进程内运行。

`{{VERSION}}` 是本地 `billion-context-proxy` 的内部发布流水号，不是上游 `billion-context`
的版本：npm 包 `billion-context`（TypeScript 项目）为 `0.1.216`，本仓库 Rust
workspace 为 `0.6.0`，`billion-context-proxy` 未发布到任何 registry。三者互不换算。

## 本版构建身份

{{VERSION}} 起，二进制能自报构建身份，`--version` 与 HTTP 端点给出同一组值：

```powershell
.\billion-context-proxy.exe --version
# billion-context-proxy {{VERSION}}
# commit {{COMMIT}}
# profile release
# dirty false
```

`/__bili/status`、`/__bili/health`、`/__bili/plugin/runtime-info`、
`/__bili/plugin/manifest`、`/__bili/update/state` 均新增 `build` 对象：

```json
"build": { "version": "{{VERSION}}", "commit": "{{COMMIT}}", "profile": "release", "dirty": false }
```

- `version`：`Cargo.toml` 的包版本。
- `commit`：编译时检出提交的短 SHA；无 git 时写 `unknown`，构建不会因此失败。
- `profile`：`release` 或 `debug`。
- `dirty`：编译时工作区是否有未提交改动。`true` 表示这不是干净签出的产物。

身份**不含**每次构建都变的时间戳，因此同一提交重建得到同一身份。
既有字段含义未变：`version` 仍是包版本，`commit` 仍是 `"rust"` 运行时种类标记
（不是 SHA，真 SHA 在 `build.commit`）。

校验分发包完整性：

```powershell
Get-FileHash .\billion-context-proxy.exe -Algorithm SHA256
Get-Content .\SHA256SUMS.txt
```

## 快速开始

```powershell
# 试跑：监听 18113，调试日志
.\billion-context-proxy.exe start --port 18113 --debug

# 健康检查
Invoke-RestMethod http://127.0.0.1:18113/__bili/status | ConvertTo-Json -Depth 3
```

宿主把 `/bili/<绝对上游 URL>` 的请求转发到上游；例如：

```powershell
Invoke-WebRequest `
  -Uri "http://127.0.0.1:18113/bili/http://127.0.0.1:18110/v1/chat/completions" `
  -Method Post -ContentType "application/json" `
  -Headers @{
    "x-bili-plugin" = "dsh"
    "x-bili-plugin-conversation" = "conv-demo"
    "x-bili-plugin-context-window" = "120000"
  } `
  -Body '{"model":"gpt-4o","messages":[{"role":"user","content":"hello"}]}'
```

## 支持的协议

- OpenAI Chat Completions：JSON 与 SSE，包含文本、reasoning、多模态、工具调用、
  usage（`prompt_tokens` / `completion_tokens` / `cached_tokens`）。
- Anthropic Messages：JSON 与 SSE，system、tools、cache_control 断点、usage。
- 插件模式（`x-bili-plugin` 头）：`/__bili/plugin/manifest`、`/__bili/plugin/tool`、
  `/__bili/plugin/status`、`/__bili/plugin/runtime-info`。
- 管理面：`/__bili/`（轻量 HTML 面板）、`/__bili/status`、`/__bili/stats`、
  `/__bili/overview`（累计/净节省、折叠次数、按协议和最近会话）、
  `/__bili/upstreams`（仅列出实际失败的上游，避免启动时伪造“上游不可用”）、
  `/__bili/pricing`（会话模型清单 + `billing.prices` 覆盖）、
  `/__bili/config`（GET 原始配置；PUT 全量 `file` 或 `dsh` 合并）、
  `/__bili/notices`、`/__bili/update/state`、`/__bili/update/check`、
  `/__bili/session-report?session=<id>&format=md|svg`。
- 持久化：`rust-sessions/<hash>.json` 保存完整 kernel 状态，旁边
  `<hash>.meta.json` 是启动索引。启动只读 meta，会话首次被访问时才读完整
  JSON；旧快照首次启动会迁移生成 meta。`/__bili/status` 的
  `storage.loadedSessions` / `indexedSessions` 可用于确认。

## 接入 DSH

已安装的 `billion-context` 原生 DSH 插件有两种接法，都要在 **启动 DSH 之前** 设置环境变量。

推荐 attach（插件不再 spawn 第二个代理进程）：

```powershell
$env:BILLION_CONTEXT_PROXY = "http://127.0.0.1:18113"
.\billion-context-proxy.exe start --port 18113
```

或让插件 spawn 直接使用 Rust：

```powershell
$env:BILLION_CONTEXT_NODE = "<本目录>\billion-context-proxy.exe"
```

插件会以 `<运行时> <proxy-script> start --host … --port …` 启动；Rust 忽略插件传入的
脚本参数并识别 `start`/`--host`/`--port`。源码包中的 `tests/shim-attach.ps1` 与
`tests/shim-spawn.ps1` 分别覆盖两种接法。

## DSH 原生面板

`@local/bili-rust-panel` 是一个独立的 DSH Client bundle，以 `link:` 方式接入
`desktop` profile（源码在仓库外，不在本分发包内）。刷新 DSH 后可在
“设置 → Rust 面板”与会话页签“Rust 上下文”看到原生数据视图；设置页新增配置、
价格、更新、通知标签。它直接读取上述 `/__bili/` 端点，不嵌入 `/__bili/` 网页，
也不修改已安装的 `billion-context`。配置页的“隐藏旧账本页签”只写
`billion-context` 的 `dsh.contextTab=false`，代理与压缩宿主保持启用。

## 压缩

- 模型看到的消息带 `<acp …>mNNNNN</acp>` ref 标注；`compress` 工具接收
  `content` 字符串（JSON 数组：`startRef`/`endRef`/`topic`/`summary`）。
- 宿主随请求注入 ACP standing instructions（ref 规则、压缩/恢复/检索工具说明），
  与 Node 行为对齐。
- 上下文按折后投影发送；折叠后的摘要留在原位置，原消息不再上行。
- 会话快照写入 `rust-sessions/<sha256(key)>.json`（原子替换），重启后恢复块、refs 和
  计数；Node 的 SQLite 历史不导入，`last_view` 视图对账不持久化。
- 外部摘要预检（2026-11-23 移植）：DSH 的 compaction 调用（头
  `x-bili-plugin(-agent): dsh` + `x-bili-plugin-purpose: compaction`，或请求体最后一条
  user 文本以 `You are now acting as a compaction engine…` 开头）不再转发，直接以 bili
  快照本地应答：`<bili-snapshot v1 conv="…" seq="…" blocks="b1+b2" />` + 每个 active
  block 的 `[Compressed conversation section]` 摘要；wire 形状按请求 `stream` 字段给出
  openai `chat.completion`/SSE 或 anthropic `message`/SSE。没有 active block 时返回 422
  （`x-bili-gate: dsh-compaction-refused`，#1729）而不是让 DSH 写出遮蔽原文历史的
  checkpoint；诊断需要时可用 `dsh.authoredCompactionRefuse=false` 恢复转发。
- 外部摘要链（`compress.externalSummary`，2026-11-23）：插件 `compress`/`acp_compress` 在折叠前
  先调用配置里的 Provider 模型逐 range 写摘要，覆盖模型自己的 `summary`（之后仍过占位/拒答闸门，
  并追加 host-carried 用户意图）；失败 fail-open 保留模型摘要。配置与 Node 版共用
  `~/.config/billion-context/billion-context.json` 的 `providers.<name>`（`baseUrl`、`api` 仅
  `openai`、`apiKeyEnv` 或 `credentialRef`、`models.<model>`）与
  `compress.externalSummary{enabled,targets,budget}`；凭据回退读 `~/.dsh/.credentials.yaml` 的
  `refs:`。目标不可解析或不支持时整链禁用并警告。
- 已知限制：Node 的 SQLite 历史、图片缩放、完整价格注册表、packs/blog/changelog、
  在线更新安装和完整 557KB 管理 SPA 尚未移植；配置页与更新/通知只提供 Rust 端点
  支持的安全子集。

## 命令行与状态目录

```
billion-context-proxy [start] [--host H] [--port N] [--passthrough] [--debug] [--strict-port]
billion-context-proxy --version | -V     # 版本与构建身份
```

环境变量：

| 变量 | 含义 |
| --- | --- |
| `BILI_HOST` / `BILI_PORT` | 默认监听地址（默认 `127.0.0.1:8765`） |
| `BILI_DEBUG=1` | 调试日志 |
| `BILI_PASSTHROUGH=1` | 只转发，不做压缩 |
| `BILI_UPSTREAM` | 可选的上游基址提示；默认不设置，不伪造启动上游 |
| `BILI_DSH_AUTHORED_COMPACTION` | 存在且不为 `0` 时启用外部摘要预检；否则读配置 `dsh.authoredCompaction`（默认启用） |
| `BILI_DSH_COMPACTION_REFUSE` | 存在且不为 `0` 时无 active block 也拒绝 compaction 调用（默认拒绝；`0` = 转发，仅诊断用） |
| `XDG_STATE_HOME` / `XDG_DATA_HOME` | 覆盖状态/数据目录 |

状态目录默认 `~/.local/state/billion-context`，与 Node 版同级，但本包不会修改
Node 的数据库或配置。

## 测试

源码包内：

```powershell
cargo test
node tests/mock-upstream.js   # 本地 mock 上游，默认 18110
```

`cargo test` 当前 {{TEST_COUNT}} passed，覆盖：Anthropic/OpenAI 双向转换、工具注入、
usage 语义、ref 标签、standing instructions、空闲 nudge 抑制、会话快照恢复、构建身份。

## 替换现有 Node 宿主

本包**不自动覆盖**已安装程序。确认后手动停掉占用目标端口的 Node 代理，再以相同
端口启动：

```powershell
.\billion-context-proxy.exe start --port 18787
```

DSH 的 profile、bundle 与 `proxy-origin` 不被本包改写；如需回滚，重新启动原
Node 代理即可。

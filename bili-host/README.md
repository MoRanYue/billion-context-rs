# bili-host (Rust)

billion-context 宿主/代理的 Rust 实现。把 OpenAI Chat Completions 与 Anthropic
Messages 请求接入 workspace 中的 `billion-context` 内核，按折后投影上行，并在本地
提供插件 manifest/tool/status、统计、经济账和轻量管理面板。

## 构建与运行

```powershell
cargo build -p bili-host --release
cargo test -p bili-host
target\release\bili-host.exe start --port 18113 --debug
```

默认 `127.0.0.1:8765`；`--passthrough` 关闭压缩，`--strict-port` 端口占用时退出。
状态目录为 `~/.local/state/billion-context`（可用 `XDG_STATE_HOME` 覆盖），与 Node
版同级但不读写其数据库。

## 模块

| 文件 | 职责 |
| --- | --- |
| `src/main.rs` | CLI、axum 路由、`/bili/` 转发、插件与管理端点、HTML 面板 |
| `src/anthropic.rs` | Anthropic wire ↔ 内核消息、工具注入、cache 断点、ref 标签 |
| `src/openai.rs` | OpenAI wire ↔ 内核消息、system 指令注入、工具注入 |
| `src/session_store.rs` | 会话表、prepare/tool/status、range 预检、快照持久化与懒加载 |
| `src/usage.rs` | SSE/JSON usage 旁路采集（按协议解释 cached/context） |
| `src/session_report.rs` | Markdown/SVG 会话报告 |

## 接入 DSH

已安装的 `billion-context` DSH 原生插件支持两种接法；都要在 **启动 DSH 之前**
设置环境变量。

推荐 attach（不让插件再 spawn 一个代理进程）：

```powershell
$env:BILLION_CONTEXT_PROXY = "http://127.0.0.1:18187"
.\bili-host.exe start --port 18187
```

或让插件的 spawn 路径直接使用 Rust（插件把 `BILLION_CONTEXT_NODE` 当作 node
运行时，并以 `<exe> <proxy-script> start --host … --port …` 启动；Rust 会忽略
前面的脚本参数）：

```powershell
$env:BILLION_CONTEXT_NODE = "<path-to-bili-host.exe>"
```

## 端点

- `POST /bili/<absolute-url>`：转发；模型请求会先经过 prepare/投影。
- `GET /__bili/`、`GET /__bili/status`、`GET /__bili/stats`、`GET /__bili/overview`
- `GET /__bili/pricing`（仅返回会话模型，未内置价格表）
- `GET /__bili/session-report?session=<id>&format=md|svg`
- `GET /__bili/plugin/manifest`、`POST /__bili/plugin/tool`、
  `GET /__bili/plugin/status`、`GET /__bili/plugin/runtime-info`

`GET /__bili/status` 的 `storage` 包含 `loadedSessions` 与 `indexedSessions`：
宿主启动只读 `rust-sessions/<hash>.meta.json` 轻量索引，会话首次被访问时才读取
完整快照；旧快照会在首次启动时一次性迁移生成 meta。

## 已知限制

- 会话快照在 `rust-sessions/<sha256(key)>.json`，重启按需恢复块、refs 和计数；
  Node 的 SQLite 历史不导入，视图对账（`last_view`）不持久化。
- 非插件直连 lane 不会自动拦截模型返回的 `compress` 工具调用；压缩走插件
  `/__bili/plugin/tool` 或宿主注入的 nudge/tool 面。
- 管理 SPA、外部摘要预检、图片缩放、更新/通知/配置等 Node 端点未实现；
  `pricing` 仅返回会话模型，价格注册表未内置。

# billion-context-proxy (Rust)

billion-context 宿主/代理的 Rust 实现。把 OpenAI Chat Completions 与 Anthropic
Messages 请求接入 workspace 中的 `billion-context` 内核，按折后投影上行，并在本地
提供插件 manifest/tool/status、统计、经济账和轻量管理面板。

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
- `/__bili/*` 管理面要求回环 `Host`；浏览器 `Origin` 只接受回环来源或
  `BILI_ADMIN_ORIGINS`（逗号分隔），CORS 不再回 `*`。
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
| `src/session_store.rs` | 会话表、prepare/tool/status、range 预检、快照持久化与懒加载 |
| `src/usage.rs` | SSE/JSON usage 旁路采集（按协议解释 cached/context） |
| `src/session_report.rs` | Markdown/SVG 会话报告 |

## 接入 DSH

已安装的 `billion-context` DSH 原生插件支持两种接法；都要在 **启动 DSH 之前**
设置环境变量。

推荐 attach（不让插件再 spawn 一个代理进程）：

```powershell
$env:BILLION_CONTEXT_PROXY = "http://127.0.0.1:18187"
.\billion-context-proxy.exe start --port 18187
```

或让插件的 spawn 路径直接使用 Rust（插件把 `BILLION_CONTEXT_NODE` 当作 node
运行时，并以 `<exe> <proxy-script> start --host … --port …` 启动；Rust 会忽略
前面的脚本参数）：

```powershell
$env:BILLION_CONTEXT_NODE = "<path-to-billion-context-proxy.exe>"
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

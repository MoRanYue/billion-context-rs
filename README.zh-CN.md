# billion-context-rs

[![crates.io](https://img.shields.io/crates/v/billion-context.svg)](https://crates.io/crates/billion-context)
[![docs.rs](https://docs.rs/billion-context/badge.svg)](https://docs.rs/billion-context)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[English](README.md) · **简体中文**

[`ranxianglei/billion-context`](https://github.com/ranxianglei/billion-context) 的核心机制（引用分配、折叠、3 层 LSM、nudge 节奏、跨回合去重）用 Rust 重写，供任意 Rust 程序嵌入使用。

原项目是 TypeScript 的 ACP 代理，内核（`kernel/`，npm 包 `acp-kernel`）是框架无关的压缩引擎。这里只移植内核，不移植代理、Web 控制台、宿主适配层。

```
billion-context/               纯引擎：消息 → 引用 → 折叠 → 恢复，零 IO、零异步、零 unsafe
billion-context-wire/          协议层：Anthropic / OpenAI 的 wire types，与 IR 无损互转
billion-context-tool/          工具层：把折叠/恢复包装成模型可直接调用的 tool
```

默认构建**零依赖**（`serde` 与真分词器都是可选 feature），无 `unsafe`（`#![forbid(unsafe_code)]`），全部同步。

> 英文简介见 [`billion-context/README.md`](billion-context/README.md) —— crates.io 上展示的就是它。
>
> 发布已自动化，走 crates.io Trusted Publishing（OIDC，不保存 API token）：见 [`RELEASING.md`](RELEASING.md)。

---

## 一、七条设计判断

这几条决定了 API 的形状，不是风格问题。

### 1. `Config` 是纯数据，代码走 `Policy`

TS 的 `Config` 里塞着闭包（`shouldCompress`、`isToolFoldable`…），于是它既不能序列化，也不能跨进程传。这里拆成两半：

```rust
let config = Config::new(200_000);          // 纯数据，serde 往返
let session = Session::new(config).with_policy(MyPolicy); // 代码
```

`Policy` 的每个方法都有默认实现，宿主只覆盖真正需要的钩子。

### 2. `&mut self`，没有 `Arc<Mutex<Session>>`

会话状态是排他的。Rust 里把它做成 `Send` 的值，由持有者独占，而不是内部加锁——需要并发就 `fork()`（拷贝开销是几个 map + 一个短 vector）。

```rust
let snapshot = session.fork();   // 之后互不影响
```

### 3. 引擎不跑循环，宿主驱动

TS 版是 `while (round < 10) { call model; apply }`——引擎自己发请求。Rust 版把循环翻出来：

```rust
let mut step = engine.begin(input);
loop {
    let reply = llm.call(&step.messages).await?;      // 你的异步
    match engine.resume(input, &reply.requests) {
        Next::Send(next) => step = next,
        Next::Done(turn) => break turn,
        Next::Exhausted(turn) => break turn,
    }
}
```

引擎不知道有异步、不知道有网络，也就没有 `async fn` 污染和 `dyn Future` 装箱。

### 4. 失败是值，不是异常

批量折叠里一条 ref 过期，不能丢掉另外九条。所以返回 `Applied`，逐条报结果：

```rust
pub struct Applied {
    pub blocks: Vec<BlockId>,          // 成功的
    pub errors: Vec<RangeError>,       // 失败的，逐条带 kind + detail
    pub tokens_compressed: u32,
    pub partial: bool,
}
```

`RangeErrorKind` 的每个变体都带 `hint()`，直接喂回模型——这是自愈回路的一部分，不是给人看的日志。

### 5. 引用是 newtype，不是 `u32`

```rust
MsgRef::new(1)          // Some(m00001)
BlockId::new(1)         // Some(b1)
MsgRef::parse("m00003") // 模型的输出也能解析
```

`MsgRef` 和 `BlockId` 都是 `1..=9_999_999` 的非零值，`m00001` / `b1` 是它们唯一的 wire 形态。编号永不复用——模型引用一个已经死掉的编号是合法的，它会得到 `UnknownRef` 提示，而不是被静默忽略。

### 6. 内核不碰内容存储

摘要存在 `Block` 里，但**原始消息**不存——那是宿主的事（TS 版是 `ContentStore`，Rust 版是 `Retrieval` 回调给宿主）。内核只保证"摘要 → 原文"的映射能重建，怎么存由你决定：文件、SQLite、对象存储都行。

### 7. 渲染与逻辑分离

`process_turn` 只做投影和计数；`<acp tokens="…">m00001</acp>` 这层包装由 `render` 策略控制，也可以完全关掉自己渲染。内核给出的 `Turn` 里，消息的 id 和 token 快照是权威的，字符串格式不是。

---

## 二、一次完整调用

```rust
use billion_context::{Boundary, CharsAsTokens, CompressRequest, Config, MsgRef,
               RenderStrategy, RequestedRange, Session, TurnInput};

let mut session = Session::new(Config::new(200_000)).with_estimator(CharsAsTokens);

// 1. 投影：分配引用、算出可折叠区间、判断是否该提醒
let turn = session.process_turn(TurnInput {
    messages: &history,
    token_count: estimate(&history),
    render: RenderStrategy::All,
    now: now(),
});

// 2. 模型看到了 turn.messages，可能收到 turn.nudge
//    它回答"把 m00012-m00030 折掉，摘要如下"
let applied = session.apply_compression(CompressRequest {
    ranges: vec![RequestedRange {
        start: Boundary::Message(MsgRef::new(12).unwrap()),
        end:   Boundary::Message(MsgRef::new(30).unwrap()),
        topic: Some("build setup".into()),
        summary: "…".into(),
    }],
    now: now(),
});

// 3. 下一回合：原消息自动消失，编号保持不变
let turn = session.process_turn(TurnInput { messages: &history, .. });

// 4. 模型要看摘要原文（它引用了 b3）
let hits = session.retrieve(&[MsgRef::new(12).unwrap()]);

// 5. 或者反过来，撤销一次折叠
session.decompress(&[BlockId::new(3)]);
```

### 谁来改写消息数组：`ViewMode`

一次折叠意味着「m00001–m00004 现在是块 b1」，但块不是消息——总得有人把这句话变成一个消息数组。两种做法都合法，产出却不同，所以它是个显式开关而不是隐含行为：

```rust
// 默认：内核交付可直接发送的视图——覆盖的消息被丢弃，换成一条持有摘要的 carrier
let session = Session::new(cfg);

// 或者：内核不动数组，只通过 blocks 报告覆盖关系，改写交给宿主
let session = Session::new(cfg).with_view_mode(ViewMode::Host);
```

`ViewMode::Host` 是上游 TS 内核的模型，此时 `turn.messages` 仍包含被折叠的原文（只是打上了 `<acp …>` 标签）。

### 无状态宿主的坑

宿主每回合重发整段 transcript 时，已折叠的原文会带着**新的 message id** 回来。内核靠内容指纹认领它们（`reconcile` 节点），把它们映射回原来的 id 再 prune 掉——否则压缩等于没做。这一条在 `lib.rs` 的 `resent_originals_are_reconciled_not_double_counted` 里锁死。

指纹只在**折叠那一刻**登记（`apply_one`），因为那时"哪些内容进了块"是确定的；对活消息建索引会把两条内容相同的活消息搞混。

反过来，**从不重发折叠原件**的宿主（自己做了 surface 替换的那种）应该关掉它：

```rust
let session = Session::new(cfg).with_reconcile(false);
```

它本来就没有东西可归一，而按内容归一反而有把两条内容相同的活消息并成一条的风险。`ViewMode::Host` 下这一项自动失效。

---

## 三、API 一览

| 类型 | 方法 | 说明 |
|---|---|---|
| `Session` | `new(Config)` / `try_new` | `try_new` 做配置校验，失败给 `Vec<ConfigError>` |
| | `with_policy` / `with_estimator` / `with_pipeline` | 覆盖钩子、分词器、节点序列 |
| | `process_turn(TurnInput)` | 唯一的热路径，同步、纯 |
| | `apply_compression(CompressRequest)` | 逐条成败 |
| | `retrieve(&[MsgRef])` / `search(&str, n)` | 摘要查询 |
| | `decompress(&[BlockId])` | 撤销折叠，原文回到视图 |
| | `status()` / `state()` / `fork()` | 观测与快照 |
| `Engine` | `begin` / `resume` / `round` / `with_max_rounds` | 宿主驱动循环 |
| `pipeline` | `default_pipeline()` / `Node` | 13 个节点，可替换 |
| `persist` | `to_json` / `from_json` / `save` / `load` | 版本化快照，引用按 wire 形态存 |
| `billion_context_wire` | `Message` / `Block` / `Usage` / `OpenAiUsage` | 未知字段保留、thinking 不解析、tool 参数不重序列化 |
| `billion_context_tool` | `definitions` / `parse_compress` / `render_compress` | 模型可调用的 `acp_compress` + `acp_retrieve` |

### 节点顺序（`pipeline_order_is_pinned` 锁死）

```
reconcile-live-ids → assign-refs → sync-blocks → prune → ccr-store
  → absorb-hide → crush → absorb-prompt → filter → hide-compress-calls
  → recommend → nudge-inject → emergency-truncate
```

前三个是"记账"，中间是各种吸收/折叠，最后三个是"决策 + 注入 + 兜底"。

`absorb-*`、`crush`、`ccr-store` 在这个移植里是**有意的空实现**：它们需要模型调用或内容存储，纯内核给不了。参考配置里它们本来就 `enabled: false`，所以默认行为与上游一致；宿主实现了对应钩子再打开。

### 默认配置（与 TS 版逐值一致）

| 项 | 值 |
|---|---|
| tier2 / tier3 触发 | 1000 / 2000 tokens |
| nudge 区间 | 45% ~ 75%，每 5 轮一次，第 15 轮起强制 |
| 折叠门槛 | 区间 ≥ 5000 字符，摘要 50 ~ 20000 字符 |
| 保留近期 | 3 条消息 / 5000 tokens |
| 紧急截断 | 95% 水位，连续 3 次无效则硬截 |
| 引用上限 | `m00001` ~ `m9999999`，`b1` ~ `b9999999` |

---

## 四、接进 agent 循环（`billion-context-tool`）

模型看不到你的 `Session`，它只看到两个工具。

```rust
use billion_context_tool::{definitions, parse_compress, parse_retrieve, render_compress, render_retrieval};

// 1. 注册工具（两个 JSON schema，直接丢给 provider）
let tools = definitions();

// 2. 模型调了 acp_compress
if call.name == billion_context_tool::COMPRESS {
    match parse_compress(&call.input) {
        Ok(ranges) => {
            let applied = session.apply_compression(CompressRequest {
                ranges, now: now(),
            });
            // 成败一起回给模型：成功的别重折，失败的按 hint 重试
            tool_result(render_compress(&applied.blocks, &applied.errors))
        }
        // 形状错误（ref 打错、start 在 end 后面）在本地就挡掉
        Err(error) => tool_result(error.to_string()),
    }
}
```

两个刻意的设计：

- **schema 本身就是 prompt。** 折叠规则、ref 格式、摘要长度全写在 `description` 里，因为模型对工具描述的注意力远高于 4 万 token 之外的 system prompt。
- **拒绝是指导，不是 400。** 失败区间回的是「m00099 不在当前视图里，换一个」，模型下一轮会自己改；回 HTTP 错误就是死一回合。

## 五、精确 token 计量（可选）

内核自己不做分词 —— `TokenEstimator` 是宿主必须注入的那一项能力，默认实现
`CharsPerToken4` 只是启发式。它的误差不小，而且**误差符号随内容翻转**：

| 内容 | 真实 tokens | 启发式 | 误差 |
| --- | ---: | ---: | ---: |
| 英文散文 241 字 | 45 | 61 | +35.6% |
| 中文散文 90 字 | 58 | 85 | +46.6% |
| 中英技术混排 140 字 | 65 | 59 | −9.2% |
| Rust 代码 273 字 | 101 | 69 | −31.7% |
| JSON 168 字 | 67 | 42 | −37.3% |
| 工具日志 182 字 | 60 | 46 | −23.3% |

中文实际约 **0.64 token/字**（不是 1.0），代码约 **2.7 字/token**（不是 4.0）。
两个方向的偏差抵消不了，任何单一缩放系数都修不好。

需要精确值就打开 `hf-tokenizer` feature —— 它把 `deepseek-recipe-encoding`
（DeepSeek 官方的 prompt 编码库）接到同一个 trait 上，内核一行不改：

```toml
[dependencies]
billion-context = { path = "../billion-context", features = ["hf-tokenizer"] }
```

```rust
use billion_context::tokenizer::HfTokenizer;

// 词表不随 crate 发布（6 MB、归模型所有），按部署的模型取对应的那份
let tokenizer = HfTokenizer::from_file("assets/tokenizer.json")?;
let session = Session::new(Config::new(1_000_000)).with_estimator(tokenizer);
```

**词表由你选，不写死。** `tokenizer.json` 是 HuggingFace `tokenizers` 的格式，
主流模型家族都会发布，所以 `HfTokenizer` 对任何一份词表都成立 —— DeepSeek 只是
它实测过的那份。DeepSeek 自己按模型家族发布、MIT 授权，在
[`deepseek-recipe`](https://github.com/deepseek-ai/deepseek-recipe) 的
`static/tokenizers/` 下：`v4/` 对应 DeepSeek-V4-Pro，`v41/` 对应
DeepSeek-V4-Flash-Vision-Exp。

四点要注意：

- **词表必须匹配实际服务的模型。** 拿错词表对某些内容比启发式还差，所以这是
  部署决策，不该做进内核默认值。
- **它只让「文本 → token」精确，不让「消息 → prompt」精确。** 内核逐条计数再
  求和，而模板的 role 标记、工具定义 schema、渲染出来的 `<acp …>` 标签都不属于
  任何一条消息的正文。这部分是几百到几千 token 的固定开销：在 1M 窗口里是噪声，
  在 128k 窗口里是几个百分点。
- **代价**：`tokenizers` 会带进 `hf-hub`/`ureq`/`rustls`/`esaxx-rs`，冷编译约
  22 s，二进制增加数 MB。所以它是 `billion-context` 的一个**可选 feature**：默认构建
  一个依赖都不加，`cargo test` 也不碰它。
- **有一个全局副作用**：`deepseek-recipe-encoding` 打开了
  `serde_json/preserve_order`，而 Cargo feature 是全局相加的 —— 开这个 feature
  会把你整个依赖图里 `serde_json::Value` 的 map 从 `BTreeMap` 换成 `IndexMap`。
  通常无害，但如果有代码依赖 `Value` 会重排键（或依赖那个更小的 map），行为会变。
  这也是 `billion-context-wire` 那条 `value_round_trip_reorders_keys_and_changes_bytes`
  用例在开 feature 后会打印 skip 而不是通过的原因：重排风险消失了，不是被修好了。
  介意这点就跳过这个 feature，直接基于 `tokenizers` 自己实现 `TokenEstimator`
  —— 计数文本时两条路结果一致，而 `tokenizers` 不动 `serde_json`。

计数本身有缓存：每条消息在**第一次分到 ref 时**计一次，写进
`state.token_snapshot` 后不再重算（重算会让渲染出来的标签字节变化，把上游前缀
缓存打掉）。所以精确分词器每轮只为**新消息**付钱 —— 实测 865 ns/token，
一条 4k token 的新消息约 3.5 ms。

这个缓存也意味着**换分词器是一次迁移**：`token_snapshot` 随状态一起持久化，
老消息的旧计数不会因为换了 estimator 而重算。真要中途切换，得清空已存的
`token_snapshot`（或整体重算一遍），否则同一份状态里会混着两种口径。

### 自己实现 TokenEstimator

`TokenEstimator` 是一个方法的 trait，十几行就能实现。用别的技术栈计数
（`tiktoken-rs`、sentencepiece 绑定、你为了别的目的已经持有的分词器）就直接实现
它，完全不碰 `hf-tokenizer`：

```rust
use billion_context::{Config, Session, TokenEstimator};

struct TiktokenEstimator { bpe: tiktoken_rs::CoreBPE }

impl TokenEstimator for TiktokenEstimator {
    fn count(&self, text: &str) -> u32 {
        self.bpe.encode_ordinary(text).len() as u32
    }
}

let session = Session::new(Config::new(128_000))
    .with_estimator(TiktokenEstimator { bpe });
```

`with_estimator` 收 `impl TokenEstimator`，而下面几种形状内核已经替你实现了，
不用自己包一层：

| 你手里有 | 直接传 |
| --- | --- |
| 具体类型 | `with_estimator(MyEstimator)` |
| `Arc<dyn TokenEstimator>` | 同上（已 `impl TokenEstimator for Arc<T>`） |
| `Box<dyn TokenEstimator>` | 同上 |
| 一个闭包 | `with_estimator(FnEstimator(\|t: &str\| ...))` |

闭包要 `FnEstimator` 包一层、而不是裸 `impl Fn` blanket impl，是因为 `Arc<F>` 和
`Box<F>` 自己也实现了 `Fn` —— 裸 blanket impl 会和上面两个实现重叠，直接过不了
coherence。

## 六、验证

```
cargo test                                        # 52：31 + 4 + 10 + 5 + 2 doctest
cargo test --features hf-tokenizer                # 59：再加 5 个分词用例与 2 个 doctest
cargo run --example basic                         # 完整走一遍
cargo clippy --all-targets                        # 无警告（两种配置都跑）
```

精确分词的那 5 个用例需要词表，缺了就自动跳过并打印一行提示：

```
cd billion-context && ..\assets\fetch-tokenizer.ps1      # 拉 6 MB 词表到 assets/
```


测试覆盖的是机制本身，不是覆盖率：编号不复用、折叠后原文消失而编号存活、重发的原文被认领而不是重复计入、reasoning 块逐字节不变、部分失败保住成功的部分、轮数上限能终止固执的模型、工具参数里打错的 ref 能被指到具体字段。

### 差分测试

自测只能证明「自己和自己一致」。真正的正确性基准是上游 TS 内核，所以有一套
差分测试（`diff/`）：同一份 scenario 喂给两边，各输出同形 canonical JSON，逐字段
diff。

```
cd D:\GITHUB\billion-context
node --import tsx D:\GITHUB\billion-context-rs\diff\run-ts.mts \
     D:\GITHUB\billion-context-rs\diff\scenario-a.json D:\GITHUB\billion-context-rs\diff\out-ts.json

cd D:\GITHUB\billion-context-rs
cargo run --offline -q --example diff_harness -- diff\scenario-a.json diff\out-rust.json
cd diff && node compare.mjs out-ts.json out-rust.json 500
```

当前 **0 处差异**（5 轮 `processTurn` + 2 次 `applyCompression` + `decompress` + `status`，
含 `view` / `refs` / `blocks` / `nudge` 全部字段）。这套测试抓出了 9 个自测发现不了的
缺陷，其中「块吸收根本没实现」「折叠后丢首条 user 消息」是纯逻辑错误。细节见
[`diff/REPORT.md`](diff/REPORT.md)。

分词器单独有一套，因为它引入的是**数字来源**而不是新逻辑——分歧不会崩，只会让每个
阈值悄悄挪位：

```
node diff/tok-ts.mts diff/tok-ts.json                 # 语料 + TS 参考值
cargo run --example tok_diff -- diff/tok-ts.json diff/tok-rust.json
node diff/compare-tok.mjs diff/tok-ts.json diff/tok-rust.json
```

73 条语料（含 CJK 每个区间的边界、非 BMP 的 emoji 与 U+20000、组合字符、纯空白）
**逐值 0 处不一致**。再往上一层，两边注入同一个合成计数器重跑整个 scenario：注入
本身会改变 60 个字段，而两边的结果仍然 **0 处差异**——证明计数器不是只在某一处生效。
其中一条容易踩的坑：JS 的 `text.length` 是 UTF-16 单元数，Rust 侧必须用
`ch.len_utf16()` 累加，用 `chars().count()` 在含 emoji 的文本上直接差一倍。

---

## 七、工具调用（包括并行调用）

一条 `Message` 只装一个 `Content`。所以 provider 的一条约含有 **N 个 `tool_use` 块**的消息
（一次发起三个 grep），在核心里就是 N 条消息：

```rust,ignore
// Anthropic：一条 assistant 消息，三个 tool_use 块。
// 核心：三条消息，每块一条。身份是 call id，不是数组位置。
let mut history = vec![Message::text("u1", Role::User, question)];
for call in calls {
    history.push(Message::tool_call(call.id.clone(), call.name, call.call_id, call.arguments));
}
```

拆开之后有两件事仍然成立，而它们才是要紧的：

- **配对**。结果靠 `call_id` 找自己的调用，不靠数组位置。`turn_groups` 会把整个 burst
  （reasoning run + 连续所有 assistant 动作 + 这些 call 指向的全部结果）收进**同一组**，
  所以并行的一批要么一起折、要么都不折。结果的顺序无关紧要。
- **可折叠**。范围可以跨过这些调用和它们的结果：它们是普通的可折叠内容，
  `apply_compression` 会为这一轮生成一条摘要。

拆开不是免费的——宿主把投影还给 provider 时，得把这 N 条重新合并成一条 assistant 消息，
否则 Anthropic 直接拒绝、OpenAI 丢掉批次形状。`billion-context-wire` 直接建模了线格式
（`Content::Blocks(Vec<Block>)`，含 `Block::ToolUse`），而它**不依赖**核心 crate，
所以这个拆分/合并适配器属于宿主，跟"当前对接的是哪家 provider"这个知识放在一起。

两个值得知道的细节：

- **调用不是散文**。它的 arguments 是宿主将要执行的用户意图，所以折叠从不改写它们
  （`Content::foldable_text` 对 `ToolCall` 返回 `None`）。但它照样参与计量——
  `Content::tool_arguments` 是池子、范围、消息指纹所度量的东西，因为那才是 provider 计费的字节。
- **调用是可折叠内容**。它可以被*摘掉*，尽管不可以被*改写*。早先一版把它当成 hole——
  结果配对不变式把每一对都读成"被拆开"，于是撤掉每一个工具结果，
  工具密集的对话最后只剩调用之间那些散文可折，而那恰恰是最需要折的对话。

和媒体保护一样，配对保护是**建议而非否决**：推荐器绝不建议一条会拆散配对的范围，
但 `apply_compression` 不会拒绝调用方自己点名的范围。如果你的模型自己写范围，请自行检查：

```rust,ignore
use std::collections::HashSet;

let calls: HashSet<&str> = span
    .iter()
    .filter(|m| m.role == Role::Assistant)
    .filter_map(|m| m.content.tool_call_id())
    .collect();
let results: HashSet<&str> = span
    .iter()
    .filter(|m| m.role == Role::Tool)
    .filter_map(|m| m.content.tool_call_id())
    .collect();
assert_eq!(calls, results, "a range must take a call and its result together");
```

宿主一旦开始拆分消息，还会碰到两件事：

- **渲染标签会压在 arguments 上。** `RenderStrategy::All` 把标签盖在消息所携带的载荷上——
  arguments 也不例外，这与 TypeScript 内核逐字节一致——于是载荷变成
  `<acp tokens="4" type="grep">m00002</acp>\n{"q":"alpha"}`。如果你要把它解析成
  provider 的 `tool_use` input，先调 `strip_ref_tag`；如果你希望内核根本不碰结构化内容，
  改用 `RenderStrategy::TextOnly` 渲染。
- **id 可能指代的是"投影"而不是"消息"。** 把一条 provider 消息拆成内容块的宿主，
  会把碎片命名为 `base#c0`、`base#r0`（即 TypeScript 内核的 `baseIdOf` 约定）。
  覆盖判定、内容索引、块存活、截断预算四处都按 `MessageId::base` 比较，
  所以折掉其中一片，下一轮该原件其余碎片也会一起消失。怎么命名随你，但同一条原件要共用一个 base。

---

## 八、还没做的

- **异步适配**：`Engine` 已经是宿主驱动的，包一层 `async` 是薄胶水，没必要进内核。
- **TS 版的 cache ledger**（前缀缓存账本）：属于代理层的观测逻辑，不属于压缩内核。`billion-context-wire` 里只保留了它依赖的"保序序列化"。
- **原文存储**：`retrieve` 只返回摘要和它替换的区间；原文在宿主手里，怎么存（文件 / SQLite / 对象存储）由你决定，内核不替你选。


# Rust 内核 vs TypeScript 内核：差分测试报告

同一份 `scenario-a.json` 喂给两个实现，各自输出同形 canonical JSON，再逐字段 diff。

```
# TS 侧（用 tsx 直接跑 kernel/src，不依赖 dist/）
# 注意：必须在仓库根目录跑，tsx 装在根 node_modules 里
cd D:\GITHUB\billion-context
node --import tsx D:\GITHUB\billion-context-rs\diff\run-ts.mts \
     D:\GITHUB\billion-context-rs\diff\scenario-a.json \
     D:\GITHUB\billion-context-rs\diff\out-ts.json

# Rust 侧
cd D:\GITHUB\billion-context-rs
cargo run --offline -q --example diff_harness -- diff\scenario-a.json diff\out-rust.json

# 对比
cd diff
node compare.mjs out-ts.json out-rust.json 500
node hist.mjs          # 按字段归并
```

## 当前结果：111 → **0 处差异**

七个步骤（5 轮 processTurn + 2 次 applyCompression + decompress + status）的全部字段逐字节一致：
`view`、`refs`、`blocks`（块数/tier/compressedTokens/survived/active）、`tokensCompressed`、
`modelContextLimit`、`nudge`（`reason` / `tier` / `maxPending` / `growth` / `ranges` 全字段）。

---

## 收敛路径

### 111 → 45：搞清「谁来改写消息数组」

初版报告把这一条判成「TS 用 surface-replacing、Rust 用 carrier」，**判错了**。
事实是 TS 在折叠生效的下一步（step4）**同样**丢掉被覆盖的消息并插入摘要载体
（10 → 8 条）；两边的真实分歧只在 step5——宿主用**全新 id** 重发了同一批内容，
Rust 的 `reconcile-live-ids` 靠内容指纹把它认回原件并再次剪掉，TS 不做这件事。

因此新增了 `ViewMode`（`billion-context/src/pipeline/mod.rs`）：

- `ViewMode::Send`（默认）：内核交付可直接发送的视图——覆盖的消息被丢弃，换成
  一条持有摘要的 carrier 消息。
- `ViewMode::Host`：内核不动数组，只通过 `blocks` 报告覆盖关系，改写交给宿主。

配套的 `Session::with_reconcile(false)` 关掉内容指纹归一。一个从不重发折叠原件的
宿主本来就没有东西可归一，而按内容归一反而会把两条内容相同的活消息并成一条。

### 45 → 30：`reconcile` 才是 `refs` 那 64 处差异的根因

关掉之后 `refs[][]`、`refs.length` 一次性全绿。

### 30 → 23：`compressedTokens` 3599 vs 3555 是**测试脚手架**的错，不是内核的错

初版报告说「TS 按渲染后的带标签文本计 token」。查 `tokenize.ts` 后发现
`countMessageTokens = countTokens(text) + thinkingTokenValue`，只算正文。
44 的差额来自 `run-ts.mts` 把 `processTurn` 返回的**渲染视图**（带 `<acp …>` 标签）
喂给了 `applyCompression`；真实宿主传的是自己那份未渲染的 transcript。

改为在 TS 侧单独保存 `lastRaw` 后，`compressedTokens` 与 `tokensCompressed` 全部一致。
**Rust 的计数口径从头到尾是对的。**

### 23 → 18：块必须随被覆盖消息一起退场

`sync-blocks` 原来对每个 active 块无条件 `survived += 1`。TS 的行为是：若块的
`directMessageIds` 已经不在当前 transcript 里，说明宿主换掉了 surface，这个块不再
代表任何东西 → 置 `active: false` 且**不再推进存活计数**。

改在 `SyncBlocks` 里做这件事，一次性修好 `active`(1) + `survived`(3) + `view.length`(1)。

### 18 → 12：tier 是推导出来的，不是调用方填的

初版报告归因为「`modelContextLimit` 上报 0 导致 usage 不同」。也**判错了**。
真因是 `CompressRequest.tier` 让调用方自己声明层级，而脚手架填了 T2、TS 判 T1。

TS 的规则（`unit-projection.ts:171`）：

```ts
const outputTier = isBlockBoundary ? Math.min(3, targetTier + 1) : 1;
```

即：折消息 → T1；折块 → 比被吸收的最高块高一层，封顶 3。这是内核知识，不该外包。

于是从 `CompressRequest` 删掉 `tier` 字段，改由 `Session::derive_tier` 推导。
顺带补上了此前完全缺失的**块吸收**（`direct_blocks`、吸收块置 inactive、吸收摘要的
token 计入 `compressedTokens`）——在此之前 T2/T3 永远不可能产生，三层金字塔是装饰。

`Status` 补上 `model_context_limit` 字段，`status` 步骤不再上报 0。

### 12 → 0：`recommend.ts` 的二次筛选全部移植

这 12 处全在 `nudge`。它不是「建议性输出可以放过」——保护逻辑决定*不能折什么*
（否则过度压缩丢信息），可折性筛选决定*折了会不会被拒*（否则漏消息、甚至死循环）。
所以整段移植，新增 `billion-context/src/recommend.rs`：

| 环节 | 作用 |
| --- | --- |
| `protected_refs` | 规则 1（最近 N 条）+ 规则 2（最近 N token，**从尾部回卷**）+ 规则 3（最近 N 个 user turn）+ 工具配对闭包 |
| `integrity_withdrawals` | INV1（reasoning 与它的 tool_call 必须同折同留）/ INV2（call/result 不可分），不动点迭代 |
| `span_owns_foldable` | 区间必须覆盖其内每条摘要载体的全部直接 id，否则会被 `evaluateRefold` 判死循环 |
| 分段与分批 | 遇 user 消息且组 ≥3 条即断开；每个 batch 单独过 `minCompressRange`，不足的尾巴并进前一批，最前面的不足段丢弃 |
| `screenRecommendedUnits` | 逐单元重投影：fatal 或无产出的单元剔除 |
| `pendingByTier` | T1 = 有效区间合并后 chars；T2 = T1 块摘要 tokens；T3 = T2 块摘要 tokens |

还有三处细节契约：

1. **growth 的参考点是「上次提示」而不是「基线」**：
   `ref = lastNudgeShownTokens > 0 ? lastNudgeShownTokens : (lastPerMessageNudgeTokens > 0 ? 后者 : tokenCount)`。
   低水位锚点必须在 `decideNudge` **之后**更新，且压缩成功后两者归零。
2. **`resolveAdaptiveGrowth` 里两个 floor 是不同的字段**：
   `min(growthCap, max(growthFloor, round(limit × growthRatio)))` 用的是 `growthFloor`；
   而提示阈值 `max(minGrowthFloor, minGrowthRatio × 上式的值)` 才用 `minGrowthFloor`。
3. **`defaultCountTokens` 是 `cjk + ceil((len_utf16 − cjk) / 4)`**，不是 `ceil(len/4)`；
   CJK 判定 `0x4E00–0x9FFF / 0x3040–0x30FF / 0xAC00–0xD7AF`。中文下两者能差 3 倍，
   直接影响什么时候开始提示压缩。

---

## 分词器差分：估计器一致性与注入管线

`hf-tokenizer` 这个 feature 引入的是**数字来源**，不是新逻辑。所以它的分歧不会崩，
只会让每个阈值悄悄挪位——更要单独对比。

### 一、逐值对比：估计器必须给出同一个数

```
node diff/tok-ts.mts diff/tok-ts.json                 # 语料 + TS 参考值
cargo run --example tok_diff -- diff/tok-ts.json diff/tok-rust.json
node diff/compare-tok.mjs diff/tok-ts.json diff/tok-rust.json
```

语料 **73 条**，在 TS 侧构造（两边数的是同一批字符串），特意覆盖手写字符类最容易漂的地方：

- CJK 三个区间的**每个边界**各取一点（`0x303f/0x3040/0x3041`、`0x30fe/0x30ff/0x3100`、
  `0x4dff/0x4e00/0x9ffe/0x9fff/0xa000`、`0xabff/0xac00/0xd7ae/0xd7af/0xd7b0`），
  单字符、重复 8 次、夹在 ASCII 中间三种形态
- 非 BMP：`😀`、ZWJ 家庭表情、`𠀀`（U+20000，**2 个 UTF-16 单元但不在 CJK 区间**）
- 组合字符 `e` + `U+0301`、纯空白、纯换行、1000 字长串

**结果：0 处不一致。** 关键点是 `text.length` 在 JS 里是 **UTF-16 单元数**，Rust 侧
必须用 `ch.len_utf16()` 累加而不是 `chars().count()`——这一条对含 emoji 的文本
直接差一倍，而「自己测自己」永远不会暴露。

### 二、注入管线对比：估计器要流到每一个计数点

只对上估计器还不够，得证明它确实到达了所有该到的地方（渲染标签、nudge、块
`compressedTokens`、`status`）。两边注入**同一个**合成计数器：

```js
// diff/run-ts.mts，第 4 个参数 inject
const synthetic = (text) => {
  let sum = 0;
  for (const ch of text) sum += ch.codePointAt(0);
  return Math.ceil(text.length / 3) + (sum % 11);
};
createCore({ countTokens: synthetic });
```

```rust
// billion-context/examples/diff_harness.rs，同样第 4 个参数
struct SyntheticCounter;
impl TokenEstimator for SyntheticCounter {
    fn count(&self, text: &str) -> u32 {
        let units: usize = text.chars().map(|c| c.len_utf16()).sum();
        let sum: u32 = text.chars().map(|c| c as u32).sum();
        units.div_ceil(3) as u32 + sum % 11
    }
}
```

**注入确实有影响**：TS 自己 baseline vs injected 差 **60 个字段**（渲染标签从 `900`
变 `1.2K`、nudge、status 全动）。**而 Rust 与 TS 的 injected 结果：0 处差异。**

也就是说「默认启发式」和「注入自定义计数器」两条路径都逐字节一致，计数器不是
只在某一处生效。

### 三、真词表对照（不是一致性测试，是标定）

同一条命令加上 `assets/tokenizer.json`，可看到启发式离真词表多远：

```
heuristic vs real vocabulary: total 2027 vs 2784 (-27.2%), over on 9, under on 53
  worst: "     " +100.0%
```

偏差方向随内容翻转（代码/emoji 低估、纯空白高估），印证了单靠缩放系数修不好，
也说明默认走启发式、需要精确时开 feature 的设计是对的。

### 四、这个测试的边界

环境里**没有**独立的 tokenizer 参考实现（无 npm `tokenizers`、无 Python `tokenizers`、
无 `@anthropic-ai/tokenizer`），所以 `HfTokenizer` 本身没有做「与另一份实现逐 id
对比」——只做了内部一致性（编码-解码往返、`count == ids.len()`、确定性）以及
**下游全部计数点的等价性**。真要闭这个环，需要装一份 `tokenizers` 做三方对照。

---

## 结论

差分测试抓出了 **9 个自测永远发现不了的缺陷**：

1. 渲染标签缺尾部换行
2. token 未走 `formatTokens`（`1.5K` 而非 `1500`）
3. `type` 用了 `contentType` 而非 `toolName`
4. 折叠后没保留首条 user 消息（上游会直接拒请求）
5. nudge 在「不建议」时被吞掉，而 TS 永远返回决策对象
6. 块不随被覆盖消息退场，留下幽灵 carrier
7. **块吸收根本没实现**——T2/T3 永远不可达
8. nudge 的 growth 基准取错，导致该提示时不提示
9. 自适应阈值用错字段，50000 变成 45000

其中 4 和 7 是纯逻辑缺陷，只靠「自己测自己」不可能发现。

同时也纠正了初版报告里 3 条**误判**（视图模型、token 口径、tier 归因），
其中 2 条其实是脚手架自身的 bug。这本身就是差分测试的价值：它连测试工具一起校准。

最终状态：**差分 0 处**（默认启发式与注入自定义计数器两条路径各自 0 处），
分词器逐值对比 73/73 一致，`cargo test` 默认 52 passed / 开 `hf-tokenizer` 59 passed，
`cargo clippy` 两种配置零警告。

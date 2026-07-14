# 下一步:几个方法方向 + 怎么思考才能做出有价值的东西

日期: 2026-07-14 · 目标: 一个能发顶会的、**真加速**的 method(不是分析论文)

---

## 0. 先说"怎么思考"(避免重蹈两个月的覆辙)

前两个月的教训极其明确,提炼成三条铁律:

1. **先钉死 regime,再谈加速**。所有验证侧 idea 都死于"单请求 memory-bound 下验证近免费"。
   任何新 method 第一步必须回答:**我加速的是哪个 regime 的哪个瓶颈?**(单请求 memory-bound
   的瓶颈只有两个:target 前向次数、draft 成本。)
2. **先测奖品在不在(oracle 上界),再动手**。路线 B 若先算 oracle(47%)和可实现(0.4%)的差,
   就不用写 kernel 才发现是精度墙。**每个 method 上来先给 oracle 上界 + 一个便宜 de-risk + 硬
   kill gate**。
3. **必须有能超越的强基线**。"比 AR 快"没意义(EAGLE 随便 2×+)。基线必须是 **vanilla EAGLE-3
   固定链 / UMbreLLa 树**。novelty = 在同等准确率下把 tok/s 顶过它们。

一句话判据:**value = (真加速 over 强基线) × (novelty) × (regime 说得清)**。三者缺一不可。

---

## 1. 方向 A(首选,风险最低):把 head 训强 → 顺带验证自适应深度

**做什么**:更多语料(几万条,含长 CoT)、不截断(max_len 8k)、训到接受长度 4–5。

**为什么有价值**:
- 直接把主数字从 2.21× 顶到预期 3×+(论文头条)。
- 是 `method_confidence_gated_adaptive_depth.md` 里那个方法的**前提验证**:强 head 若恢复长 run,
  自适应深度立刻有 headroom → 从"复现"升级成"method"。

**怎么做 + kill gate**:训好后大 cap(γ=16/32)采集**逐 token 置信-接受**;若高置信区(prob>0.95)
的连续接受 run 中位数 ≥ 8 → 自适应深度成立,落解码循环测 tok/s;否则单层 head 结构性短视,转方向 B/C。

**代价**:清盘 + 生成语料(~1h) + 训练(~2–3h)。**必经之路,建议立刻做**。

---

## 2. 方向 B:级联 drafter(短跑者 + 马拉松者)

**洞察来源**:我们实测 EAGLE head 是"短跑者"(第一步 0.63、深层崩),完整 1.5B 是"马拉松者"
(长 run 多、但每 token 贵)。

**做什么**:便宜 EAGLE head 冲前 1–2 步(高命中);**当 head 置信度高(预示进入易跨度)时,才
切换到更深/更贵的 draft(或让 head 继续多步)去延伸长 run**。用校准置信度(AUC 0.86)做切换门。

**为什么 novel**:现有工作要么纯 EAGLE(短跑)、要么纯大 draft(马拉松);**按置信度在两种 drafter
间动态路由**、专门吃推理长易跨度,是没被系统做过的。

**kill gate**:先在数据上模拟"两级 drafter 的 committed-per-(target+draft)-cost",若打不过单用
EAGLE head 就毙。**中风险**(切换开销可能吃掉收益)。

---

## 3. 方向 C:EAGLE head 上的树(把接受长度从 2.4 顶向 4.8)

**洞察**:我们的 head 用的是 vLLM 线性链(top-1),接受长度 2.36;UMbreLLa 树接受长度 4.84。
树在每个位置探多个候选,能救"top-1 错但 top-2 对"的情形。

**做什么**:在便宜 EAGLE head 上做**动态树**(EAGLE-2 风格),但用我们的置信校准做**内容自适应
的树形状**(易跨度窄而深、硬拐点宽而浅)。

**风险**:EAGLE-2 树是**已知技术**,纯树 novelty 不足;novelty 必须来自"置信自适应树形状 +
推理结构"。**中高风险**——容易被审稿人说增量。

---

## 4. 方向 D(换 regime,高风险高回报):step-level 投机 for reasoning

**洞察**:token-level 有 ~1.4× 算法天花板;领域前沿(Lookahead Reasoning / SpecReason /
ConfSpec)是 **step-level 投机**——draft 提议整个推理步,target 验证步。

**我们独有的资产**:V3.6 verification gap、V4.0 draft 置信、T2 hard-negative——都是"如何判断一个
推理步该不该接受"的积累。核心难点是 **confident hallucination**(draft 自信但事实错),这也是
我们 de-risk 里量化过的(SD-safety AUC 0.68)。

**做什么**:用 draft 置信 + 廉价 hard-negative 捕捉做**顺序(非并行)step 接受策略**,目标在同等
准确率下比 ConfSpec 更高接受率/加速。

**风险**:step 验证有 verification gap,lossy 需要严格的准确率守恒证据;**高风险**,但一旦成立
是顶会级 story(打破 token 天花板)。

---

## 5. 推荐路径(决策树)

```
现在
 └─ 方向 A:训强 head(必做,升级主数字 + 验证自适应深度前提)
      ├─ 强 head 有长 run  → 落"置信自适应深度"method,测 tok/s vs EAGLE 固定链  →(赢)顶会 method
      └─ 强 head 无长 run  → 方向 C(置信自适应树) 或 方向 B(级联 drafter)
 └─(并行备胎)方向 D:step-level,若 token 侧全部触顶再上,用我们 verification 侧的积累
```

**一句话**:先把 head 训强(无论如何都赚),它同时决定"置信自适应深度"这个最干净的 method 能否
成立;成立就冲它,不成立就走树/级联;token 侧真触顶再考虑 step-level。

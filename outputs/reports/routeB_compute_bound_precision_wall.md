# 路线 B —— Compute-Bound 下的 Layer-Adaptive Early-Reject:精度墙

日期: 2026-07-14 · 数据: `outputs/vsignal/tokens.jsonl`(11520 token, 1.5B draft γ=8, 32B verify)
· 分析: `action_study/sd_audit/run_b_layeradaptive_derisk.py` · 结果: `outputs/vsignal/b_layeradaptive.json`

> **一句话**:换到 compute-bound(大 batch 服务)regime 重审"利用 T1(拒绝早解析)剪掉
> 验证 tail"。发现**奖品是真的**(oracle 上界省 47.5% 验证 FLOPs),但**拿不到**:无损剪枝
> 要求对"首个 reject 边界"做近确定性判定,浅层表示只能**排序**(AUC 0.86)做不到**判定**
> → 严格无损只能省 **0.4%**。这是一道新的、比经济墙更本质的**精度墙**。

---

## 1. 动机:给 T1 一个能赚钱的 regime

之前(`sd4_layerwise_trajectory.md`)证明:T1"拒绝早、接受晚"在**单请求 memory-bound**下做
early-exit 无效(验证被丢弃 token 几乎免费)。但在 **compute-bound / 大 batch 服务** regime,
验证是真花算力的。若能在浅层就识别"首个 reject",就能把它之后(反正会被丢弃)的 tail 从深层
计算里剪掉,省 target FLOPs。因果掩码保证:剪掉 first-reject 之后位置的深层计算,不影响之前
被接受 token 的 logits(它们只 attend 到更早位置)→ **无损**。

前一版审计(`run_t1_earlyexit_analyze.py`)用**单层原始 margin 阈值**当检测器(最弱形式)。
本次给它**最强的公平机会**:多层浅特征 + **免费的 draft 自信号** 训分类器。

---

## 2. 方法(三层)

1. **Oracle 上界(先知)**:知道每个 cycle 的真实首-reject r,保留 0..r 满深度(接受 token +
   校正位需要 target 完整分布),tail 全丢。这是**任何** early-reject 方法的算力天花板。
2. **学习型浅探针**:numpy logistic regression,特征 = 前 m 个采样层的 logit-lens margin +
   faithful-lens delta + dec_match,**外加免费 draft 自信号**(top1_prob/margin/logprob/entropy)。
   一半 cycle 训、一半测,用分类器分数驱动剪枝模拟。
3. **kill gate**:学习探针在**零精度损失**(不减投机)下 compute_saving > 10%。

> 无损语义:剪掉已 commit 的首-reject 之后位置,不改变输出(与标准贪心 SD 逐 token 一致);
> **误触发**(在真 reject 前提前停)只是让该块少投机 → 效率损失,非正确性损失。所以正确坐标
> 是"给定接受损失预算下的 compute_saving"。

---

## 3. 结果

**Oracle compute_saving 上界 = 47.5%**(accept_len_mean 3.2, n_layers 64)。tail 确实很大。

| probe 层 | reject AUC | 严格无损 save/speedup | 容忍~2%少投机 save/speedup | 容忍~10% save/speedup |
|---|---|---|---|---|
| L0  | 0.862 | 0.4% / ×1.004 | 0.4% / ×1.004 | 12.9% / ×1.123 |
| L9  | 0.862 | 0.3% / ×1.003 | 4.6% / ×1.044 | 11.4% / ×1.107 |
| L18 | 0.862 | 0.3% / ×1.003 | 3.7% / ×1.036 | 9.5% / ×1.083 |

---

## 4. 判决:KILL(精度墙)

- **kill gate 未过**:严格无损 compute_saving = **0.4%**(×1.004),远 < 10%。
- **根因是信息/精度,不是经济**(与 memory-bound 经济墙不同,是新的一层):
  - reject 率仅 0.21,每块平均 3.2 个 accept 在前;**任何一次提前误触发都砍掉整块投机**。
  - 要零精度损失 → 阈值必须高到几乎不触发 → recall→0 → 省不下东西。
  - 浅层能给 reject **排序**(AUC 0.86),但做不到无损剪枝所需的**首-reject 近确定性判定**。
- 想拿 ×1.12 必须牺牲 ~2.3% 投机量,而且这还是**理想化"每层 FLOP"模型**;真实要靠变长/ragged
  分层剪枝 kernel,headroom 会被 kernel 开销吃掉,叠在快速的 EAGLE-3 上净收益微薄。

---

## 5. 对全局的意义(论文价值)

这道精度墙 + 之前的经济墙,构成"**为什么投机解码的验证侧加速这么难**"的完整双 regime 论证:

> 验证侧存在 ~47% 的算力奖品(oracle 可达),但它在两个 regime 都被锁死:
> memory-bound 下是**经济墙**(验证近免费,无 headroom);compute-bound 下是**精度墙**
> (浅层表示能排序 AUC 0.86 却无法无损判定首-reject)。T1 的层间决策动力学给出了机制解释。

这条结论本身是 actionable 的负向指导:**把预算投到 drafter 侧而非验证侧**——正是我们转向
EAGLE-3 的依据。

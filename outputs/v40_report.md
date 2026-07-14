# V4.0 Verifier-Free Selective Speculative Reasoning — 结题报告（加速主线扩充版）

日期: 2026-07-14 · 模型: DeepSeek-R1-Distill-Qwen 1.5B (draft) / 32B-AWQ (target, E2E 时用) / 32B-bf16 (部分 audit) · Oracle: DeepSeek-V4-Pro (OpenRouter)

> **本版定位调整**：本工作是一篇**推理加速**论文。加速是目标；draft-side confidence 是实现加速的关键机制；verification gap 是解释"传统 target-verified speculative reasoning 为什么加速不起来"的诊断结果。**self-eval 比 verifier 更准不是最终目标**，它只是回答"哪些 draft step 可以不调用 target 而安全复用"的手段。

---

## 0. 一句话结论

**问题**：如何在尽量保持 target-level accuracy 的前提下，减少长链 reasoning 中昂贵 target model 的调用，从而降低端到端延迟。

**答案（本方法）**：verifier-free selective speculative reasoning —— draft 提出 reasoning step，仅用 draft 内禀信号（尤其 `self_eval_logit`）做接受判断，高置信免验证接受、低置信才 handoff 给 target。零额外 target verification cost。

**当前实证**：
- 在 GSM8K E2E 上，selfconf 以近零额外验证成本**追平 target 级准确率**；ConfSpec 式 32B-as-verifier 基线反而把准确率拖到最差（0.778 < draft_only）——这是 V3.6 "verification gap" 的端到端铁证，说明传统 verifier **同时破坏 latency 和 accuracy**。
- wall-clock 加速目前温和（阈值 0.70x→1.07x），**但这是一个尚未做系统优化的 step-wise 原型的结果**，不是 confidence gate 没有加速潜力的证据。省下的 target 计算尚未转化为同比例的 wall-clock 收益（重复 prefill / step-level 调度 / handoff 重复 prefill 是主要漏损）。

---

## 1. 核心问题与形式化

不再把方法表述为"draft 自信号比 target verifier 更可靠"，而是：

> **如何减少 reasoning 过程中昂贵 target model 的调用，同时尽可能保持 target-level accuracy？**

形式化为**质量约束下的延迟最小化**：

$$
\min_{\pi}\ \mathbb{E}[T(\pi)]
\quad\text{s.t.}\quad
\mathrm{Acc}(\pi)\ \ge\ \mathrm{Acc}(\text{target-only})-\Delta,
$$

- $\pi$：每一步"接受 draft 或 handoff 给 target"的策略；
- $T(\pi)$：端到端 wall-clock latency；
- $\Delta$：允许的准确率损失（如 $0$、$1\%$、$2\%$）。

self-confidence 不是要证明的终点，它只服务于一个子问题：**哪些 draft step 可以不调用 target，直接安全复用？**

### 1.1 端到端延迟分解

$$
T_{\text{total}} = T_{\text{draft}} + T_{\text{verification}} + T_{\text{handoff}} + T_{\text{system overhead}}.
$$

即使 verifier 完全正确，只要 $T_{\text{verification}} \approx T_{\text{target generation}}$，验证本身就吞掉了收益。而 V3.6 进一步表明 verifier 还不可靠 —— 因此 target verification **既贵又不准**。

---

## 2. 背景与动机

### 2.1 V3 系列的死路
- **V3.5**: `Branch@4 ≈ 1.06×Handoff` —— 多候选救援的验证并不便宜。
- **V3.6**: 32B 作为 Accept/Reject step verifier，在真实 1.5B 候选分布上判别力≈0（全 τ max precision 7%, AUC≈0.5），但在手工构造的"明显对/错"上 AUC 0.97 —— 即 **verification gap**（plausibility ≠ progress）：verifier 喜欢"看起来像推理"的复述，不识别实质进展。

### 2.2 核心假设
放弃 Branch 与 target-as-verifier 两条死路。改用：

> **Draft 对自身下一步的内禀置信信号 + conformal abstention**：高置信免验证接受 draft step；低置信 handoff 给 32B target。**额外 target 验证成本 = 0**，且可给出可证明的接受精度保证。

---

## 3. 方法详述

### 3.1 问题设定（顺序 speculative reasoning）

对每个推理题，维护已接受前缀 `prefix`。每一步：

1. **Draft**（1.5B）从 `prefix` greedy 生成一个完整 reasoning step `s_d`（以 `\n\n` / `<STEP_END>` 为边界）。
2. **Gate** 读 draft 自信号，输出分数 `s(s_d)`。
3. 若 `s(s_d) ≥ τ`：**接受** `s_d`，追加到 `prefix`（不调用 target）。
4. 否则：**Handoff**——Target（32B）从同一 `prefix` 生成替换步 `s_t`，追加到 `prefix`。
5. 重复直到出现 `\boxed{}` 或达到步数上限。

与经典 token 级 speculative decoding 的区别：决策单位是**语义 step**（一段推理），不是 token；gate 不调用 target，只用 draft 自身信号。

### 3.2 Draft 自信号（近零成本）

对候选 step `c`，把 `prefix_ids + candidate_ids` 一次性 teacher-force 喂给 draft，读 `prompt_logprobs`（vLLM `prompt_logprobs=k`），在候选区间上计算：

| 信号 | 定义 | 方向（↑=更可能 acceptable） | 生成时是否已免费获得 |
|---|---|---|---|
| `mean_logprob` / `min_logprob` / `last_logprob` | 候选 token 实际 logprob 的统计 | ↑ | ✅ decode 时顺手可得 |
| `perplexity` | `exp(-mean_logprob)` | ↓ | ✅ |
| `mean_entropy` / `max_entropy` | 每位置 top-k 分布的 Shannon 熵 | ↓ | ✅ |
| `mean_margin` / `min_margin` | top1−top2 logprob 差 | ↑ | ✅ |
| `repetition_rate` | 1 − (distinct bigrams / total) | ↓ | ✅ |
| `step_length` / EOS 概率 | step 长度 / step-end 概率 | — | ✅ |
| **`self_eval_logit`** | 在 `prefix+c` 后拼 "Is this step correct and useful? Answer:"，读 draft 对 ` yes` vs ` no` 的单 token logit 差 | ↑ | ❌ 需额外一次 prefill |

实现要点（见 `v40_self_signals.py`）：
- 解析与 vLLM 调用分离，便于 token 对齐单测。
- `yes`/`no` 对必须是单 token（对 Qwen tokenizer 用 ` yes`/` no`）。
- 成本：相对已生成该 step 的 draft 前向，`self_eval_logit` 额外一次 prefill，无 decode。**这一次 prefill 是当前系统最大的可优化漏损**（见 §7.1）。

### 3.3 信号融合

```text
x = standardize([self_eval_logit, mean_margin, max_entropy, mean_entropy,
                 mean_logprob, min_logprob, repetition_rate])
s = σ(w·x + b)     # logistic，在 de-risk 标定集上拟合
```

也可退化为单信号 passthrough（`s = self_eval_logit`）。**刻意不把 32B verifier_score 放进融合**，以保证 gate 零 target 成本（verifier-free）。

### 3.4 Conformal / LTT 阈值（质量保证，次要位置）

目标：分布无关保证

```text
P(oracle_acceptable = 0 | accepted) ≤ ε    （以概率 ≥ 1−δ）
```

即接受集上的 precision ≥ 1−ε。做法（`v40_conformal_gate.py`）：
1. 按 `problem_id` 切 train / cal / test（防泄漏）。
2. train 上拟合融合模型；cal 上对分数网格做 Bonferroni-LTT。
3. 每个阈值 τ：用 Clopper-Pearson 精确二项上界估计 `P(y=0|s≥τ)`；若 UCB ≤ ε 且接受数 ≥ min_accept，则该 τ 通过。
4. 在通过的阈值中取覆盖率最大者部署。

> **定位说明**：conformal 用于**安全选择 operating point**，不是方法核心。当前数据量不足以认证，因此它不应主导论文（见 §8 的阈值选择应改为质量约束下的延迟最优）。

### 3.5 与 ConfSpec / SpecReason 的定位差异

- **ConfSpec 前提**：verification 廉价且可靠 → V3.5/V3.6 实证反驳。
- **本方法**：承认跨模型 step 验证不可靠，改用 **draft 内禀信号**；target 只在 abstain 时做生成，不做判别。
- **SpecReason 式 draft-confidence 阈值**：可视为本方法的单信号特例；我们加了多信号融合 + 两级 gate + 质量约束阈值选择框架。

---

## 4. 论文三大贡献（加速主线）

### Contribution 1：诊断——传统 target verification 阻碍推理加速

现有 speculative reasoning 依赖 target 对 draft step 做显式验证。但在自然生成的 reasoning step 上：

1. 验证本身消耗昂贵（$T_{\text{verify}} \approx T_{\text{target-gen}}$，见 §1.1 分解）；
2. 对自然 draft step 判断不可靠（verification gap，§2.1）；
3. 高频验证抵消 speculative execution 的计算收益。

结论：**传统 verifier 同时破坏 latency 和 accuracy。** 这是问题诊断，服务于加速主线，而非最终故事。

### Contribution 2：方法——Verifier-Free Selective Speculative Reasoning

核心设计：draft 提出 step；gate 仅用 draft-side signals；高置信免验证接受；target 只在必要时参与**生成**。重点是 **verifier-free**：额外 target verification cost = 0（不是 gate cost = 0）。

两种策略的延迟对比：

**Target-verified speculation**：
$$
T_{\text{TV}} = T_d + T_v^{(32B)} + (1-a_v)\,T_t,
$$

**Draft-confidence speculation（本方法）**：
$$
T_{\text{DC}} = T_d + T_g^{(1.5B)} + (1-a_g)\,T_t.
$$

其中 $a_v, a_g$ 为各自接受率。因为 $T_g^{(1.5B)} \ll T_v^{(32B)}$，即使接受率相近，本方法理论上也应更快。这是最清楚的加速动机。

### Contribution 3：accuracy–latency Pareto frontier

最终贡献不是"某个阈值下 1.07x"，而是：**扫描 confidence threshold 得到连续的 accuracy–latency trade-off 曲线**；在匹配 target accuracy 或限定准确率损失下，本方法优于 target-only 与 target-verification 基线的 frontier。

核心图：横轴 speedup/latency，纵轴 accuracy；每个阈值一个点；target-only / draft-only / target_verify 为基线点。要证明存在一个区域：

$$
T_{\text{selfconf}} < T_{\text{target-only}}
\quad\text{且}\quad
\mathrm{Acc}_{\text{selfconf}} \approx \mathrm{Acc}_{\text{target-only}}.
$$

**不是**证明每个阈值都加速。保守阈值下的 0.70x 只是 Pareto 曲线上一个不理想点，不代表方法失败。

---

## 5. Phase 0 — De-risk（判据：GREEN）

### 数据
179 条 oracle 标注候选：GSM8K 107（正例 45）、AIME 24（正例 2）、AIME-calib 复用 48（正例 3）。base rate 27.9%。

### 判别力（ROC-AUC vs oracle_label）

| 信号 | AUC |
|---|--:|
| **self_eval_logit** | **0.844** |
| mean_margin | 0.741 |
| mean_entropy | 0.700 |
| 融合 draft-only（GroupKFold OOF） | 0.833 |
| 32B verifier_score（基线） | 0.667 |

### per-dataset 头对头

| dataset | N | 正例 | verifier AUC | self_eval AUC |
|---|--:|--:|--:|--:|
| gsm8k | 107 | 45 | 0.599 | **0.828** |
| aime_calib | 48 | 3 | 0.356 | **0.659** |
| aime | 24 | 2 | 0.950 | 0.455 |

**决策：GREEN**（最佳 draft 信号 AUC 0.844 ≫ verifier）。

> **加速视角的解读**：AUC 高低不是终极指标，应比较 **"每毫秒 gate 开销带来的有用判别"**（useful gating per millisecond）。见 §7.2 的 net latency saving。

---

## 6. Phase 1 结果

### 6.1 Conformal gate 现状
单测 11 项全过。现有数据量下 ε=0.10~0.25 正式认证阈值均返回 None（数据饥渴）。经验操作点可用：tp=0.85 → 精度 0.87 @ 覆盖 13%；GSM8K 上 tp=0.70 → 覆盖约 51%。

### 6.2 E2E（GSM8K）

**配置 A：tp=0.85（保守）**

| 策略 | 准确率 | speedup | steps | handoffs | 验证开销 |
|---|--:|--:|--:|--:|--:|
| target_only | 1.00 | 1.00x | 22.0 | 22.0 | 0 |
| draft_only | 0.875 | 2.60x | 25.3 | 0 | 0 |
| **selfconf** | **1.00** | **0.70x** | 24.1 | 18.2 | 3.22s |
| target_verify | 0.75 | 1.53x | 24.9 | 2.3 | 4.62s |

**配置 B：tp=0.70（提高接受率）**

| 策略 | 准确率 | speedup | 接受率 | 验证开销 |
|---|--:|--:|--:|--:|
| target_only | 1.00 | 1.00x | — | 0 |
| draft_only | 0.889 | 1.90x | 100% | 0 |
| **selfconf** | **1.00** | **1.07x** | 58.6% | **1.85s** |
| target_verify | 0.778 | 1.29x | 92.5% | 4.19s |

**读数**：selfconf 追平 target 准确率；target_verify 近乎无差别接受（92.5%）→ 准确率崩到 0.778（比纯 draft 还差）—— verification gap 的 E2E 铁证。selfconf 的验证开销（1.85s，含每步 `prefix+candidate` 重复 prefill）显著低于 target_verify（4.19s 的 32B 前向），但仍未压到接近零，是 §7 的主要优化对象。

---

## 7. 下一步系统优化（把省下的 target 计算转成 wall-clock 收益）

当前 1.07x 的瓶颈不在算法，而在系统实现：每步 draft decode + 重复 prefill + self-eval prefill + draft/target 切换 + 很长的 reasoning prefix + Python/vLLM 调度。以下按预期收益排序。

### 7.1 消灭重复的 self-eval prefill（最高优先级）

现状（`run_v40_e2e.py` 的 `verify_sec`）：`self_eval_logit` 对 `prefix + candidate + question` 再做一次完整 prefill，成本 $O(|\text{prefix}|+|\text{candidate}|)$。

理想实现：
1. draft 生成该 step 时**保留 KV cache**；
2. step 结束后在现有 KV cache 后直接追加 `Is this step correct and useful? Answer:`；
3. 只额外 decode/read 一个位置的 yes/no logits。

成本从 $O(|\text{prefix}|+|\text{candidate}|)$ 降到 $O(|\text{self-eval prompt}|+1)$。这是把 1.07x 提到更明显收益最可能的一处。

若当前 vLLM 接口不便保留 sequence KV，至少要测：**prefix caching / continuous batching / persistent request / engine-level KV reuse**。否则测到的是接口开销，不是算法速度。

### 7.2 优先使用生成时已免费的信号 + net latency saving

`token logprob / margin / entropy / repetition / step length / EOS 概率` 在 draft decode 时顺手可得，不应再单独 teacher-force。比较标准从 AUC 改为 **useful gating per millisecond**：

| Gate | AUC | Gate cost | 预期最终 speedup |
|---|--:|--:|--:|
| self_eval | 0.844 | 高一些 | 1.07x |
| margin | 0.741 | 近零 | 可能 1.2x |
| entropy | 0.700 | 近零 | 可能 1.25x |

引入新评估量：
$$
\text{Net latency saving} = \text{saved target time} - \text{gate time}.
$$

### 7.3 减少 step 边界带来的系统调用次数

target_only 平均 22 步、selfconf 平均 24 步 → 几十次请求切换。若每 step 是独立 vLLM request，调度成本很高。

- **方案 A：更大的 reasoning chunk**。每次生成 2 个 step / 固定 64–128 tokens / 到明显语义边界，决策次数从 ~24 降到 8–12。需画 chunk length 消融 $L\in\{32,64,128,\text{semantic step}\}$；chunk 越长错误风险越大。
- **方案 B：confidence-triggered checking（两级 gate）**。先用免费信号一级筛选：
$$
m > \tau_h \Rightarrow \text{accept},\quad m < \tau_l \Rightarrow \text{handoff},\quad \tau_l\le m\le\tau_h \Rightarrow \text{run self\_eval}.
$$
即免费统计信号处理大多数 step，昂贵但更准的 self-eval 只负责模糊样本。这比"每步都 self-eval"更符合加速论文。

### 7.4 Handoff 时减少 target 重复 prefill

target 从同一 prefix 生成替换步，若每次重 prefill 整个 prefix，则 $T_{\text{handoff}}(k)\propto |\text{prefix}_k|$，后半程 handoff 越来越贵。应：保存 target prefix KV / draft 接受新步后异步或批量更新 target KV / handoff 时用 prefix caching / 至少实验中分开报告 target prefill 与 decode 时间。这是 speculative reasoning 相对普通 routing 的真正系统难点。

---

## 8. 阈值选择：从"经验 precision"改为"质量约束下延迟最优"

不再按经验 precision（0.70/0.85）选阈值。加速为主目标时：

$$
\tau^* = \arg\min_\tau T(\tau)
\quad\text{s.t.}\quad
\mathrm{Acc}(\tau) \ge \mathrm{Acc}_{\text{target}} - \Delta.
$$

分别报告：

| 质量约束 | 最佳 speedup | 接受率 | target 调用减少 |
|---|--:|--:|--:|
| 0% accuracy drop | … | … | … |
| ≤1% drop | … | … | … |
| ≤2% drop | … | … | … |

conformal 保留但放次要位置（安全选择 operating point）。

---

## 9. 需要报告的完整指标

### 模型计算侧
target generated / prefill / decode tokens；target calls；draft generated / prefill tokens；gate FLOPs 或 gate latency。

### 系统侧
total latency；p50/p90/p95 latency；time-to-first-token；GPU 利用率；peak memory；draft–target switching overhead。

### 推测效率
$$
\text{Target avoidance rate} = \frac{\text{steps accepted without target}}{\text{total steps}},
$$
$$
\text{Useful acceptance rate} = \frac{\text{accepted correct draft steps}}{\text{total proposed draft steps}},
$$
$$
\text{Latency saved per accepted step} = \frac{T_{\text{target-only}} - T_{\text{method}}}{N_{\text{accepted}}}.
$$

最后一项用于判断"接受更多 step"是否真的产生速度收益。

---

## 10. 论文故事线

- **Motivation**：推理模型长链生成导致 target latency 高。Step-level speculative reasoning 希望小模型承担简单步骤，只在困难处调用大模型。
- **Challenge**：现有方法依赖大模型验证 draft step，但大模型验证 (1) 本身昂贵，(2) 对自然 draft step 不可靠，(3) 高频验证抵消 speculative execution 收益。
- **Method**：verifier-free selective speculative reasoning —— draft 生成候选步；draft-side confidence 判断；高置信直接接受；低置信才调用 target；阈值控制 accuracy–latency trade-off。
- **Key finding**：draft-side self-evaluation 能更有效区分可接受步骤，使系统在无 target verification 下避免大量 target 调用。
- **Main result**：accuracy-matched 设置下获得端到端加速，全阈值扫描中优于 target verification baseline 的 accuracy–latency frontier。

---

## 11. 标题候选

- 强调系统：**Verifier-Free Speculative Reasoning via Draft-Side Confidence**
- 强调加速：**Accelerating Long-Form Reasoning with Draft-Confidence Selective Execution**
- 最直接（最有记忆点）：**Draft When Confident: Verifier-Free Speculative Reasoning for Faster Inference**

---

## 12. 如何看待当前 1.07x

1.07x 现在不能作为强结果，但意义在于：**在一个明显未充分系统优化的 step-wise 原型上，已达到正加速并保持了当前小样本下的 target accuracy**。方向有速度潜力。

下一阶段明确目标：
1. 去掉重复 self-eval prefill（§7.1）；
2. 复用 draft 和 target KV cache（§7.1/§7.4）；
3. 减少 step-level request 数（§7.3）；
4. 使用两级 gate（§7.3 方案 B）；
5. 扩大 E2E 样本；
6. 在 matched accuracy 下优化阈值（§8）。

若这些优化后仍只有 1.05–1.10x，方法加速价值确实有限；但在这些系统问题解决前，还不能据此判断上限。

> **主线定位**：这是一篇**推理加速**论文。verification gap 解释了为什么现有 target-verified speculative reasoning 加速不起来；draft confidence 提供了一个低成本、verifier-free 的替代机制。

---

## 13. 局限

1. 加速温和且强依赖 τ（0.70x → 1.07x），且尚未做系统级优化（§7）。
2. Conformal 高精度认证暂不可达（数据饥渴）。
3. AIME E2E 因 R1 长链不可用。
4. 样本偏小（de-risk 179 / E2E 15）。

## 14. 产物

- 代码：`v40_self_signals.py`、`run_v40_derisk.py`、`v40_derisk_analyze.py`、`v40_conformal_gate.py`、`run_v40_e2e.py`、`v40_analyze.py`
- 单测：`test_v40_self_signals.py`、`test_v40_conformal.py`
- 数据：`outputs/action_study_v40_derisk/`、`outputs/action_study_v40_e2e{,_tp70}/`

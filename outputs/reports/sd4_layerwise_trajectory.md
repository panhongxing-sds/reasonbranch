# SD④ — Target Decision Resolution Depth（Layerwise Verification Trajectory）

## 1. 想法

不是用 target **最终** entropy/margin，而是用：

> Target 在第几层才稳定地接受或推翻 draft token。

对 draft token \(d_t\) 与 target 最终 token \(y_t\)，在采样层 \(\ell\) 上：

\[
\delta_t^{(\ell)} = \left\langle \mathrm{LN}(h_t^{(\ell)}),\; e_{y_t}-e_{d_t} \right\rangle
\]

三个量：

| 量 | 含义 |
|---|---|
| \(\ell^{\mathrm{flip}}\) | 决策反转层：此后层全部 \(\delta>0\) |
| \(N^{\mathrm{flip}}\) | 符号翻转次数（纠结程度） |
| \(S_t\) | 路径变分 \(\sum\|\Delta\delta\|\)（收敛速度） |

同样最终 margin=5，可能 early-resolve，也可能 late-flip —— 最终 logits 看不到的维度。

用途：作为**下一轮** speculative length / 保守度的控制信号（不是替代本轮 Accept/Reject）。

## 2. 新颖性表述（准确版）

不能说“第一次用 target hidden”。应说：

> 使用 target 对 draft–target 分歧的**层间决策反转轨迹**，作为 SD 的反馈控制信号。

相邻工作（Lever / EAGLE / early-exit / 层间 KL 分析）用途不同；未见到把 flip trajectory 用于控制下一轮 γ。

## 3. 最小验证设计

收集线性 SD cycles，特征：

- M1：\(A_c,\gamma\)  
- M2：+ draft confidence  
- M3：+ final target \(H_T\), margin, KL  
- M4：+ \(\ell^{\mathrm{flip}}, N^{\mathrm{flip}}, S\), late_resolve_frac  

预测 \(A_{c+1}\)。Kill gate：**M4 必须显著超过 M3**。

另做 matched-margin：相近最终 margin 下 early vs late 的下一轮 A。

## 4. 实验结果（v1，方法有误 — 见 §5 更正）

设定：1.5B/32B，γ=8，16 prompts × 16 cycles = **240** 条；采样 8 层（含末层）。

| 模型 | RMSE | MAE | R² |
|---|--:|--:|--:|
| M1 | 2.681 | 2.201 | −0.003 |
| M2 | 2.565 | 2.117 | 0.082 |
| M3 | 2.549 | 2.092 | 0.093 |
| **M4** | **2.542** | **2.085** | **0.099** |

初判：M4 vs M3 RMSE +0.3%，判 FAIL。**此判决基于有 bug 的特征，已作废，见下。**

## 5. 方法论更正（关键）

复盘发现 v1 的 \(\delta\) 计算有两处致命错误，导致特征本身是噪声，"无信号"结论不成立：

1. **归一化用错**：v1 用 `F.layer_norm(h)`，但 Qwen 用的是 **RMSNorm**，且没有走模型自己的 final norm。后果：被拒 token 在**最后一层**的 \(\delta\) 符号竟与真实 `final_margin` 相反（实测 v2 修正前 28/28 号不一致的反例）。用错误符号算出的 flip_depth/flip_count 自然是随机数。
2. **接受 token 无轨迹**：v1 对 accepted token 令 \(e_y-e_d=0\)，轨迹恒为 0，无法回答"接受在哪层锁定"。

**v2 修正**：改用忠实 **logit-lens**——每层 hidden 走模型自己的 final norm，再投影到 lm_head 的两行；并对所有 token 记录**全词表 logit-lens argmax 是否等于 draft token**（决策解析深度 `dec_depth`，接受/拒绝都适用）。修正后最后一层 \(\delta\) 与真实 margin **28/28 同号**。

数据：60 prompts × 24 cycles = **11520 token**；代码 `sd_audit/run_vsignal_collect.py`、`run_sd4_redo_analyze.py`；产物 `outputs/vsignal/tokens.jsonl`、`sd4_redo.json`。

### T1 —— 决策解析非对称性（强、干净、新）

| | dec_depth / 深度（mean） | 中位数 |
|---|--:|--:|
| **接受** token | **0.914** | 0.984（≈ 最后一层） |
| **拒绝** token | **0.145** | **0.0（第 0 层）** |

Mann–Whitney \(p \approx 0\)（n=11520）。即 **"拒绝早、接受晚"**：target 的**浅层就能否掉一个 draft token**，但要**确认接受必须跑到顶层**。这是 v1 因 bug 完全错过的真实表征级发现。

### T2 —— hard-negative 解析更晚（显著但效应小）

hard = draft 自信却被拒的 token（draft_top1_prob ≥ 中位数，即 draft-confidence gate 会误接受的那批）：

| 特征 | hard mean | easy mean | p | AUC(hard vs easy) |
|---|--:|--:|--:|--:|
| path_speed | 20.69 | 18.08 | 4.8e-11 | 0.577 |
| flip_depth | 44.6 | 39.8 | 2.7e-7 | 0.560 |
| dec_depth | 11.5 | 7.0 | 1.6e-3 | 0.537 |
| final_margin | 1.96 | 2.25 | 1.2e-7 | 0.562（反向） |

方向一致且极显著：**draft 自信的误判确实在 target 里解析得更晚 / 路径更纠结**——这是 V3.6"verification gap"的表征级解释。但效应小（AUC≈0.56–0.58），且 `final_margin` 也带同量级信息，轨迹并不碾压最终 margin。

### T3 —— 次周期预测（无泄漏，确认死亡）

用第 c 周期特征预测第 c+1 周期接受长度（OOF ridge，n=1380）：

| 模型 | RMSE | R² |
|---|--:|--:|
| M_draft | 2.896 | 0.080 |
| M_draft+final | 2.897 | 0.079 |
| M_draft+final+**traj** | 2.910 | **0.071** |

加轨迹特征 R² **反降**。用修正后的特征仍无效 ⇒ "用轨迹预测未来 γ"这条路是真的死，不是特征 bug。

### Early-exit 可行性（确认不可行）

截断到第 L 层的决策一致率：L=45 → 0.28，L=54 → 0.47，只有 L=63 才 1.0。**接受决策集中在顶层**，故"浅层截断确认接受"不可行。

## 6. 判决（更正后）

**部分翻案：一个强发现 + 明确负面。**

- ✅ **T1（拒绝早/接受晚非对称）**：统计无懈可击、v1 因 bug 错过，是可写进论文的表征级观察。
- ⚠️ **T2**：hard-negative 解析更晚，显著但效应小，未超过 final margin，作为"verification gap 机制解释"的佐证。
- ❌ **T3 / early-exit / 控制下一轮 γ**：确认无效。

**由 T1 引出的"自预算浅层探针"机制 —— 已离线证伪（n=11520）**：

| L 层 | 拒绝探针 precision | recall | flag_rate | 接受探针 precision | 可自信接受占比 |
|--:|--:|--:|--:|--:|--:|
| 45 (70%) | 0.219 | 0.952 | 0.912 | 0.886 | 0.088 |
| 54 (85%) | 0.269 | 0.887 | 0.691 | 0.923 | 0.309 |
| 63 (100%) | 1.000 | 1.000 | 0.210 | 1.000 | 0.790 |

- **浅层拒绝探针**：精确率 ≈ 基础拒绝率 0.21，把"尚未解析的接受"也误判为拒绝（flag_rate 0.91）⇒ 无法廉价定位 reject 边界，**死路**。
- **浅层接受探针**：高精度（0.88–0.95）但低召回——直到 85% 深度也只能自信接受 31% 的 token ⇒ 只有微小 early-exit 空间。

**结论**：T1 非对称性中"要省算力就得省接受确认"，而接受恰恰晚解析，故 **T1 是表征级科学发现，不构成验证加速**。SD①×T1 统一路线（自预算浅层探针）到此证伪。

## 7. 让 T1"有用"的第二轮尝试：早拒绝剪枝验证器（证伪，含净加速模拟）

用**连续 margin**（每层 logit-lens `best_token − draft_token`，对接受/拒绝都定义）替代二值 argmax，做一个**保正确性**的机制：浅层探针一旦高精度判某位置为拒绝，则剪掉其后位置的剩余层（这些位置在标准 SD 里本就被丢弃）。误报只会让块提前结束（接受变短、需重抽），**不改输出**，故用"每单位算力提交 token 数"评估净加速。

代码 `sd_audit/run_t1_earlyexit_analyze.py`；数据 `outputs/vsignal/t1_earlyexit.json`（60 prompts / 1440 cycles / accept_len 均值 3.2）。

**净加速 Pareto（扫 8 探针层 × 10 阈值）**：

| 接受损失预算 (token/cycle) | 最优算力节省 | 净加速 |
|--:|--:|--:|
| 0.0 | **0.0%** | 1.00 |
| 0.10 | 0.9% | **0.987（更慢）** |
| 0.25 | 2.2% | 0.982（更慢） |

**零损失 ⇒ 零节省**；任何早剪都得不偿失。根因由逐层 AUC 给出：margin 区分"真拒绝 vs 尚未成熟的接受"的能力只在顶层涌现——

| 层(深度) | 0 | 45(70%) | 54(85%) | 63(100%) |
|---|--:|--:|--:|--:|
| AUC(区分接受/拒绝) | 0.505 | 0.564 | 0.711 | **0.989** |
| 接受 margin 均值 | 11.52 | 5.25 | 3.65 | 0.00 |
| 拒绝 margin 均值 | 11.58 | 6.04 | 7.29 | 2.10 |

第 0 层接受/拒绝 margin 几乎相同（11.52 vs 11.58）。**"拒绝早解析"只体现在二值 argmax≠d（对错误 token 平凡成立），而真正用于剪枝的区分信息只在顶层出现** ⇒ 早停在物理上不可能保正确地省算力。

## 8. T1 唯一被数据支持的"用途"（非运行时加速）

既然运行时省不了，T1/T2 的可行用途是**离线**的：hard-negative（draft 自信却被 target 深层推翻的 token）是 V4.0 draft-confidence gate 会误接受、且危害最大的样本。用**完整 target 的晚解析信号**离线挖掘这些 hard-negative，做**定向蒸馏/难例课程**改进 drafter，使其自信度更校准 ⇒ V4.0 gate 精度提升 ⇒ 运行时更快，**且运行时不需要 target 验证**。这把 T1/T2 变成"改进 drafter 的离线挖矿信号"，闭环回到运行时收益。前提（T2 显示 late-resolution 能否比 draft 信号更好地识别 hard-negative）目前 AUC≈0.57，偏弱，需专门实验确认是否值得。

## 9. 第三轮：接受边界预测 + 自适应 γ（真信号，但被 SD 经济学证伪）

换角度：被接受的 token 若"勉强/晚解析"，是否预示投机块即将结束（下一位置拒绝）？在 7924 个"接受后仍有下一位置"的样本上（下一拒绝基率 0.193）：

| 信号 | AUC(→下一拒绝) | 可用时点 |
|---|--:|---|
| target `dec_depth` | **0.620** | 验证后 |
| target `final_margin` | 0.595 | 验证后 |
| **draft `entropy`** | **0.601** | **验证前（draft 侧）** |
| draft `top1_prob` | 0.600 | 验证前 |

`final_margin` 三分位：低 margin 接受后 P(下一拒绝)=0.251，高 margin 仅 0.137（≈2×，单调）。**这是整条 T1/T3 线里最强的预测信号，且 draft 自己就能算（AUC 0.60），target 层间只多 0.02。**

用途设想：draft 用自身 entropy 早停 → target 少验证注定丢弃的 token → **自适应 γ**。决定性检验：自适应必须打败**最优固定 γ**。两种成本模型（`run_t1_adaptive_gamma.py`，1440 cycles）：

| 成本模型 | 最优固定 γ | 该模型下 passes-or-verify/token | 自适应 vs 最优固定 |
|---|---|--:|--:|
| 内存受限（单请求真实） | **γ=8** | 0.250 passes/token | **0.987（更差）** |
| 计算受限（大 batch） | **γ=1** | 1.0 verify/token | 0.999（持平） |

**两头都打不过最优固定 γ。** 数据/代码：`outputs/vsignal/t1_adaptive_gamma.json`、`sd_audit/run_t1_adaptive_gamma.py`。

## 10. 统一根因（本轮最大收获，可作论文"lessons"）

四条 verification-side idea（SD①缩短验证、SD③跳 LM-head、SD④/T1 early-exit、自适应 γ）失败**同源**：

> **单请求 SD 验证是内存带宽受限的**——一次 32B forward 的成本由权重加载主导，验证 8 个 token ≈ 验证 1 个 token。因此"验证注定被丢弃的 token"几乎免费，任何"少验证/更聪明地验证/早停"都没有 headroom，最优策略平凡地是"γ 拉满"。换到大 batch 变计算受限后，最优又平凡地退化为"γ=1 不投机"。**两个极端都不给耍聪明的空间。**

推论：要在验证侧拿到真加速，必须离开"单请求 memory-bound"设定——异构 INT4 硬件、真实大 batch 服务、或让**起草**（而非验证）成为瓶颈的场景。

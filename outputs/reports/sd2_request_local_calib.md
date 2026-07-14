# SD② — Request-Local Residual Calibration

## 1. 想法

每次 target verification 后免费得到位置 t 的 draft/target logits：

\[
r_t = z_T^{(t)} - z_D^{(t)}
\]

问题：同一次请求前半段的 \(r_t\)，能否改善后半段 drafting？

第一版：**per-request、无梯度、无持久参数**的 EMA / token-bias：

\[
\tilde z_D = z_D + g_{\phi_{\text{request}}}(x)
\]

请求结束即删。部署仍只有一个 drafter。

## 2. 新颖性核实

| 工作 | 关键特征 |
|---|---|
| OSD | 在线 KD，**梯度**，跨请求持久 |
| TIDE | serving 内适配，**更新参数**，跨请求 locality |
| OnlineSpec / EvoSpec | 在线学习 / LoRA，持久 |

→ “单请求、无梯度、无持久、EMA/闭式”在文献中是**相对干净的空白点**。新颖性在三者中最强。

## 3. Kill Gate（设计）

前 25% verification 位置拟合校准器，测后 75%：

- target top-1 coverage  
- KL(draft∥target)  
必须**稳定改善**（非训练集过拟合）。

## 4. Audit B（执行）

- 6 prompts，γ=8，线性 greedy SD  
- 768 个位置级记录  
- 校准：EMA residual + per-token bias  

| 集合 | Δ top1 (EMA) | Δ KL |
|---|--:|--:|
| train（前 25%） | −0.016 | −0.125（KL 变差） |
| **test（后 75%）** | **−0.043** | **−0.159** |

- 仅 **2/6** prompt 的 test top1 改善  
- 判决：**FAIL**

代码：`action_study/sd_audit/audit_b_residual.py`  
原始：`outputs/sd_audit_b.json`

## 5. 重做 Audit（v2，修复 v1 三处问题）

v1 的问题：全词表 152k 维 EMA（少样本估计不可能准）、词表未对齐、只有 6 prompt、且没有区分"信号不存在"与"估计器太弱"。

v2 修复：
1. **限定活跃 token 集**（校准段出现过的 target/draft argmax + |残差| top-64），而非全词表。
2. **留出校准片段选收缩系数 λ∈{0,.25,.5,.75,1}**，其中 **λ=0 = 基线** ⇒ 校准在期望意义上**不可能比基线差**（不泛化时自动 no-op）。
3. 正确 `align_logits`。
4. **40 prompts**，每请求 ~110–160 位置，配对 sign test。
5. **新增 oracle 上界**：直接在 eval 段选最优 λ（作弊），量化"任何请求内加性偏置的天花板"。

代码：`action_study/sd_audit/run_sd2_redo.py`；数据：`outputs/vsignal/sd2_redo.json`。

### 结果（40 请求，base agreement 已 0.849）

| 指标 | 值 |
|---|--:|
| 诚实校准 mean Δagreement | **−0.001** |
| 改善 / 变差 / 持平 | 1 / 3 / 36 |
| sign-test p | 0.625（无效） |
| λ 选择分布 | λ=0 占 26/40 |
| **oracle 上界 mean Δ** | **+0.0155** |
| **oracle 上界 max Δ** | **+0.079** |

**决定性诊断**：连"在 eval 上作弊选 λ"的 oracle 也只有平均 +1.5%、最高 +7.9% 的 agreement 提升，且诚实估计器连这点都拿不到（因为校准段的微弱偏置不能泛化到评估段）。

## 6. 判决（更正后）

**FAIL —— 但这次是"信号本身不存在"的强证伪，不是估计器问题。**

根因：1.5B 与 32B 的 logit 残差**不是请求内稳定的 per-token 常数偏移**，而是**上下文/位置相关**的。因此常数加性校准的**天花板本身**就只有 ~1.5% agreement（对 GSM8K 数学推理）。

真正可能复活的方向（都已超出"无梯度/无持久参数"的初衷，等价于训练小适配器，故不属于本 idea）：
- 上下文条件化的 bias（需小网络/低秩映射 + 梯度）；
- 换到"格式/实体强重复"workload（代码补全、结构化生成），残差的 per-token 稳定性可能更高——这是唯一值得另开的实验，但不适用于当前数学推理设定。

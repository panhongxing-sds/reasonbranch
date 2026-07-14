# SD③ — Certified Candidate-Only Projection

## 1. 想法

Target 验证时每个位置通常算完整 LM-head：

\[
z_t = W_{\mathrm{lm}} h_t \in \mathbb{R}^{|V|}
\]

但 greedy 验证 draft token \(y_t\) 只需知道它是否 argmax。用 embedding cluster 上界：

\[
U_c(h) = h^\top\mu_c + |h|\,r_c
\]

若 \(\ell_y > \max_{c\not\ni y} U_c(h)\)，则**证书证明** \(y\) 是 top-1，跳过全词表 GEMM；否则回退完整投影。输出与原 target 完全一致。

## 2. 新颖性核实

| 工作 | 对象 |
|---|---|
| NanoSpec / DynaSpec / FR-Spec | 优化 **drafter** LM-head（小模型 head 占比大） |
| “When Is a Draft Accepted?” (2606.30265) | greedy 接受的 **exact KL / margin 证书理论**（已在 Qwen3 评测） |

→ 理论证书大体被 2606.30265 覆盖；剩余空间是 **target 侧跳 GEMM 的系统实现**。新颖性弱于 SD②。

## 3. 预先风险

32B target 的计算量在 transformer body；LM-head 可能只占 verify 的很小比例 → 即使证书常触发也省不了多少。

## 4. Audit C（执行）

Profile \(\rho_{\mathrm{head}} = T_{\mathrm{lm\text{-}head}} / T_{\mathrm{verify}}\)：

| 模型 | ρ_head | lm_head (ms) | total (ms) |
|---|--:|--:|--:|
| **32B target** | **1.9%** | 1.2 | 61.4 |
| 1.5B draft | 3.6% | — | — |

Kill gate：ρ_head ≥ 10% 才值得做 certificate → **1.9% ≪ 10%**。

代码：`action_study/sd_audit/audit_c_profile.py`  
原始：`outputs/sd_audit_c.json`

## 5. 重做 Audit（v2，batch 扫描 + 物理核算）

针对"large-batch serving regime 下 head 会不会变瓶颈"的复活假设，实测 ρ_head 随 batch 变化（seq=128，batch∈{1,4,16,64,128}）。

物理预期：一次 forward 处理 N=batch×seq 个 token，body ~ N·layers·(attn+mlp)、head ~ N·H·V，**两者都线性于 N** ⇒ ρ_head 应与 batch 无关。

| batch | 32B target ρ_head | 1.5B draft ρ_head |
|--:|--:|--:|
| 1 | 0.0196 | 0.0356 |
| 4 | 0.0182 | 0.0601 |
| 16 | 0.0194 | 0.0937 |
| 64 | 0.0199 | 0.1000 |
| 128 | 0.0204 | 0.0974 |

- **32B target：ρ_head 恒定 ~2%，完全与 batch 无关** —— large-batch 假设被证伪。
- 1.5B draft：head 随 batch 升到 ~10%，确实可观 —— **但 drafter 是自回归生成、没有候选 token，candidate-only / certificate 用不上**。

代码：`action_study/sd_audit/run_sd3_reframe.py`；数据：`outputs/vsignal/sd3_reframe.json`。

## 6. 判决（更正后）

**KILL —— 物理上封死（含 large-batch regime）。**

核心矛盾：**能用 candidate-only 的地方（target 验证）head 只占 2% 且 batch 无关；head 占比高的地方（drafter）又用不上 candidate-only（生成需要全 argmax）。**

唯一可能翻身的 regime：**大词表 + 中小验证器**（如 256k+ 多语词表、V/H 比极高的模型）。对 DeepSeek-R1-Distill-Qwen（V=152k, target H=5120）不成立。

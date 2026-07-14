# V3.5 — Cost–Rescue Gate（Branch vs Direct Handoff）

## 1. 问题

在 greedy draft step 被拒绝后，比较两条路径的 wall-clock：

- **Path H**：Direct Handoff —— 32B 直接生成替换步  
- **Path B(K)**：Branch@K —— 1.5B 采样 K 个候选，32B 批量验证，选最优；全拒则 fallback 到 Handoff

核心问题：Branch 是否能在**不慢于 Handoff** 的前提下提供救援？

## 2. 方法

- Dual-resident：1.5B draft + 32B-AWQ target 同卡  
- 对 rejected states 做配对计时（warmup 后）  
- 指标：`T_B(K)` vs `T_H`，`γ = max(50ms, 5%·T_H)` 容差；救援率（存在/被选/安全）

## 3. 关键结果

Smoke / microbenchmark 一致显示：

| 路径 | 相对 Handoff |
|---|---|
| Branch@4 | **≈ 1.06 × T_H** |

即 Branch **并不比 Handoff 便宜**。验证开销抵消了 draft 采样的廉价。

详细原始记录：`outputs/pilot_v3_5_smoke_findings.md`、`outputs/action_study_v35_*`。

## 4. 判决

**工程先验：FAIL（作为加速手段）**

Branch 可能仍有语义救援价值，但作为“比 Handoff 更快”的假设不成立。后续不再把 Branch 当主加速路线。

## 5. 对后续的含义

- 直接催生 V3.6：把问题收窄到“验证是否可靠”  
- 也解释了为何 ConfSpec 式“验证廉价”前提在本设置下不成立

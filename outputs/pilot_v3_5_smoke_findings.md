# V3.5 Smoke Findings — Cost–Rescue Gate

> 4-state smoke on AutoDL: `1.5B` + `32B-AWQ`, `step_max_tokens=96`.  
> **Scope-limited**：当前实现 / 当前长度 / 当前硬件下的 Branch@4；**不是**“Branch 永远没用”。

## Measured (overall mean)

| Metric | Value |
|--------|------:|
| $C_T$ | **1.033s** |
| $C_{D4}$ | 0.562s |
| $C_{V4}$ | 0.536s |
| $C_{D4}+C_{V4}$ | **1.098s** |
| $r_4^*=(C_{D4}+C_{V4})/C_T$ | **106.3%** |

即使 $r_4=100\%$，Branch@4 仍慢约 6.3%。完美 router 也无法让 Branch@4 赢过 Handoff。

$$
\boxed{\text{当前实现下 Branch@4 被 Handoff 严格支配（}r_4^*>100\%\text{）}}
$$

## What this does / does not prove

**证明了**

- 乐观先验 $C_{D4}+C_{V4}\ll C_T$ 不成立（实测 ≈1.06$C_T$）
- 现在训 Branch router **没有意义**（瓶颈是 Branch 成本，不是触发时机）
- Draft 并行 OK：$C_{D1}\approx C_{D4}$

**尚未证明**

- Branch@1 / @2 无用（$r_1^*\approx 70\%<100\%$）
- 更长 natural step 下仍支配
- prefix caching / 1-token verifier 优化后仍支配
- 所有 prefix×step bucket 都支配

## Next = V3.6（不要继续 V3.5a）

V3.5 smoke / V3.5a 分解成本 **不能**作为终局 never-branch 证据。  
正式机制实验改为：

[`docs/v3_6_one_step_cost_rescue.md`](../docs/v3_6_one_step_cost_rescue.md)

在 greedy-rejected prefix 上配对测 $T_H$ vs $T_B^{(K)}$，用 $R^{\mathrm{safe}}$ 约束质量。

# SD Audits B → A → C — 执行报告

执行顺序: 2(B) → 1(A) → 3(C)

## Audit B — 请求内 residual 稳定性（方向②）

- 决策: **FAIL**
- 记录数: 768 (6 prompts)
- pooled test delta_top1 (EMA): **-0.043**
- pooled test delta_KL: **-0.159**
- 改善 prompt 比例: 2/6

## Audit A — 两阶段验证回本（方向①）

- 决策: **FAIL**
- Both approx and exact timing use same AWQ target at different verify lengths; measures whether shortening exact verify after budget scan can beat one full pass. INT4 scan adds extra full-length forward — likely negative on memory-bound single-request.

| gamma | baseline(s) | twostage(s) | savings | r_hat | pass |
|--:|--:|--:|--:|--:|:--:|
| 4 | 0.0608 | 0.1215 | -100.0% | 1.5 | False |
| 8 | 0.0612 | 0.1219 | -99.4% | 2.5 | False |
| 16 | 0.0618 | 0.1226 | -98.5% | 2.5 | False |

## Audit C — LM-head profile（方向③）

- 决策: **KILL**
- target ρ_head (mean): **1.9%**
- draft ρ_head (mean): **3.6%**
- target verify total: 61.4 ms, lm_head: 1.2 ms

## 综合建议

方向② **暂停**：residual 时间稳定性不足或未通过 kill gate。
方向① **封存**：额外 INT4 扫描 + 短 verify 未回本（符合 memory-bound 预期）。
方向③ **封存**：target LM-head 占比 <10%，跳 head 收益不足。

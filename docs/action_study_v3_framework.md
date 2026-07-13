# Action Study V3 — SpecReason Utility Oracle

> **定位**：SpecReason 执行骨架 + ConfSpec 式 confidence routing + **conditional Branch**  
> 不是 ConfSpec tree drafting，也不是 token-level acceptance replay。

## 核心问题

当小模型当前 step 不可靠时，uncertainty 能否区分：

- **可探索恢复** → Branch（多候选，target 只打 cheap utility 分）
- **能力不足** → Handoff（target 真正生成 replacement step）

## 动作空间

| 动作 | 条件（utility oracle, τ∈{5,6,7,8}） |
|------|-------------------------------------|
| **Continue** | `u_0 ≥ τ` |
| **Branch** | `u_0 < τ` 且 `max(u_1..4) ≥ τ` |
| **Handoff** | `max(u_0..4) < τ` |

## 数据（复用 pilot v2，不重跑 4B）

- `admission_main` = 1548 prefixes
- 每 prefix：`continue` (greedy) + `branch` ×4
- QwQ-32B 对每个候选的**第一个 reasoning step** 打 0–9 分（SpecReason prompt）

## 运行

```bash
bash reasoning_branch_dataset/scripts/run_utility_scoring_v3.sh
```

产出：

- `outputs/action_study_pilot_v3/utility_scores_QwQ-32B.jsonl`
- `outputs/action_study_pilot_v3/utility_oracle_report.md`
- `outputs/action_study_pilot_v3/oracle_summary.json`

## 与 V2 区别

| | V2 (reachable / token replay) | V3 (utility oracle) |
|--|-------------------------------|---------------------|
| Target 输出 | token greedy acceptance 长度 | 单 token utility 0–9 |
| Branch 价值 | `A_best4 - A_single` | Branch-rescuable 区域是否存在 |
| 数据需求 | target-reachable prefix | 现有 1+4 candidates 即可 |

## 下一步（V3 之后）

1. 用 hidden/logits 预测 oracle label（problem_id split）
2. Branch-rescuable / Handoff-required PR-AUC
3. 延迟模型：`P(rescue)·T_target` vs `T_branch + T_batched_score`

# V3.4 — Sequential Oracle Policy Rollout Report

> GPT-5.5-guided sequential policy from problem prompt; actions modify prefix cascade.

- rollouts completed: **4**
- step records: **4**
- Branch events: **0**
- Oracle step API error rate: **100.0%**

## 18.1 Policy-level results

| Policy | N | Accuracy | Avg Continue | Avg Branch | Avg Handoff | Target steps | Proxy latency |
|--------|--:|---------:|-------------:|-----------:|------------:|-------------:|--------------:|
| SPECREASON | 2 | 0.0% | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 |
| CONDITIONAL_BRANCH | 2 | 0.0% | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 |

## 18.2 Cascade metrics

- P(Continue | Branch): **0.0%**
- P(Handoff | Branch): **0.0%**
- P(Branch | Branch): **0.0%**
- Mean L_B (Continue run after Branch): **0.00**
- Median L_B: **0.0**
- P(L_B≥1): **0.0%**
- P(L_B≥3): **0.0%**
- Mean L_H (Continue run after Handoff): **0.00**
- Mean ΔH (SpecReason − CondBranch handoffs/problem): **0.000**
- Mean ΔH (valid paired only): **0.000**
- Mean cascade C_q (ΔH / N_B): **0.000**

## 18.3 Paired SpecReason vs Conditional Branch (seed=1)

- ΔHandoff mean **0.000** (95% CI [0.000, 0.000])

## Action transition matrix

| From \ To | Continue | Branch | Handoff |
|-----------|----------|--------|---------|
| CONTINUE | 0.0% | 0.0% | 0.0% |
| BRANCH | 0.0% | 0.0% | 0.0% |
| HANDOFF | 0.0% | 0.0% | 0.0% |
| ORACLE_API_ERROR | 0.0% | 0.0% | 0.0% |

## V3.4 success criteria

- Branch events exist: **False**
- E[ΔH] > 0 (fewer handoffs with Branch): **False**

> GPT oracle is offline only — not included in deployment latency.


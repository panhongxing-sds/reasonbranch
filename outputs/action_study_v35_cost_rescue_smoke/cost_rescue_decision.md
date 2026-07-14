# V3.5 — Cost–Rescue Gate Decision

> SpecExit-style: measure signals/costs first; set operating point by threshold,
> do **not** train a Branch/Handoff action classifier until break-even requires it.

- latency summary: `/root/autodl-tmp/reasonbranch/outputs/action_study_v35_latency_smoke/latency_summary.json`
- rescue rates: `/root/autodl-tmp/reasonbranch/outputs/action_study_v35_rescue/rescue_rates.json` (source: `v3.3_gpt_step_oracle_provisional_4B_draft`)

## Primary recommendation (K=4)

**Decision: `never_branch`**

r_4=44.6% ≪ r_4^*=106.3% (margin=-61.7%) → skip fixed Branch; try smaller K or SpecReason

## Cost–rescue table

| K | $r_K$ | $r_K^*$ | margin | E[$C_{branch}$] | $C_T$ | speedup | decision |
|--:|------:|--------:|-------:|----------------:|------:|--------:|----------|
| 1 | 26.5% | 69.0% | -42.5% | 1.472s | 1.033s | 0.70× | `never_branch` |
| 2 | 36.5% | 75.7% | -39.2% | 1.438s | 1.033s | 0.72× | `never_branch` |
| 4 | 44.6% | 106.3% | -61.7% | 1.670s | 1.033s | 0.62× | `never_branch` |

## Measured costs (overall)

| Metric | Value |
|--------|------:|
| $C_T$ | 1.033s |
| $C_{D4}$ | 0.562s |
| $C_{V4}$ | 0.536s |
| $(C_{D4}+C_{V4})/C_T$ | 106.3% |

## What to do next

1. Do not pay Branch@4 by default.
2. Re-measure with K=1/2, or improve batch verify / shared-prefix KV.
3. Only then consider a selective Branch predictor.

> Note: rescue rates are **provisional** (V3.3 / different draft stack).
> Re-run Experiment B on final 1.5B+32B before locking the decision.

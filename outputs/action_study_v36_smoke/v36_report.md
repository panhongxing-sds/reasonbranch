# V3.6 — One-Step Counterfactual Cost–Rescue Report

- trials: **2**
- decision: `fixed_handoff`
- rationale: no K shows reliable positive safe Δ — Reject→Handoff; do not train Branch router

## Profitability / Δ

| K | median Δ | mean Δ (boot CI) | P(Δ>0) | P(profitable) | Safe Rescue |
|--:|---------:|-----------------:|-------:|--------------:|------------:|
| 1 | -1139ms | -1139ms [-1980ms,-297ms] | 0.0% | 0.0% | 0.0% |
| 2 | 455ms | 455ms [-272ms,1182ms] | 50.0% | 0.0% | 0.0% |
| 4 | 310ms | 310ms [-278ms,898ms] | 50.0% | 0.0% | 0.0% |

## Rescue decomposition

| K | Exist | Accepted | Safe | Selector Gap |
|--:|------:|---------:|-----:|-------------:|
| 1 | 0.0% | 50.0% | 0.0% | 0.0% |
| 2 | 0.0% | 50.0% | 0.0% | 0.0% |
| 4 | 0.0% | 50.0% | 0.0% | 0.0% |

## Latency

| K | Handoff med | Branch pipe med | Success med | Fail med |
|--:|------------:|----------------:|------------:|---------:|
| 1 | 1151ms | 2290ms | — | 2290ms |
| 2 | 1151ms | 696ms | 584ms | 809ms |
| 4 | 1151ms | 842ms | 868ms | 815ms |

## Next

- `fixed_handoff` → SpecReason first version; skip Branch router
- `fixed_branch` → sequential V3.7 with Fixed Branch@K*
- `need_router` → train $Y^{profitable}$ then V3.7

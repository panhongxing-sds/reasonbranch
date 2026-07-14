# V3.6 — One-Step Counterfactual Cost–Rescue Report

- trials: **4**
- decision: `fixed_handoff`
- rationale: no K shows reliable positive safe Δ — Reject→Handoff; do not train Branch router

## Profitability / Δ

| K | median Δ | mean Δ (boot CI) | P(Δ>0) | P(profitable) | Safe Rescue |
|--:|---------:|-----------------:|-------:|--------------:|------------:|
| 1 | -407ms | -828ms [-2209ms,-348ms] | 0.0% | 0.0% | 0.0% |
| 2 | -518ms | -245ms [-572ms,639ms] | 25.0% | 0.0% | 0.0% |
| 4 | -865ms | -601ms [-869ms,199ms] | 25.0% | 0.0% | 0.0% |

## Rescue decomposition

| K | Exist | Accepted | Safe | Selector Gap |
|--:|------:|---------:|-----:|-------------:|
| 1 | 0.0% | 25.0% | 0.0% | 0.0% |
| 2 | 0.0% | 25.0% | 0.0% | 0.0% |
| 4 | 0.0% | 25.0% | 0.0% | 0.0% |

## Latency

| K | Handoff med | Branch pipe med | Success med | Fail med |
|--:|------------:|----------------:|------------:|---------:|
| 1 | 551ms | 958ms | — | 958ms |
| 2 | 551ms | 1104ms | 1111ms | 1097ms |
| 4 | 551ms | 1417ms | 1550ms | 1411ms |

## Next

- `fixed_handoff` → SpecReason first version; skip Branch router
- `fixed_branch` → sequential V3.7 with Fixed Branch@K*
- `need_router` → train $Y^{profitable}$ then V3.7

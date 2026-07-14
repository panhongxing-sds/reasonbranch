# V3.6 — One-Step Counterfactual Cost–Rescue Report

- trials: **4**
- decision: `fixed_handoff`
- rationale: no K shows reliable positive safe Δ — Reject→Handoff; do not train Branch router

## Profitability / Δ

| K | median Δ | mean Δ (boot CI) | P(Δ>0) | P(profitable) | Safe Rescue |
|--:|---------:|-----------------:|-------:|--------------:|------------:|
| 1 | -415ms | -848ms [-2265ms,-356ms] | 0.0% | 0.0% | 0.0% |
| 2 | -518ms | -242ms [-562ms,654ms] | 25.0% | 0.0% | 0.0% |
| 4 | -861ms | -597ms [-867ms,208ms] | 25.0% | 0.0% | 0.0% |

## Rescue decomposition

| K | Exist | Accepted | Safe | Selector Gap |
|--:|------:|---------:|-----:|-------------:|
| 1 | 0.0% | 25.0% | 0.0% | 0.0% |
| 2 | 0.0% | 25.0% | 0.0% | 0.0% |
| 4 | 0.0% | 25.0% | 0.0% | 0.0% |

## Latency

| K | Handoff med | Branch pipe med | Success med | Fail med |
|--:|------------:|----------------:|------------:|---------:|
| 1 | 558ms | 972ms | — | 972ms |
| 2 | 558ms | 1105ms | 1102ms | 1108ms |
| 4 | 558ms | 1419ms | 1548ms | 1414ms |

## Next

- `fixed_handoff` → SpecReason first version; skip Branch router
- `fixed_branch` → sequential V3.7 with Fixed Branch@K*
- `need_router` → train $Y^{profitable}$ then V3.7

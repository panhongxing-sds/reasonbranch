# Oracle Action-Matching Study Report

## Main Table (Oracle Feasibility)
| prefix_state   |   n_prefixes |   continue_at_1 |   branch_pass_at_4 |   rollback_pass_at_4 |   continue_tokens |   branch_tokens |   rollback_tokens |   continue_latency |   branch_latency |   rollback_latency |
|:---------------|-------------:|----------------:|-------------------:|---------------------:|------------------:|----------------:|------------------:|-------------------:|-----------------:|-------------------:|
| Unclear        |            2 |               0 |                  1 |                    1 |               388 |            1495 |              1682 |            12.2867 |          49.3107 |             52.929 |

## Paired Δ(Branch − Rollback) with Bootstrap 95% CI
- **all** (n=1): Δ=0.000 [0.000, 0.000], branch wins 0.0%

## Go / No-Go
- **overall_go**: False
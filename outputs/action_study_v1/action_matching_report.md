# Oracle Action-Matching Study Report

> Phase 1 measures **oracle recoverability** R(a,s), not E2E latency.
> Draft-model seconds are debug-only; target-side latency belongs in Phase 3.

## Main Table (Oracle Recoverability)
| prefix_state   |   n_prefixes |   continue_oracle_recoverable |   branch_oracle_recoverable |   rollback_oracle_recoverable |   continue_draft_tokens |   branch_draft_tokens |   rollback_draft_tokens |   rollback_discarded_prefix_tokens |
|:---------------|-------------:|------------------------------:|----------------------------:|------------------------------:|------------------------:|----------------------:|------------------------:|-----------------------------------:|
| Unclear        |           18 |                             1 |                           1 |                             1 |                 352.333 |               1441.89 |                 1559.54 |                                  0 |

## Paired Δ(Branch − Rollback) Oracle Recoverability
- **all** (n=13): Δ_oracle=0.000 [0.000, 0.000], branch oracle wins 0.0%

## Go / No-Go (oracle mechanism only)
- **overall_go**: False
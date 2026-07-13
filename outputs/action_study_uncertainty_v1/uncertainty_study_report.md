# Phase-1 Uncertainty Study Report

> Small model only. Labels v2: behavior state + recovery profile.
> **Do not train Hidden Probe until Decision-sensitive and Corrupted-stuck exist.**

## Behavior State Table (substantive prefixes only)
| state                 |   n_prefixes |   n_continue_evaluated |   n_branch_evaluated |   n_continue_errors |   n_branch_errors |   continue_accuracy |   branch_pass_at_4 |   branch_accuracy_at_4 |   branch_gain |
|:----------------------|-------------:|-----------------------:|---------------------:|--------------------:|------------------:|--------------------:|-------------------:|-----------------------:|--------------:|
| Stable                |           27 |                     26 |                   26 |                   1 |                 1 |                1    |                  1 |                 1      |          0    |
| Decision-sensitive    |            1 |                      0 |                    0 |                   1 |                 1 |              nan    |                nan |               nan      |        nan    |
| Corrupted-recoverable |            8 |                      8 |                    8 |                   0 |                 0 |                0.75 |                  1 |                 0.8125 |          0.25 |
| Corrupted-stuck       |           14 |                      4 |                    4 |                  10 |                10 |                0    |                  0 |                 0      |          0    |

## Legacy v1 Table (for comparison)
| state                           |   n_prefixes |   n_continue_evaluated |   n_branch_evaluated |   n_continue_errors |   n_branch_errors |   continue_accuracy |   branch_pass_at_4 |   branch_accuracy_at_4 |   branch_gain |
|:--------------------------------|-------------:|-----------------------:|---------------------:|--------------------:|------------------:|--------------------:|-------------------:|-----------------------:|--------------:|
| Valid + Low Diversity (legacy)  |           24 |                     23 |                   23 |                   1 |                 1 |            1        |           1        |               1        |      0        |
| Valid + High Diversity (legacy) |            4 |                      3 |                    3 |                   1 |                 1 |            1        |           1        |               1        |      0        |
| Invalid Prefix (legacy)         |           22 |                     12 |                   12 |                  10 |                10 |            0.5      |           0.666667 |               0.541667 |      0.166667 |
| Unclear                         |           10 |                      9 |                    9 |                   1 |                 1 |            0.888889 |           1        |               0.916667 |      0.111111 |

## Labeling Notes
- Strategy diversity uses API strategy-level clustering (cluster_v2); heuristic is conservative.
- `NO_COMMITMENT` prefixes excluded from behavior table.
- `branch_accuracy_at_4` = mean(correct_branches / evaluated_branches); distinct from pass@4.
- Report `n_*_evaluated` and `n_*_errors` — do not compare metrics with mismatched denominators.

## Pilot Readiness
- stable_continue_approx_branch: True
- decision_sensitive_exists: False
- corrupted_recoverable_exists: True
- corrupted_stuck_exists: True
- **ready_for_scale_up**: False
- **ready_for_probe**: False

## Next Step
Fix strategy clustering → add branch correct count → distinguish recoverable vs stuck → re-run 50–100 harder problems (split by problem_id). Hidden Probe only after pilot gates pass.
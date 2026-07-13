# Phase-1 Uncertainty Study Report

> Small model only. Measures state discoverability + Continue vs Branch oracle feasibility.
> Not E2E latency. Rollback is not a primary action.

## Main Table
| state   |   n_prefixes |   continue_accuracy |   branch_pass_at_4 |   branch_gain |
|:--------|-------------:|--------------------:|-------------------:|--------------:|
| Unclear |           18 |                   1 |                  1 |             0 |

## Expected Patterns
- **Stable**: continue_accuracy ≈ branch_pass@4 (Branch unnecessary)
- **Future-diverse**: branch_pass@4 > continue_accuracy (Branch oracle gain)
- **Current-unreliable**: continue_accuracy low (path contamination)

## Hypothesis Checks
- **overall_go**: False

## Next Step: Hidden Probe
Train probe on hidden/logits to predict Stable / Future-diverse / Current-unreliable.
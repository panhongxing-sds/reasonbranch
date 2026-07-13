# Reasoning Branch/Rollback Analysis Report

## 1. Branch rate by prefix type
| prefix_type   |   n |   branch_rate |   mean_utility |
|:--------------|----:|--------------:|---------------:|
| ENTROPY_SPIKE |   1 |             0 |           -0.1 |
| PARAGRAPH_END |   4 |             0 |           -0.1 |
| RANDOM        |   2 |             0 |           -0.1 |

## 2. Entropy vs branch utility
- corr_entropy_utility: -0.0000
- auc_entropy: nan

## 3. Hidden → branch utility probe
- draft_layer16_last: AUROC=nan, PR-AUC=nan
- draft_layer28_last: AUROC=nan, PR-AUC=nan
- draft_layer32_last: AUROC=nan, PR-AUC=nan
- target_layer16_last: AUROC=nan, PR-AUC=nan
- target_layer28_last: AUROC=nan, PR-AUC=nan
- target_layer32_last: AUROC=nan, PR-AUC=nan

## 4. Hidden → rollback probe
- hidden_draft_layer32: AUROC=nan, PR-AUC=nan

## 5. API prefix enrichment
- api_enrichment: 0.0000
- random_branch_rate: 0.0000
- api_top_branch_rate: 0.0000
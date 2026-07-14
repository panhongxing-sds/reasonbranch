# Layerwise Verification Trajectory — Kill-Gate Report

- Decision: **FAIL**
- N cycles: 240
- M4 vs M3 RMSE gain: **0.3%**
- M4 vs M3 R² gain: **+0.006**

## Predictor comparison (OOF next acceptance length)

| model | RMSE | MAE | R² |
|---|--:|--:|--:|
| M1 | 2.681 | 2.201 | -0.003 |
| M2 | 2.565 | 2.117 | 0.082 |
| M3 | 2.549 | 2.092 | 0.093 |
| M4 | 2.542 | 2.085 | 0.099 |

## Matched-margin early vs late resolve

```
{
  "n_early": 70,
  "n_late": 69,
  "n_matched_pairs": 70,
  "matched_early_A_next": 1.9714285714285715,
  "matched_late_A_next": 3.4,
  "matched_delta": -1.4285714285714284,
  "early_mean_m_T": 3.811049107142857,
  "late_mean_m_T": 3.8060267857142858
}
```

Kill gate failed: resolution depth / flip count do not significantly improve next-cycle acceptance prediction over final-logit features (M3). Stop.

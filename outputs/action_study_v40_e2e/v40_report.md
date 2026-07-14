# V4.0 End-to-End: Draft-Confidence Selective Speculative Reasoning

- Paired problems (attempted by all policies): 15

## Per-policy (on paired set, speedup vs target_only)

| policy | accuracy | wall(s) | speedup | steps | handoffs | accept_rate | verify(s) |
|---|--:|--:|--:|--:|--:|--:|--:|
| target_only | 1.000 | 18.392 | 1.000x | 22.000 | 22.000 | 0.000 | 0.000 |
| draft_only | 0.875 | 7.061 | 2.605x | 25.333 | 0.000 | 1.000 | 0.000 |
| selfconf | 1.000 | 26.215 | 0.702x | 24.067 | 18.200 | 0.262 | 3.221 |
| target_verify | 0.750 | 12.043 | 1.527x | 24.933 | 2.333 | 0.925 | 4.623 |

## By dataset (accuracy / wall)

| dataset | target_only | draft_only | selfconf | target_verify |
|---|--:|--:|--:|--:|
| aime | n/a / 30.897s | n/a / 9.094s | n/a / 47.138s | n/a / 19.988s |
| gsm8k | 1.000 / 15.265s | 0.875 / 6.553s | 1.000 / 20.985s | 0.750 / 10.056s |

## Interpretation

- selfconf (OURS) reaches 1.000 accuracy at 0.702x speedup vs target_only (1.000), with near-zero verification overhead (3.221s).
- target_verify pays a 32B verification pass (4.623s) yet reaches 0.750 accuracy at 1.527x -- its accept decisions are unreliable (V3.6 verification gap).

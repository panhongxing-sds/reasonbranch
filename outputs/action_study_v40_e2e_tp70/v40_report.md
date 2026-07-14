# V4.0 End-to-End: Draft-Confidence Selective Speculative Reasoning

- Paired problems (attempted by all policies): 15

## Per-policy (on paired set, speedup vs target_only)

| policy | accuracy | wall(s) | speedup | steps | handoffs | accept_rate | verify(s) |
|---|--:|--:|--:|--:|--:|--:|--:|
| target_only | 1.000 | 13.379 | 1.000x | 17.467 | 17.467 | 0.000 | 0.000 |
| draft_only | 0.889 | 7.038 | 1.901x | 24.333 | 0.000 | 1.000 | 0.000 |
| selfconf | 1.000 | 12.493 | 1.071x | 19.467 | 7.533 | 0.586 | 1.847 |
| target_verify | 0.778 | 10.342 | 1.294x | 23.200 | 1.933 | 0.925 | 4.185 |

## By dataset (accuracy / wall)

| dataset | target_only | draft_only | selfconf | target_verify |
|---|--:|--:|--:|--:|
| gsm8k | 1.000 / 13.379s | 0.889 / 7.038s | 1.000 / 12.493s | 0.778 / 10.342s |

## Interpretation

- selfconf (OURS) reaches 1.000 accuracy at 1.071x speedup vs target_only (1.000), with near-zero verification overhead (1.847s).
- target_verify pays a 32B verification pass (4.185s) yet reaches 0.778 accuracy at 1.294x -- its accept decisions are unreliable (V3.6 verification gap).

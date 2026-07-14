# V3.5 Experiment A — Latency Microbenchmark

> Goal: measure whether Fixed Branch@K is systemically cheaper than Handoff,
> before training any Branch/Handoff classifier.

- draft: `/root/autodl-tmp/specreason/models/DeepSeek-R1-Distill-Qwen-1.5B`
- target: `/root/autodl-tmp/specreason/models/DeepSeek-R1-Distill-Qwen-32B-AWQ`
- states measured: **4**
- step_max_tokens: 96

## Break-even formula

$$r_K^* = (C_{DK} + C_{VK}) / C_T$$

If empirical $r_K \gg r_K^*$, use **always Branch@K** (no router).

## Overall means

| Metric | Value |
|--------|------:|
| $C_T$ | 1.033s |
| $C_{D1}$ | 0.477s |
| $C_{D2}$ | 0.452s |
| $C_{D4}$ | 0.562s |
| $C_{V1}$ | 0.235s |
| $C_{V2}$ | 0.330s |
| $C_{V4}$ | 0.536s |
| $r_1^*$ | 69.0% |
| $r_2^*$ | 75.7% |
| $r_4^*$ | 106.3% |

## By prefix × step bucket

| Prefix | Step | N | $C_T$ | $C_{D4}$ | $C_{V4}$ | $r_4^*$ |
|-------:|-----:|--:|------:|---------:|---------:|-------:|
| medium | medium | 2 | 1.328s | 0.691s | 0.541s | 92.8% |
| medium | short | 2 | 0.738s | 0.433s | 0.531s | 130.5% |

## Next

1. Run Experiment B to estimate final-stack $r_1,r_2,r_4$ (or use provisional V3.3).
2. Compare $r_K$ vs $r_K^*$ via `run_v3_5_cost_rescue.py`.
3. Only train a Branch predictor if near break-even.

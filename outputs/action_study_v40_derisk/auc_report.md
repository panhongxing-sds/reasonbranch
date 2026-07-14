# V4.0 Phase-0 De-risk: Draft Self-Confidence Discriminability

- Decision: **GREEN**
- N=179 (pos=50, neg=129, base_rate=0.279)
- 32B verifier AUC (baseline, expected ~0.5): **0.667**
- Best single draft signal: `self_eval_logit` AUC=0.844 power=0.844 AP=0.662
- Fused draft-only (GroupKFold OOF): AUC=0.833 AP=0.606
- Fused draft+verifier (OOF): AUC=0.830 AP=0.609
- Best single signal op: coverage@P90=0.000, maxP@cov10=0.789
- Best discriminative power: **0.844**
- Coverage at precision>=0.90 (best of fused/single): **0.006**

## Head-to-head by dataset (verifier vs draft self-eval)

| dataset | N | pos | base_rate | verifier AUC | self_eval AUC |
|---|--:|--:|--:|--:|--:|
| aime | 24 | 2 | 0.083 | 0.950 | 0.455 |
| aime_calib | 48 | 3 | 0.062 | 0.356 | 0.659 |
| gsm8k | 107 | 45 | 0.421 | 0.599 | 0.828 |

## Per-signal discriminability (sorted by power)

| signal | orient | AUC | power | AP |
|---|--:|--:|--:|--:|
| self_eval_logit | +1 | 0.844 | 0.844 | 0.662 |
| mean_margin | +1 | 0.741 | 0.741 | 0.513 |
| repetition_rate | -1 | 0.272 | 0.728 | 0.191 |
| mean_entropy | -1 | 0.700 | 0.700 | 0.412 |
| max_entropy | -1 | 0.673 | 0.673 | 0.433 |
| mean_logprob | +1 | 0.669 | 0.669 | 0.360 |
| perplexity | -1 | 0.669 | 0.669 | 0.360 |
| verifier_score | +1 | 0.667 | 0.667 | 0.400 |
| n_tokens | +1 | 0.607 | 0.607 | 0.320 |
| min_margin | +1 | 0.564 | 0.564 | 0.292 |
| last_logprob | +1 | 0.452 | 0.548 | 0.340 |
| min_logprob | +1 | 0.467 | 0.533 | 0.290 |
| char_len | +1 | 0.505 | 0.505 | 0.262 |

## Interpretation

Draft self-confidence separates oracle-acceptable from unacceptable steps where the 32B verifier cannot. Proceed to Phase 1 (conformal abstention gate + E2E).

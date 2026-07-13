# V3.3 — GPT Local Next-Step Action Oracle Report

> GPT-5.5 offline oracle: independently assess 1 greedy + 4 branch **next steps** at fixed prefix.
> Dual-pass stability on greedy acceptability and any-branch-exists (E).

- prefixes labeled: **1548** / 1548
- prompt version: `gpt_step_oracle_v2`

## 15.1 Data quality

| Metric | Count | Rate |
|--------|------:|-----:|
| Total prefixes | 1548 | 100% |
| Greedy complete | 1548 | 100.0% |
| All branches complete | 1548 | 100.0% |
| Action-stable dual pass | 1464 | 94.6% |
| Oracle unstable | 84 | 5.4% |
| Partial branch evidence | 0 | 0.0% |
| Data error (greedy incomplete) | 0 | 0.0% |
| **Final eligible prefixes** | **1395** | **90.1%** |

## 15.2 Oracle action distribution (stable, eligible)

| Action | Count | Rate | 95% CI (cluster) |
|--------|------:|-----:|-----------------:|
| CONTINUE | 1229 | 88.1% | [86.0%, 90.2%] |
| BRANCH | 74 | 5.3% | [4.1%, 6.6%] |
| HANDOFF | 92 | 6.6% | [5.1%, 8.1%] |
| PREFIX_INVALID | 0 | 0.0% | — |
| PARTIAL_BRANCH_EVIDENCE | 0 | 0.0% | — |

## 15.3 Acceptable branches (greedy rejected only)

| Acceptable branches | Count | Rate |
|--------------------:|------:|-----:|
| 0/4 | 92 | 55.4% |
| 1/4 | 18 | 10.8% |
| 2/4 | 25 | 15.1% |
| 3/4 | 17 | 10.2% |
| 4/4 | 14 | 8.4% |

## 15.4 Branch width (Rescue@K | G=0)

| Width K | Rescue rate |
|--------:|------------:|
| 1 | 26.5% |
| 2 | 36.5% |
| 4 | 44.6% |

## 16.1 QwQ weak Branch vs GPT V3.3

- Precision P(GPT Branch | QwQ weak): **14.5%**
- Recall P(QwQ weak | GPT Branch): **28.4%**
- QwQ weak (eligible): 145
- GPT Branch (eligible): 74

## 16.2 V2 behavior_state × V3.3 action

| V2 state | N | Continue | Branch | Handoff | %Branch |
|----------|--:|---------:|-------:|--------:|--------:|
| Corrupted-recoverable | 94 | 71 | 9 | 14 | 9.6% |
| Corrupted-stuck | 34 | 20 | 4 | 10 | 11.8% |
| Decision-sensitive | 24 | 21 | 2 | 1 | 8.3% |
| Stable | 1243 | 1117 | 59 | 67 | 4.7% |

## 16.3 Final answer correctness (auxiliary)

- **BRANCH**: n=74, P(final correct)=79.7%
- **CONTINUE**: n=1229, P(final correct)=87.2%
- **HANDOFF**: n=92, P(final correct)=72.8%

## 17 Research questions

- **RQ1** Branch-rescuable exists: **True** (Branch count=74)
- **RQ2** Branch rate (eligible): **5.30%**
- **RQ3** Rescue@4 vs Rescue@1: **44.6%** vs **26.5%**
- **RQ4** QwQ precision/recall: **14.5%** / **28.4%**
- **RQ5** Probe: blocked until stability ≥85% and N_Branch≥50

## 18 Probe unlock

- Dual-pass stability: **94.6%** (need ≥85%)
- Stable Branch N: **74** (need ≥50)

## Definition

- **CONTINUE**: G=1
- **BRANCH**: G=0 ∧ ∃k B_k=1
- **HANDOFF**: G=0 ∧ all complete branches rejected
- V3.3 does NOT claim online latency or cascade effects (see V3.4).


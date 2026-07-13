# V3.2 — GPT-5.5 Pairwise Oracle Report

> Offline structured judge (NOT latency path). Dual-pass A/B swap required for stable labels.

- total reviewed: **312**
- QwQ weak Branch pool: **162**
- dual-pass stable: **290** (92.9%)

## Key metrics

- P(GPT true Branch | QwQ weak): **8.6%**
- Stable true Branch count: **14**
- Est. rate over 1548: **0.90%**

## Stable canonical verdicts

| verdict | count |
|---------|------:|
| BOTH_ACCEPTABLE_EQUIVALENT | 172 |
| BOTH_OK_BRANCH_PREFERRED | 54 |
| BRANCH_ONLY_ACCEPTABLE | 19 |
| BOTH_OK_GREEDY_PREFERRED | 17 |
| GREEDY_ONLY_ACCEPTABLE | 14 |
| BOTH_UNACCEPTABLE | 14 |

## GPT action mapping (stable only)

- `GREEDY_ONLY_ACCEPTABLE` / `BOTH_OK_GREEDY_PREFERRED` → **Continue**
- `BRANCH_ONLY_ACCEPTABLE` / `BOTH_OK_BRANCH_PREFERRED` → **Branch**
- `BOTH_ACCEPTABLE_EQUIVALENT` → **Continue** (no Branch cost)
- `BOTH_UNACCEPTABLE` → **Handoff**

## Probe unlock (v3.2)

- Require dual-pass stable rate ≥ 85%
- Require stable true Branch N ≥ 50 for 3-way probe; else Continue vs Non-Continue

## Audit cases: QwQ weak but GPT equivalent

### Case 1: `deepscaler_01245_p00_paragraph_end`
- QwQ: u₀=1 u_best=9
- GPT reason: Both correctly clarify that only numerical digits of the time count, not the AM/PM letters.

### Case 2: `deepscaler_01360_p01_paragraph_end`
- QwQ: u₀=5 u_best=7
- GPT reason: Both correctly derive that pink carnations are 0.4T using the given pink flower information.

### Case 3: `deepscaler_01192_p01_paragraph_end`
- QwQ: u₀=5 u_best=7
- GPT reason: Both correctly set up choosing three nonconsecutive periods using equivalent inequality conditions.


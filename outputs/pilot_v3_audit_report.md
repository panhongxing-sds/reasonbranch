# V3 Oracle Audit Report

> **Do not train probe until this audit passes.** Validates utility scorer + step extraction.

- scored prefixes: **1548**
- audit sample (stratified): **200** → `action_study_pilot_v3/audit_sample.jsonl`

## Verdict (preliminary)

- Weak Branch @ τ=7: **10.47%** (u₀<τ, max≥τ)
- After complete-step filter: **10.61%** (1527/1548 prefixes eligible)
- **Strong Branch** (u₀≤4, max≥7, Δ≥3): **28** (1.8%)
- DATA_ERROR (incomplete candidates): **21** prefixes — exclude from Handoff
- Score–length correlation: **-0.067** (|r|>0.15 → length bias risk)
- Inadmissible prefixes (any incomplete step): **21** (1.4%)

**Recommendation**: probe training blocked until shuffle-rescore stability + manual review of audit cases.

## 1. Score distribution P(u)

| u | count | % |
|---|------:|--:|
| 0 | 9 | 0.1% |
| 1 | 28 | 0.4% |
| 2 | 21 | 0.3% |
| 3 | 377 | 4.9% |
| 5 | 1332 | 17.2% |
| 6 | 7 | 0.1% |
| 7 | 3896 | 50.3% |
| 8 | 1891 | 24.4% |
| 9 | 179 | 2.3% |

## 2. Step quality (per candidate)

| quality | count | % |
|---------|------:|--:|
| COMPLETE_SUBSTANTIVE_STEP | 7656 | 98.9% |
| TRUNCATED_STEP | 84 | 1.1% |

## 3. Oracle table — raw vs filtered (complete steps only)

| τ | raw Continue | raw Branch | raw Handoff | filt Continue | filt Branch | filt Handoff |
|---|-------------:|-----------:|------------:|--------------:|------------:|-------------:|
| 5 | 93.93% | 4.2% | 1.87% | 93.91% | 4.26% | 1.83% |
| 6 | 76.74% | 10.47% | 12.79% | 76.49% | 10.61% | 12.9% |
| 7 | 76.61% | 10.47% | 12.92% | 76.36% | 10.61% | 13.03% |
| 8 | 27.13% | 9.75% | 63.11% | 26.78% | 9.82% | 63.39% |

## 4. Length bias

- corr(u, step_chars): **-0.067**
- mean chars | u≥7: **256.91099564197117**
- mean chars | u<7: **272.3579481397971**

## 5. V2 behavior_state × V3 oracle (raw τ=7)

| behavior_state | n | Continue | Branch | Handoff | %Branch |
|----------------|--:|---------:|-------:|--------:|--------:|
| Corrupted-recoverable | 128 | 78 | 19 | 31 | 14.84% |
| Corrupted-stuck | 74 | 47 | 10 | 16 | 13.51% |
| Decision-sensitive | 28 | 22 | 3 | 3 | 10.71% |
| Stable | 1318 | 1026 | 130 | 150 | 9.86% |

## 6. V2 × V3 (filtered, complete steps only)

| behavior_state | n | Continue | Branch | Handoff | %Branch |
|----------------|--:|---------:|-------:|--------:|--------:|
| Corrupted-recoverable | 126 | 76 | 19 | 31 | 15.08% |
| Corrupted-stuck | 73 | 47 | 10 | 16 | 13.7% |
| Decision-sensitive | 27 | 22 | 3 | 2 | 11.11% |
| Stable | 1301 | 1021 | 130 | 150 | 9.99% |

- P(V3 Branch | V2 Decision-sensitive) = **10.71%**
- P(V3 Branch | V2 Stable) = **9.86%**

## 7. Next checks (not yet run)

1. **Shuffle rescore stability**: re-score audit sample with QwQ, check score agreement.
2. **Independent judge**: binary ACCEPT/REJECT on Branch-rescuable cases.
3. **Re-score with hardened prompt** (already in specreason_scorer for future runs).


## Audit Flag Cases

### Audit Case 1: `length_sensitive_branch_rescue` (`deepscaler_01245_p00_paragraph_end`)

- greedy u=1 (126 chars) vs branch u=9 (367 chars)

**Greedy step:**
```text
The digits displayed are the numerical digits of $H$ and $M$. The letters "AM" or "PM" do not contribute to the sum of digits.
```

**Best branch step:**
```text
The question asks for the sum of the digits in the display. We interpret "digits" as the numerical characters used to represent the hours and minutes. The letters "A", "M", "P" in "AM" and "PM" are typically considered letters, not digits, unless specified otherwise. In standard math problems of this type, we sum the numerical digits ($0-9$) displayed for the time.
```

### Audit Case 2: `length_sensitive_branch_rescue` (`deepscaler_01045_p00_paragraph_end`)

- greedy u=1 (233 chars) vs branch u=8 (300 chars)

**Greedy step:**
```text
### Step 1: Identify the Pigeons and Pigeonholes
- **Pigeons**: The people in the room. There are $52$ people.
- **Pigeonholes**: The possible months in which a birthday can fall. In the standard calendar year, there are $12$ months.
```

**Best branch step:**
```text
### Step 1: Identify the number of categories (pigeonholes) and the number of items (pigeons)
*   **Categories (Pigeonholes)**: There are 12 months in a year. Let's denote them as the possible months for birthdays. So, $k = 12$.
*   **Items (Pigeons)**: There are 52 people in the room. So, $N = 52$.
```

### Audit Case 3: `handoff_with_incomplete_steps` (`deepscaler_01401_p02_paragraph_end`)

- step qualities: `['TRUNCATED_STEP', 'TRUNCATED_STEP', 'COMPLETE_SUBSTANTIVE_STEP', 'TRUNCATED_STEP', 'TRUNCATED_STEP']`

**continue** u=7 quality=TRUNCATED_STEP
```text
Both methods yield the same result.
```

**branch_0** u=7 quality=TRUNCATED_STEP
```text
Both methods yield the same result.
```

**branch_1** u=5 quality=COMPLETE_SUBSTANTIVE_STEP
```text
\boxed{25}
```

**branch_2** u=7 quality=TRUNCATED_STEP
```text
Both methods yield the same result.
```

**branch_3** u=7 quality=TRUNCATED_STEP
```text
Both methods yield the same result.
```


# V3 Oracle Audit — Phase 2

> Shuffle-rescore stability + pairwise judge on raw weak Branch cases.

## Oracle tiers (reminder)

| tier | definition | n (pilot) |
|------|------------|----------:|
| **Weak Branch** | u₀<τ, max≥τ | ~162 |
| **Strong Branch** | u₀≤4, max≥7, Δ≥3 | 28 |
| **DATA_ERROR** | incomplete candidates | 21 |

## Unlock gates (probe training)

| gate | threshold | current | pass |
|------|-----------|---------|------|
| Accept/Reject agreement (shuffle) | ≥90% | 82.7% | ✗ |
| Oracle action agreement (shuffle) | ≥85% | 67.0% | ✗ |
| Branch precision (pairwise) | ≥70% | 0.0% | ✗ |
| Cleaned Branch count | ≥50 | 0 | ✗ |

## 1. Shuffle-rescore stability

- prefixes rescored: **200**
- exact score agreement: **65.9%**
- within-1 agreement: **76.3%**
- accept agreement (u≥7): **82.7%**
- oracle action agreement: **67.0%**

## 2. Pairwise judge (raw weak Branch cases)

- reviewed: **162** / ~162
- BRANCH_BETTER: **0** (0.0%)
- EQUIVALENT: **41**
- GREEDY_BETTER: **0**
- BOTH_REJECT: **0**

- **Precision** (QwQ weak ∧ judge=BRANCH_BETTER): **0.0%**
- **Estimated cleaned Branch rate**: **0.0%** of all prefixes

### Interpretation

- `EQUIVALENT` on Case 1/2-style pairs → absolute utility oracle noise, not real Branch.
- True Branch = QwQ weak **and** pairwise `BRANCH_BETTER`.
- If cleaned N<50 → train **Continue vs Non-Continue** first, or rare Branch detector.

## Verdict

**Probe training: BLOCKED**

Next if blocked: hardened-prompt rescore on weak Branch + sample Handoff/Continue; then re-run pairwise on disagreements.


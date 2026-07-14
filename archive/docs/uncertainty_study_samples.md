# Phase-1 Uncertainty Study 完整样本

> 生成时间：2026-07-11 18:38:37  
> 数据：`outputs/action_study_uncertainty_v1/`  
> 框架：Stable / Future-diverse / Current-unreliable × Continue + Branch

---

## 数据集概览

- 题目数：15  
- prefix 数：60  
- 状态分布：{'Stable': 27, 'Future-diverse': 13, 'Current-unreliable': 20}

## 主表（analyze 输出）

# Phase-1 Uncertainty Study Report

> Small model only. Measures state discoverability + Continue vs Branch oracle feasibility.
> Not E2E latency. Rollback is not a primary action.

## Main Table
| state                  |   n_prefixes |   continue_accuracy |   branch_pass_at_4 |   branch_gain |
|:-----------------------|-------------:|--------------------:|-------------------:|--------------:|
| Valid + Low Diversity  |           27 |                0.96 |           1        |      0.04     |
| Valid + High Diversity |           13 |                1    |           1        |      0        |
| Invalid Prefix         |           20 |                0.5  |           0.666667 |      0.166667 |

## Expected Patterns
- **Stable**: continue_accuracy ≈ branch_pass@4 (Branch unnecessary)
- **Future-diverse**: branch_pass@4 > continue_accuracy (Branch oracle gain)
- **Current-unreliable**: continue_accuracy low (path contamination)

## Hypothesis Checks
- stable_continue_approx_branch: True
- future_diverse_branch_gain_positive: False
- current_unreliable_continue_low: False
- **overall_go**: False

## Next Step: Hidden Probe
Train probe on hidden/logits to predict Stable / Future-diverse / Current-unreliable.

---

## Sample — Branch Gain=1 (Current-unreliable, 65.7%)

**prefix_id**: `math500_0010_p02_paragraph_end`  
**state_bucket**: `Current-unreliable`  
**future_system_action**: `Handoff`

### 题目

What is the least positive integer multiple of 30 that can be written with only the digits 0 and 2?

**Gold**: `2220`

### Prefix 状态

| 字段 | 值 |
|------|-----|
| reasoning_progress | 0.657 |
| prefix_validity | INVALID |
| diversity_label | LOW_DIVERSITY |
| num_clusters | 2 |
| diversity_entropy | 0.562 |

### API Validity
```json

{
  "prefix_id": "math500_0010_p02_paragraph_end",
  "prefix_validity": "INVALID",
  "prefix_status": "INVALID",
  "first_error_step": 3,
  "error_type": "arithmetic",
  "confidence": 0.99,
  "explanation": "The prefix incorrectly claims that 2022 ends in 0, but it ends in 2, so it is not divisible by 10 or 30.",
  "api_model": "",
  "api_prompt_version": "validity_v2",
  "timestamp": "2026-07-11T18:37:11.931240+00:00"
}

```

### API Cluster
```json

{
  "prefix_id": "math500_0010_p02_paragraph_end",
  "clusters": [
    "could there be a smaller number?",
    "could there be a smaller number?",
    "is there a smaller number?",
    "could there be a smaller number?"
  ],
  "num_semantic_clusters": 2,
  "cluster_source": "heuristic"
}

```

### Next Steps ×4

**sample 0**:
```
Could there be a smaller number?
- Length 1: Impossible (needs three 2s).
- Length 2: Impossible.
- Length 3: Needs three 2s, but only 3 digits available. Arrangement would be 222. Sum=6 (divisible by 3), but doesn't end in 0. Not divisible by 10.
- Length 4: We found 2022.
Any number with fewer than three 2s will not be divisible by 3 (sum would be 0, 2, or 4, none
```

**sample 1**:
```
Could there be a smaller number?
- Length 1: Only 0 or 2. Neither is divisible by 30.
- Length 2: Must end in 0. Digits {0, 0} -> 00 (not positive). {0, 2} -> 20 (sum=2, no).
- Length 3: Must end in 0. Digits from {0, 2}. Sum must be multiple of 3.
  - Possible sums: 0 (000), 2 (200), 4 (220
```

**sample 2**:
```
Is there a smaller number?
- Length 1: Only 0 (not positive) or 2 (not div by 10).
- Length 2: Ends in 0. Digits {2, 0}. Sum = 2 (not div by 3).
- Length 3: Ends in 0. Digits must sum to a multiple of 3.
  - Two 2s and one 0: Sum = 4 (No).
  - Three 0s: Sum = 0 (Number is 000, not positive).
  - One
```

**sample 3**:
```
Could there be a smaller number?
- Length 1: Impossible (cannot end in 0 and have sum divisible by 3 with only 0s or 2s).
- Length 2: Possible candidates ending in 0: 20. Sum = 2. Not divisible by 3.
- Length 3: Candidates ending in 0 using {0, 2}:
  - 200: Sum = 2. No.
  - 020: Not a valid 3-digit number (starts with 0).
  - 000: Not
```

### Oracle Outcomes（Continue + Branch）

| operation | oracle_recoverable | draft_generated_tokens |
|-----------|-------------------|------------------------|
| continue | 0 | 493 |
| branch | 1 | 2744 |

**Branch Gain** = 1


---

## Sample — Future-diverse (8.9%)

**prefix_id**: `math500_0001_p00_paragraph_end`  
**state_bucket**: `Future-diverse`  
**future_system_action**: `Branch`

### 题目

Define
\[p = \sum_{k = 1}^\infty \frac{1}{k^2} \quad \text{and} \quad q = \sum_{k = 1}^\infty \frac{1}{k^3}.\]Find a way to write
\[\sum_{j = 1}^\infty \sum_{k = 1}^\infty \frac{1}{(j + k)^3}\]in terms of $p$ and $q.$

**Gold**: `p - q`

### Prefix 状态

| 字段 | 值 |
|------|-----|
| reasoning_progress | 0.089 |
| prefix_validity | VALID |
| diversity_label | HIGH_DIVERSITY |
| num_clusters | 3 |
| diversity_entropy | 1.040 |

### API Validity
```json

{
  "prefix_id": "math500_0001_p00_paragraph_end",
  "prefix_validity": "VALID",
  "prefix_status": "VALID",
  "first_error_step": null,
  "error_type": "none",
  "confidence": 1.0,
  "explanation": "The prefix only states the goal and introduces a forthcoming method, with no mathematical claim or error yet.",
  "api_model": "",
  "api_prompt_version": "validity_v2",
  "timestamp": "2026-07-11T18:37:00.746141+00:00"
}

```

### API Cluster
```json

{
  "prefix_id": "math500_0001_p00_paragraph_end",
  "clusters": [
    "### step 1: change of variables",
    "### step 1: simplify the double sum by changing ",
    "algebra",
    "### step 1: change of variables"
  ],
  "num_semantic_clusters": 3,
  "cluster_source": "heuristic"
}

```

### Next Steps ×4

**sample 0**:
```
### Step 1: Change of Variables
Let $n = j + k$. Since $j \ge 1$ and $k \ge 1$, the smallest possible value for $n$ is $2$.
For a fixed integer $n \ge 2$, the possible pairs $(j, k)$ are $(1, n-1), (2, n-2), \dots, (n-1, 1)$.
The number of such pairs is $n-1$.
We can rewrite the double sum by grouping terms with the same value of $n$:
\[
\sum_{
```

**sample 1**:
```
### Step 1: Simplify the Double Sum by Changing Variables
Let $S = \sum_{j = 1}^\infty \sum_{k = 1}^\infty \frac{1}{(j + k)^3}$.
Notice that the term depends only on the sum $n = j + k$. Since $j \ge 1$ and $k \ge 1$, the minimum value for $n$ is $1+1=2$.
For a fixed integer $n \ge 2$, the possible pairs $(j, k)$ such that $j+k=n$
```

**sample 2**:
```
### Step 1: Simplify the Double Sum using Index Substitution
Let $S = \sum_{j = 1}^\infty \sum_{k = 1}^\infty \frac{1}{(j + k)^3}$.
The term depends only on the sum $n = j + k$.
Since $j \ge 1$ and $k \ge 1$, the possible values for $n$ range from $1+1=2$ to $\infty$.
For a fixed integer $n \ge 2$, the number of pairs $(j, k)$
```

**sample 3**:
```
### Step 1: Change of Variables
Let $n = j + k$. Since $j \ge 1$ and $k \ge 1$, the minimum value for $n$ is $1+1=2$.
For a fixed integer $n \ge 2$, the possible pairs $(j, k)$ such that $j+k=n$ are $(1, n-1), (2, n-2), \dots, (n-1, 1)$.
The number of such pairs is $n-1$.
We can rewrite the double sum by grouping terms with the same value
```

### Oracle Outcomes（Continue + Branch）

| operation | oracle_recoverable | draft_generated_tokens |
|-----------|-------------------|------------------------|
| continue | 1 | 865 |
| branch | 1 | 3109 |

**Branch Gain** = 0


---

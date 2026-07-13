# Pilot v2 — Uncertainty Study Report

> 4B draft only. Labels: behavior state + recovery profile (correctness auxiliary).
> **Admission re-derived from final traces/actions (ignores stale pipeline admission_pass).**

## 相关报告（同 `outputs/` 目录）

| 版本 | 报告文件 | 状态 |
|------|----------|------|
| v2（本报告） | `pilot_v2_report.md` | 数据采集 + 正确率 |
| v3 utility oracle | `pilot_v3_report.md` | 已完成 |
| reachable-state | `reachable_state_report.md` | 已完成 |

### 本报告范围

- **包含**：4B draft Continue/Branch、终答正确率、behavior state、完整 case。
- **不含**：主指标 G_B（见 `reachable_state_report.md`）；utility oracle（见 `pilot_v3_report.md`）。
- **原始数据**：`action_study_pilot_v2/*.jsonl`（本报告不重复列出）。

## Admission Summary
| metric | count |
|--------|------:|
| total_prefixes | 1857 |
| complete_trace_prefixes | 1857 |
| substantive_prefixes | 1589 |
| continue_ok | 1847 |
| branch_4of4_ok | 1799 |
| branch_partial | 55 |
| admission_main | 1548 |
| admission_partial | 1575 |
| excluded_missing_context | 0 |
| excluded_truncated_trace | 0 |
| excluded_action_incomplete | 41 |
| excluded_non_substantive | 268 |
| traces_ok | 520 |
| traces_truncated | 219 |
| excluded_problems_missing_figure | 61 |
| unique_complete_trace_problems | 517 |

**Branch valid count histogram**

| valid_branches | prefixes |
|----------------|---------:|
| 0/4 | 3 |
| 1/4 | 4 |
| 2/4 | 11 |
| 3/4 | 40 |
| 4/4 | 1799 |

## Behavior State Table (`admission_main` only)
| state                 |   n_prefixes |   n_continue_evaluated |   n_branch_evaluated |   n_continue_errors |   n_branch_errors |   continue_accuracy |   branch_pass_at_4 |   branch_accuracy_at_4 |   branch_gain |
|:----------------------|-------------:|-----------------------:|---------------------:|--------------------:|------------------:|--------------------:|-------------------:|-----------------------:|--------------:|
| Stable                |         1318 |                   1318 |                 1318 |                   0 |                 0 |           0.888467  |           0.902883 |              0.887709  |     0.0144158 |
| Decision-sensitive    |           28 |                     28 |                   28 |                   0 |                 0 |           0.0357143 |           0.142857 |              0.0803571 |     0.107143  |
| Corrupted-recoverable |          128 |                    128 |                  128 |                   0 |                 0 |           0.921875  |           1        |              0.939453  |     0.078125  |
| Corrupted-stuck       |           74 |                     74 |                   74 |                   0 |                 0 |           0.027027  |           0        |              0         |    -0.027027  |

## Behavior State Table (`admission_partial`, branch≥3/4)
| state                 |   n_prefixes |   n_continue_evaluated |   n_branch_evaluated |   n_continue_errors |   n_branch_errors |   continue_accuracy |   branch_pass_at_4 |   branch_accuracy_at_4 |   branch_gain |
|:----------------------|-------------:|-----------------------:|---------------------:|--------------------:|------------------:|--------------------:|-------------------:|-----------------------:|--------------:|
| Stable                |         1335 |                   1335 |                 1335 |                   0 |                 0 |           0.886891  |           0.902622 |               0.886642 |     0.0157303 |
| Decision-sensitive    |           31 |                     31 |                   31 |                   0 |                 0 |           0.0645161 |           0.193548 |               0.115591 |     0.129032  |
| Corrupted-recoverable |          133 |                    133 |                  133 |                   0 |                 0 |           0.909774  |           1        |               0.929198 |     0.0902256 |
| Corrupted-stuck       |           76 |                     76 |                   76 |                   0 |                 0 |           0.0263158 |           0        |               0        |    -0.0263158 |

## Legacy v1 Table (for comparison)
| state                           |   n_prefixes |   n_continue_evaluated |   n_branch_evaluated |   n_continue_errors |   n_branch_errors |   continue_accuracy |   branch_pass_at_4 |   branch_accuracy_at_4 |   branch_gain |
|:--------------------------------|-------------:|-----------------------:|---------------------:|--------------------:|------------------:|--------------------:|-------------------:|-----------------------:|--------------:|
| Valid + Low Diversity (legacy)  |         1149 |                   1148 |                 1148 |                   1 |                 1 |            0.864983 |           0.885017 |               0.866725 |     0.0200348 |
| Valid + High Diversity (legacy) |          223 |                    223 |                  223 |                   0 |                 0 |            0.869955 |           0.887892 |               0.871076 |     0.0179372 |
| Invalid Prefix (legacy)         |          218 |                    214 |                  217 |                   4 |                 1 |            0.579439 |           0.62212  |               0.575269 |     0.0426806 |
| Unclear                         |          267 |                    262 |                  266 |                   5 |                 1 |            0.755725 |           0.834586 |               0.771303 |     0.0788613 |

## Labeling Notes
- Strategy diversity uses API strategy-level clustering (cluster_v2); heuristic is conservative.
- Main table uses `admission_main`: complete trace + substantive + continue OK + branch 4/4.
- Partial table uses `admission_partial`: branch≥3/4 (not standard Pass@4).
- **`branch_pass_at_4` / correctness `branch_gain` are auxiliary only** — not Branch utility oracle.
- Primary Branch metric: `target_acceptance_gain = max_j A_j - A_single` (requires target replay).
- Behavior states are exploratory; see `docs/target_acceptance_framework.md`.
- Truncated base traces produce no prefixes in this pilot.
- `branch_accuracy_at_4` = mean(correct_branches / evaluated_branches); distinct from pass@4.
- Report `n_*_evaluated` and `n_*_errors` — do not compare metrics with mismatched denominators.

## Pilot Readiness (`admission_main`, exploratory)
- stable_continue_approx_branch: True
- decision_sensitive_correctness_gain_positive: True
- decision_sensitive_exists: True
- **ready_for_scale_up**: False
- **ready_for_target_replay**: False
- **ready_for_probe**: False (requires target acceptance labels)

## Current Status & Next Step

1. **v2 数据采集**：完成（本报告）。
2. **draft-only target replay**：已尝试，4B draft prefix 不在 QwQ reachable 状态，无有效 G_B。
3. **reachable-state**：完成 → 同目录 `reachable_state_report.md`。
4. **v3 utility oracle**：完成 → 同目录 `pilot_v3_report.md`。
5. **下一步**：先做 V3 oracle audit → `pilot_v3_audit_report.md`；**审计通过前不训练 probe**。

_Correctness `branch_gain` 仅作辅助；主 Branch 信号看 reachable G_B 或 v3 utility oracle。_

## Illustrative Cases (correctness auxiliary)

### Case 1: correctness_branch_only_rescue (`deepscaler_01182_p01_paragraph_end`)

**Problem** (deepscaler_01182):
- `behavior_state` = **Stable** (draft-side label)
- correctness: continue **False**, branch **4/4**

> In the sequence
\[..., a, b, c, d, 0, 1, 1, 2, 3, 5, 8,...\]
each term is the sum of the two terms to its left. Find $a$.

- gold: `-3` | continue answer: `-1`

**Prefix tail:**

```text
Solve the following math problem efficiently and clearly. Please reason step by step, separate logical reasoning steps with two newline characters (\n\n), and put your final answer within \boxed{}.
Problem: In the sequence
\[..., a, b, c, d, 0, 1, 1, 2, 3, 5, 8,...\]
each term is the sum of the two terms to its left. Find $a$.

<think>

</think>

To find the value of $a$, let's analyze the given sequence and the rule provided.

**1. Understand the Sequence Rule**
The problem states that "each...
```

**Continue continuation (first step block):**

```text
So, the sequence from the term $0$ onwards is:
$x_k, x_{k+1}, x_{k+2}, \dots$ where $x_k = 0, x_{k+1} = 1, \dots$

**3. Work Backwards to Find $a$ and $b$**
Since each term is the sum of the two preceding terms, we can rearrange the formula to find a preceding term:
$x_{n-2} = x_n - x_{n-1}$.

Let's apply this rule step-by-step backwards from the known values:

- We know the term before $0$ is $d$.
  $0 = d + (\text{term before } d)$. This doesn't help immediately unless we know the term before $d$.
  Let's look at the pair $(0, 1)$.
  $1 = 0 + (\text{term before } 0)$.
  So, the term before $0$ is $1 - 0 = 1$.
  Therefore, $d = 1$.

- Now we have the pair $(1, 1)$ (which are the terms $d...
```

**Rescuing branch answer `-3`:**

```text
So, the segment $d, 0, 1, 1, 2, 3, 5, 8$ corresponds to $F_k, F_1, F_2, F_3, F_4, F_5, F_6, F_7$ for some index $k$.
Specifically:
$d = F_k$
$0 = F_1$
$1 = F_2$
$1 = F_3$
$2 = F_4$
$3 = F_5$
$5 = F_6$
$8 = F_7$

**3. Work Backwards to Find $d, c, b, a$**
Using the reverse of the addition rule ($x_{n-2} = x_n - x_{n-1}$):
- Since $d + 0 = 1$, we have $d = 1 - 0 = 1$.
- Since $0 + 1 = 1$, this checks out ($F_3 = F_2 + F_1 \Rightarrow 1 = 1 + 0$).
- Since $1 + 1 = 2$, this checks out ($F_4 = F_3 + F_2 \Rightarrow 2 = 1 + 1$).

Now we continue backwards to find $c, b, a$:
- To find $c$: We know $c + d = 0$. Since $d = 1$, then $c + 1 = 0 \Rightarrow c = -1$.
- To find $b$: We know $b + c = d$...
```

### Case 2: correctness_branch_only_rescue (`deepscaler_01501_p01_paragraph_end`)

**Problem** (deepscaler_01501):
- `behavior_state` = **Stable** (draft-side label)
- correctness: continue **False**, branch **4/4**

> $R$ varies directly as $S$ and inversely as $T$. When $R = \frac{4}{3}$ and $T = \frac{9}{14}$, $S = \frac{3}{7}$. Find $S$ when $R = \sqrt{48}$ and $T = \sqrt{75}$.

- gold: `30` | continue answer: `90`

**Prefix tail:**

```text
Solve the following math problem efficiently and clearly. Please reason step by step, separate logical reasoning steps with two newline characters (\n\n), and put your final answer within \boxed{}.
Problem: $R$ varies directly as $S$ and inversely as $T$. When $R = \frac{4}{3}$ and $T = \frac{9}{14}$, $S = \frac{3}{7}$. Find $S$ when $R = \sqrt{48}$ and $T = \sqrt{75}$.

<think>

</think>

To solve this problem, we will follow these steps:
1.  Establish the mathematical relationship between $...
```

**Continue continuation (first step block):**

```text
First, simplify the fraction $\frac{S}{T}$:
$$\frac{\frac{3}{7}}{\frac{9}{14}} = \frac{3}{7} \cdot \frac{14}{9} = \frac{3 \cdot 14}{7 \cdot 9} = \frac{3 \cdot 2}{1 \cdot 3} = 2$$

Now substitute this back into the equation for $R$:
$$\frac{4}{3} = k \cdot 2$$

Solve for $k$:
$$k = \frac{4}{3} \div 2 = \frac{4}{3} \cdot \frac{1}{2} = \frac{2}{3}$$

So, the specific equation relating the variables is:
$$R = \frac{2}{3} \cdot \frac{S}{T}$$
Or equivalently:
$$S = \frac{R \cdot T}{\frac{2}{3}} = \frac{3}{2} R T$$

**Step 3: Calculate the new value of $S$**
We are given the new conditions:
$$R = \sqrt{48}, \quad T = \sqrt{75}$$

First, simplify the square roots:
$$\sqrt{48} = \sqrt{16 \cdot 3}...
```

**Rescuing branch answer `30`:**

```text
First, simplify the fraction $\frac{S}{T}$:
$$\frac{S}{T} = \frac{3}{7} \div \frac{9}{14} = \frac{3}{7} \cdot \frac{14}{9}$$
$$= \frac{3 \cdot 14}{7 \cdot 9} = \frac{3 \cdot 2}{1 \cdot 9} = \frac{6}{9} = \frac{2}{3}$$

Now substitute this back into the main equation:
$$\frac{4}{3} = k \cdot \frac{2}{3}$$

Solve for $k$:
$$k = \frac{4}{3} \div \frac{2}{3} = \frac{4}{3} \cdot \frac{3}{2} = \frac{4}{2} = 2$$

So, the constant of proportionality is $k = 2$. The specific equation for this problem is:
$$R = 2 \cdot \frac{S}{T} \quad \text{or} \quad S = \frac{R \cdot T}{2}$$

**Step 3: Calculate the new value of $S$**
We are given the new conditions:
$$R = \sqrt{48}, \quad T = \sqrt{75}$$

First...
```

### Case 3: correctness_continue_sufficient (`deepscaler_01043_p02_paragraph_end`)

**Problem** (deepscaler_01043):
- `behavior_state` = **Corrupted-stuck** (draft-side label)
- correctness: continue **True**, branch **0/4**

> Marvin had a birthday on Tuesday, May 27 in the leap year $2008$. In what year will his birthday next fall on a Saturday?

- gold: `2017` | continue answer: `2017`

**Prefix tail:**

```text
Solve the following math problem efficiently and clearly. Please reason step by step, separate logical reasoning steps with two newline characters (\n\n), and put your final answer within \boxed{}.
Problem: Marvin had a birthday on Tuesday, May 27 in the leap year $2008$. In what year will his birthday next fall on a Saturday?

<think>

</think>

To find the year when Marvin's birthday next falls on a Saturday, we need to determine the day of the week for May 27 in subsequent years.

**Step 1...
```

**Continue continuation (first step block):**

```text
Note: Even though 2012 is a leap year, the extra day (Feb 29, 2012) occurs before May 27. Thus, the shift from May 27, 2011 to May 27, 2012 is still just 1 day. The shift of 2 days only happens when calculating the day for May 27 of a leap year based on the *previous* leap year's May 27 (e.g., May 27, 2008 to May 27, 2009 is 1 day; May 27, 2009 to May 27, 2010 is 1 day; ... May 27, 2011 to May 27, 2012 is 1 day because Feb 29, 2012 is in the future relative to May 27, 2011? No, wait.
Let's re-evaluate the leap year logic carefully.
The shift depends on the number of days in the interval.
Interval 1: May 27, 2008 to May 27, 2009. Days = 365. Shift = +1. (Result: Wed)
Interval 2: May 27, 20...
```

### Case 4: correctness_continue_sufficient (`deepscaler_01173_p00_paragraph_end`)

**Problem** (deepscaler_01173):
- `behavior_state` = **Stable** (draft-side label)
- correctness: continue **True**, branch **0/4**

> The table below displays the grade distribution of the $30$ students in a mathematics class on the last two tests.  For example, exactly one student received a 'D' on Test 1 and a 'C' on Test 2.  What percent of the students received the same grade on both tests?

- gold: `40\%` | continue answer: `40\%`

**Prefix tail:**

```text
Solve the following math problem efficiently and clearly. Please reason step by step, separate logical reasoning steps with two newline characters (\n\n), and put your final answer within \boxed{}.
Problem: The table below displays the grade distribution of the $30$ students in a mathematics class on the last two tests.  For example, exactly one student received a 'D' on Test 1 and a 'C' on Test 2.  What percent of the students received the same grade on both tests?

<think>

</think>

To fin...
```

**Continue continuation (first step block):**

```text
**Step 2: Identify the number of students who received the same grade on both tests.**
We look for the entries in the table where the grade on Test 1 matches the grade on Test 2. These are the cells along the main diagonal of the table:
- Grade A on both tests: 2 students
- Grade B on both tests: 3 students
- Grade C on both tests: 4 students
- Grade D on both tests: 1 student
- Grade F on both tests: 2 students

**Step 3: Calculate the total number of students with matching grades.**
Sum = $2 + 3 + 4 + 1 + 2$
Sum = $12$

**Step 4: Calculate the percentage.**
The percentage is calculated as:
$$ \text{Percentage} = \left( \frac{\text{Number of students with same grades}}{\text{Total number...
```

### Case 5: correctness_both_fail (`deepscaler_01007_p00_paragraph_end`)

**Problem** (deepscaler_01007):
- `behavior_state` = **Stable** (draft-side label)
- correctness: continue **False**, branch **0/4**

> Let $f(n) = \frac{x_1 + x_2 + \cdots + x_n}{n}$, where $n$ is a positive integer. If $x_k = (-1)^k, k = 1, 2, \cdots, n$, the set of possible values of $f(n)$ is:

- gold: `$\{0, -\frac{1}{n}\}$` | continue answer: `\left\{ 0, -\frac{1}{n} \mid n \text{ is an odd positive integer} \right\}`

**Prefix tail:**

```text
Solve the following math problem efficiently and clearly. Please reason step by step, separate logical reasoning steps with two newline characters (\n\n), and put your final answer within \boxed{}.
Problem: Let $f(n) = \frac{x_1 + x_2 + \cdots + x_n}{n}$, where $n$ is a positive integer. If $x_k = (-1)^k, k = 1, 2, \cdots, n$, the set of possible values of $f(n)$ is:

<think>

</think>

To find the set of possible values for $f(n)$, let's analyze the definition and the properties of the seque...
```

**Continue continuation (first step block):**

```text
**Step 2: Calculate the sum $S_n = \sum_{k=1}^{n} x_k$ for different cases of $n$.**

*Case 1: $n$ is even.*
Let $n = 2m$ for some positive integer $m$.
The sum consists of $m$ pairs of $(-1, 1)$.
$$S_{2m} = (-1 + 1) + (-1 + 1) + \cdots + (-1 + 1) = 0 + 0 + \cdots + 0 = 0$$
Therefore,
$$f(2m) = \frac{0}{2m} = 0$$

*Case 2: $n$ is odd.*
Let $n = 2m + 1$ for some non-negative integer $m$ (since $n \ge 1$, $m \ge 0$).
The sum consists of $m$ pairs of $(-1, 1)$ followed by the last term $x_{2m+1} = -1$.
$$S_{2m+1} = [(-1 + 1) + \cdots + (-1 + 1)] + (-1) = 0 + (-1) = -1$$
Therefore,
$$f(2m+1) = \frac{-1}{2m+1}$$

**Step 3: Determine the set of possible values.**
The set of values depends on wh...
```

### Case 6: correctness_both_fail (`deepscaler_01007_p01_paragraph_end`)

**Problem** (deepscaler_01007):
- `behavior_state` = **Stable** (draft-side label)
- correctness: continue **False**, branch **0/4**

> Let $f(n) = \frac{x_1 + x_2 + \cdots + x_n}{n}$, where $n$ is a positive integer. If $x_k = (-1)^k, k = 1, 2, \cdots, n$, the set of possible values of $f(n)$ is:

- gold: `$\{0, -\frac{1}{n}\}$` | continue answer: `\left\{ 0 \right\} \cup \left\{ -\frac{1}{n} \mid n \text{ is an odd positive integer} \right\}`

**Prefix tail:**

```text
Solve the following math problem efficiently and clearly. Please reason step by step, separate logical reasoning steps with two newline characters (\n\n), and put your final answer within \boxed{}.
Problem: Let $f(n) = \frac{x_1 + x_2 + \cdots + x_n}{n}$, where $n$ is a positive integer. If $x_k = (-1)^k, k = 1, 2, \cdots, n$, the set of possible values of $f(n)$ is:

<think>

</think>

To find the set of possible values for $f(n)$, let's analyze the definition and the properties of the seque...
```

**Continue continuation (first step block):**

```text
**Step 3: Determine the set of all possible values.**
The set of possible values corresponds to the set of values $f(n)$ can take as $n$ ranges over all positive integers $\{1, 2, 3, \dots\}$.
- If $n$ is even, $f(n) = 0$.
- If $n$ is odd, $f(n) = -\frac{1}{n}$.

Since $n$ can be any odd positive integer ($1, 3, 5, \dots$), the values are:
- For $n=1$: $f(1) = -1/1 = -1$
- For $n=3$: $f(3) = -1/3$
- For $n=5$: $f(5) = -1/5$
- ...
- For any odd $n$, the value is $-\frac{1}{n}$.

Combining both cases, the set of possible values is $\{0\} \cup \left\{ -\frac{1}{n} \mid n \in \mathbb{Z}^+, n \text{ is odd} \right\}$.

This can be written compactly as:
$$ \left\{ 0 \right\} \cup \left\{ -\frac...
```

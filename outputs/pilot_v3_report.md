# Pilot v3 — Utility Oracle Report

> SpecReason-style QwQ utility scoring on v2 admission_main prefixes (1 greedy + 4 branch).

- v2 data: `action_study_pilot_v2/`
- scores: `action_study_pilot_v3/utility_scores_QwQ-32B.jsonl`
- target: `/mnt/afs/L202500372/specreason/models/QwQ-32B`
- scored prefixes: **1548**
- score method: `greedy`

## Oracle table by τ

| τ | Continue-sufficient | Branch-rescuable | Handoff-required |
|---|--------------------:|-----------------:|-----------------:|
| 5 | 1454 (93.93%) | 65 (4.2%) | 29 (1.87%) |
| 6 | 1188 (76.74%) | 162 (10.47%) | 198 (12.79%) |
| 7 | 1186 (76.61%) | 162 (10.47%) | 200 (12.92%) |
| 8 | 420 (27.13%) | 151 (9.75%) | 977 (63.11%) |

## Interpretation

- **Continue-sufficient**: `u_0 ≥ τ` — greedy draft step acceptable.
- **Branch-rescuable**: `u_0 < τ` but `max(u_1..4) ≥ τ` — selective Branch may avoid target generation.
- **Handoff-required**: all candidates below τ — target should take over.

This is SpecReason-style utility scoring, **not** token-level acceptance replay.

> **Probe training blocked** until `pilot_v3_audit_report.md` passes length-bias + step-quality checks.

## Illustrative Cases (τ=7)

### Case 1: continue_sufficient (`deepscaler_01005_p00_paragraph_end`)

**Problem** (deepscaler_01005):

> Two is $10 \%$ of $x$ and $20 \%$ of $y$. What is $x - y$?

- scores: greedy=9, branches=[9, 9, 9, 9], τ=7

**Prefix tail:**

```text
Solve the following math problem efficiently and clearly. Please reason step by step, separate logical reasoning steps with two newline characters (\n\n), and put your final answer within \boxed{}.
Problem: Two is $10 \%$ of $x$ and $20 \%$ of $y$. What is $x - y$?

<think>

</think>

To find the value of $x - y$, we will translate the given statements into algebraic equations and solve for the difference.

**Step 1: Translate the first statement into an equation.**
The problem states: "Two i...
```

**continue** — utility **9**

```text
**Step 2: Translate the second statement into an equation.**
The problem states: "Two is $20 \%$ of $y$."
Mathematically, this can be written as:
$$2 = 20\% \cdot y$$
Convert the percentage to a decimal:
$$2 = 0.20y$$
Solve for $y$:
$$y = \frac{2}{0.20}$$
$$y = 10$$
```

**branch_0** — utility **9**

```text
**Step 2: Translate the second statement into an equation.**
The problem states: "Two is $20 \%$ of $y$."
Mathematically, this can be written as:
$$2 = 20\% \cdot y$$
Convert the percentage to a decimal:
$$2 = 0.20y$$
Solve for $y$:
$$y = \frac{2}{0.20}$$
$$y = 10$$
```

**branch_1** — utility **9**

```text
**Step 2: Translate the second statement into an equation.**
The problem states: "Two is $20 \%$ of $y$."
Mathematically, this can be written as:
$$2 = 20\% \cdot y$$
Convert the percentage to a decimal:
$$2 = 0.20y$$
Solve for $y$:
$$y = \frac{2}{0.20}$$
$$y = 10$$
```

**branch_2** — utility **9**

```text
**Step 2: Translate the second statement into an equation.**
The problem states: "Two is $20 \%$ of $y$."
Mathematically, this can be written as:
$$2 = 20\% \cdot y$$
Convert the percentage to a decimal:
$$2 = 0.20y$$
Solve for $y$:
$$y = \frac{2}{0.20}$$
$$y = 10$$
```

**branch_3** — utility **9**

```text
**Step 2: Translate the second statement into an equation.**
The problem states: "Two is $20 \%$ of $y$."
Mathematically, this can be written as:
$$2 = 20\% \cdot y$$
Convert the percentage to a decimal:
$$2 = 0.20y$$
Solve for $y$:
$$y = \frac{2}{0.20}$$
$$y = 10$$
```

### Case 2: continue_sufficient (`deepscaler_01020_p00_paragraph_end`)

**Problem** (deepscaler_01020):

> If $y=x^2+px+q$, then if the least possible value of $y$ is zero $q$ is equal to:

- scores: greedy=9, branches=[9, 7, 9, 9], τ=7

**Prefix tail:**

```text
Solve the following math problem efficiently and clearly. Please reason step by step, separate logical reasoning steps with two newline characters (\n\n), and put your final answer within \boxed{}.
Problem: If $y=x^2+px+q$, then if the least possible value of $y$ is zero $q$ is equal to:

<think>

</think>

To find the value of $q$, let's analyze the given quadratic equation and the condition provided.

**Step 1: Analyze the quadratic function**
The given function is $y = x^2 + px + q$.
This...
```

**continue** — utility **9**

```text
Since the coefficient of $x^2$ ($a=1$) is positive, the parabola opens upwards. This means the function has a minimum value at its vertex.
```

**branch_0** — utility **9**

```text
Since the coefficient of $x^2$ ($a=1$) is positive, the parabola opens upwards. Therefore, the function has a minimum value (the vertex).
```

**branch_1** — utility **7**

```text
Since the coefficient of $x^2$ ($a=1$) is positive, the parabola opens upwards. Therefore, the function has a minimum value (a vertex), and this minimum value is less than or equal to the value of the function for any real $x$.
```

**branch_2** — utility **9**

```text
Since the coefficient of $x^2$ is positive ($a = 1 > 0$), the parabola opens upwards. Therefore, the function has a minimum value (the vertex).
```

**branch_3** — utility **9**

```text
Since the coefficient of $x^2$ ($a=1$) is positive, the parabola opens upwards. This means the function has a minimum value (the vertex) rather than a maximum.
```

### Case 3: weak_branch_rescuable (`deepscaler_01245_p00_paragraph_end`)

**Problem** (deepscaler_01245):

> A digital watch displays hours and minutes with AM and PM. What is the largest possible sum of the digits in the display?

- scores: greedy=1, branches=[7, 9, 1, 5], τ=7

**Prefix tail:**

```text
Solve the following math problem efficiently and clearly. Please reason step by step, separate logical reasoning steps with two newline characters (\n\n), and put your final answer within \boxed{}.
Problem: A digital watch displays hours and minutes with AM and PM. What is the largest possible sum of the digits in the display?

<think>

</think>

To find the largest possible sum of the digits in the display of a digital watch, we need to analyze the format of the time and the range of values...
```

**continue** — utility **1**

```text
The digits displayed are the numerical digits of $H$ and $M$. The letters "AM" or "PM" do not contribute to the sum of digits.
```

**branch_0** — utility **7**

```text
The question asks for the sum of the **digits** in the display. This typically refers to the numerical digits ($0-9$) shown for the hours and minutes. The "AM" or "PM" indicator consists of letters, not digits, so they do not contribute to the sum of digits. We need to maximize the sum of the digits of the hour and the minute.
```

**branch_1** — utility **9**

```text
The question asks for the sum of the digits in the display. We interpret "digits" as the numerical characters used to represent the hours and minutes. The letters "A", "M", "P" in "AM" and "PM" are typically considered letters, not digits, unless specified otherwise. In standard math problems of this type, we sum the numerical digits ($0-9$) displayed for the time.
```

**branch_2** — utility **1**

```text
The question asks for the sum of the **digits** in the display. This means we only count the numerical characters (0-9) and ignore the letters "A", "M", "P".
```

**branch_3** — utility **5**

```text
The problem asks for the sum of the **digits**. The letters "AM" or "PM" are not digits, so they do not contribute to the numerical sum. We only sum the digits of the hour and the minute.
```

### Case 4: weak_branch_rescuable (`deepscaler_01045_p00_paragraph_end`)

**Problem** (deepscaler_01045):

> There are $52$ people in a room. what is the largest value of $n$ such that the statement "At least $n$ people in this room have birthdays falling in the same month" is always true?

- scores: greedy=1, branches=[5, 7, 8, 8], τ=7

**Prefix tail:**

```text
Solve the following math problem efficiently and clearly. Please reason step by step, separate logical reasoning steps with two newline characters (\n\n), and put your final answer within \boxed{}.
Problem: There are $52$ people in a room. what is the largest value of $n$ such that the statement "At least $n$ people in this room have birthdays falling in the same month" is always true?

<think>

</think>

To find the largest value of $n$ such that "At least $n$ people in this room have birthd...
```

**continue** — utility **1**

```text
### Step 1: Identify the Pigeons and Pigeonholes
- **Pigeons**: The people in the room. There are $52$ people.
- **Pigeonholes**: The possible months in which a birthday can fall. In the standard calendar year, there are $12$ months.
```

**branch_0** — utility **5**

```text
### Step 1: Define the Pigeons and Pigeonholes
- **Pigeons**: The people in the room. There are $52$ people.
- **Pigeonholes**: The possible months in which a birthday can fall. Since we are dealing with calendar months, there are $12$ months in a year.
```

**branch_1** — utility **7**

```text
### Step 1: Identify the "Pigeonholes" and "Pigeons"
*   **Pigeonholes**: These represent the possible categories for the birthdays. Since there are 12 months in a year, there are $k = 12$ pigeonholes.
*   **Pigeons**: These represent the people in the room. There are $N = 52$ pigeons.
```

**branch_2** — utility **8**

```text
### Step 1: Identify the number of categories (pigeonholes) and the number of items (pigeons)
*   **Categories (Pigeonholes)**: There are 12 months in a year. Let's denote them as the possible months for birthdays. So, $k = 12$.
*   **Items (Pigeons)**: There are 52 people in the room. So, $N = 52$.
```

**branch_3** — utility **8**

```text
### Step 1: Identify the categories (pigeonholes) and the items (pigeons)
*   **Categories (Pigeonholes):** There are 12 months in a year. Let's denote them as $M_1, M_2, \dots, M_{12}$.
*   **Items (Pigeons):** There are 52 people in the room. Let's denote them as $P_1, P_2, \dots, P_{52}$.
```

### Case 5: handoff_required (`deepscaler_01150_p00_paragraph_end`)

**Problem** (deepscaler_01150):

> For $x$ real, the inequality $1 \le |x-2| \le 7$ is equivalent to

- scores: greedy=0, branches=[0, 0, 0, 0], τ=7

**Prefix tail:**

```text
Solve the following math problem efficiently and clearly. Please reason step by step, separate logical reasoning steps with two newline characters (\n\n), and put your final answer within \boxed{}.
Problem: For $x$ real, the inequality $1 \le |x-2| \le 7$ is equivalent to the union of two intervals. For $x$ real, the inequality $1 \le |x-2| \le 7$ is equivalent to the union of two intervals. For $x$ real, the inequality $1 \atop 1 \le |x-2| \le 7$ is equivalent to the union of two intervals....
```

**continue** — utility **0**

```text
<think>
```

**branch_0** — utility **0**

```text
<think>
```

**branch_1** — utility **0**

```text
<think>
```

**branch_2** — utility **0**

```text
<think>
```

**branch_3** — utility **0**

```text
<think>
```

### Case 6: handoff_required (`deepscaler_01529_p00_paragraph_end`)

**Problem** (deepscaler_01529):

> Handy Aaron helped a neighbor $1 \frac{1}{4}$ hours on Monday, $50$ minutes on Tuesday, from 8:20 to 10:45 on Wednesday morning, and a half-hour on Friday. He is paid $\textdollar 3$ per hour. How much did he earn for the week?

- scores: greedy=1, branches=[1, 1, 1, 1], τ=7

**Prefix tail:**

```text
Solve the following math problem efficiently and clearly. Please reason step by step, separate logical reasoning steps with two newline characters (\n\n), and put your final answer within \boxed{}.
Problem: Handy Aaron helped a neighbor $1 \frac{1}{4}$ hours on Monday, $50$ minutes on Tuesday, from 8:20 to 10:45 on Wednesday morning, and a half-hour on Friday. He is paid $\textdollar 3$ per hour. How much did he earn for the week?

<think>

</think>

To find out how much Handy Aaron earned fo...
```

**continue** — utility **1**

```text
**Step 1: Convert all time durations into hours.**
```

**branch_0** — utility **1**

```text
**Step 1: Convert all work durations into hours.**
```

**branch_1** — utility **1**

```text
**Step 1: Convert all time durations into hours.**
```

**branch_2** — utility **1**

```text
**Step 1: Convert all work durations into hours.**
```

**branch_3** — utility **1**

```text
**Step 1: Convert all work durations into hours.**
```


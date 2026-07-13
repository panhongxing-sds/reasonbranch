# Reachable-State Report (Scheme A)

> QwQ greedy trace → checkpoint → 4B draft (γ tokens) → QwQ verify acceptance length.

- data: `reachable_state_pilot/`
- target: `/mnt/afs/L202500372/specreason/models/QwQ-32B`
- draft: `/mnt/afs/L202500372/models/Qwen3.5-4B`
- gamma: 32
- n_checkpoints_evaluated: 288

## Summary

| metric | value |
|--------|-------|
| mean A_single | 6.688 |
| mean A_best4 | 8.073 |
| mean G_B | 1.385 |
| P(G_B > 0) | 0.316 |
| median G_B | 0.000 |

## By checkpoint position

| checkpoint_tokens | mean G_B | n |
|-------------------|----------|---|
| 32 | 1.531 | 96 |
| 96 | 1.240 | 96 |
| 192 | 1.385 | 96 |

## Illustrative Cases

### Case 1: branch_large_gain (`deepscaler_01750@ck192`)

**Problem** (deepscaler_01750):

> Each valve $A$, $B$, and $C$, when open, releases water into a tank at its own constant rate. With all three valves open, the tank fills in 1 hour, with only valves $A$ and $C$ open it takes 1.5 hours, and with only valves $B$ and $C$ open it takes 2 hours. The number of hours required with only valves $A$ and $B$ open is

- checkpoint: token 192 | γ=32
- A_single=2 | A_best4=31 | **G_B=29**
- branch accept lengths: `[31, 2, 2, 2]` (best branch index=0)

**Prefix tail (target-reachable context):**

```text
Solve the following math problem efficiently and clearly. Please reason step by step, separate logical reasoning steps with two newline characters (\n\n), and put your final answer within \boxed{}.
Problem: Each valve $A$, $B$, and $C$, when open, releases water into a tank at its own constant rate. With all three valves open, the tank fills in 1 hour, with only valves $A$ and $C$ open it takes 1.5 hours, and with only valves $B$ and $C$ open it takes 2 hours. The number of hours required wit...
```

**Greedy draft block (first γ tokens):**

```text
combined rate is:
\[
(a + c) \times \frac{3}{2} = 1 \quad \Rightarrow \quad a + c
```

**Best branch draft block:**

```text
combined rate times time equals 1 tank:
\[
(a + c) \times \frac{3}{2} = 1 \quad \Rightarrow
```

### Case 2: branch_small_gain (`deepscaler_01126@ck96`)

**Problem** (deepscaler_01126):

> In a jar of red, green, and blue marbles, all but 6 are red marbles, all but 8 are green, and all but 4 are blue. How many marbles are in the jar?

- checkpoint: token 96 | γ=32
- A_single=4 | A_best4=5 | **G_B=1**
- branch accept lengths: `[4, 4, 4, 5]` (best branch index=3)

**Prefix tail (target-reachable context):**

```text
Solve the following math problem efficiently and clearly. Please reason step by step, separate logical reasoning steps with two newline characters (\n\n), and put your final answer within \boxed{}.
Problem: In a jar of red, green, and blue marbles, all but 6 are red marbles, all but 8 are green, and all but 4 are blue. How many marbles are in the jar? Step 1: Let's denote the number of red marbles as R, green marbles as G, and blue marbles as B. The total number of marbles will be T = R + G +...
```

**Greedy draft block (first γ tokens):**

```text
are green marbles. This means that the sum of red and blue marbles is 8. So, R + B = 8.

Step
```

**Best branch draft block:**

```text
are green marbles, which means the sum of red and blue marbles is 8. So, R + B = 8.

Step 4
```

### Case 3: branch_no_gain (`deepscaler_01365@ck96`)

**Problem** (deepscaler_01365):

> In a bag of marbles, $\frac{3}{5}$ of the marbles are blue and the rest are red. If the number of red marbles is doubled and the number of blue marbles stays the same, what fraction of the marbles will be red?

- checkpoint: token 96 | γ=32
- A_single=1 | A_best4=1 | **G_B=0**
- branch accept lengths: `[1, 1, 1, 1]` (best branch index=0)

**Prefix tail (target-reachable context):**

```text
Solve the following math problem efficiently and clearly. Please reason step by step, separate logical reasoning steps with two newline characters (\n\n), and put your final answer within \boxed{}.
Problem: In a bag of marbles, $\frac{3}{5}$ of the marbles are blue and the rest are red. If the number of red marbles is doubled and the number of blue marbles stays the same, what fraction of the marbles will be red? \n
Okay, let's see. The problem says that in a bag of marbles, 3/5 are blue and...
```

**Greedy draft block (first γ tokens):**

```text
maybe the total number of marbles is 5. That way, 3/5 of 5 is 3, which is a whole number. So
```


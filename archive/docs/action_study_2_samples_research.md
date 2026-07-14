# Action Study 有研究价值的样本（2 题）

> 生成时间：2026-07-11 18:10:25  
> 筛选：从 5 题 / 18 prefix 中按 state_bucket、动作可用性、语义多样性选取  
> 判分：rescored grader；API：修复 temperature 后 gpt-5.5

---

## 为什么换样本？

先前 p03/p04（极坐标题）的问题：

- 三种动作全部 pass@k=1，**无动作差异**
- p04 接近 TERMINAL，**无决策研究价值**

当前 pilot（5 题）修复 grader 后：**尚无 Continue=0 且 Branch=1 的 case**。
因此选取两类仍有分析价值的样本：

1. **动作可用性差异**（Rollback 不可用）
2. **不确定性状态差异**（唯一 Forward-uncertain + 语义分支）

---

## Sample 1 — math500_0004 EARLY（8.6%）Rollback 不可用

**prefix_id**: `math500_0004_p00_paragraph_end`

**选取理由：**

- 推理最早期（progress=0.086），路线尚未展开
- Rollback=None：无 previous_checkpoint，只能 Continue / Branch
- 展示「动作可用性」随 prefix 位置变化

### 0. 题目

The results of a cross-country team's training run are graphed below. Which student has the greatest average speed? [asy]
for ( int i = 1; i <= 7; ++i )
{

draw((i,0)--(i,6));
}

for ( int i = 1; i <= 5; ++i )
{

draw((0,i)--(8,i));
}
draw((-0.5,0)--(8,0), linewidth(1));
draw((0,-0.5)--(0,6), linewidth(1));
label("$O$", (0,0), SW);
label(scale(.85)*rotate(90)*"distance", (0, 3), W);
label(scale(.85)*"time", (4, 0), S);
dot((1.25, 4.5));
label(scale(.85)*"Evelyn", (1.25, 4.8), N);
dot((2.5, 2.2))...(truncated)

**Gold**: `\text{Evelyn}`

### 1. Greedy trace

| 字段 | 值 |
|------|-----|
| predicted_answer | `Evelyn` |
| is_correct | **1** |
| token_count | 755 |

### 2. Prefix

| 字段 | 值 |
|------|-----|
| reasoning_progress | **0.086** |
| previous_checkpoint | **无（Rollback 不可用）** |

```
);
draw((0,-0.5)--(0,6), linewidth(1));
label("$O$", (0,0), SW);
label(scale(.85)*rotate(90)*"distance", (0, 3), W);
label(scale(.85)*"time", (4, 0), S);
dot((1.25, 4.5));
label(scale(.85)*"Evelyn", (1.25, 4.8), N);
dot((2.5, 2.2));
label(scale(.85)*"Briana", (2.5, 2.2), S);
dot((4.25,5.2));
label(scale(.85)*"Carla", (4.25, 5.2), SE);
dot((5.6, 2.8));
label(scale(.85)*"Debra", (5.6, 2.8), N);
dot((6.8, 1.4));
label(scale(.85)*"Angela", (6.8, 1.4), E);
[/asy]
<think>

</think>

To determine which student has the greatest average speed, we need to analyze the relationship between distance and time for each student based on the provided graph.


```

### 3. Next steps ×4

**sample 0**:
```
**1. Understand the Graph:**
*   **Horizontal Axis (x-axis):** Represents **time**. The grid lines are spaced by 1 unit.
*   **Vertical Axis (y-axis):** Represents **distance**. The grid lines are spaced by 1 unit.
*   **Data Points:** Each dot represents a student's position at a specific time.
```

**sample 1**:
```
**1. Understand the Graph Axes:**
*   The horizontal axis (x-axis) represents **time**. The grid lines are spaced at intervals of 1 unit.
*   The vertical axis (y-axis) represents **distance**. The grid lines are spaced at intervals of 1 unit.
*   The origin $O$ is at $(0,0)$.
```

**sample 2**:
```
**1. Understand the Axes:**
*   The horizontal axis (x-axis) represents **Time**.
*   The vertical axis (y-axis) represents **Distance**.
*   The origin $(0,0)$ is labeled $O$.
*   The grid lines indicate increments. Looking at the labels and positions:
    *   The x-axis goes from 1 to 8. Let's assume each unit represents 1 minute (or a consistent time unit).
    *   The y-axis goes from 0 to 6. Let's assume each unit represents 1 km (or a consistent distance unit).
```

**sample 3**:
```
**1. Understand the Graph:**
*   **X-axis:** Represents **Time** (in some unit, likely minutes or hours, though the unit isn't explicitly stated, the relative magnitude is what matters). The grid lines are spaced at intervals of 1 unit.
*   **Y-axis:** Represents **Distance** (in miles or kilometers). The grid lines are spaced at intervals of 1 unit.
*   **Speed Definition:** Average speed is defined as the total distance traveled divided by the total time taken ($ \text{Speed} = \frac{\text{Distance}}{\text{Time}} $).

```

### 4. API Validity
```json

{
  "prefix_id": "math500_0004_p00_paragraph_end",
  "prefix_validity": "VALID",
  "prefix_status": "VALID",
  "first_error_step": null,
  "error_type": "none",
  "confidence": 0.99,
  "explanation": "The prefix correctly states that average speed should be determined from the distance-time relationship on the graph.",
  "api_model": "",
  "api_prompt_version": "validity_v2",
  "timestamp": "2026-07-11T18:10:25.181379+00:00"
}

```

### 5. Cluster + state_bucket
```json

{
  "prefix_id": "math500_0004_p00_paragraph_end",
  "clusters": [
    0,
    0,
    0,
    0
  ],
  "num_semantic_clusters": 1,
  "strategy_descriptions": [
    "Interpret the graph axes and setup before comparing average speeds"
  ],
  "api_model": "gpt-5.5",
  "api_prompt_version": "cluster_v1",
  "diversity_label": "LOW_DIVERSITY",
  "state_bucket": "Stable"
}

```

### 6. 三动作 Oracle Outcome（Phase 1 主指标）

| action | oracle_recoverable | draft_generated_tokens | discarded_prefix_tokens | action_start |
|--------|-------------------|------------------------|-------------------------|--------------|
| continue | 1 | 720 | 0 | current_prefix |
| branch | 1 | 2750 | 0 | current_prefix |

> `debug_latency_sec` 仅作工程排查，**不用于论文延迟结论**。

### 7. 动作明细（各举 1 条）

**continue** → `Evelyn` correct=1

```
**1. Understand the Graph Axes:**
*   The horizontal axis (x-axis) represents **time**.
*   The vertical axis (y-axis) represents **distance**.
*   The origin $O$ is at $(0,0)$.

**2. Recall the Formula for Average Speed:**
Average speed is defined as the total distance traveled divided by the total time taken:
$$ \text{Average Speed} = \frac{\text{Distance}}{\text{Time}} $$
On a distance-time graph, this value corresponds to the **slope** of the line segment connecting the origin $(0,0)$ to the
```

**branch** → `Evelyn` correct=1

```
**1. Understand the Graph:**
- The horizontal axis represents **Time**.
- The vertical axis represents **Distance**.
- The graph shows a cross-country training run, which implies that distance is directly proportional to time (assuming constant speed for simplicity in this context, or simply looking at the ratio $\frac{\text{Distance}}{\text{Time}}$).
- The students are represented by dots at specific coordinates $(Time, Distance)$.

**2. Formula for Average Speed:**
$$ \text{Average Speed} = \f
```

### 8. 研究解读

- state_bucket = `Stable`（validity=VALID, diversity=LOW_DIVERSITY）
- **Rollback 动作不可执行**：研究问题退化为 Continue vs Branch
- Branch draft tokens ≈ Continue 的 **3.8×**（计算量参考，非 E2E 延迟）

---

## Sample 2 — math500_0004 MIDDLE（45.5%）唯一 Forward-uncertain

**prefix_id**: `math500_0004_p01_paragraph_end`

**选取理由：**

- 当前 18 个 prefix 中唯一的 Forward-uncertain（VALID + HIGH_DIVERSITY）
- next step 聚为 2 个语义簇：「逐个算速度」vs「先算 Evelyn」
- Continue@1 480 tok vs Branch@4 2082 tok — 精度-成本权衡可研究

### 0. 题目

The results of a cross-country team's training run are graphed below. Which student has the greatest average speed? [asy]
for ( int i = 1; i <= 7; ++i )
{

draw((i,0)--(i,6));
}

for ( int i = 1; i <= 5; ++i )
{

draw((0,i)--(8,i));
}
draw((-0.5,0)--(8,0), linewidth(1));
draw((0,-0.5)--(0,6), linewidth(1));
label("$O$", (0,0), SW);
label(scale(.85)*rotate(90)*"distance", (0, 3), W);
label(scale(.85)*"time", (4, 0), S);
dot((1.25, 4.5));
label(scale(.85)*"Evelyn", (1.25, 4.8), N);
dot((2.5, 2.2))...(truncated)

**Gold**: `\text{Evelyn}`

### 1. Greedy trace

| 字段 | 值 |
|------|-----|
| predicted_answer | `Evelyn` |
| is_correct | **1** |
| token_count | 755 |

### 2. Prefix

| 字段 | 值 |
|------|-----|
| reasoning_progress | **0.455** |
| previous_checkpoint | 有（可 Rollback） |

```
$ is at $(0,0)$.

**2. Recall the Formula for Average Speed:**
Average speed is defined as the total distance traveled divided by the total time taken:
$$ \text{Average Speed} = \frac{\text{Distance}}{\text{Time}} $$
On a distance-time graph, this value corresponds to the **slope** of the line segment connecting the origin $(0,0)$ to the student's data point. A steeper slope indicates a higher speed.

**3. Analyze the Data Points:**
Let's estimate the coordinates $(\text{time}, \text{distance})$ for each student based on the grid lines (where vertical lines are at $x=1, 2, 3, 4, 5, 6, 7, 8$ and horizontal lines are at $y=1, 2, 3, 4, 5, 6$).


```

### 3. Next steps ×4

**sample 0**:
```
*   **Evelyn**: Located at approximately $x = 1.25$, $y = 4.5$.
    *   Speed $\approx \frac{4.5}{1.25} = 3.6$
*   **Briana**: Located at approximately $x = 2.5$, $y = 2.2$.
    *   Speed $\approx \frac{2.2}{2.5} = 0.88$
*   **Carla**: Located at approximately $x = 4.25$, $y = 5
```

**sample 1**:
```
*   **Evelyn**: Located at approximately $x = 1.25$, $y = 4.5$.
    *   $\text{Speed} \approx \frac{4.5}{1.25} = 3.6$
```

**sample 2**:
```
*   **Evelyn**: Located at approximately $x = 1.25$, $y = 4.5$.
    $$ \text{Speed}_{\text{Evelyn}} = \frac{4.5}{1.25} $$
    Calculation: $4.5 \div 1.25 = 4.5 \div \frac{5}{4} = 4.5 \times 0.8 = 3.6$
```

**sample 3**:
```
*   **Evelyn**: Located at approximately $x = 1.25$, $y = 4.5$.
    *   Speed $\approx \frac{4.5}{1.25} = 3.6$
*   **Briana**: Located at approximately $x = 2.5$, $y = 2.2$.
    *   Speed $\approx \frac{2.2}{2.5} = 0.88$
*   **Carla**: Located at approximately $x = 4.25$, $y = 5
```

### 4. API Validity
```json

{
  "prefix_id": "math500_0004_p01_paragraph_end",
  "prefix_validity": "VALID",
  "prefix_status": "VALID",
  "first_error_step": null,
  "error_type": "none",
  "confidence": 0.93,
  "explanation": "The reasoning correctly identifies average speed as distance divided by time, corresponding to the slope from the origin to each student's point.",
  "api_model": "",
  "api_prompt_version": "validity_v2",
  "timestamp": "2026-07-11T18:10:25.194913+00:00"
}

```

### 5. Cluster + state_bucket
```json

{
  "prefix_id": "math500_0004_p01_paragraph_end",
  "clusters": [
    0,
    1,
    1,
    0
  ],
  "num_semantic_clusters": 2,
  "strategy_descriptions": [
    "Compute approximate speeds for multiple students to compare ratios",
    "Compute Evelyn's average speed from her coordinate ratio"
  ],
  "api_model": "gpt-5.5",
  "api_prompt_version": "cluster_v1",
  "diversity_label": "HIGH_DIVERSITY",
  "state_bucket": "Forward-uncertain"
}

```

### 6. 三动作 Oracle Outcome（Phase 1 主指标）

| action | oracle_recoverable | draft_generated_tokens | discarded_prefix_tokens | action_start |
|--------|-------------------|------------------------|-------------------------|--------------|
| continue | 1 | 480 | 0 | current_prefix |
| branch | 1 | 2082 | 0 | current_prefix |
| rollback | 1 | 2702 | 0 | previous_checkpoint |

> `debug_latency_sec` 仅作工程排查，**不用于论文延迟结论**。

### 7. 动作明细（各举 1 条）

**continue** → `Evelyn` correct=1

```
*   **Evelyn**: Located at approximately $x = 1.25$, $y = 4.5$.
    *   Speed $\approx \frac{4.5}{1.25} = \frac{4.5}{5/4} = 4.5 \times 0.8 = 3.6$ units/time.

*   **Briana**: Located at approximately $x = 2.5$, $y = 2.2$.
    *   Speed $\approx \frac{2.2}{2.5} = \frac{2.2}{2.5} = 0.88$ units/time.

*   **Carla**: Located at approximately $x = 4.25$, $y = 5.2$.
    *   Speed $\approx \frac{5.2}{4.25} = \frac{5.2}{4.25} \approx 1.22$ units/time.

*   **Debra**: Located at approximately $x = 5.6$, 
```

**branch** → `Evelyn` correct=1

```
*   **Evelyn**: Located at approximately $x = 1.25$, $y = 4.5$.
    $$ \text{Speed} \approx \frac{4.5}{1.25} = 3.6 $$
*   **Briana**: Located at approximately $x = 2.5$, $y = 2.2$.
    $$ \text{Speed} \approx \frac{2.2}{2.5} = 0.88 $$
*   **Carla**: Located at approximately $x = 4.25$, $y = 5.2$.
    $$ \text{Speed} \approx \frac{5.2}{4.25} \approx 1.22 $$
*   **Debra**: Located at approximately $x = 5.6$, $y = 2.8$.
    $$ \text{Speed} \approx \frac{2.8}{5.6} = 0.5 $$
*   **Angela**: Located at
```

**rollback** → `Evelyn` correct=1

```
**3. Extract Data Points for Each Student:**
By observing the coordinates of the dots on the grid (where each major grid line represents 1 unit):
*   **Evelyn**: Located at approximately $(1.25, 4.5)$.
*   **Briana**: Located at approximately $(2.5, 2.2)$.
*   **Carla**: Located at approximately $(4.25, 5.2)$.
*   **Debra**: Located at approximately $(5.6, 2.8)$.
*   **Angela**: Located at approximately $(6.8, 1.4)$.

**4. Calculate the Slope (Speed) for Each Student:**

*   **Evelyn**:
    $$ \
```

### 8. 研究解读

- state_bucket = `Forward-uncertain`（validity=VALID, diversity=HIGH_DIVERSITY）
- 语义分支数 = 2：适合研究 Branch 并行探索的价值
- Branch draft tokens ≈ Continue 的 **4.3×**（计算量参考，非 E2E 延迟）
- 当前判分：三动作 oracle recoverability 均为 1 → **无准确率差异**，需更多难题/错误 prefix

---

## 数据局限 & 下一步

| 现状 | 影响 |
|------|------|
| 仅完成 5/200 题 | 样本太少，无 INVALID bucket |
| 修复 grader 后全 pass@k=1 | 尚无动作 outcome 差异 |
| 无 Continue=0 案例 | 无法验证 rescue 效应 |

**建议**：Phase 0 pilot 刻意纳入 greedy 答错的题 + early/middle checkpoint，才能产生有差异的 oracle action label。

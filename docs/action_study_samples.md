# Action Study 完整样本导出（2 题）

生成时间：2026-07-11 17:41
数据来源：`outputs/action_study_v1/`

---

## 全局数据流逻辑

```
题目 (MATH500)
  ↓ build_prompt()
小模型 greedy 生成 full_reasoning → traces.jsonl
  ↓ extract_study_prefixes() 按 \n\n / Wait / But 切
每个 prefix:
  ├─ vLLM 采样 4 个 next_step → next_step_samples.jsonl
  ├─ API 标 validity (V_i) → validity_labels.jsonl
  ├─ API 聚类 next steps (D_i) → cluster_labels.jsonl
  ├─ state_bucket = f(validity, diversity) → prefixes.jsonl
  ├─ Continue (greedy ×1)
  ├─ Branch (sample ×4)
  ├─ Rollback (回退上一 \n\n，sample ×4)
  └─ gold answer 判对错 → action_results.jsonl + actions.jsonl
  ↓ analyze.py → action_matching_report.md
```

## Sample 1：math500_0000（极坐标转换）+ prefix p03

### Step 0 题目

**题目**：Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\theta),$ where $r > 0$ and $0 \le \theta < 2 \pi.$

**Gold**：`\left( 3, \frac{\pi}{2} \right)`

### Step 1 小模型 greedy 完整推理 → `traces.jsonl`

- `predicted_answer`: `(3, \frac{\pi}{2})`
- `is_correct`: **0**（格式未完全匹配 gold）
- `token_count`: 457, `generation_latency`: 32.6s

推理结尾：
```
, using the arctangent function with quadrant adjustment:
    Since $x=0$ and $y>0$, the point is on the positive y-axis, so $\theta = \frac{\pi}{2}$.

3.  **Check constraints**:
    The problem requires $0 \le \theta < 2\pi$.
    Our calculated angle is $\frac{\pi}{2}$, which satisfies $0 \le \frac{\pi}{2} < 2\pi$.

Thus, the polar coordinates are $(3, \frac{\pi}{2})$.

\boxed{(3, \frac{\pi}{2})}
```

### Step 2 Prefix 切分 → `prefixes.jsonl`

- `prefix_id`: `math500_0000_p03_paragraph_end`
- `prefix_type`: `PARAGRAPH_END`（在 `\n\n` 处切断）
- `reasoning_progress`: **0.269**（推理进度 26.9%）
- `previous_checkpoint`: 有（可 rollback）
- `state_bucket`: `UNCLEAR` = validity(`UNCLEAR`) + diversity(`LOW_DIVERSITY`)

prefix 末尾（已做完 step1 算 r=3）：
```
k>

To convert the point $(0, 3)$ from rectangular coordinates $(x, y)$ to polar coordinates $(r, \theta)$, we follow these steps:

1.  **Calculate the radius $r$**:
    The formula for $r$ is $r = \sqrt{x^2 + y^2}$.
    Given $x = 0$ and $y = 3$:
    $$r = \sqrt{0^2 + 3^2} = \sqrt{9} = 3$$
    Since the problem requires $r > 0$, we have $r = 3$.


```

### Step 3 采样 next steps → `next_step_samples.jsonl`

小模型在 prefix 后采样 4 条「下一步」：

- sample 0: `2.  **Calculate the angle $\theta$**:     The formula for $\theta$ involves $\tan \theta = \frac{y}{x}$. However, we mus...` (128 tokens)

- sample 1: `2.  **Calculate the angle $\theta$**:     The formula for $\theta$ is $\theta = \arctan\left(\frac{y}{x}\right)$, but we...` (128 tokens)

- sample 2: `2.  **Calculate the angle $\theta$**:     The formula for $\theta$ is $\theta = \arctan\left(\frac{y}{x}\right)$, but we...` (128 tokens)

- sample 3: `2.  **Calculate the angle $\theta$**:     The formula for $\theta$ involves the tangent function: $\tan(\theta) = \frac{...` (128 tokens)

### Step 4 API 标注 → `validity_labels.jsonl` + `cluster_labels.jsonl`

- validity: `UNCLEAR`, confidence: 0.0
- explanation: Error code: 400 - {'error': {'type': 'invalid_request_error', 'code': 400, 'message': 'Bad request. Please check your input parameters.'}}

- clusters: [] (source: heuristic fallback，API 400 失败)

### Step 5 三动作执行 → `action_results.jsonl` + `actions.jsonl`

| action | pass@k | is_correct | tokens | latency |
|--------|--------|------------|--------|---------|
| continue | 0 | 0 | 320 | 10.2s |
| branch | 1 | 1 | 1239 | 39.8s |
| rollback | 1 | 1 | 1738 | 56.8s |

**解读**：在 progress=0.269 处，Continue 仍错（pass@k=0），Branch/Rollback 均 rescue 成功（pass@k=1）。

Continue 动作逻辑：prefix 不变，greedy 续写至 `\boxed{}`，再 `math_equal` 判分。

Branch 动作逻辑：在 `start_checkpoint`（当前 prefix 末尾）重新 sample×4，取 pass@k。

Rollback 动作逻辑：回退到 `previous_checkpoint`（上一个 `\n\n`），再 sample×4。


Branch sample 0 最终答案片段：
```
  Therefore, the angle is $\theta = \frac{\pi}{2}$.

3.  **Check constraints**:
    -   $r = 3 > 0$ (Satisfied)
    -   $0 \le \frac{\pi}{2} < 2\pi$ (Satisfied)

The polar coordinates are $(3, \frac{\pi}{2})$.

\boxed{(3, \frac{\pi}{2})}<|endoftext|>
```

### Step 6 API 复核 → `api_reviews.jsonl`

- reason: `low_confidence_validity` (validity confidence 低，触发复核)

---

## Sample 2：math500_0001（双重求和化简）

### Step 0 题目

**题目**：
```
Define
\[p = \sum_{k = 1}^\infty \frac{1}{k^2} \quad \text{and} \quad q = \sum_{k = 1}^\infty \frac{1}{k^3}.\]Find a way to write
\[\sum_{j = 1}^\infty \sum_{k = 1}^\infty \frac{1}{(j + k)^3}\]in terms of $p$ and $q.$
```

**Gold**：`p - q`

### Step 1 小模型 greedy 完整推理 → `traces.jsonl`

- `predicted_answer`: `p - q`
- `is_correct`: **1** ✓
- `token_count`: 939, `generation_latency`: 4.9s

完整推理：
```


<think>

</think>

To find the value of the double sum $\sum_{j = 1}^\infty \sum_{k = 1}^\infty \frac{1}{(j + k)^3}$ in terms of $p$ and $q$, we proceed with the following steps:

### Step 1: Change of Variables
Let $n = j + k$. Since $j \ge 1$ and $k \ge 1$, the minimum value for $n$ is $2$.
For a fixed integer $n \ge 2$, the possible pairs $(j, k)$ such that $j + k = n$ are:
- If $n=2$, $(1, 1)$ (1 pair)
- If $n=3$, $(1, 2), (2, 1)$ (2 pairs)
- If $n=4$, $(1, 3), (2, 2), (3, 1)$ (3 pairs)
In general, for a fixed $n$, there are $n-1$ pairs of positive integers $(j, k)$ that satisfy $j+k=n$.

We can rewrite the double sum by grouping terms with the same value of $n$:
\[
S = \sum_{j = 1}^\infty \sum_{k = 1}^\infty \frac{1}{(j + k)^3} = \sum_{n = 2}^\infty \sum_{\substack{j, k \ge 1 \\ j+k=n}} \frac{1}{n^3}
\]
Since the term $\frac{1}{n^3}$ is constant for all pairs summing to $n$, we can factor it out:
\[
S = \sum_{n = 2}^\infty \left( \sum_{\substack{j, k \ge 1 \\ j+k=n}} 1 \right) \frac{1}{n^3}
\]
The inner sum counts the number of solutions, which is $n-1$. Thus:
\[
S = \sum_{n = 2}^\infty \frac{n-1}{n^3}
\]

### Step 2: Simplify the Series
We can split the fraction $\frac{n-1}{n^3}$ into two parts:
\[
\frac{n-1}{n^3} = \frac{n}{n^3} - \frac{1}{n^3} = \frac{1}{n^2} - \frac{1}{n^3}
\]
Substituting this back into the sum:
\[
S = \sum_{n = 2}^\infty \left( \frac{1}{n^2} - \frac{1}{n^3} \right) = \sum_{n = 2}^\infty \frac{1}{n^2} - \sum_{n = 2}^\infty \frac{1}{n^3}
\]

### Step 3: Relate to $p$ and $q$
We are given:
\[
p = \sum_{k = 1}^\infty \frac{1}{k^2} = 1 + \sum_{k = 2}^\infty \frac{1}{k^2}
\]
\[
q = \sum_{k = 1}^\infty \frac{1}{k^3} = 1 + \sum_{k = 2}^\infty \frac{1}{k^3}
\]
Therefore, the sums starting from $n=2$ can be expressed as:
\[
\sum_{n = 2}^\infty \frac{1}{n^2} = p - 1
\]
\[
\sum_{n = 2}^\infty \frac{1}{n^3} = q - 1
\]

### Step 4: Final Calculation
Substitute these expressions into the equation for $S$:
\[
S = (p - 1) - (q - 1)
\]
\[
S = p - 1 - q + 1
\]
\[
S = p - q
\]

\boxed{p - q}
```

### Step 2–6（进行中）

该题 trace 已落盘，prefix 切分 + 三动作实验**尚未完成**（vLLM 正式跑进度 2/200）。

完成后将按 Sample 1 同样逻辑写入：
- `prefixes.jsonl`（预计 3–5 个 prefix，在 step1/step2 换变量、结论处切断）
- `next_step_samples.jsonl` / `validity_labels.jsonl` / `cluster_labels.jsonl`
- `action_results.jsonl` / `actions.jsonl`

**预期 prefix 切分点**（根据推理结构）：
1. intro 结束（`\n\n` 后进入 Step 1）
2. Step 1 换变量推导结束
3. Step 2 关联 p/q 结束
4. 最终结论前

---

## 落盘文件索引

| 文件 | 含义 |
|------|------|
| `problems.jsonl` | 题目元数据 |
| `traces.jsonl` | greedy 完整推理 + 判分 |
| `prefixes.jsonl` | prefix 状态（validity/diversity/state_bucket） |
| `next_step_samples.jsonl` | prefix 后 4 条 next step |
| `validity_labels.jsonl` | API validity 标注 |
| `cluster_labels.jsonl` | API/heuristic 聚类 |
| `action_results.jsonl` | 三动作汇总（pass@k） |
| `actions.jsonl` | 三动作明细（每条 continuation） |
| `api_reviews.jsonl` | 冲突案例 API 复核 |
| `hidden.safetensors` | HF 后处理 hidden/logits |

## 原始 JSON 导出

- `outputs/action_study_v1/sample_export_math500_0000_p03.json`
- `outputs/action_study_v1/sample_export_math500_0001.json`
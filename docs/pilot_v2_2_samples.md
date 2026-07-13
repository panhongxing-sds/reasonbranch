# Pilot v2 DeepScaler 样本（前 2 题）— 审阅结论

> **这两题目前不能进入 Pilot v2 正式统计。**  
> 数据：`outputs/action_study_pilot_v2/`（旧 run，修复前生成）

---

## 审阅结论

| 题号 | 问题 | 状态 |
|------|------|------|
| `deepscaler_01000` | 缺图（`missing_figure`） | **整题排除** |
| `deepscaler_01001` | Greedy trace 截断（`TRUNCATED`） | 待重跑（`max_new_tokens=4096`） |

**不是**「Branch 没有收益」，而是：

> 第 1 题输入不完整；第 2 题及所有 action 生成预算不足，Oracle outcome 尚未被观测。

---

## 1. `deepscaler_01000` — Missing-context / Unsolvable input

**dataset**: deepscaler  
**Gold**: `6`

题目依赖 *"the figure"* / *"nine positions indicated"*，但 prompt 无图。

Greedy 推理特征（无研究意义）：
- `Where is the 9th?` / `Perhaps ...` / `Let's try a different orientation.`
- 9 个 prefix 均来自同一种错误猜测，**全部不应进入统计**

**修复后处理**：`input_complete=false`, `exclusion_reason=missing_figure`，跳过 trace/prefix/action 生成。

---

## 2. `deepscaler_01001` — TRUNCATED（非 INCORRECT）

**dataset**: deepscaler  
**Gold**: `279`

Greedy 结束于 `So if $0 < f < 1 - 2\` — 明显 `finish_reason=length` 截断。

| 字段 | 旧值（错误） | 新值（正确） |
|------|-------------|-------------|
| `evaluation_status` | ERROR / 记为错 | `TRUNCATED` |
| `is_correct` | `0` | `None` |
| `reasoning_progress` | 43%/76% 等 | `None`（trace 不完整） |
| `branch_gain` | `0` | `NA` |

**修复后处理**：trace `max_new_tokens=4096`；action continuation `2048+2048` 二阶段续写；截断 trace 不生成 prefix。

---

## 已实现的 Pipeline 修复（v2.1）

1. **缺图过滤** — `visual_input.py` + `datasets.load_deepscaler()`
2. **评估状态** — `TRUNCATED` / `NO_FINAL_ANSWER` / `OK`（`grading.classify_generation_outcome`）
3. **准入门槛** — `admission.py`：`admission_pass` 字段
4. **Branch Gain** — 无有效终答时为 `NA`，非 `0`
5. **Wait/But 合并** — 相邻 marker 聚为 uncertainty event，每 event 最多 1 BEFORE + 1 AFTER
6. **生成预算** — trace 4096；continuation 2048 + retry 2048

Pilot 800 题范围内预计约 **8%** 缺图题被自动排除（offset=1000，实测 61/800）。

---

## 重跑建议

```bash
# 清掉旧 checkpoint 中已错误处理的题（或 FRESH=1 全量重跑）
FRESH=1 bash reasoning_branch_dataset/scripts/run_pilot_v2.sh
```

`deepscaler_01000` 将写入 `excluded_problems.jsonl`；`deepscaler_01001` 用 4096 token 重跑后若完整终答才可进入 prefix study。

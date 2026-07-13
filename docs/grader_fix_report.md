# Grader 修复报告 + 重评分结果

> 生成时间：2026-07-12  
> 状态：**评分器已修复并审计；正式实验已暂停，待 Phase 0 人工核查后再扩规模**

---

## 一、你指出的问题 — 确认属实

### 1. 评分器坏了（已修）

**原 bug**：`\left(3,\frac{\pi}{2}\right)` 与 `(3,\frac{\pi}{2})` 语义等价，但 `math_equal()` 只做去空格 + 子串匹配，判为 `is_correct=0`。

**根因**：
- 未剥离 `\left`/`\right`
- 未做 tuple 分量比较
- 未用 sympy 处理 `\frac{1}{2}` vs `0.5`
- continuation 无 `\boxed{}` 时 fallback 到最后一行文本，产生空 `predicted_answer` / `None`

**修复**（`reasoning_branch_dataset/grading.py`）：
- LaTeX 归一化（`\left/\right/\text{}` 等）
- tuple 逐元素 `math_equal`
- sympy 符号/数值等价
- `grade_math_answer()` 返回 `evaluation_status: OK | ERROR`
- 动作评分默认 `require_marker=True`（必须有 `\boxed{}` 或 `####`）

**单元测试**：`reasoning_branch_dataset/tests/test_grading.py`

```python
# 全部通过
(r"\left(3,\frac{\pi}{2}\right)", r"(3,\frac{\pi}{2})", True)
(r"\boxed{(3,\frac{\pi}{2})}", r"\left(3,\frac{\pi}{2}\right)", True)
(r"p-q", r"p - q", True)
(r"\frac{1}{2}", r"0.5", True)
```

### 2. API 400 被误标为 UNCLEAR（已修）

**修复**（`api_validity.py` + `diversity.py`）：
- API 失败 → `prefix_validity = API_ERROR`
- `state_bucket = API_ERROR`（不再进入 UNCLEAR 分析桶）

### 3. pass@k / 字段一致性（部分已修）

- `actions.jsonl` 现统一写入 `predicted_answer`, `is_correct`, `evaluation_status`
- Branch/Rollback 汇总增加 `oracle_branch_recoverable` / `oracle_rollback_recoverable`
- 增加 `latency_parallel_sec`（vLLM batch 墙钟延迟）

**尚未实现**（按你的建议，下一阶段再做）：
- Continue@1 vs Branch@4 的公平对照（或效用函数 \(U(a)\)）
- `selected_acc`（在线 selector 准确率）
- 语义 rollback（Rollback-1/2/token-based）
- TERMINAL prefix 过滤

---

## 二、重评分结果（不重新生成，只修 grader 后复判）

对 `outputs/action_study_v1/` 已有 continuation 用新 grader 重判：

```bash
python reasoning_branch_dataset/scripts/rescore_action_study.py \
  --output-dir reasoning_branch_dataset/outputs/action_study_v1
```

输出文件：
- `traces.rescored.jsonl`
- `actions.rescored.jsonl`
- `action_results.rescored.jsonl`

### 变更统计

| 指标 | 数量 |
|------|------|
| trace 判分更正 | **1**（math500_0000: 0→1） |
| action 判分更正 | **142** |

---

## 三、Sample 1：math500_0000 + prefix p03（progress=0.269）

**题目**：将 `(0,3)` 转为极坐标  
**Gold**：`\left(3, \frac{\pi}{2}\right)`

### 修复前（不可信）

| action | pass@k | 备注 |
|--------|--------|------|
| continue | 0 | grader 误判 |
| branch | 1 | "rescue" 结论不可信 |
| rollback | 1 | 同上 |

### 修复后（rescored）

| action | pass@k | evaluation_status |
|--------|--------|-------------------|
| continue | **1** | OK |
| branch | **1** | OK |
| rollback | **1** | OK |

**结论修正**：此前「p03 处 Branch 优于 Continue」是**评分器假象**。修复后三种动作均可达正确答案（oracle recoverability 均为 1），**不能**得出 Branch 更优的结论。

---

## 四、Sample 2：math500_0000 + prefix p04（progress=0.941）

### 修复前（不可信）

| action | pass@k |
|--------|--------|
| continue | 0 |
| branch | 0 |
| rollback | 0 |

动作明细 `predicted_answer` 为空、`is_correct=None` — 典型 grader 故障。

### 修复后（rescored）

| action | pass@k | evaluation_status |
|--------|--------|-------------------|
| continue | **1** | OK |
| branch | **1** | OK |
| rollback | **1** | OK |

**结论修正**：p04 是接近 TERMINAL 的状态（答案已基本确定），三种动作全部成功是预期的，**没有动作选择研究价值** — 与你指出的第 8 点一致。

---

## 五、trace 修复

```json
{
  "problem_id": "math500_0000",
  "predicted_answer": "(3, \\frac{\\pi}{2})",
  "is_correct": 1,
  "evaluation_status": "OK"
}
```

（修复前 `is_correct: 0`）

---

## 六、当前实验状态

- **已停止** 200 题全量 vLLM 跑（按你要求不扩规模）
- 已有 v1 数据保留；rescored 文件供审计
- validity 标注仍为 API 400 → 新代码会标 `API_ERROR`，需修 API 请求后再跑

---

## 七、下一步（Phase 0 最小 pilot）

1. **人工核查 50 条**：对比 `actions.rescored.jsonl` 与人工判分，目标 grader agreement ≈ 100%
2. **修 API 400**（请求参数/format），确认 VALID/INVALID/UNCLEAR 能正常产出
3. **重跑 50 题 pilot**（非 200 题）：
   - 每题 3 个语义 checkpoint：early / middle / late
   - 排除 TERMINAL
   - 记录 `oracle_*_recoverable` + `latency_parallel_sec`
4. **再谈** Continue@1 vs Branch@4 的效用比较

---

## 八、代码改动清单

| 文件 | 改动 |
|------|------|
| `grading.py` | 重写 math_equal + grade_math_answer + evaluation_status |
| `tests/test_grading.py` | 单元测试 |
| `action_study/vllm_backend.py` | score_answer → grade_math_answer |
| `action_study/actions.py` | 字段统一 + pass@k 逻辑 + oracle_* 字段 |
| `action_study/pipeline.py` | trace 评分 + action_results 新字段 |
| `action_study/api_validity.py` | API_ERROR 与 UNCLEAR 分离 |
| `action_study/diversity.py` | state_bucket 处理 API_ERROR |
| `scripts/rescore_action_study.py` | 对已有数据重评分 |

---

## 九、一句话总结

> Pipeline 能生成三种动作的 continuation，但**修复前所有 pass@k 结论不可信**。评分器修好后，math500_0000 的两个样本显示三种动作均可成功，**不存在先前报告的 Branch rescue 效应**；在扩规模前必须先完成人工 grader 审计和 API 标注修复。

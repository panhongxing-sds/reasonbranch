# Phase-1 Pilot 结论与标签修订（v2）

> 基于 `action_study_uncertainty_v1`（15 题 / 60 prefix）的严格评审结论。  
> **结论：Phase 1 跑通，但三分类定义不成立；暂不训练 Hidden Probe。**

---

## 1. 核心发现

| 发现 | 含义 |
|------|------|
| Stable 假设初步成立 | 正确 + 单策略 prefix，Continue ≈ Branch |
| Future-diverse 大量假阳性 | heuristic 把措辞差异当策略差异 |
| INVALID ≠ 不可恢复 | 错误 prefix 仍可能自我修正（Branch/Continue 混合收益） |
| Branch Gain=0 不等于 Branch 无用 | 可能是 4/4 全对，pass@4 无法区分必要性 |
| 15 题太少且 prefix 强相关 | 必须按 `problem_id` 划分 train/test |

---

## 2. v1 标签问题

### 2.1 Future-diverse 假阳性

`math500_0001_p00` 四条 next step 均为「令 n=j+k 重索引」，heuristic 误拆为 3 cluster → `HIGH_DIVERSITY`。

**修复**：策略级 API 聚类（`cluster_v2`），heuristic 改为保守默认 `ONE_STRATEGY`。

### 2.2 过早 prefix（NO_COMMITMENT）

8.9% 处 prefix 仅「下面开始求解」，API 标 VALID 无意义。

**修复**：新增 `prefix_substantiveness: NO_COMMITMENT | SUBSTANTIVE`，主实验排除 NO_COMMITMENT。

### 2.3 INVALID 与 recoverability 混淆

`math500_0012` 类样本：prefix 有明显错误，但 Branch 仍可 rescue。

**修复**：从 K=4 rollout 派生 `recovery_profile`，拆成 Corrupted-recoverable / Corrupted-stuck。

---

## 3. v2 四维标签体系

### 维度一：Current validity

```text
VALID | INVALID | UNCLEAR | NO_COMMITMENT
```

### 维度二：Strategy diversity（策略级 judge）

```text
ONE_STRATEGY | MULTIPLE_GENUINE_STRATEGIES
```

### 维度三：Recovery profile（K=4 branch rollout）

```text
ALL_SUCCEED | MOST_SUCCEED | MIXED | RARE_SUCCESS | ALL_FAIL
```

### 维度四：Behavior state（映射到系统动作）

| behavior_state | 条件概要 | 未来动作 |
|----------------|----------|----------|
| **Stable** | VALID + ONE_STRATEGY + 高成功率 | Continue |
| **Decision-sensitive** | VALID + 多策略 + 结果敏感 / Continue 失败 Branch 成功 | Branch |
| **Corrupted-recoverable** | INVALID + 部分路径可修正 | Branch 或继续探索 |
| **Corrupted-stuck** | INVALID + 几乎全部失败 | Handoff |
| **Excluded** | NO_COMMITMENT | 不进主实验 |

---

## 4. Branch 指标扩展

每个 prefix 同时报告：

- `branch_pass_at_4` = 𝟙[#correct > 0]
- `branch_correct_count` = #correct ∈ {0,1,2,3,4}
- `branch_accuracy_at_4` = #correct / #evaluated

主表同时输出：

```text
n_prefixes, n_continue_evaluated, n_branch_evaluated,
n_continue_errors, n_branch_errors
```

---

## 5. 下一步（按顺序）

```text
修复策略聚类 (cluster_v2)
  → 加入 branch_correct_count / recovery_profile
  → 区分 recoverable vs stuck
  → 50–100 道更难题 pilot（按 problem_id 划分）
  → 找到 Decision-sensitive + Corrupted-stuck 后再训 Hidden Probe
```

**不要现在训练 probe** — 当前标签噪声会学到 prefix 位置、Markdown 标题、题目文本模式。

---

## 6. 代码落点

| 文件 | 改动 |
|------|------|
| `action_study/diversity.py` | v2 behavior state + recovery + 保守 heuristic |
| `action_study/prefix_substantiveness.py` | NO_COMMITMENT 检测 |
| `action_study/api_validity.py` | validity_v3 + cluster_v2 |
| `action_study/actions.py` | branch_correct_count / accuracy |
| `action_study/pipeline.py` | 写入 v2 字段 |
| `action_study/analyze.py` | behavior 主表 + 分母 + pilot readiness |
| `scripts/backfill_uncertainty_study.py` | 并行 relabel 已有数据 |

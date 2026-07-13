# Phase-1：小模型 Uncertainty 研究框架（v2）

> **v3 修正**：Branch 主效用指标已改为 **target acceptance gain**，见 [`target_acceptance_framework.md`](target_acceptance_framework.md)。  
> 本文档描述的小模型行为状态与 correctness-based `branch_gain` **仅作探索性分析**，不是最终 controller 的 ground truth。

## 结论（pilot v1）

Phase 1 **pipeline 已跑通**，但 v1 三分类（Stable / Future-diverse / Current-unreliable）**标签不成立**。  
详见 [`phase1_pilot_findings.md`](phase1_pilot_findings.md)。

**暂不训练 Hidden Probe。** 先修标签 → target replay 回填 acceptance → 再训 probe 预测 (Ĝ_{\text{branch}})。

---

## 研究问题（v2 → v3）

**探索阶段（小模型 only）**：

\[
h_t,\ \text{logits}_t \rightarrow \{\text{Stable},\ \text{Decision-sensitive},\ \text{Corrupted-recoverable},\ \text{Corrupted-stuck}\}
\]

**最终目标（需 target replay）**：

\[
h_t \rightarrow \hat A_{\text{single}},\ \hat G_{\text{branch}},\ P_{\text{handoff}}
\]

| behavior_state（exploratory） | 小模型含义 | 未来系统动作（待 target 验证） |
|------------------------------|-----------|-------------------------------|
| **Stable** | 正确 + 单策略 | 可能 Continue-friendly |
| **Decision-sensitive** | 多策略 + 结果敏感 | 可能 Branch-helpful |
| **Corrupted-recoverable** | 错误但部分 rollout 可修正 | 待测 target 是否仍接受长 suffix |
| **Corrupted-stuck** | 错误且 rollout 全败 | 可能 Handoff-friendly |
| **Excluded** | NO_COMMITMENT | 不进主实验 |

---

## 标签维度

### 1. prefix_validity

`VALID | INVALID | UNCLEAR | NO_COMMITMENT`

### 2. prefix_substantiveness

`SUBSTANTIVE | NO_COMMITMENT` — 排除只复述题目/只说「开始求解」的 early prefix。

### 3. strategy_diversity（策略级 API 聚类 cluster_v2）

`ONE_STRATEGY | MULTIPLE_GENUINE_STRATEGIES`

Judge 问题：候选是否采用**不同数学操作/关键假设/解题策略**，而非不同措辞？

### 4. recovery_profile（K=4 branch rollout）

`ALL_SUCCEED | MOST_SUCCEED | MIXED | RARE_SUCCESS | ALL_FAIL`

---

## 小模型阶段操作

### Continue-full
Greedy 续写到 `\boxed{}`。

### Branch
Sample K=4 next steps → **策略级** API 聚类 → 每条续写到答案。

记录：
- `branch_pass_at_4`
- `branch_correct_count`
- `branch_accuracy_at_4`
- `recovery_profile`

---

## 主表（analyze 输出）

Behavior state 表（`admission_main`）— **探索性**，非 target-oracle：

| 字段 | 说明 | 层级 |
|------|------|------|
| continue_accuracy | Continue 最终正确率 | 辅助 |
| branch_pass_at_4 | mean(𝟙[#correct>0]) | **辅助（已降级）** |
| branch_gain | correctness pass@4 − continue | **辅助（已降级）** |

**主效用指标**（target replay 后）：`target_acceptance_gain = max_j A_j − A_single`

---

## Pilot readiness gates

### 探索性（小模型 only）

- `decision_sensitive_exists` ≥ 3
- `corrupted_recoverable_exists` ≥ 3
- `corrupted_stuck_exists` ≥ 3

满足 → 可扩规模、可构建 **target replay dataset**。

### 不可仅凭此声称

- `decision_sensitive_correctness_gain_positive` ≠ Branch 对 latency 有帮助
- `ready_for_probe` 需 target replay 回填 (G_{\text{branch}}) 后才可启用

**训练划分**：必须按 `problem_id` split，禁止随机按 prefix split。

---

## 命令

```bash
source /mnt/afs/L202500372/bootstrap/max_speed_env.sh
source reasoning_branch_dataset/scripts/load_api_env.sh

# 新 pilot（建议更难数据集 + 50–100 题）
OUT=outputs/action_study_pilot_v2 \
python -m reasoning_branch_dataset.action_study.pipeline \
  --output-dir $OUT --math500-limit 80 --gsm8k-limit 0 --engine vllm

# relabel 已有数据（v2 API 标签）
python reasoning_branch_dataset/scripts/backfill_uncertainty_study.py

# 分析
python -m reasoning_branch_dataset.action_study.analyze --data-dir $OUT
```

---

## 能声称 / 不能声称

**能**（pilot 后）：
- 存在 validity × strategy diversity × recoverability 三维结构（探索性）
- 小模型可低成本产生多样化候选（replay 数据集就绪）
- target replay 后 hidden 能否预测 (G_{\text{branch}})（待做）

**不能**（当前）：
- Branch Pass@4 证明 speculative latency 收益
- correctness-based `branch_gain` 作为 controller ground truth
- Future-diverse 一定适合 Branch（需 target acceptance 验证）
- INVALID 一律 Handoff
- 直接上 Hidden Probe（缺 target labels）

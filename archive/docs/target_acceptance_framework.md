# Target Acceptance 框架（v3）

> **核心修正**：Branch 的主效用不是「小模型是否找到最终正确答案」，而是「是否提高 target 对 speculative draft 的接受长度/接受概率」。

---

## 研究目标

```text
小模型生成 speculative 候选
        ↓
大模型 (target) 验证 / 选择 / 接管
        ↓
尽量接受更长 draft → 减少 target 重新生成与交互轮数
```

[
\boxed{
G_{\text{branch}} = A_{\text{branch}} - A_{\text{single}}
}
]

其中 (A_j = \text{accepted\_length}(T, s_t, b_j))，(A_{\text{branch}} = \max_j A_j)。

**最终正确率**仅作质量/安全约束，**不能**作为 Branch 主标签。

---

## 三个动作区域（target-oracle）

| 区域 | 条件 | 动作 |
|------|------|------|
| **Continue-friendly** | (A_{\text{single}} \ge \tau_A) 且 (A_{\text{branch}} - A_{\text{single}} \le \delta) | Continue |
| **Branch-helpful** | (A_{\text{branch}} - A_{\text{single}} > \delta) | Branch |
| **Handoff-friendly** | (\max_j A_j < \tau_H) | Handoff / Target takeover |

无主动 Rollback；target 拒绝后自然回到最后接受位置。

---

## 指标分层

### 主指标（需 target replay 回填）

| 字段 | 含义 |
|------|------|
| `target_accepted_length_continue` | (A_{\text{single}}) |
| `target_accepted_length_branch_max` | (A_{\text{branch}}) |
| `target_acceptance_gain` | (G_{\text{branch}}) |
| `target_accept_ratio_j` | (A_j / \|b_j\|) |
| `first_reject_position_j` | 首拒位置 |
| `target_selected_branch` | target 选择的候选 |
| `all_branches_rejected_early` | Handoff 信号 |

### 辅助指标（小模型阶段可算，**不**等于 Branch 效用）

| 字段 | 用途 |
|------|------|
| `branch_pass_at_4` | 探索性质量分析 |
| `branch_correct_count` | 安全/质量约束 |
| `continue_accuracy` | draft 质量参考 |
| `branch_gain` (correctness) | **已降级**，不可作 controller GT |

---

## Phase-1 行为状态（exploratory only）

`Stable / Decision-sensitive / Corrupted-recoverable / Corrupted-stuck` 描述**小模型内部**状态，**不是**最终 controller 的 ground truth。

| 旧标签 | 应重新解释为 |
|--------|----------------|
| Decision-sensitive | 小模型候选结果分化；**待验证**是否存在 (G_{\text{branch}} > \delta) |
| Corrupted-recoverable | 小模型可局部纠错；**待验证** target 是否仍接受较长 suffix |
| Corrupted-stuck | 小模型 rollout 全败；**待验证**是否 (\max_j A_j < \tau_H) |

Hidden Probe 最终应预测：

```text
Â_single = f₁(h_t)
Ĝ_branch = f₂(h_t)
P_handoff = f₃(h_t)
```

而非直接预测 correctness-based behavior state。

---

## 小模型阶段能做什么（无 target）

1. **候选多样性**：token overlap、策略级 diversity、branch width (K)、生成成本
2. **Proxy 假设**：hidden 是否相关于「需要更多候选覆盖」（标签待 target 回填）
3. **构建 target replay dataset**：保留 prefix + 各候选文本/token ids + 生成配置

**不能**声称：Branch Pass@4 证明 speculative latency 收益。

---

## Target Replay 数据契约

每个 prefix 需保存（供离线 target 验证，**无需重跑小模型**）：

```text
prefix_text, prefix_token_ids
decision_hidden, decision_logits (optional layers)

continue_candidate_text, continue_candidate_token_ids
branch_1..K: text, token_ids

sampling_seed, temperature, top_p, generation_config
speculative_block_length γ (future: 32/64/128, not full answer)
```

Target 回填后写入 `target_replay_results.jsonl`。

### Pilot v2 当前缺口

| 字段 | 状态 |
|------|------|
| `continuation` (text) | ✅ actions.jsonl |
| `start_checkpoint` | ✅ |
| `seed`, `temperature` | ✅ branch |
| `prefix_text` | ✅ prefixes.jsonl |
| `hidden.safetensors` | ✅ 后处理导出 |
| `continuation_token_ids` | ❌ 未落盘 |
| `prefix_token_ids` | ❌ 未落盘 |
| `target_accepted_length_*` | ❌ 待 target replay |

---

## 生成预算优化（下一阶段）

目标为 early acceptance 时，Branch **不必**续写到 `\boxed{}`：

[
\gamma \in \{32, 64, 128\} \quad \text{或一个完整 reasoning step}
]

研究 (\mathbb{E}[\max_j A_j]) 随 (K) 和 (\gamma) 的变化。

---

## E2E 效用（最终 oracle）

[
U_{\text{branch}} = \text{Target gen saved} - \text{Extra draft cost} - \text{Extra verification cost}
]

[
a^* = \arg\min_{a \in \{\text{Continue}, \text{Branch}, \text{Handoff}\}} T_{\text{E2E}}(a)
]

接受长度是机制指标；端到端延迟才是最终 action label。

### 验证形式

- **Batched verification**：([s+b_1,\ldots,s+b_K]) 一次 batch
- **Tree verification**（更优）：共享 prefix KV，tree attention 一次验证多条分支

若四条完整序列分别调四次 target，可能 accept 更长但 latency 反增。

---

## Pilot v2 数据如何使用

| 数据 | 用途 |
|------|------|
| 1548 `admission_main` prefix | target replay 主集 |
| 行为状态表 | 探索性分析（降级表述） |
| `branch_pass_at_4` / correctness `branch_gain` | 辅助质量，**不作主结论** |
| 完整续写候选 | 可截断为 (\gamma)-block 后 replay，或先全量 replay 再分析 early accept |

---

## 相关文件

- 框架 v2（小模型标签）：[`action_study_phase_framework.md`](action_study_phase_framework.md)
- Replay schema 代码：`action_study/target_replay_schema.py`
- 准入重算：`action_study/admission_derive.py`

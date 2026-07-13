# V3.4 — Sequential Oracle Policy Rollout 完整报告

> **实验类型**：从 problem prompt 起步的真实 sequential rollout；每步动作（Continue / Branch / Handoff）会改写 prefix，后续推理在更新后的 prefix 上继续。  
> **Oracle**：GPT-5.5 离线逐步评判（V3.3 `gpt_step_oracle_v2` 协议）；**不参与部署延迟统计**。  
> **完成时间**：2026-07-12 18:12 UTC+8  
> **数据目录**：`outputs/action_study_pilot_v34/`

---

## 0. 执行摘要（一句话）

$$\boxed{\text{顺序 rollout 管线跑通了，但当前结果还不能判断 Branch 是否有效。}}$$

本次实验应定义为 **V3.4 pipeline / debug pilot**，而非正式的 mechanism evaluation。V3.3 已在固定 prefix 上证明 Branch-rescuable 状态存在；V3.4 本意是验证这些局部 Branch 能否在 sequential 系统中减少后续 Handoff——**本次 pilot 的数据质量尚未达到能回答该问题的标准**。

**可以说的**：sequential 管线可运行；动作会改写 prefix；记录了 21 次 Branch 事件；双模型常驻稳定。

**不能说的**：「SpecReason 已证明优于 Conditional Branch」；「Branch 没用」；「Branch 产生负向 cascade」——当前 ΔHandoff 与转移矩阵均被工程噪声严重污染。

---

## 1. 实验目标

V3.4 在 V3.3 固定 prefix 逐步 oracle 标签的基础上，将实验推进到更贴近部署的设置：

> **从空 prefix 开始，按 policy 真实执行多步推理，观察 action cascade \(a_t \to p_{t+1} \to a_{t+1}\)。**

**设计意图**（正确）：每步 Continue / Branch / Handoff 产生新 prefix，在新 prefix 上重新判断下一动作。

**本次 pilot 实际交付**：管线跑通 + 工程问题定位；**尚未**达到可回答「Conditional Branch 是否减少 target intervention」的数据质量标准。

具体记录：

1. 五种 policy 在 30 题上的动作分布与终止模式（含污染警示）；
2. Branch 事件是否在 sequential 设置下真实发生；
3. SpecReason vs Conditional Branch 配对比较的**原始数字**（§8，不可作机制结论）；
4. 动作转移矩阵与 L_B、L_H（§7，全 policy 混合，待 per-policy 拆分）。

---

## 2. 实验设置

### 2.1 模型

| 角色 | 模型 | 路径 | 加载方式 |
|------|------|------|----------|
| **Draft（小模型）** | DeepSeek-R1-Distill-Qwen-1.5B | `specreason/models/DeepSeek-R1-Distill-Qwen-1.5B` | vLLM dual-resident |
| **Target（大模型）** | DeepSeek-R1-Distill-Qwen-14B (bf16) | `specreason/models/DeepSeek-R1-Distill-Qwen-14B` | vLLM dual-resident |

- 单卡 H100 80GB，**双模型常驻**（`--dual-resident`），Handoff 时无需卸载/重载 target，避免 V3.4 早期因 QwQ-32B 换模型导致的极端延迟。
- Draft util ≈ 0.45，Target util ≈ 0.42；`max_model_len=4096`，`max_steps=12`。

### 2.2 Oracle

- **模型**：GPT-5.5 @ `endpoint.greatrouter.com`
- **协议**：`gpt_step_oracle_v2`（与 V3.3 一致）
- **SpecReason**：仅评判 greedy 一步 → Continue 或 Handoff
- **Conditional Branch / Always Branch**：评判 greedy + 4 branch → Continue / Branch / Handoff
- **双遍稳定性** + 第三遍 tie-break；不稳定或 API 错误 → 当前实现保守标为 Handoff（`API_ERROR_HANDOFF`）——**这是主要污染来源，V3.4b 须分离为 `ORACLE_API_ERROR`**

### 2.3 数据集与抽样

- 来源：V2 pilot `action_study_pilot_v2/problems.jsonl`（DeepScaler 子集）
- **30 题**，`seed=42` 随机抽样（`input_complete=True`）
- 每题跑 **5 条 rollout**（5 policies × 1 seed）→ **150 rollouts 总计**

### 2.4 五种 Policy

| Policy | 行为摘要 |
|--------|----------|
| **DRAFT_ONLY** | 每步 greedy draft，恒 Continue |
| **TARGET_ONLY** | 每步 target 生成，恒 Handoff |
| **SPECREASON** | Oracle 评判 greedy；不可接受 → Handoff（无 branch） |
| **CONDITIONAL_BRANCH** | Oracle 评判 greedy；不可接受 → 生成 4 branch 再 oracle；仍不可接受 → Handoff |
| **ALWAYS_BRANCH** | 每步生成 4 branch + oracle 五选一（含 branch/handoff 路径） |

### 2.5 终止条件

| 原因 | 含义 |
|------|------|
| `FINAL_ANSWER` | prefix 中出现 `\boxed{}` 终答 |
| `MAX_STEP_TRUNCATED` | 达到 `max_steps=12` |
| `MALFORMED_STEP` | 步质量不合格（draft 常见） |
| `LOOP_DETECTED` | 近几步重复 |
| `PREFIX_UNCHANGED` | Handoff/Branch 后 prefix 未变化（含空步） |

---

## 3. 执行概况

| 指标 | 数值 |
|------|------|
| Rollouts 完成 | **150 / 150** |
| Step 记录 | **1,067** |
| Branch 事件（步级） | **21** |
| 运行时长 | ~4.2 h（13:58–18:12） |
| GPU 模式 | 双常驻 ~41GB（14B + 1.5B），无 per-handoff 换模型 |

**产物文件**

- `rollout_summaries.jsonl` — 150 行，每 rollout 一条
- `rollout_steps.jsonl` — 1,067 行，每步一条
- `gpt_step_oracle_cache.jsonl` — GPT 调用缓存
- `v34_summary.json` — 聚合统计 JSON

---

## 4. 主结果：Policy 级汇总

| Policy | N | Accuracy* | Avg Steps | Avg Continue | Avg Branch | Avg Handoff | Target steps | Proxy latency† |
|--------|--:|----------:|----------:|-------------:|-----------:|------------:|-------------:|---------------:|
| DRAFT_ONLY | 30 | 0.0% | 7.80 | 7.80 | 0.00 | 0.00 | 0.00 | 11.7 |
| TARGET_ONLY | 30 | 0.0% | 6.87 | 0.00 | 0.00 | 6.87 | 6.87 | 65.2 |
| SPECREASON | 30 | 0.0% | 6.80 | 3.20 | 0.00 | 3.60 | 3.60 | 39.0 |
| CONDITIONAL_BRANCH | 30 | 0.0% | 7.23 | 1.40 | 0.70 | 5.13 | 5.13 | 53.7 |
| ALWAYS_BRANCH | 30 | 0.0% | 6.87 | 0.00 | 0.00 | 6.87 | 6.87 | 65.2 |

\* **Accuracy 列全为 0% 的原因见 §6.1**（`extracted_answer` 字段未回填；仅 6 题走到 `FINAL_ANSWER`，其中 5 题 `is_correct=1`）。  
† Proxy latency = 结构延迟估计（T_D=1, T_V=0.5, T_T=8, T_BK=3），**不含 GPT oracle 时间**。

### 4.1 终止原因分布

| Policy | FINAL_ANSWER | MAX_STEP_TRUNCATED | PREFIX_UNCHANGED | MALFORMED_STEP | LOOP_DETECTED |
|--------|-------------:|-------------------:|-----------------:|---------------:|--------------:|
| DRAFT_ONLY | 2 | 11 (37%) | 0 | 12 (40%) | 5 (17%) |
| TARGET_ONLY | 0 | 16 (53%) | 14 (47%) | 0 | 0 |
| SPECREASON | 3 | 14 (47%) | 13 (43%) | 0 | 0 |
| CONDITIONAL_BRANCH | 1 | 16 (53%) | 13 (43%) | 0 | 0 |
| ALWAYS_BRANCH | 0 | 16 (53%) | 14 (47%) | 0 | 0 |

**解读（带污染警示）**

- **DRAFT_ONLY**：1.5B 小模型在 12 步内较难收敛，40% 因 malformed step 提前终止，仅 2 题走到终答。
- **TARGET_ONLY / ALWAYS_BRANCH**：轨迹**完全相同**（均 6.87 handoff/题、0 branch）——见 §6.4；不能将这两行当作有效 policy 对比。
- **SPECREASON vs CONDITIONAL_BRANCH**：表面 handoff 差异见 §8，**当前不可作机制结论**。

### 4.2 可疑结果：TARGET_ONLY ≡ ALWAYS_BRANCH

| 指标 | Target-only | Always Branch |
|------|------------:|--------------:|
| Avg Steps | 6.87 | 6.87 |
| Avg Continue | 0 | 0 |
| Avg Branch | 0 | 0 |
| Avg Handoff | 6.87 | 6.87 |
| Target steps | 6.87 | 6.87 |

Always Branch 理论上应每步生成 4 条 candidate 并可能选 Branch；实际 **30 题 0 次 Branch**。可能原因（非互斥）：

1. **Oracle 不稳定 / API 错误**（53.4%）→ `force_handoff=True` → 每步退化为 Handoff；
2. **1.5B candidate 几乎全部被拒** + greedy 不可接受 → 与 TARGET_ONLY 同轨迹；
3. 实现逻辑上 ALWAYS_BRANCH 在 `force_handoff` 时走 Handoff 路径（见 `sequential_rollout.py`），与 TARGET_ONLY 逐步 Handoff 等价。

**在未清洗 step log 前，不能把 ALWAYS_BRANCH 这一行当作有效实验。**

---

## 5. Oracle 与动作分布（步级）

### 5.1 Oracle 相关策略上的动作

（SPECREASON + CONDITIONAL_BRANCH + ALWAYS_BRANCH，共 627 oracle 步）

| 动作 | 次数 | 占比 |
|------|-----:|-----:|
| HANDOFF | 468 | 74.6% |
| CONTINUE | 138 | 22.0% |
| BRANCH | 21 | 3.4% |

对比 **V3.3 静态 prefix**（eligible & stable）：Continue 88.1% / Branch 5.3% / Handoff 6.6%。

**Sequential 设置下 handoff 率远高于 V3.3**，主要原因：

1. **Prefix 随 rollout 增长**，早期 greedy 步质量差 → oracle 更常拒绝；
2. **53.4% oracle 步带 `API_ERROR_HANDOFF`**（335/627），不稳定时保守 Handoff；
3. **仅 45.9%** oracle 步双遍稳定（288/627），远低于 V3.3 的 94.6%。

### 5.2 Greedy 可接受率

- Greedy acceptable：**138 / 627 = 22.0%**
- 即每步约 78% 概率 oracle 拒绝 greedy，触发 branch 或 handoff 路径

### 5.3 Branch 事件

- 共 **21** 次 Branch，全部来自 **CONDITIONAL_BRANCH**
- 30 题中约 **14 题** 至少发生 1 次 Branch
- Branch 后 oracle 稳定率：多数为 stable（示例：`deepscaler_01027` step 4）

---

## 6. 准确率与终答分析

### 6.1 为何主表 Accuracy = 0%

汇总脚本以 `extracted_answer` 非空 + `is_correct` 计 accuracy；本 run 中 **`extracted_answer` 字段全部为空**（grading 回填未写入 summary），故表内为 0%。

### 6.2 实际终答情况（`FINAL_ANSWER` 终止）

| 题号 | Policy | Steps | is_correct |
|------|--------|------:|:----------:|
| deepscaler_01562 | DRAFT_ONLY | 10 | ✅ |
| deepscaler_01708 | DRAFT_ONLY | — | ❌ |
| deepscaler_01027 | SPECREASON | 9 | ✅ |
| deepscaler_01098 | SPECREASON | 7 | ✅ |
| deepscaler_01604 | SPECREASON | 7 | ✅ |
| deepscaler_01098 | CONDITIONAL_BRANCH | 12 | ✅ |

- 走到终答：**6 / 150 rollouts（4%）**
- 其中正确：**5 / 6**
- **12 步上限**是主要瓶颈：大量 rollout 在到达 `\boxed{}` 前被截断

### 6.3 PREFIX_UNCHANGED 问题（54 rollouts，36%）

Handoff 后 prefix 未变化，导致 rollout 立即终止。典型案例：

- `deepscaler_01034` / SPECREASON：第 0 步 Handoff，`target_step` 与 `selected_step` **长度均为 0**（14B 未产出有效步）

这是本 run **最大的工程问题之一**：target 在 dual-resident + 较短 `max_model_len` 下，部分 handoff 生成为空，直接浪费一步并终止轨迹。由此导致的「Handoff 后几乎不 Continue」「Handoff 连续发生」**不能**解读为策略 cascade。

### 6.4 ALWAYS_BRANCH 退化为 TARGET_ONLY

见 §4.2。Always Branch 在 oracle 步中应产生 branch 候选，但本 run 中 **branch 计数为 0**。与 TARGET_ONLY 逐步固定 Handoff 的轨迹一致，说明该 policy 行**未提供独立信息**，应从机制分析中排除。

---

## 6.5 为何现在不能说 Branch 失败

综合五项污染来源，当前**不能**从表面数字得出「Branch 无效」或「SpecReason 更优」：

| # | 问题 | 规模 | 对结论的影响 |
|---|------|------|-------------|
| 1 | **API 错误被计为 Handoff** | 335/627 = **53.4%** `API_ERROR_HANDOFF` | 74.6% 步级 Handoff 率被严重污染；CondBranch 因更多 API 调用更易触发 |
| 2 | **Target 空步 → PREFIX_UNCHANGED** | **54/150 = 36%** rollout | Handoff 后轨迹异常终止，非策略行为 |
| 3 | **极低正常完成率** | **6/150 = 4%** `FINAL_ANSWER` | Accuracy 全 0% 因 grading 未回填 + 几乎无完整轨迹 |
| 4 | **V3.3 ≠ V3.4 模型配置** | 4B draft + QwQ vs **1.5B + 14B** | Continue 88.1% → 22.0% 不能全归因 sequential cascade |
| 5 | **Oracle 稳定率骤降** | 94.6% → **45.9%** | 超半数 step 无可靠双轮判断，转移矩阵不可信 |

**可以说**：Branch 后 P(Continue|Branch)=47.4%，说明 Branch **有时**能把轨迹带回可继续状态。

**不能说**：Conditional Branch 比 SpecReason 产生更多 Handoff 因此 Branch 失败——更可能是工程错误占主导。

---

## 7. Cascade 指标

### 7.1 动作转移矩阵（步级，**全 policy 混合，不可直接解读**）

| From \ To | Continue | Branch | Handoff |
|-----------|----------|--------|---------|
| **CONTINUE** | 84.3% | 1.8% | 13.9% |
| **BRANCH** | 47.4% | 26.3% | 26.3% |
| **HANDOFF** | 8.1% | 1.1% | 90.8% |

**误读警示**：P(Handoff|Handoff)=90.8% **不**等于「大模型一旦接管就会不断接管」。该数字混入了：

- TARGET_ONLY 固定 H→H→H；
- ALWAYS_BRANCH 退化为逐步 Handoff；
- API error 导致的连续 Handoff；
- target 空输出导致的异常终止。

**修复后应分别报告** SpecReason 与 Conditional Branch 的 per-policy 转移矩阵，且仅在有效完成的 paired rollouts 上计算 cascade 指标。

### 7.2 Branch / Handoff 后 Continue 游程（同样受污染）

| 指标 | 值 |
|------|---:|
| P(Continue \| Branch) | 47.4% |
| P(Handoff \| Branch) | 26.3% |
| P(Branch \| Branch) | 26.3% |
| Mean **L_B**（Branch 后连续 Continue 步数） | 1.10 |
| Median L_B | 0.0 |
| P(L_B ≥ 1) | 42.9% |
| P(L_B ≥ 3) | 14.3% |
| Mean **L_H**（Handoff 后连续 Continue 步数） | 0.15 |

Branch 后偶有短 Continue 游程（P(C|B)=47.4%），但 median L_B=0；Handoff 后几乎不接 Continue——**两者均受 §6.5 污染，暂不作机制解读**。

---

## 8. 表面结果：SpecReason vs Conditional Branch（**不可作机制结论**）

配对 30 题（同题、seed=1）的**原始数字**如下：

| 指标 | SpecReason | Conditional Branch | Δ (Spec − Cond) |
|------|----------:|-------------------:|----------------:|
| 总 Handoff 次数 | 108 | 154 | −46 |
| 平均每题 Handoff | 3.60 | 5.13 | **−1.53** |
| 平均每题 Continue | 3.20 | 1.40 | +1.80 |
| 平均每题 Branch | 0.00 | 0.70 | −0.70 |

Cluster bootstrap：Mean ΔHandoff = **−1.545**（95% CI [−2.800, −0.300]）。

### 8.1 为何现在不能解读为「SpecReason 更优」

上述 ΔHandoff 混合了两类完全不可比的 Handoff：

$$\text{观测 Handoff} = \underbrace{\text{真实 Oracle Handoff}}_{\text{GPT 判定 draft 不可接受}} + \underbrace{\text{API Error Handoff}}_{\text{请求失败 → 保守 Handoff}}$$

- Oracle 步中 **53.4%**（335/627）带 `API_ERROR_HANDOFF`；
- Conditional Branch 每步多 4 条 candidate + 更多 GPT 双轮请求 → **更易触发 API 失败**；
- 36% rollout 因 `PREFIX_UNCHANGED`（target 空步）异常终止。

因此当前更可能解释是：

$$\text{Conditional Branch 管线更容易出错} \;\neq\; \text{Branch 真实负向 cascade}$$

在 API error 率降至 <2%、空步率 <1% 之前，**不能**声称 SpecReason 显著减少 target intervention。

### 8.2 两种待区分的机制假说（修复后才可检验）

**假说 A（真实负向 cascade）**：Branch 找到局部可接受步 → 进入更脆弱路径 → 后续连续 Handoff。

**假说 B（工程假阳性）**：Branch 路径更多 API/解析失败 → 被计为 Handoff → CondBranch 虚高。

当前数据无法区分 A 与 B。

### 8.3 配对 ΔHandoff 分布（原始，未清洗）

| ΔHandoff (Spec − Cond) | 题数 |
|------------------------|-----:|
| 0 | 12 |
| −2 | 3 |
| −7 | 3 |
| −8 | 2 |
| −5 | 2 |
| +1 | 2 |
| 其他 | 6 |

### 8.4 统计检验（未清洗数据，仅供参考）

- Mean ΔHandoff = −1.545（95% CI [−2.800, −0.300]）
- **CI 不含 0 仅说明原始计数有差异，不说明 Branch 机制成立**

---

## 9. 与 V3.3 的关系

| 维度 | V3.3（静态 prefix） | V3.4（sequential） |
|------|---------------------|-------------------|
| 起点 | 固定 prefix（~1548 条） | 空 prompt → 逐步增长 |
| Oracle Continue 率 | 88.1% | 22.0%（步级） |
| Branch 率 | 5.3% | 3.4%（步级） |
| Handoff 率 | 6.6% | 74.6%（步级） |
| Oracle 稳定率 | 94.6% | 45.9% |
| API 错误致 Handoff | 少量 | **53.4%** |

**结论**：V3.3 标签在固定高质量 prefix 上行为良好；**搬到 sequential 后 prefix 质量下降 + API 不稳定**，oracle 行为分布完全不同。二者不可直接混用为训练标签。

---

## 10. Pilot 检查清单（工程 vs 机制）

| 类别 | 标准 | 结果 | 说明 |
|------|------|:----:|------|
| **工程** | Sequential 管线可运行 | ✅ | 150 rollouts、1067 steps 完整记录 |
| **工程** | 双模型常驻无换模崩溃 | ✅ | 14B+1.5B 双常驻稳定 |
| **工程** | Branch 事件可记录 | ✅ | 21 次 Branch（CONDITIONAL_BRANCH） |
| **工程** | Cascade 指标可计算 | ✅ | 转移矩阵、L_B/L_H 已实现 |
| **机制** | E[ΔH] > 0（Branch 减少 handoff） | ⏸️ | 原始 ΔH=−1.53，**数据未清洗，暂停解读** |
| **质量** | API error rate < 2% | ❌ | 实际 **53.4%** |
| **质量** | PREFIX_UNCHANGED < 1% | ❌ | 实际 **36%** |
| **质量** | 正常完成率可比较 accuracy | ❌ | 仅 4% FINAL_ANSWER |

---

## 11. 典型案例

### Case A — SpecReason 成功终答（`deepscaler_01027`）

- 9 步，3 Continue + 6 Handoff，`FINAL_ANSWER`，**is_correct=1**
- 说明在 handoff 有效时，SpecReason 可在步数上限内解出题目

### Case B — Conditional Branch 救援（`deepscaler_01027` step 4）

- Greedy 被拒 → 4 branch 中选一（oracle stable）
- 该题 CondBranch 最终 12 步打满，未在 summary 中标记终答（与 SpecReason 同题对比：SpecReason 9 步完成）

### Case C — 空 Handoff 终止（`deepscaler_01034`）

- SPECREASON 第 0 步即 Handoff，target 产出空串 → `PREFIX_UNCHANGED`
- 代表 36% rollout 的工程失败模式

### Case D — DRAFT_ONLY malformed（12/30 题）

- 1.5B draft 步格式不合格 → `MALFORMED_STEP`
- 小模型单独推理难以维持 V3 步格式约束

---

## 12. 局限与已知问题

1. **样本量小**：30 题 × 1 seed，统计功效有限；CI 宽。
2. **步数上限 12**：大量 `MAX_STEP_TRUNCATED`（49%），压制终答率与 accuracy。
3. **`extracted_answer` 未写入 summary**：主表 accuracy 失真，需修复 grading 回填。
4. **API_ERROR_HANDOFF 过高（53%）**：侵蚀 oracle 质量，需排查 API 稳定性 / 重试 / 超时。
5. **PREFIX_UNCHANGED（36%）**：target 空步问题，需查 14B handoff 生成参数与 `extract_step` 逻辑。
6. **模型配置**：本 run 实际为 **R1-1.5B + R1-14B bf16**；原计划 32B-AWQ 因下载/显存未采用。
7. **GPT 延迟未计入**：proxy latency 仅结构估计；真实部署需加 oracle 耗时。
8. **与 V3.3 分布差异大**：sequential probe 训练标签需单独构建，不能复用 V3.3 静态标签。

---

## 13. 结论

$$\boxed{\text{当前 V3.4 实验被 API、target 空输出和截断问题严重污染，暂时无法评价 Branch。}}$$

1. **V3.4 定性为 pipeline / debug pilot**，不是正式 mechanism evaluation。
2. **Sequential rollout 管线已打通**：150/150 完成；Continue/Branch/Handoff 会改写 prefix；双常驻 14B+1.5B 稳定。
3. **Branch 有时有效**：21 次 Branch 事件中 P(Continue|Branch)=47.4%，说明局部 Branch 能把轨迹带回可继续状态——但这**不等于** sequential 系统中 Branch 能减少 target intervention。
4. **不能声称 SpecReason 优于 Conditional Branch**：ΔHandoff=−1.53 混合了 API error Handoff、空 target step、截断与模型配置变化；CondBranch 管线操作更多，更易触发工程失败。
5. **不能声称 Branch 失败**：表面「CondBranch handoff 更多」更可能来自工程假阳性，而非真实负向 cascade。
6. **V3.3 与 V3.4 不可混用**：静态 prefix 标签（Continue 88.1%）与 sequential 分布（22.0%）差异巨大；probe 训练需基于清洗后的 sequential 数据重新打标。

---

## 14. 建议下一步

### P0（修复后才能做机制评估）

| 动作 | 目标 |
|------|------|
| **分离 API error 与 Handoff** | 标为 `ORACLE_API_ERROR` 并重试；多次失败则标记技术失败、从主比较排除，**不再计入策略 Handoff** | API error < **2%** |
| **修复 target 空步** | 查 14B raw output、stop token、`<STEP_END>`、extraction、`max_tokens`、context 长度 | `PREFIX_UNCHANGED` < **1%** |
| **修复 grading** | `extracted_answer` / `is_correct` 写入 summary；统一 `FINAL_ANSWER` 解析 | Accuracy 表可比较 |

### P1（修复后第一轮重跑）

| 动作 | 说明 |
|------|------|
| 提高 `max_steps` | 20–24（先修空步/循环，否则只增加异常调用） |
| **只跑两条核心 policy** | Sequential SpecReason vs Conditional Branch；暂不跑 DRAFT_ONLY / TARGET_ONLY / ALWAYS_BRANCH |
| **Per-policy 转移矩阵** | 只在有效完成的 paired rollouts 上算 ΔH |
| 扩大样本 | 100+ 题，多 seed |

### 修复后真正该看的 paired 表

| 指标 | SpecReason | Conditional Branch |
|------|----------:|-------------------:|
| 正常完成率 | | |
| Accuracy | | |
| Handoff/题 | | |
| Branch/题 | 0 | |
| Target tokens/题 | | |
| API failure rate | | |
| Empty-step rate | | |

仅在清洗后的 paired rollouts 上：若 mean(ΔH) = H_Spec − H_Cond > 0，才说明 Branch 减少 target intervention。

---

## 附录 A：复现命令

```bash
source /mnt/afs/L202500372/bootstrap/max_speed_env.sh
cd /mnt/afs/L202500372
bash reasoning_branch_dataset/scripts/run_v3_4_full.sh
# 仅重生成报告：
/tmp/vllm-cu124/bin/python -m reasoning_branch_dataset.action_study.run_v3_4_policy_rollout \
  --report-only --out-dir reasoning_branch_dataset/outputs/action_study_pilot_v34 \
  --report-path reasoning_branch_dataset/outputs/pilot_v3_4_report.md
```

## 附录 B：自动生成精简表（原始数字，未清洗）

```
实验类型: pipeline/debug pilot（非 mechanism evaluation）
rollouts: 150 | steps: 1067 | branch events: 21
API_ERROR_HANDOFF: 335/627 (53.4%) | PREFIX_UNCHANGED: 54/150 (36%)
Mean ΔH (SpecReason − CondBranch): -1.533  [不可解读，待清洗]
P(Continue|Branch): 47.4% | Mean L_B: 1.10 | Mean L_H: 0.15
```

---

*报告生成：基于 `rollout_summaries.jsonl`、`rollout_steps.jsonl`、`v34_summary.json` 全量统计。*

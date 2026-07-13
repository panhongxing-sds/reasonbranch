# ReasonBranch

> **Step-level reasoning action study** for SpecReason-style speculative decoding: when should a small **draft** model **Continue**, **Branch** (sample alternatives), or **Handoff** to a large **target** model?

Repository: [github.com/panhongxing-sds/reasonbranch](https://github.com/panhongxing-sds/reasonbranch)

---

## 研究问题

在数学推理的逐步生成中，小模型每一步可以：

| 动作 | 含义 |
|------|------|
| **Continue** | 接受当前 greedy draft 步，追加到 prefix |
| **Branch** | greedy 不可接受，但从多条 draft 候选中选一条可接受步 |
| **Handoff** | draft 路径失败，由 target 大模型生成纠正步 |

核心问题：

1. **Branch-rescuable 状态是否存在？**（局部多候选能否救回轨迹）
2. **Sequential cascade**：\(a_t \to p_{t+1} \to a_{t+1}\) 中，Branch 能否减少后续 Handoff？
3. **能否用本地 probe / verifier 替代 GPT oracle**，实现无 API 部署？

---

## 项目结构

```
reasoning_branch_dataset/
├── action_study/              # 核心实验代码
│   ├── pipeline.py            # V2 数据采集管线
│   ├── gpt_step_oracle.py       # V3.3 GPT 逐步 oracle 协议
│   ├── sequential_rollout.py    # V3.4 顺序 rollout 引擎
│   ├── build_probe_dataset.py   # 两阶段 probe 数据导出
│   ├── train_local_probe.py     # 本地 probe 训练（无 API）
│   ├── build_verifier_dataset.py
│   ├── local_step_verifier.py   # 14B 本地 ACCEPT/REJECT verifier
│   ├── target_step_diagnostic.py
│   └── technical_errors.py      # 技术失败 vs 策略动作分离
├── scripts/                   # 可执行入口（见下方）
├── tests/
├── docs/                      # 设计文档 & 样本
├── data/                      # 小数据集 + 下载说明
└── outputs/                   # 实验报告（入库）+ 大文件（gitignore）
```

**包导入方式**：本目录名是 `reasoning_branch_dataset`，需将**父目录**加入 `PYTHONPATH`：

```bash
export AFS=/path/to/workspace    # 包含 reasoning_branch_dataset/ 的目录
export PYTHONPATH="${AFS}"
```

---

## 已完成工作总览

### Phase V2 — 数据采集与不确定性刻画

- 从 DeepScaler 子集采集 **prefix + 1 greedy + 4 branch** 候选续写
- 记录每 prefix 的 **entropy / margin / diversity** 等 logit 特征
- 产出：`outputs/action_study_pilot_v2/`（problems, prefixes, actions, traces）
- 报告：[`outputs/pilot_v2_report.md`](outputs/pilot_v2_report.md)

### Phase V3 — Utility Oracle（QwQ 0–9 打分，已弃用为新标签源）

- 在固定 prefix 上用 QwQ-32B 对候选步打 utility 分
- 发现 **0–9 绝对分数不稳定**，不再用于新实验
- 报告：[`outputs/pilot_v3_report.md`](outputs/pilot_v3_report.md)

### Phase V3.2 — GPT Pairwise Oracle

- GPT-5.5 成对比较 greedy vs best branch
- 报告：[`outputs/pilot_v3_2_report.md`](outputs/pilot_v3_2_report.md)

### Phase V3.3 — GPT Step Oracle（**当前最重要标签资产**）

在 **固定 prefix** 上，GPT-5.5 独立评判 1 greedy + 4 branch **下一步**是否可接受（`gpt_step_oracle_v2` 协议）。

| 指标 | 数值 |
|------|------|
| Prefix 总数 | 1548 |
| 双遍稳定率 | **94.6%** |
| 有效标签 prefix | **1395** |
| Continue | 1229 (88.1%) |
| Branch | 74 (5.3%) |
| Handoff | 92 (6.6%) |
| Rescue@4（greedy 被拒时） | **44.6%** |
| Candidate-level 标签（稳定 pass） | **~7320** 条 ACCEPT/REJECT |

**结论**：Branch-rescuable 状态真实存在；QwQ weak-branch 与 GPT Branch 一致性差（precision 14.5%），不能替代 GPT 标签。

报告：[`outputs/pilot_v3_3_report.md`](outputs/pilot_v3_3_report.md)

### Phase V3.4 — Sequential Rollout（**管线验证 pilot**）

从 **空 prompt** 开始，真实执行多步 Continue / Branch / Handoff，观察 action cascade。

| 指标 | 数值 |
|------|------|
| Rollouts | 150（30 题 × 5 policies） |
| Steps | 1067 |
| Branch 事件 | 21 |
| 模型 | R1-1.5B draft + R1-14B target，H100 双常驻 ~41GB |

**一句话结论**：

> 顺序 rollout 管线跑通了，但当前结果**还不能**判断 Branch 是否有效。

**污染来源**（导致不能下机制结论）：

| 问题 | 比例 |
|------|------|
| API 错误被计为 Handoff | 53.4% |
| Target 空步 → PREFIX_UNCHANGED | 36% |
| 正常走到 FINAL_ANSWER | 4% |
| Oracle 稳定率（vs V3.3 94.6%） | 45.9% |

报告：[`outputs/pilot_v3_4_report.md`](outputs/pilot_v3_4_report.md)

### Phase V3.4b — 工程修复（**进行中，无需 API**）

| 修复项 | 状态 | 说明 |
|--------|------|------|
| `ORACLE_API_ERROR` 与 Handoff 分离 | ✅ | API 失败不再计入策略 Handoff |
| `TARGET_GENERATION_ERROR` / `STEP_EXTRACTION_ERROR` | ✅ | 技术失败单独标记 |
| R1 `` 块 handoff 提取 | ✅ | `extract_handoff_step()` |
| Grading 回归测试 | ✅ | 50 条 trace，88% 可评分 |
| Target 空步诊断 | ✅ | 100 prefix 采样脚本 |
| 两阶段 local probe | ✅ | 见下表 |
| Verifier 数据集导出 | ✅ | 7320 条 candidate 标签 |
| 本地 verifier zero-shot 评测 | ⏳ | 脚本就绪，待 GPU 跑 |

### Local Probe（V3.3 标签 + V2 logit 特征，**无 API**）

**两阶段设计**（problem-level GroupKFold，防泄漏）：

| Stage | 任务 | 样本 | OOF AUROC | 备注 |
|-------|------|------|-----------|------|
| S1 | Continue vs Intervention | 1395 | 0.676 | intervention recall 0.51 |
| S2 | Branch vs Handoff | 166 | 0.718 | Branch recall 0.64 |

特征：entropy, margin, top1/top2 prob, diversity, prefix 长度, Wait/But marker 等。

产物：`outputs/probe_datasets/`, `outputs/probe_models/`

### Local Verifier 蒸馏（**无 API 部署的关键路径**）

- 从 V3.3 稳定 GPT 标签导出 **7320** 条 `(problem, prefix, candidate) → ACCEPT/REJECT`
- 14B zero-shot 二分类 prompt（`local_step_verifier.py`）
- 目标 gate：action agreement ≥85%，Branch precision ≥70%
- 不达标则 LoRA/SFT

---

## 实验入口

| 阶段 | 命令 | 需要 API |
|------|------|----------|
| V2 数据采集 | `bash scripts/run_batch.sh` | 可选 |
| V3.3 GPT step oracle | `bash scripts/run_v3_3_gpt_step_oracle.sh` | 是 |
| V3.4 sequential rollout | `bash scripts/run_v3_4b.sh` | 是 |
| **本地管线（推荐）** | `bash scripts/run_local_pipeline.sh` | **否** |

### 环境配置

```bash
export AFS=/path/to/workspace
export PYTHONPATH="${AFS}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# API（仅 oracle 实验需要）
cp scripts/env.example .env
# 设置 TEACHER_API_KEY 或 TEACHER_KEYFILE
source scripts/load_api_env.sh
```

### 模型路径（不在 repo 内，需自行下载）

| 角色 | 推荐模型 | 单卡显存（bf16 双常驻） |
|------|----------|-------------------------|
| Draft | DeepSeek-R1-Distill-Qwen-1.5B | ~4 GB |
| Target | DeepSeek-R1-Distill-Qwen-14B | ~28 GB |
| Target（更强） | DeepSeek-R1-Distill-Qwen-32B | ~64 GB |

**显存参考**：

- H100 80GB：14B + 1.5B 双常驻 ✅（~41GB，已验证）
- H100 80GB：32B bf16 + 1.5B ❌ OOM
- A6000 110GB：32B bf16 + 1.5B 双常驻 ✅（预计 ~80–90GB）
- 2× 40GB：推荐 GPU0=32B AWQ/bf16，GPU1=1.5B 分卡

```bash
bash scripts/download_r1_14b.sh
bash scripts/download_r1_32b_awq.sh   # 可选
```

---

## 报告索引

完整报告列表：[`outputs/INDEX.md`](outputs/INDEX.md)

| 版本 | 报告 | 内容 |
|------|------|------|
| V2 | `pilot_v2_report.md` | 数据采集、不确定性 |
| V3 | `pilot_v3_report.md` | QwQ utility（已弃用） |
| V3.2 | `pilot_v3_2_report.md` | GPT pairwise |
| **V3.3** | `pilot_v3_3_report.md` | **GPT step oracle 主标签** |
| **V3.4** | `pilot_v3_4_report.md` | Sequential rollout pilot |
| Reachable state | `reachable_state_report.md` | Target replay 可达性 |

---

## 路线图

当前优先级（**无需 API 直到 local verifier 达标**）：

```
修管线 → 训练 V3.3 local probe → GPT 标签蒸馏本地 verifier → V3.4b local rollout
```

详见 [`docs/ROADMAP.md`](docs/ROADMAP.md)

### 明确不做

- 用污染后的 V3.4 数字下「Branch 无效」结论
- 把 V3.3 静态标签直接当 sequential 训练标签
- 继续用 QwQ 0–9 构造新标签

---

## 测试

```bash
cd "${AFS}"
python -m pytest reasoning_branch_dataset/tests/ -q
```

---

## 大文件与复现

以下**不入库**（见 `.gitignore`），需本地生成或下载：

- `outputs/**/*.jsonl`（实验原始数据）
- `logs/`
- `data/deepscaler_preview.jsonl`（~21MB，见 [`data/README.md`](data/README.md)）
- 模型权重

入库内容：全部 Python 代码、shell 脚本、设计文档、**实验报告 markdown**、小 summary JSON。

---

## 引用与联系

- GitHub: [panhongxing-sds/reasonbranch](https://github.com/panhongxing-sds/reasonbranch)
- 基于 SpecReason 风格的 speculative reasoning + step-level action control

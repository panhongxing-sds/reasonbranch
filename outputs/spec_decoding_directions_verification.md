# Verification-Centric Speculative Decoding — 三方向验证报告

日期: 2026-07-14 · 方法: 文献核实（WebSearch 实证） + 技术正确性分析 + 现有基础设施勘察

## TL;DR 结论表

| 方向 | 新颖性 | 技术风险 | 最先该做的 Audit | 我的建议优先级 |
|---|---|---|---|---|
| ① 低精度预算预测 | **窄但真**（vs Quasar/HierSpec 有明确边界） | **高**：单请求 memory-bound 下缩短 verify 长度≈不省时 | Audit A（两阶段回本） | 2 |
| ② 请求内局部校准 | **最干净、最新** | 中：全押"请求内 residual 时间稳定性" | Audit B（稳定性） | **1（上调）** |
| ③ Certified candidate-only projection | **理论已被抢先**（2606.30265 Cor 4.7） | **最高**：大 target 的 LM-head 占比很小 | Audit C（profile ρ_head） | 3 |

核心修正：**建议把优先级从 1>2>3 调成 2>1>3**。理由见下。

---

## 方向 ① 低精度预算预测（Approximate Budgeting + Exact Verification）

### 新颖性核实
- **Quasar (2603.01399)**：用低比特量化做 verification，但**量化模型本身执行验证**（m2 算法保证 KL 可忽略），BF16 不再介入。1.28× 吞吐。
- **HierSpec / "SD Meets Quantization" (2505.22179)**：4-bit target 下插入一个小的**中间模型**做（树→序列）验证。也是量化模型直接验证。
- **SGLang / SpecDec++**：动态调 speculation **length**（draft 侧），lossless。

→ 你的边界成立且清晰：**低精度模型不做最终验证，只预测/收缩 BF16 精确验证的长度或候选集，BF16 保留最终权威**。这与 Quasar/HierSpec（量化模型即验证器）确实不同。

### ⚠️ 技术正确性的致命疑点（必须先证伪）
标准 SD 里验证 γ 个 token 是**一次并行 forward**，其 wall-clock 由**权重加载（memory-bound）**主导，几乎与位置数 γ 无关。因此：
- "只让 BF16 精确验证前 r̂+s 个"**并不能缩短那一次 forward**（一趟就是一趟）；
- 反而多付了一趟 INT4 forward。

即 `T_INT4(γ) + T_BF16(r̂)` ≈ `0.3~0.5·T_BF16 + T_BF16` = **1.3~1.5·T_BF16 > T_BF16**，单请求下**大概率变慢**。

**唯一可能回本的 regime**：compute-bound（大 batch / 长 γ / 树验证），此时减少精确验证的**位置数或候选数**才真省 FLOPs。所以：
- Audit A **必须在 compute-bound 设定下也测**（大 batch 或 tree），否则纯线性单请求几乎注定 <5% 甚至负收益。
- 更稳的价值不是"缩短一趟 BF16"，而是"**用 INT4 预测该 draft 多长 / 该验证多少候选**"来减少 rollback 浪费——这更接近 SpecDec++ 的 length 自适应，需要把差异化讲清楚。

**判词**：新颖性 OK，但**可行性存疑**，Audit A 的 kill gate（≥5% 真 wall-clock）很可能在线性单请求上失败。

---

## 方向 ② 请求内局部校准（Request-Local Residual Calibration）

### 新颖性核实（这是三者里最干净的）
- **OSD (2310.07177)**：在线 KD，**梯度**微调 draft，**跨请求持久**。
- **TIDE (2602.05145)**：serving-engine 原生在线适配，复用 hidden states，但仍**更新 draft 参数**、**持久**、吃跨请求 workload locality。
- **OnlineSpec / "When Drafts Evolve" (2603.12617)**：在线学习 + dynamic regret + 历史梯度复用 + ensemble。
- **EvoSpec**：在线 LoRA + 动态词表。

→ 全部是**跨请求 + 梯度 + 持久参数**。你的定义——**per-request、无梯度、无持久参数、闭式/EMA 局部校准，请求结束即删**——是设计空间里**明确未被占据**的点。新颖性最强。

### 技术判断
- 第一版用 residual EMA / token-bias / ridge，不需要神经网络，实现最轻。
- **唯一的 Gate 你已经点中**：单请求内部的 draft–target residual 是否具**时间稳定性**（前 25% 学到的偏差能改善后 75%）。这不是"能不能拟合"，而是"有没有可迁移的稳定结构"。
- 直觉支撑：长生成（代码/推理链）内确有重复的局部偏好（变量名、缩进、API、格式），reasoning 链尤其如此——与我们 V4.0 观察到的 draft 系统性偏差一致。

**判词**：**最推荐先做**。新颖边界清晰、实现最轻、Audit B 设计正确且能快速证伪。建议上调为 #1。

---

## 方向 ③ Certified candidate-only projection

### 新颖性核实（⚠️ 理论已被抢先）
- **NanoSpec (2605.26444)** / **DynaSpec (2510.13847)** / **FR-Spec**：都在**优化 drafter 的 LM-head**（小模型大词表，head 占比高），**verification 仍走全词表**。与你的 target 侧不同——这点你说得对。
- **但 "When Is a Draft Accepted?" (2606.30265)** 已经系统给出**greedy 接受的 exact KL 证书 + sharp margin bounds**（Cor 4.7 "Exact KL certificate for greedy agreement"：draft token 是否等于 target argmax，由 margin 决定），并在 **Qwen3** 上评测。这**正是你"证明 candidate 是 top-1"的理论**。

→ 你的理论新意（margin/argmax 证书）**基本被 2606.30265 覆盖**。剩下的差异只能收缩为**计算/系统实现**：用 cluster 上界跳过 **target LM-head 的整块 GEMM**（该论文只做理论与统计，不做 kernel 级跳算）。

### ⚠️ 技术正确性的致命疑点
- NanoSpec 之所以有效，是因为它砍的是**小 drafter 的 head**（head 在小模型里占比大，砍它省 51.6% draft 时间）。
- 但你要砍的是 **32B target 的 head**——32B 的 transformer 层极重，**LM-head 只占 target 前向的很小一块**。所以 `ρ_head = T_lm-head / T_verification` 很可能 <10%，Audit C 会**直接判死**（与你自己的担心一致）。
- 即便 ρ_head 不小，GPU 上一次 dense GEMM 常比 cluster 检索 + 稀疏 gather 更快——kernel 效率会再补一刀。

**判词**：**理论被抢先 + 系统收益对大 target 很可能为负**。风险最高、期望回报最低。Audit C（纯 profile）应作为**最便宜的一票否决**先跑。

---

## 现有基础设施（Audit 可行性）

`reasonv4/umbrella/` 是一套完整 SD 引擎，可复用：
- `engine/static_speculation_engine.py`、`dynamic_speculation_engine.py`、`ar_engine.py`（AR baseline）
- `quantization/awq_utils.py`、`fbgemm_utils.py`（INT4/W8A8 量化，方向①要用）
- `models/qwen.py` 等 + `attn/cache.py`（KV cache）
- `policies/tree_policy.py`、growmap（树/线性 draft 结构）

注意：`static_speculation_engine` 实为 growmap 驱动（含树）。做**纯线性 block SD** 的干净 wall-clock 测量，建议**另起一个精简 HF/vLLM micro-harness**（固定 γ∈{4,8,16}，只测 draft-forward / target-verify-forward / lm-head 三段计时），比改造 umbrella 更快更可控；量化部分可借 `awq_utils`。

---

## 建议的 Audit 执行顺序（便宜的一票否决优先）

1. **Audit C（半天，纯 profile，最先跑）**：加载 32B（AWQ 与 BF16 各测），量 `ρ_head = T_lm-head / T_verify` 与 draft 侧对照。<10% → 方向③直接封存。**成本最低、否决力最强。**
2. **Audit A（1 天）**：线性 SD 下测 `T_INT4(γ)+T_BF16(r̂)` vs `T_BF16(γ)`，**必须同时测大 batch / tree（compute-bound）**，否则单请求几乎注定失败。kill gate ≥5% 真 wall-clock 且 exact output。
3. **Audit B（1–2 天，最推荐投入）**：单请求前半段 verification feedback 校准（EMA/ridge），测后半段的 target top-1 coverage / acceptance length / KL 改善。kill gate：跨位置改善稳定（非训练位置过拟合）。

> Tree 确实可完全拿掉；三方向都先在线性 block SD 上验证，某机制确证后再看能否自然扩到 tree。

# V3.6 — One-Step Counterfactual & Verification Gap

## 1. 问题

在 greedy-rejected 状态上，做一步反事实：

- **TH**：Direct Handoff（32B 生成）  
- **TB(K)**：Branch@K（1.5B 候选 + 32B logit Accept/Reject verifier）

并用离线 Oracle（DeepSeek-V4-Pro）标注 step 是否“数学正确、一致、有实质进展、可安全追加”。

核心科学问题：**32B 作为 step verifier，在真实 1.5B 候选分布上有没有判别力？**

## 2. 方法

### 2.1 Verifier
Prompt stem + candidate，对 ` Accept` / ` Reject`（单 token）做 paired `prompt_logprobs`，分数：

```text
score = logP(Accept) − logP(Reject)
```

### 2.2 Oracle
离线 API 语义标注；保留 `None`（未知），不强制转 False。

### 2.3 数据与 τ 标定
- 收集 rejected states → 候选池 → oracle 标注  
- τ 扫描：precision / coverage / AUC

## 3. 关键结果

### 3.1 技术 sanity
手工构造明显对/错：verifier **AUC ≈ 0.97** —— 接口与评分逻辑正确。

### 3.2 真实 1.5B 候选分布
- 全 τ 最大 precision ≈ **7%**  
- AUC ≈ **0.5**（无判别力）  
- 负例分数常高于正例  

### 3.3 定性
Oracle 大量拒绝“看起来像推理”的复述 / 无实质进展步；verifier 仍给高分 → **plausibility ≠ progress**。

工程修复（smoke3）：种子塌缩、候选双重生成、K=1 温度不一致等，详见 `outputs/v36_smoke_review.md`。

## 4. 判决

**Verification Gap 成立：Target-as-step-verifier 主线死亡。**

32B 能分清“明显对错”，但分不清“真实 draft 分布上的好坏 step”。ConfSpec 的“verification 是廉价可靠判别任务”前提在此被反驳。

## 5. 对后续的含义

催生 V4.0：放弃跨模型 step 验证，改用 **draft 自信号 + abstention**。

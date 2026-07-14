# V4.0 — Draft-Confidence Selective Speculative Reasoning

## 1. 一句话

用 **draft 自身置信信号** 决定“免费接受本步 / handoff 给 32B”，不做 Branch，不做 target-as-verifier；用 conformal/LTT 框架追求接受集上的精度保证。

## 2. 动机

| 前序发现 | 含义 |
|---|---|
| V3.5 Branch@4 ≈ 1.06×Handoff | 验证不便宜 |
| V3.6 verifier AUC≈0.5 on real drafts | 跨模型 step 验证不可靠 |

→ 主线切换：**近零成本的 draft-intrinsic signals + abstention**。

## 3. 方法（完整）

### 3.1 系统循环（语义 step 级）

每步：
1. Draft（1.5B）greedy 生成下一步 `s_d`  
2. 提取自信号 → 融合分数 `s(s_d)`  
3. `s ≥ τ` → 接受 `s_d`（不调 target）  
4. 否则 Handoff：Target（32B）生成 `s_t`  
5. 直到 `\boxed{}` 或步数上限

### 3.2 Draft 自信号（teacher-forcing）

把 `prefix+candidate` 喂 draft，读 `prompt_logprobs`：

| 信号 | 含义 |
|---|---|
| mean/min/last logprob, perplexity | 流畅度 |
| mean/max entropy, mean/min margin | 预测尖锐度 |
| repetition_rate | 退化复述 |
| **self_eval_logit** | 拼 “Is this step correct/useful? yes/no”，logit(yes)−logit(no) |

实现：`action_study/v40_self_signals.py`（解析与 vLLM 分离，有单测）。

### 3.3 融合 + 阈值

- Logistic 融合（draft-only 特征；**不含** verifier_score）  
- Bonferroni-LTT + Clopper-Pearson UCB → `P(bad|accepted)≤ε`  
- 认证数据不足时用经验 `target_precision` 操作点  

实现：`v40_conformal_gate.py`。

### 3.4 E2E 四策略

`target_only` / `draft_only` / `selfconf`(本方法) / `target_verify`(ConfSpec 式反面基线)。

## 4. Phase 0 De-risk

**数据**: 179 条 oracle 标注（GSM8K 抬高正例率 + AIME calib）。

| 信号 | AUC |
|---|--:|
| **self_eval_logit** | **0.844** |
| fused draft-only OOF | 0.833 |
| 32B verifier | 0.667 |

per-dataset：GSM8K self_eval 0.828 vs verifier 0.599；aime_calib 0.659 vs 0.356。

**判决: GREEN**（进 Phase 1）。

## 5. Phase 1 E2E（GSM8K）

### tp=0.85（保守）
selfconf 准确率 1.00，但 **0.70×**（比 target 慢）—— handoff 过多。

### tp=0.70（提高接受）

| 策略 | 准确率 | speedup | 接受率 | 验证开销 |
|---|--:|--:|--:|--:|
| target_only | 1.00 | 1.00x | — | 0 |
| draft_only | 0.889 | 1.90x | 100% | 0 |
| **selfconf** | **1.00** | **1.07x** | 58.6% | **1.85s** |
| target_verify | 0.778 | 1.29x | 92.5% | 4.19s |

要点：
- selfconf **追平 target 准确率**  
- target_verify 几乎全接受 → 准确率最差（verification gap 的 E2E 铁证）  
- 加速温和且强依赖 τ  

## 6. 局限

1. Speedup 非强项（1.07x）  
2. Conformal 高精度认证数据饥渴  
3. AIME E2E 因长链不可用  
4. 样本偏小  

## 7. 产物路径

- 代码：`v40_*.py`、`run_v40_*.py`  
- 数据：`outputs/action_study_v40_derisk/`、`action_study_v40_e2e{,_tp70}/`  
- 单测：`tests/test_v40_*.py`（11 passed）  

## 8. 判决

**方法成立（判别力 + 准确率保护）；系统加速叙事弱。**  
可继续工程优化，不宜夸大 speedup。

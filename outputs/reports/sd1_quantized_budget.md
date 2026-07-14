# SD① — Quantized Budget Predictor（Approximate Budgeting + Exact Verification）

## 1. 想法

Draft 生成长度 γ 的 block 后，传统做法让 BF16 target 一次验证全部 γ。

本方向：先用**低精度 target** 估计可能接受边界 \(\hat r_q\)，再只让 BF16 精确验证前 \(\hat r_q+s\) 个 token。

```text
Draft block γ=16
    → INT4/低精度估计前 5 个可能有效
    → BF16 只精确验证前 7 个（+安全余量）
```

最终 Accept/Reject 仍由 BF16 决定 → **输出分布不变**（低估只少接受；高估由 BF16 正常拒）。

差异化边界（vs Quasar / HierSpec）：低精度模型**不执行最终验证**，只做预算预测。

## 2. 新颖性核实

| 工作 | 做法 | 与本方向 |
|---|---|---|
| Quasar | 量化模型**直接做 verification** | 不同：我们保留 BF16 权威 |
| HierSpec | 中间小模型验证树→序列 | 不同：量化/小模型即验证器 |
| SpecDec++ / SGLang | 调 draft 侧 speculative length | 相近精神，但不是“低精度预验证长度” |

→ 新颖性**窄但成立**。

## 3. 预先风险

验证 γ 个 token 通常是**一次 memory-bound 并行 forward**，wall-clock 几乎与 γ 无关。缩短精确验证长度**不一定省时**，却多付一趟低精度 forward → 单请求可能变慢。

## 4. Audit A（执行）

设定：线性 SD，γ∈{4,8,16}；用 32B 前向做“全长 verify” vs “全长扫描 + 短 verify”（同骨干上的长度缩放代理）。

| γ | baseline (s) | twostage (s) | savings | mean \(\hat r\) |
|--:|--:|--:|--:|--:|
| 4 | 0.061 | 0.122 | **−100%** | 1.5 |
| 8 | 0.061 | 0.122 | **−99%** | 2.5 |
| 16 | 0.062 | 0.123 | **−99%** | 2.5 |

Kill gate：≥5% wall-clock 节省 → **未达到**。

代码：`action_study/sd_audit/audit_a_budget.py`  
原始：`outputs/sd_audit_a.json`

## 5. 判决

**FAIL / 封存（当前单请求线性设定）**

两阶段 ≈ 2× 单次 verify，符合 memory-bound 预期。唯一可能回本 regime：大 batch / tree / 真 compute-bound；未测前不继续。

## 6. 诚实备注

本 Audit 用同一 32B 骨干做“approx+exact”长度代理，**不是**真 INT4+BF16 异构。若上真 INT4，常数因子可能改善，但单请求下仍难抵消“多一趟全长扫描”。

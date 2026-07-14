# ReasonBranch

> **为 DeepSeek-R1-Distill-Qwen-32B 训 EAGLE-3 推理草稿 head,并在 vLLM 上实现无损加速。**

Repository: [github.com/panhongxing-sds/reasonbranch](https://github.com/panhongxing-sds/reasonbranch)

---

## 项目现状(先看这个)

**一句话:有真加速(2.21×),但还没有自己的 method。**

### 确定有的(真东西)

| 进展 | 状态 | 说明 |
|---|---|---|
| **真加速 2.21×** | ✅ 已验证 | 32B 自生成 21 tok/s → EAGLE-3 head **47 tok/s**,vLLM 同引擎可复现 |
| 超过 UMbreLLa 树基线 | ✅ | 绝对 tok/s 47 > 40(树 SD 2.23×),尽管接受长度更低 |
| 首个 Qwen-32B EAGLE-3 推理 head | ✅ | epoch6 收敛,严重欠训(1600 条/2048 截断),headroom 大 |

**但要诚实**:这是 **EAGLE-3(别人已有的方法)** 应用到一个尚无公开 head 的 32B 推理模型。
价值 = 工程 + 首个可用 head,**不是**"发明了一个新算法"。

### 还没有的(顶会 method 所需)

| 目标 | 状态 |
|---|---|
| **自己的**、能发顶会的 method | ❌ 还没有 |
| 比 vanilla EAGLE-3 **更快**的新机制 | ❌ 还没证明 |
| 3×+ 稳定加速(官方 8B head 参考 3.24×) | ❌ 目前 2.21×,head 欠训 |

### 方法候选(有信号,未落地)

**Confidence-Gated Adaptive-Depth**(置信度自适应草稿深度):
- draft 置信度预测接受 AUC **0.86**,校准干净(prob>0.95 → accept 0.97) → **信号是真的**
- **还没**写成能跑的解码器,**还没**证明比 vanilla EAGLE-3 更快
- 当前欠训 head 是"短跑者"(几乎无长 run),前提暂不成立

→ 不能当主结果,不能当论文 method。详见 [`method_confidence_gated_adaptive_depth.md`](outputs/reports/method_confidence_gated_adaptive_depth.md)

### 进度

```
[████████░░░░░░░░░░░░] ~40%

✅ 验证侧全灭,搞清楚了什么路走不通
✅ 转到 drafter 侧,训出可用 EAGLE-3 head,2.21× 真加速
⬜ 训更强的 head(预期 3×+,工程)
⬜ 实现 method(置信自适应深度等)
⬜ 证明 method 跑赢 vanilla EAGLE-3 → 这才算顶会 story
```

**下一步唯一有意义的路径**:训强 head → 在上面实现并 benchmark method → **必须超过 vanilla EAGLE-3 固定链**。

---

## 当前主结果

| 配置 | 引擎 | tok/s | speedup |
|---|---|---|---|
| target-only AR 32B | vLLM | 21.14 | 1.0× |
| **自训 EAGLE-3 head (k=5)** | vLLM | **46.69** | **2.21×** |
| 动态树 SD (1.5B→32B) | UMbreLLa | 40.6 | 2.23× |

详细报告: [`outputs/reports/eagle3_drafter_pivot.md`](outputs/reports/eagle3_drafter_pivot.md)

---

## 快速开始

```bash
pip install -r requirements.txt
export VLLM_USE_FLASHINFER_SAMPLER=0   # Blackwell 必须

# EAGLE-3 head benchmark
bash eagle3/scripts/run_bench.sh

# target-only 基线
bash eagle3/scripts/run_bench.sh ar-only
```

完整说明: [`eagle3/README.md`](eagle3/README.md)

---

## 项目结构

```
reasonbranch/
├── eagle3/                  # ★ 当前主线代码
│   ├── bench/               #   vLLM benchmark
│   ├── train/               #   语料生成 + draft 配置
│   ├── method/              #   方法候选 de-risk
│   └── scripts/             #   一键入口
├── outputs/reports/         # ★ 研究报告
├── archive/                 # 早期 V2–V4/SD 代码(已证伪,归档)
└── data/                    # 小数据集
```

---

## 报告

| 报告 | 内容 |
|---|---|
| [`eagle3_drafter_pivot.md`](outputs/reports/eagle3_drafter_pivot.md) | **主结果** 2.21× |
| [`method_confidence_gated_adaptive_depth.md`](outputs/reports/method_confidence_gated_adaptive_depth.md) | 方法候选 |
| [`99_next_directions.md`](outputs/reports/99_next_directions.md) | 下一步 |

索引: [`outputs/reports/README.md`](outputs/reports/README.md)

早期验证侧探索(V3.5–V4.0 / SD①–④)代码在 [`archive/`](archive/),报告在 `outputs/reports/`。

---

## 模型与产物(不入 git)

| 类型 | 路径 |
|---|---|
| Target 32B | 自行下载 DeepSeek-R1-Distill-Qwen-32B |
| 自训 EAGLE head | `SpecForge/outputs/.../epoch_6_step_10000/` (~1.5G) |
| 训练语料 | `reasonv4/data/cot_corpus.jsonl` |

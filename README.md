# ReasonBranch

> **为 DeepSeek-R1-Distill-Qwen-32B 训 EAGLE-3 推理草稿 head,并在 vLLM 上实现无损加速。**

Repository: [github.com/panhongxing-sds/reasonbranch](https://github.com/panhongxing-sds/reasonbranch)

---

## 当前主结果

在验证侧加速全部证伪后,项目转向 **drafter 侧**:用 EAGLE-3 抬高单步接受率 α,减少昂贵的 target 前向次数。

| 配置 | 引擎 | tok/s | TPOT | 平均接受长度 | speedup |
|---|---|---|---|---|---|
| target-only AR 32B | vLLM | **21.14** | 47.3 ms | 1.0 | 1.0× |
| **自训 EAGLE-3 head (k=5)** | vLLM | **46.69** | 21.4 ms | 2.36 | **2.21×** |
| 动态树 SD (1.5B→32B) | UMbreLLa | 40.6 | 24.6 ms | 4.84 | 2.23× |
| 官方 EAGLE-3 head (8B 参考) | vLLM | 282.4 | 3.54 ms | — | 3.24× |

**要点**:
- 这是 **DeepSeek-R1-Distill-Qwen-32B 上据我们所知首个 EAGLE-3 推理 head**。
- vLLM 同引擎 **2.21× 无损加速**,绝对 tok/s **超过** UMbreLLa 树基线(40.6),尽管接受长度更低(2.36 vs 4.84)——EAGLE head 草稿成本极低。
- head **严重欠训**(1600 条语料 / max_len 2048 / 7 有效 epoch),接受长度还有很大 headroom;更强 head 预期 3×+。

详细报告: [`outputs/reports/eagle3_drafter_pivot.md`](outputs/reports/eagle3_drafter_pivot.md)

---

## 方法候选(进行中)

**Confidence-Gated Adaptive-Depth Speculative Decoding** —— 按校准置信度动态延伸草稿深度:高置信往深挖(塌缩多次 target 前向),置信一掉立刻收。

de-risk 信号(完整 1.5B draft 上):
- draft 置信度预测接受 AUC **0.86**,prob>0.95 时接受率 **0.97**
- 20% 的块吃满 cap = 长易跨度真实存在

但在当前欠训 EAGLE head 上前提暂不成立(head 是"短跑者",几乎无长 run)。**下一步:训强 head → 验证长 run → 落解码循环测 tok/s vs vanilla EAGLE-3。**

报告: [`outputs/reports/method_confidence_gated_adaptive_depth.md`](outputs/reports/method_confidence_gated_adaptive_depth.md)  
方向规划: [`outputs/reports/99_next_directions.md`](outputs/reports/99_next_directions.md)

---

## 复现

### 环境

```bash
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export VLLM_USE_FLASHINFER_SAMPLER=0   # Blackwell(sm120) 必须
```

### Benchmark(主结果)

```bash
python3 eagle3/vllm_eagle_bench.py \
  --base /path/to/DeepSeek-R1-Distill-Qwen-32B \
  --eagle /path/to/epoch_6_step_10000 \
  --n 15 --max-tokens 512 --spec-tokens 5
```

去掉 `--eagle` 即 target-only 基线。脚本与训练说明见 [`eagle3/README.md`](eagle3/README.md)。

### 训练 head(三阶段)

| 阶段 | 内容 | 状态 |
|---|---|---|
| A | vLLM EAGLE-3 栈验证(官方 8B head → 3.24×) | ✅ |
| B | 32B 自蒸馏数学 CoT 语料(1600 条) | ✅ |
| C | SpecForge 训 Qwen-32B EAGLE-3 head | ✅ epoch6 收敛 |

训练产物(head 不入 git,~1.5G):`SpecForge/outputs/r1-qwen-32b-eagle3-math/epoch_6_step_10000/`

---

## 项目结构

```
reasonbranch/
├── eagle3/                  # ★ EAGLE-3 训练 + benchmark 脚本
│   ├── vllm_eagle_bench.py
│   ├── gen_cot_corpus.py
│   └── README.md
├── outputs/reports/         # ★ 研究报告(从这里读)
│   ├── eagle3_drafter_pivot.md          # 主结果
│   ├── method_confidence_gated_adaptive_depth.md
│   ├── 99_next_directions.md
│   └── README.md                        # 报告索引
├── action_study/sd_audit/ # de-risk 分析脚本
└── action_study/          # 历史实验代码(归档,非主线)
```

---

## 报告索引

**从这里读**: [`outputs/reports/README.md`](outputs/reports/README.md)

| 报告 | 内容 |
|---|---|
| [`eagle3_drafter_pivot.md`](outputs/reports/eagle3_drafter_pivot.md) | **主结果**:阶段 A/B/C + 2.21× + 短跑者/马拉松者 |
| [`method_confidence_gated_adaptive_depth.md`](outputs/reports/method_confidence_gated_adaptive_depth.md) | 方法候选 de-risk |
| [`99_next_directions.md`](outputs/reports/99_next_directions.md) | 下一步方向 + 怎么思考 |

早期 V3.5–V4.0 / SD①–④ 验证侧探索(均已证伪)的完整记录保留在 `outputs/reports/` 和 `action_study/`,不作为当前主线。

---

## 模型路径(不在 repo 内)

| 角色 | 模型 | 显存(bf16) |
|---|---|---|
| Target | DeepSeek-R1-Distill-Qwen-32B | ~64 GB |
| Draft(EAGLE head) | 自训 1 层 Llama head | ~1.5 GB |

---

## 大文件

以下不入库(见 `.gitignore`),需本地生成:

| 类型 | 说明 |
|---|---|
| head 权重 | `model.safetensors` ~1.5G |
| 训练语料 | `cot_corpus.jsonl` |
| 实验原始数据 | `outputs/**/*.jsonl` |
| 密钥 | `key/`, `.env` |

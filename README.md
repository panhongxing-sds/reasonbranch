# ReasonBranch

> **为 DeepSeek-R1-Distill-Qwen-32B 训 EAGLE-3 推理草稿 head,并在 vLLM 上实现无损加速。**

Repository: [github.com/panhongxing-sds/reasonbranch](https://github.com/panhongxing-sds/reasonbranch)

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

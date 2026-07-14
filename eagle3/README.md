# EAGLE-3 — DeepSeek-R1-Distill-Qwen-32B 推理加速

主结果报告: [`../outputs/reports/eagle3_drafter_pivot.md`](../outputs/reports/eagle3_drafter_pivot.md)

## 目录

```
eagle3/
├── bench/           # 评测
│   └── vllm_eagle_bench.py
├── train/           # 语料 + head 训练配置
│   ├── gen_cot_corpus.py
│   ├── convert_cot_corpus.py
│   └── configs/r1-distill-qwen-32b-eagle3.json
├── method/          # 方法候选 de-risk
│   └── adaptive_depth_derisk.py
└── scripts/
    └── run_bench.sh
```

## 快速开始

### Benchmark(主结果 2.21×)

```bash
# EAGLE-3 head
bash eagle3/scripts/run_bench.sh

# target-only 基线
bash eagle3/scripts/run_bench.sh ar-only
```

或直接:

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
python3 eagle3/bench/vllm_eagle_bench.py \
  --base /path/to/DeepSeek-R1-Distill-Qwen-32B \
  --eagle /path/to/epoch_6_step_10000 \
  --n 15 --max-tokens 512 --spec-tokens 5
```

### 阶段 B:生成训练语料

```bash
python3 eagle3/train/gen_cot_corpus.py \
  --sources gsm8k aime math --limit 1600 \
  --out /path/to/cot_corpus.jsonl
python3 eagle3/train/convert_cot_corpus.py \
  --in /path/to/cot_corpus.jsonl --out /path/to/r1_math_cot.jsonl
```

### 阶段 C:SpecForge 训练 head

在 [SpecForge](https://github.com/sgl-project/SpecForge) 仓库中:

```bash
torchrun --standalone --nproc_per_node 1 scripts/train_eagle3.py \
  --target-model-path /path/to/DeepSeek-R1-Distill-Qwen-32B \
  --draft-model-config /path/to/eagle3/train/configs/r1-distill-qwen-32b-eagle3.json \
  --train-data-path cache/dataset/r1_math_cot.jsonl \
  --chat-template deepseek-r1-distill --target-model-backend hf \
  --attention-backend sdpa --embedding-key model.embed_tokens.weight \
  --max-length 2048 --num-epochs 10 --batch-size 1
```

**SpecForge 必要补丁**(否则 Qwen-32B 训不通):
1. `specforge/data/template.py`: `deepseek-r1-distill` 的 `end_of_turn_token` 改为 `"<｜end▁of▁sentence｜>"`
2. `specforge/args.py`: `sglang` import 包 try/except
3. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; `max_length=2048`(4096 OOM)

### 方法 de-risk:自适应深度

```bash
python3 eagle3/method/adaptive_depth_derisk.py \
  --tokens ../outputs/vsignal/tokens.jsonl
```

## 主结果(vLLM, batch=1, MATH-500)

| 配置 | tok/s | speedup |
|---|---|---|
| target-only AR 32B | 21.14 | 1.0× |
| 自训 EAGLE-3 head (k=5) | **46.69** | **2.21×** |

head 权重(~1.5G)不入 git,本地路径示例:
`SpecForge/outputs/r1-qwen-32b-eagle3-math/epoch_6_step_10000/`

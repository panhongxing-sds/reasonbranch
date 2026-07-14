# EAGLE-3 Drafter for DeepSeek-R1-Distill-Qwen-32B

当前主结果的代码(报告见 `../outputs/reports/eagle3_drafter_pivot.md`)。

## 文件
- `gen_cot_corpus.py` —— 阶段B:32B 自蒸馏数学 CoT 语料(ShareGPT 格式)。
- `convert_cot_corpus.py` —— ShareGPT `{from,value}` → SpecForge `{role,content}`。
- `r1-distill-qwen-32b-eagle3.json` —— SpecForge draft(EAGLE-3 head)配置,单层 Llama,hidden 5120,draft_vocab 32000。
- `vllm_eagle_bench.py` —— vLLM EAGLE-3 vs target-only 基准(单请求计时 + 接受率)。

## 训练(SpecForge,HF 后端)
```bash
torchrun --standalone --nproc_per_node 1 scripts/train_eagle3.py \
  --target-model-path .../DeepSeek-R1-Distill-Qwen-32B \
  --draft-model-config configs/r1-distill-qwen-32b-eagle3.json \
  --train-data-path cache/dataset/r1_math_cot.jsonl \
  --chat-template deepseek-r1-distill --target-model-backend hf --attention-backend sdpa \
  --embedding-key model.embed_tokens.weight --max-length 2048 --num-epochs 10
```

## SpecForge 必要补丁(否则跑不通)
1. `specforge/data/template.py` 的 `deepseek-r1-distill` 模板:`end_of_turn_token=None`
   → 改成 `"<｜end▁of▁sentence｜>"`(否则 `re.escape(None)` 崩)。
2. `specforge/args.py`:把 `sglang` import 包 try/except(HF 后端无需 sglang)。
3. 显存:`max_length` 用 2048(4096 会 OOM),`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

## 产出 head(不入 git,太大)
`SpecForge/outputs/r1-qwen-32b-eagle3-math/epoch_6_step_10000/`(config.json + model.safetensors 1.5G)

## 关键结果
vLLM 同引擎 batch=1 MATH-500:target-only 21.1 tok/s → EAGLE-3 head **46.7 tok/s(2.21×)**,accept 2.36。

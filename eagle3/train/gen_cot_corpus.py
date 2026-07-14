#!/usr/bin/env python3
"""Stage B: generate target (32B) self-distilled math CoT traces in ShareGPT format
for EAGLE-3 training. Uses vLLM for throughput.

Output: jsonl with {"id":..., "conversations":[{"from":"human","value":problem},
                                                {"from":"gpt","value":solution}]}
"""
import argparse, json, os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def load_problems(sources, limit):
    probs = []
    for src in sources:
        if src == "gsm8k":
            p = "/root/autodl-tmp/reasonbranch/data/gsm8k_test.jsonl"
            for l in open(p):
                r = json.loads(l)
                probs.append(r["question"])
        elif src == "aime":
            p = "/root/autodl-tmp/reasonbranch/data/aime_train.jsonl"
            for l in open(p):
                probs.append(json.loads(l)["problem"])
        elif src == "math":
            from datasets import load_dataset
            ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
            probs += [r["problem"] for r in ds]
    if limit:
        probs = probs[:limit]
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/autodl-tmp/specreason/models/DeepSeek-R1-Distill-Qwen-32B")
    ap.add_argument("--sources", nargs="+", default=["gsm8k"])
    ap.add_argument("--limit", type=int, default=800)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--out", default="/root/autodl-tmp/reasonv4/data/cot_corpus.jsonl")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    probs = load_problems(args.sources, args.limit)
    print(f"loaded {len(probs)} problems from {args.sources}")

    tmpl = ("<｜begin▁of▁sentence｜><｜User｜>{q}\nPlease reason step by step, "
            "and put your final answer within \\boxed{{}}.<｜Assistant｜>")
    prompts = [tmpl.format(q=q) for q in probs]

    llm = LLM(model=args.model, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem, trust_remote_code=True)
    sp = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=args.max_tokens)
    outs = llm.generate(prompts, sp, use_tqdm=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n = 0
    with open(args.out, "w") as f:
        for i, (q, o) in enumerate(zip(probs, outs)):
            sol = o.outputs[0].text
            if len(o.outputs[0].token_ids) < 16:
                continue
            rec = {"id": f"cot_{i}", "conversations": [
                {"from": "human", "value": q},
                {"from": "gpt", "value": sol},
            ]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} traces -> {args.out}")


if __name__ == "__main__":
    main()

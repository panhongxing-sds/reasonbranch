#!/usr/bin/env python3
"""Benchmark vLLM EAGLE-3 speculative decoding vs target-only on MATH-500.

Usage:
  python3 vllm_eagle_bench.py --base <base_model> [--eagle <eagle3_head>] \
      --n 20 --max-tokens 512 --spec-tokens 5

If --eagle is omitted -> target-only baseline.
Reports: decode tok/s, mean output tokens, and (spec) mean accepted length from vLLM metrics.
"""
import argparse, time, os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--eagle", default=None)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--spec-tokens", type=int, default=5)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split=f"test[:{args.n}]")
    tmpl = ("<｜begin▁of▁sentence｜><｜User｜>{q}\nPlease reason step by step, "
            "and put your final answer within \\boxed{{}}.<｜Assistant｜><think>\n")
    prompts = [tmpl.format(q=r["problem"]) for r in ds]

    kw = dict(model=args.base, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem, enforce_eager=False,
              trust_remote_code=True, disable_log_stats=False)
    tag = "target-only"
    if args.eagle:
        kw["speculative_config"] = {
            "method": "eagle3",
            "model": args.eagle,
            "num_speculative_tokens": args.spec_tokens,
        }
        tag = f"eagle3(k={args.spec_tokens})"

    llm = LLM(**kw)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    # warmup
    llm.generate(prompts[:1], sp, use_tqdm=False)

    # per-request (batch=1) timing -> this is where spec decoding latency win shows
    per_req = []
    tot_out = 0
    t_all = time.time()
    for p in prompts:
        t0 = time.time()
        o = llm.generate([p], sp, use_tqdm=False)
        dt = time.time() - t0
        n = len(o[0].outputs[0].token_ids)
        if n >= 32:
            per_req.append((n, dt))
            tot_out += n
    wall = time.time() - t_all
    import statistics
    tps = [n / dt for n, dt in per_req]
    mean_tps = statistics.mean(tps) if tps else 0.0

    print("\n================ RESULT ================")
    print(f"config          : {tag}")
    print(f"base            : {args.base}")
    if args.eagle:
        print(f"eagle head      : {args.eagle}")
    print(f"reqs(>=32tok)   : {len(per_req)}/{len(prompts)}  max_tokens={args.max_tokens}")
    print(f"total out tokens: {tot_out}")
    print(f"per-request tok/s (batch=1): mean={mean_tps:.2f}  "
          f"min={min(tps):.1f} max={max(tps):.1f}")
    print(f"per-request TPOT (ms/token): {1000.0/mean_tps:.2f}")

    # Pull spec-decode acceptance metrics
    try:
        acc = draft = steps = None
        for metric in llm.get_metrics():
            nm = getattr(metric, "name", "")
            val = getattr(metric, "value", None)
            if "spec_decode_num_accepted_tokens" in nm:
                acc = val
            elif "spec_decode_num_draft_tokens" in nm:
                draft = val
            elif "num_accepted_tokens_per_pos" in nm:
                print(f"  {nm} = {val}")
        if acc is not None and draft is not None:
            print(f"spec: accepted={acc} draft={draft} "
                  f"accept_rate={acc/max(draft,1):.3f} "
                  f"mean_accept_len≈{1 + acc/max(draft/args.spec_tokens,1):.2f}")
    except Exception as e:
        print(f"(metrics unavailable: {e})")


if __name__ == "__main__":
    main()

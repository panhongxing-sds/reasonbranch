"""V3.5 Experiment A — Latency microbenchmark for Branch cost–rescue.

Measures wall-clock:
  C_T   : 32B generates one reasoning step (autoregressive)
  C_DK  : 1.5B generates K Branch steps (K in {1,2,4})
  C_VK  : 32B batch-verifies K complete steps (ACCEPT/REJECT only)

Then reports break-even rescue rates:
  r_K^* = (C_DK + C_VK) / C_T

No API required.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

from reasoning_branch_dataset.action_study.batch_step_verifier import BatchStepVerifier
from reasoning_branch_dataset.action_study.cost_rescue import (
    CostSample,
    aggregate_break_even,
    length_bucket,
    overall_break_even,
)
from reasoning_branch_dataset.action_study.target_verifier import build_target_verifier
from reasoning_branch_dataset.action_study.vllm_backend import VLLMEngine
from reasoning_branch_dataset.model_utils import build_prompt


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def load_problems(path: Path, n: int, seed: int) -> list[dict[str, Any]]:
    rows = _load_jsonl(path)
    if not rows:
        raise FileNotFoundError(f"No problems in {path}")
    cleaned: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        q = r.get("problem") or r.get("question") or r.get("prompt") or ""
        if not q.strip():
            continue
        cleaned.append(
            {
                "problem_id": str(r.get("id", r.get("problem_id", f"p{i:04d}"))),
                "question": q.strip(),
            }
        )
    rng = random.Random(seed)
    if len(cleaned) > n:
        cleaned = rng.sample(cleaned, n)
    return cleaned


def sample_prefix_states(
    draft: VLLMEngine,
    problems: list[dict[str, Any]],
    *,
    n_states: int,
    seed: int,
    warm_tokens: tuple[int, ...] = (64, 192, 384, 768),
    step_max_tokens: int = 128,
) -> list[dict[str, Any]]:
    """Build short/medium/long prefix states from draft warm-up traces."""
    rng = random.Random(seed)
    states: list[dict[str, Any]] = []
    # Cycle problems until we have enough states.
    idx = 0
    while len(states) < n_states and problems:
        prob = problems[idx % len(problems)]
        idx += 1
        prompt = build_prompt(prob["question"])
        target_warm = warm_tokens[len(states) % len(warm_tokens)]
        # Generate a longer trace then truncate to target warm length.
        gen = draft.generate_trace(prompt, max_tokens=max(target_warm + 64, 128))
        text = gen["response_text"]
        # Prefer paragraph boundaries for prefix cuts.
        blocks = text.split("\n\n")
        if len(blocks) >= 2:
            keep = max(1, len(blocks) // 2)
            prefix_tail = "\n\n".join(blocks[:keep]).rstrip() + "\n\n"
        else:
            # Fallback: character cut approximating token budget.
            cut = max(80, int(len(text) * target_warm / max(gen["num_tokens"], 1)))
            prefix_tail = text[:cut].rstrip() + "\n\n"
        prefix = prompt + prefix_tail
        # One greedy step for step-length bucketing / verify payloads.
        step_rows = draft.generate_next_steps(
            prefix,
            k=1,
            max_tokens=step_max_tokens,
            temperature=0.0,
            top_p=1.0,
        )
        step = (step_rows[0]["text"] if step_rows else "").strip()
        if not step:
            # Force a synthetic short step so verify still runs.
            step = "Let me reconsider the previous calculation carefully."
        tok = draft.llm.get_tokenizer() if hasattr(draft, "llm") else None
        if tok is not None:
            prefix_tokens = len(tok.encode(prefix))
            step_tokens = len(tok.encode(step))
        else:
            prefix_tokens = max(1, len(prefix) // 4)
            step_tokens = max(1, len(step) // 4)
        states.append(
            {
                "state_id": f"{prob['problem_id']}_s{len(states):03d}",
                "problem_id": prob["problem_id"],
                "question": prob["question"],
                "prefix_text": prefix,
                "example_step": step,
                "prefix_tokens": prefix_tokens,
                "step_tokens": step_tokens,
                "prefix_bucket": length_bucket(prefix_tokens),
                "step_bucket": length_bucket(step_tokens, short=48, medium=96),
                "warm_target_tokens": target_warm,
            }
        )
        # Light shuffle of warm targets via rng (side effect: consume entropy).
        _ = rng.random()
    return states


def measure_draft_costs(
    draft: VLLMEngine,
    state: dict[str, Any],
    *,
    ks: tuple[int, ...] = (1, 2, 4),
    step_max_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.95,
    warmup: bool = True,
) -> dict[str, Any]:
    prefix = state["prefix_text"]
    out: dict[str, Any] = {"branches": {}}
    if warmup:
        # One discarded call to stabilize kernels / CUDA graphs.
        _ = draft.generate_next_steps(
            prefix, k=1, max_tokens=16, temperature=0.0, top_p=1.0
        )
    for k in ks:
        t0 = time.perf_counter()
        rows = draft.generate_next_steps(
            prefix,
            k=k,
            max_tokens=step_max_tokens,
            temperature=temperature if k > 1 else 0.0,
            top_p=top_p if k > 1 else 1.0,
        )
        latency = time.perf_counter() - t0
        texts = [r["text"].strip() for r in rows]
        # Pad if model returned fewer than k (should not happen with n=k).
        while len(texts) < k:
            texts.append(state["example_step"])
        out["branches"][str(k)] = {
            "latency_sec": latency,
            "texts": texts[:k],
            "num_tokens": [r.get("num_tokens", 0) for r in rows[:k]],
        }
    return out


def measure_target_costs(
    verifier: BatchStepVerifier,
    target_llm: Any,
    target_tok: Any,
    state: dict[str, Any],
    branch_payloads: dict[str, Any],
    *,
    ks: tuple[int, ...] = (1, 2, 4),
    step_max_tokens: int = 128,
    warmup: bool = True,
) -> dict[str, Any]:
    from vllm import SamplingParams

    prefix = state["prefix_text"]
    question = state["question"]
    prefix_ids = target_tok.encode(prefix, add_special_tokens=False)

    if warmup:
        _ = target_llm.generate(
            [{"prompt_token_ids": prefix_ids}],
            SamplingParams(max_tokens=8, temperature=0.0, top_p=1.0, detokenize=False),
        )

    # C_T: autoregressive generation of one reasoning step.
    # vLLM>=0.25 requires detokenize=True when using stop strings.
    params = SamplingParams(
        max_tokens=step_max_tokens,
        temperature=0.0,
        top_p=1.0,
        stop=["\n\n"],
        detokenize=True,
    )
    t0 = time.perf_counter()
    gen = target_llm.generate([{"prompt_token_ids": prefix_ids}], params)[0]
    c_t = time.perf_counter() - t0
    gen_tokens = list(gen.outputs[0].token_ids)

    out: dict[str, Any] = {
        "c_t": c_t,
        "target_step_tokens": len(gen_tokens),
        "verify": {},
    }

    for k in ks:
        texts = branch_payloads.get(str(k), {}).get("texts") or [state["example_step"]] * k
        texts = (texts + [state["example_step"]] * k)[:k]
        res = verifier.verify_batch(
            question=question,
            prefix_text=prefix,
            candidates=texts,
        )
        out["verify"][str(k)] = {
            "latency_sec": res.latency_sec,
            "parsed_rate": res.parsed_rate,
            "n_accepted": sum(1 for a in res.acceptable if a is True),
        }
    return out


def run_latency_bench(
    *,
    draft_model: str,
    target_model: str,
    problems_path: Path,
    out_dir: Path,
    n_states: int = 60,
    n_problems: int = 30,
    seed: int = 42,
    step_max_tokens: int = 128,
    draft_gpu_util: float = 0.90,
    target_gpu_util: float = 0.92,
    target_quantization: str | None = None,
    max_model_len: int = 4096,
    ks: tuple[int, ...] = (1, 2, 4),
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    states_path = out_dir / "prefix_states.jsonl"
    samples_path = out_dir / "latency_samples.jsonl"
    summary_path = out_dir / "latency_summary.json"
    report_path = out_dir / "cost_rescue_latency_report.md"

    problems = load_problems(problems_path, n_problems, seed)

    # --- Phase 1: draft-only (build states + measure C_DK) ---
    print(f"[v3.5A] loading draft: {draft_model}")
    draft = VLLMEngine(
        draft_model,
        gpu_memory_utilization=draft_gpu_util,
        max_model_len=max_model_len,
    )
    if states_path.exists():
        states = _load_jsonl(states_path)
        print(f"[v3.5A] resumed {len(states)} prefix states")
    else:
        print(f"[v3.5A] sampling {n_states} prefix states")
        states = sample_prefix_states(
            draft, problems, n_states=n_states, seed=seed, step_max_tokens=step_max_tokens
        )
        _write_jsonl(states_path, states)

    draft_costs: dict[str, dict[str, Any]] = {}
    done_draft = {r["state_id"] for r in _load_jsonl(out_dir / "draft_costs.jsonl") if "state_id" in r}
    draft_cost_path = out_dir / "draft_costs.jsonl"
    for st in states:
        sid = st["state_id"]
        if sid in done_draft:
            continue
        measured = measure_draft_costs(
            draft, st, ks=ks, step_max_tokens=step_max_tokens, warmup=True
        )
        row = {"state_id": sid, **measured}
        _append_jsonl(draft_cost_path, row)
        draft_costs[sid] = measured
        print(f"[v3.5A] draft costs {sid}: " + ", ".join(
            f"C_D{k}={measured['branches'][str(k)]['latency_sec']:.3f}s" for k in ks
        ))

    # Reload any previously measured draft costs.
    for r in _load_jsonl(draft_cost_path):
        draft_costs[r["state_id"]] = {"branches": r["branches"]}

    # Free draft before loading 32B.
    del draft
    import gc

    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass

    # --- Phase 2: target-only (C_T + C_VK) ---
    print(f"[v3.5A] loading target: {target_model}")
    target = build_target_verifier(
        target_model,
        engine="vllm",
        gpu_memory_utilization=target_gpu_util,
        max_model_len=max_model_len,
        quantization=target_quantization,
        debug=False,
        enable_prefix_caching=True,
    )
    verifier = BatchStepVerifier(target.llm, target.tokenizer)

    done_samples = {r["state_id"] for r in _load_jsonl(samples_path) if "state_id" in r}
    cost_samples: list[CostSample] = []
    for st in states:
        sid = st["state_id"]
        if sid in done_samples:
            continue
        payload = draft_costs.get(sid, {"branches": {}})
        measured = measure_target_costs(
            verifier,
            target.llm,
            target.tokenizer,
            st,
            payload.get("branches", {}),
            ks=ks,
            step_max_tokens=step_max_tokens,
            warmup=True,
        )
        branches = payload.get("branches", {})
        sample = CostSample(
            prefix_bucket=st["prefix_bucket"],
            step_bucket=st["step_bucket"],
            prefix_tokens=int(st["prefix_tokens"]),
            step_tokens=int(st["step_tokens"]),
            c_t=float(measured["c_t"]),
            c_d1=float(branches["1"]["latency_sec"]) if "1" in branches else None,
            c_d2=float(branches["2"]["latency_sec"]) if "2" in branches else None,
            c_d4=float(branches["4"]["latency_sec"]) if "4" in branches else None,
            c_v1=float(measured["verify"]["1"]["latency_sec"]) if "1" in measured["verify"] else None,
            c_v2=float(measured["verify"]["2"]["latency_sec"]) if "2" in measured["verify"] else None,
            c_v4=float(measured["verify"]["4"]["latency_sec"]) if "4" in measured["verify"] else None,
        )
        row = {
            "state_id": sid,
            "problem_id": st["problem_id"],
            **sample.__dict__,
            "target_step_tokens": measured["target_step_tokens"],
            "verify_meta": measured["verify"],
        }
        _append_jsonl(samples_path, row)
        cost_samples.append(sample)
        print(
            f"[v3.5A] target {sid}: C_T={sample.c_t:.3f}s "
            f"C_V4={sample.c_v4:.3f}s" if sample.c_v4 is not None else
            f"[v3.5A] target {sid}: C_T={sample.c_t:.3f}s"
        )

    # Rebuild full sample list from disk for aggregation.
    all_rows = _load_jsonl(samples_path)
    cost_samples = [
        CostSample(
            prefix_bucket=r["prefix_bucket"],
            step_bucket=r["step_bucket"],
            prefix_tokens=int(r["prefix_tokens"]),
            step_tokens=int(r["step_tokens"]),
            c_t=float(r["c_t"]),
            c_d1=r.get("c_d1"),
            c_d2=r.get("c_d2"),
            c_d4=r.get("c_d4"),
            c_v1=r.get("c_v1"),
            c_v2=r.get("c_v2"),
            c_v4=r.get("c_v4"),
        )
        for r in all_rows
    ]

    by_bucket = [r.to_dict() for r in aggregate_break_even(cost_samples)]
    overall = overall_break_even(cost_samples).to_dict()
    summary = {
        "n_states": len(cost_samples),
        "draft_model": draft_model,
        "target_model": target_model,
        "ks": list(ks),
        "step_max_tokens": step_max_tokens,
        "overall": overall,
        "by_bucket": by_bucket,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(render_latency_report(summary), encoding="utf-8")
    print(f"[v3.5A] wrote {summary_path}")
    print(f"[v3.5A] wrote {report_path}")
    return summary


def render_latency_report(summary: dict[str, Any]) -> str:
    o = summary.get("overall") or {}
    lines = [
        "# V3.5 Experiment A — Latency Microbenchmark",
        "",
        "> Goal: measure whether Fixed Branch@K is systemically cheaper than Handoff,",
        "> before training any Branch/Handoff classifier.",
        "",
        f"- draft: `{summary.get('draft_model')}`",
        f"- target: `{summary.get('target_model')}`",
        f"- states measured: **{summary.get('n_states', 0)}**",
        f"- step_max_tokens: {summary.get('step_max_tokens')}",
        "",
        "## Break-even formula",
        "",
        "$$r_K^* = (C_{DK} + C_{VK}) / C_T$$",
        "",
        "If empirical $r_K \\gg r_K^*$, use **always Branch@K** (no router).",
        "",
        "## Overall means",
        "",
        "| Metric | Value |",
        "|--------|------:|",
        f"| $C_T$ | {_fmt_s(o.get('c_t'))} |",
        f"| $C_{{D1}}$ | {_fmt_s(o.get('c_d1'))} |",
        f"| $C_{{D2}}$ | {_fmt_s(o.get('c_d2'))} |",
        f"| $C_{{D4}}$ | {_fmt_s(o.get('c_d4'))} |",
        f"| $C_{{V1}}$ | {_fmt_s(o.get('c_v1'))} |",
        f"| $C_{{V2}}$ | {_fmt_s(o.get('c_v2'))} |",
        f"| $C_{{V4}}$ | {_fmt_s(o.get('c_v4'))} |",
        f"| $r_1^*$ | {_fmt_pct(o.get('r1_star'))} |",
        f"| $r_2^*$ | {_fmt_pct(o.get('r2_star'))} |",
        f"| $r_4^*$ | {_fmt_pct(o.get('r4_star'))} |",
        "",
        "## By prefix × step bucket",
        "",
        "| Prefix | Step | N | $C_T$ | $C_{D4}$ | $C_{V4}$ | $r_4^*$ |",
        "|-------:|-----:|--:|------:|---------:|---------:|-------:|",
    ]
    for row in summary.get("by_bucket") or []:
        lines.append(
            f"| {row['prefix_bucket']} | {row['step_bucket']} | {row['n']} | "
            f"{_fmt_s(row.get('c_t'))} | {_fmt_s(row.get('c_d4'))} | "
            f"{_fmt_s(row.get('c_v4'))} | {_fmt_pct(row.get('r4_star'))} |"
        )
    lines += [
        "",
        "## Next",
        "",
        "1. Run Experiment B to estimate final-stack $r_1,r_2,r_4$ (or use provisional V3.3).",
        "2. Compare $r_K$ vs $r_K^*$ via `run_v3_5_cost_rescue.py`.",
        "3. Only train a Branch predictor if near break-even.",
        "",
    ]
    return "\n".join(lines)


def _fmt_s(x: Any) -> str:
    if x is None:
        return "—"
    return f"{float(x):.3f}s"


def _fmt_pct(x: Any) -> str:
    if x is None:
        return "—"
    return f"{100 * float(x):.1f}%"


def main() -> None:
    p = argparse.ArgumentParser(description="V3.5 Experiment A: latency microbenchmark")
    p.add_argument(
        "--draft-model",
        default="/root/autodl-tmp/specreason/models/DeepSeek-R1-Distill-Qwen-1.5B",
    )
    p.add_argument(
        "--target-model",
        default="/root/autodl-tmp/specreason/models/DeepSeek-R1-Distill-Qwen-32B-AWQ",
    )
    p.add_argument("--target-quantization", default="awq")
    p.add_argument(
        "--problems",
        default="/root/autodl-tmp/reasonbranch/data/aime_train.jsonl",
    )
    p.add_argument(
        "--out-dir",
        default="/root/autodl-tmp/reasonbranch/outputs/action_study_v35_latency",
    )
    p.add_argument("--n-states", type=int, default=48)
    p.add_argument("--n-problems", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--step-max-tokens", type=int, default=128)
    p.add_argument("--draft-gpu-util", type=float, default=0.90)
    p.add_argument("--target-gpu-util", type=float, default=0.92)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--no-quant", action="store_true", help="Disable AWQ quantization kwarg")
    args = p.parse_args()

    quant = None if args.no_quant else (args.target_quantization or None)
    # If user points at non-AWQ path, don't force awq.
    if quant and "awq" not in Path(args.target_model).name.lower():
        quant = None

    run_latency_bench(
        draft_model=args.draft_model,
        target_model=args.target_model,
        problems_path=Path(args.problems),
        out_dir=Path(args.out_dir),
        n_states=args.n_states,
        n_problems=args.n_problems,
        seed=args.seed,
        step_max_tokens=args.step_max_tokens,
        draft_gpu_util=args.draft_gpu_util,
        target_gpu_util=args.target_gpu_util,
        target_quantization=quant,
        max_model_len=args.max_model_len,
    )


if __name__ == "__main__":
    main()

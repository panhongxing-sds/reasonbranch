"""Unit tests for greedy target acceptance / prompt_logprobs alignment."""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from reasoning_branch_dataset.action_study.logprob_align import (
    hf_greedy_token,
    vllm_candidate_rank,
    vllm_greedy_token,
)
from reasoning_branch_dataset.action_study.target_verifier import (
    build_target_verifier,
    greedy_generate_ids_vllm,
)
from reasoning_branch_dataset.model_utils import build_prompt, load_model_and_tokenizer


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def collect_env_info(model_path: str, *, debug: bool) -> dict[str, Any]:
    import transformers
    import vllm

    info: dict[str, Any] = {
        "timestamp": _utc_now(),
        "model_path": model_path,
        "vllm_version": vllm.__version__,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "debug_mode": debug,
        "enable_prefix_caching": False if debug else "default",
        "enable_chunked_prefill": False if debug else "default",
        "enforce_eager": True if debug else False,
    }
    try:
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        info["tokenizer_class"] = tok.__class__.__name__
        info["tokenizer_revision"] = getattr(tok, "revision", None)
    except Exception as exc:
        info["tokenizer_error"] = str(exc)
    return info


def _sample_questions(data_dir: Path | None, n: int) -> list[str]:
    fallback = [
        "What is 17 + 25?",
        "If x + 3 = 10, what is x?",
        "How many sides does a hexagon have?",
        "What is the square root of 144?",
        "Solve 2x = 14 for x.",
    ]
    if data_dir and (data_dir / "problems.jsonl").exists():
        rows = [
            json.loads(line)["question"]
            for line in (data_dir / "problems.jsonl").read_text().splitlines()
            if line.strip()
        ]
        if len(rows) >= n:
            rng = random.Random(42)
            return rng.sample(rows, n)
    while len(fallback) < n:
        fallback = fallback + fallback
    return fallback[:n]


def test_self_replay_vllm(
    model_path: str,
    *,
    n_tokens: int = 32,
    prompt: str,
    debug: bool,
) -> dict[str, Any]:
    verifier = build_target_verifier(model_path, engine="vllm", debug=debug)
    try:
        prefix_ids = verifier.tokenize(prompt)
        gen_ids = greedy_generate_ids_vllm(verifier.llm, prefix_ids, max_tokens=n_tokens)
        result = verifier.greedy_acceptance_ids(prefix_ids, gen_ids)
        passed = result.accepted_length == len(gen_ids) == n_tokens
        return {
            "test": "self_replay_vllm",
            "model": model_path,
            "n_tokens_requested": n_tokens,
            "n_tokens_generated": len(gen_ids),
            "accepted_length": result.accepted_length,
            "first_reject_position": result.first_reject_position,
            "passed": passed,
            "prefix_token_count": len(prefix_ids),
            "generated_ids_head": gen_ids[:8],
        }
    finally:
        del verifier
        _free_cuda()


def test_single_token_vllm(
    model_path: str,
    *,
    prompt: str,
    debug: bool,
) -> dict[str, Any]:
    verifier = build_target_verifier(model_path, engine="vllm", debug=debug)
    prefix_ids = verifier.tokenize(prompt)
    gen_ids = greedy_generate_ids_vllm(verifier.llm, prefix_ids, max_tokens=1)
    if not gen_ids:
        return {"test": "single_token_vllm", "model": model_path, "passed": False, "error": "no token generated"}
    result = verifier.greedy_acceptance_ids(prefix_ids, gen_ids)
    passed = result.accepted_length >= 1
    return {
        "test": "single_token_vllm",
        "model": model_path,
        "first_token_id": gen_ids[0],
        "first_token_text": verifier.tokenizer.decode([gen_ids[0]]),
        "accepted_length": result.accepted_length,
        "passed": passed,
    }


def _hf_top1_at(model, prefix_ids: list[int], continuation_ids: list[int], pos: int, device: str) -> int:
    full_ids = prefix_ids + continuation_ids[: pos + 1]
    input_ids = torch.tensor([full_ids], device=device, dtype=torch.long)
    with torch.inference_mode():
        logits = model(input_ids=input_ids).logits[0]
    return hf_greedy_token(logits, len(prefix_ids) - 1 + pos)


def test_hf_vllm_alignment(
    model_path: str,
    *,
    prompts: list[str],
    n_positions: int = 16,
    debug: bool,
    device: str = "cuda",
) -> dict[str, Any]:
    hf_model, hf_tok = load_model_and_tokenizer(model_path, device, "bfloat16")
    hf_model.eval()

    prepared: list[dict[str, Any]] = []
    for pi, prompt in enumerate(prompts):
        prefix_ids = hf_tok.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([prefix_ids], device=device, dtype=torch.long)
        with torch.inference_mode():
            gen = hf_model.generate(
                input_ids=input_ids,
                max_new_tokens=n_positions,
                do_sample=False,
                pad_token_id=hf_tok.pad_token_id or hf_tok.eos_token_id,
            )
        continuation_ids = gen[0, len(prefix_ids) :].tolist()
        full_ids = prefix_ids + continuation_ids
        with torch.inference_mode():
            logits = hf_model(
                torch.tensor([full_ids], device=device, dtype=torch.long)
            ).logits[0]
        hf_top1 = [hf_greedy_token(logits, len(prefix_ids) - 1 + pos) for pos in range(len(continuation_ids))]
        prepared.append(
            {
                "prompt_index": pi,
                "prefix_ids": prefix_ids,
                "continuation_ids": continuation_ids,
                "full_ids": full_ids,
                "hf_top1": hf_top1,
            }
        )

    del hf_model
    _free_cuda()

    verifier = build_target_verifier(model_path, engine="vllm", debug=debug)
    rows: list[dict[str, Any]] = []
    mismatches = 0
    for item in prepared:
        lp_steps = verifier.prompt_logprobs_at(item["full_ids"])
        base = len(item["prefix_ids"])
        for pos, cand_id in enumerate(item["continuation_ids"]):
            hf_id = item["hf_top1"][pos]
            step_idx = base + pos
            lp_map = lp_steps[step_idx] if step_idx < len(lp_steps) else None
            vllm_id = vllm_greedy_token(lp_map)
            rank = vllm_candidate_rank(lp_map, cand_id)
            prev_id = item["full_ids"][step_idx - 1] if step_idx >= 1 else None
            match = hf_id == vllm_id
            if not match:
                mismatches += 1
            cand_lp = None
            if lp_map and cand_id in lp_map:
                cand_lp = float(lp_map[cand_id].logprob)
            rows.append(
                {
                    "prompt_index": item["prompt_index"],
                    "position": pos,
                    "previous_token_id": prev_id,
                    "candidate_token_id": cand_id,
                    "hf_argmax_id": hf_id,
                    "vllm_argmax_id": vllm_id,
                    "candidate_rank": rank,
                    "candidate_logprob": cand_lp,
                    "top1_match": match,
                }
            )

    del verifier
    _free_cuda()

    return {
        "test": "hf_vllm_alignment",
        "model": model_path,
        "n_prompts": len(prompts),
        "n_positions": n_positions,
        "n_rows": len(rows),
        "n_top1_mismatches": mismatches,
        "passed": mismatches == 0,
        "rows": rows,
    }


def _free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_all(
    *,
    target_model: str,
    draft_model: str,
    data_dir: Path | None,
    out_dir: Path,
    debug: bool = True,
    skip_hf_align: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = _sample_questions(data_dir, 10)
    probe_prompt = build_prompt(prompts[0])

    report: dict[str, Any] = {
        "meta": {
            "target_model": target_model,
            "draft_model": draft_model,
            "debug": debug,
        },
        "tests": [],
    }

    print("[test1] QwQ self-replay (run first while GPU is clean)")
    qwq = build_target_verifier(target_model, engine="vllm", debug=debug)
    prefix_ids = qwq.tokenize(probe_prompt)
    gen_ids = greedy_generate_ids_vllm(qwq.llm, prefix_ids, max_tokens=32)
    r1 = qwq.greedy_acceptance_ids(prefix_ids, gen_ids)
    t1 = {
        "test": "self_replay_vllm",
        "model": target_model,
        "n_tokens_requested": 32,
        "n_tokens_generated": len(gen_ids),
        "accepted_length": r1.accepted_length,
        "first_reject_position": r1.first_reject_position,
        "passed": r1.accepted_length == len(gen_ids) == 32,
        "prefix_token_count": len(prefix_ids),
        "generated_ids_head": gen_ids[:8],
    }
    report["tests"].append(t1)
    print(f"  passed={t1['passed']} accepted={t1['accepted_length']}/32")

    print("[test4] QwQ single-token replay")
    gen1 = greedy_generate_ids_vllm(qwq.llm, prefix_ids, max_tokens=1)
    r4 = qwq.greedy_acceptance_ids(prefix_ids, gen1)
    t4 = {
        "test": "single_token_vllm",
        "model": target_model,
        "first_token_id": gen1[0] if gen1 else None,
        "first_token_text": qwq.tokenizer.decode([gen1[0]]) if gen1 else None,
        "accepted_length": r4.accepted_length,
        "passed": r4.accepted_length >= 1,
    }
    report["tests"].append(t4)
    print(f"  passed={t4['passed']} token={t4.get('first_token_text')!r}")
    del qwq
    _free_cuda()

    print("[test2] 4B self-replay")
    t2 = test_self_replay_vllm(draft_model, n_tokens=32, prompt=probe_prompt, debug=debug)
    report["tests"].append(t2)
    print(f"  passed={t2['passed']} accepted={t2['accepted_length']}/32")
    _free_cuda()

    print("[test3] HF vs vLLM alignment (4B, 10 prompts x 16 positions)")
    short_prompts = [build_prompt(q) for q in prompts]
    if skip_hf_align:
        t3 = {
            "test": "hf_vllm_alignment",
            "model": draft_model,
            "passed": None,
            "skipped": True,
            "note": "skipped via --skip-hf-align",
        }
        print("  skipped")
    else:
        t3 = test_hf_vllm_alignment(draft_model, prompts=short_prompts, n_positions=16, debug=debug)
        print(f"  passed={t3['passed']} mismatches={t3['n_top1_mismatches']}/{t3['n_rows']}")
    report["tests"].append(t3)
    _free_cuda()

    report["meta"]["env_target"] = collect_env_info(target_model, debug=debug)
    report["meta"]["env_draft"] = collect_env_info(draft_model, debug=debug)
    report["all_passed"] = all(t.get("passed") for t in report["tests"] if t.get("passed") is not None)

    json_path = out_dir / "target_acceptance_debug_report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# Target Acceptance Debug Report",
        "",
        f"- generated: {_utc_now()}",
        f"- target: `{target_model}`",
        f"- draft: `{draft_model}`",
        f"- debug mode: {debug}",
        f"- **all passed: {report['all_passed']}**",
        "",
        "## Summary",
        "",
        "| test | passed | detail |",
        "|------|--------|--------|",
    ]
    for t in report["tests"]:
        name = t["test"]
        if name == "self_replay_vllm":
            label = "QwQ self" if target_model in t.get("model", "") else "4B self"
        elif name == "single_token_vllm":
            label = "single token"
        else:
            label = name
        detail = ""
        if "accepted_length" in t:
            detail = f"A={t['accepted_length']}"
        elif "n_top1_mismatches" in t:
            detail = f"mismatches={t['n_top1_mismatches']}/{t['n_rows']}"
        md_lines.append(f"| {label} | {t.get('passed')} | {detail} |")

    if not t3.get("passed"):
        md_lines.extend(["", "## HF/vLLM mismatches (first 20)", ""])
        bad = [r for r in t3.get("rows", []) if not r.get("top1_match")][:20]
        for r in bad:
            md_lines.append(
                f"- prompt={r['prompt_index']} pos={r['position']}: "
                f"HF={r['hf_argmax_id']} vLLM={r['vllm_argmax_id']} cand={r['candidate_token_id']}"
            )

    md_path = out_dir / "target_acceptance_debug_report.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"\nWrote {json_path}\nWrote {md_path}")
    return report


def main() -> None:
    # Line-buffered logs when redirected to file.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description="Greedy acceptance alignment unit tests")
    parser.add_argument("--target-model", default="/mnt/afs/L202500372/specreason/models/QwQ-32B")
    parser.add_argument("--draft-model", default="/mnt/afs/L202500372/models/Qwen3.5-4B")
    parser.add_argument(
        "--data-dir",
        default="/mnt/afs/L202500372/reasoning_branch_dataset/outputs/action_study_pilot_v2",
    )
    parser.add_argument(
        "--out-dir",
        default="/mnt/afs/L202500372/reasoning_branch_dataset/outputs/action_study_pilot_v2",
    )
    parser.add_argument("--no-debug", action="store_true", help="Allow prefix caching / eager off")
    parser.add_argument("--skip-hf-align", action="store_true", help="Skip slow HF/vLLM alignment test")
    args = parser.parse_args()

    run_all(
        target_model=args.target_model,
        draft_model=args.draft_model,
        data_dir=Path(args.data_dir) if args.data_dir else None,
        out_dir=Path(args.out_dir),
        debug=not args.no_debug,
        skip_hf_align=args.skip_hf_align,
    )


if __name__ == "__main__":
    main()

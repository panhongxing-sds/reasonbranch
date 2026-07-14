"""Diagnose 14B target next-step generation on sampled prefixes (no API)."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from reasoning_branch_dataset.action_study.run_utility_scoring import _load_jsonl
from reasoning_branch_dataset.action_study.step_extraction import (
    extract_handoff_step,
    strip_model_thinking,
)
from reasoning_branch_dataset.action_study.target_verifier import build_target_verifier, greedy_generate_vllm
from reasoning_branch_dataset.model_utils import build_prompt


_THINK_ONLY_RE = re.compile(
    r"^\s*(?:\x3cthink\x3e[\s\S]*?\x3c/think\x3e|<think>\s*</think>)\s*$",
    re.IGNORECASE,
)


def classify_empty_reason(
    *,
    raw_output: str,
    extracted_step: str,
    prompt_length: int,
    generated_length: int,
    finish_reason: str | None,
    max_model_len: int,
) -> str:
    if extracted_step.strip():
        return "OK"
    if prompt_length >= max_model_len - 16:
        return "CONTEXT_OVERFLOW"
    if generated_length == 0:
        return "EMPTY_RAW_OUTPUT"
    if finish_reason == "length":
        return "MAX_TOKENS_TRUNCATED"
    visible = strip_model_thinking(raw_output)
    if not visible.strip():
        return "THINKING_ONLY"
    if _THINK_ONLY_RE.match(raw_output.strip()):
        return "THINKING_ONLY"
    if raw_output.strip() and not extracted_step.strip():
        return "EXTRACTION_FAILED"
    return "UNKNOWN_EMPTY"


def sample_prefixes(
    v2_dir: Path,
    *,
    n: int,
    seed: int,
    rollout_steps: Path | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if rollout_steps and rollout_steps.exists():
        seen: set[str] = set()
        for st in _load_jsonl(rollout_steps):
            key = st.get("prefix_hash") or st.get("prefix_text", "")[:80]
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "prefix_id": f"rollout:{st.get('rollout_id')}:{st.get('step_index')}",
                    "problem_id": st.get("problem_id", ""),
                    "prefix_text": st.get("prefix_text", ""),
                    "source": "v34_rollout",
                }
            )

    for p in _load_jsonl(v2_dir / "prefixes.jsonl"):
        if p.get("admission_main") or p.get("include_in_main_experiment"):
            rows.append(
                {
                    "prefix_id": p["prefix_id"],
                    "problem_id": p.get("problem_id", ""),
                    "prefix_text": p.get("prefix_text", ""),
                    "source": "v2_prefix",
                }
            )

    rng = random.Random(seed)
    if len(rows) <= n:
        return rows
    return rng.sample(rows, n)


def run_diagnostic(
    prefixes: list[dict[str, Any]],
    *,
    target_model: str,
    max_tokens: int,
    max_model_len: int,
    out_path: Path,
) -> dict[str, Any]:
    verifier = build_target_verifier(
        target_model,
        engine="vllm",
        gpu_memory_utilization=0.90,
        max_model_len=max_model_len,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    reasons = Counter()

    for item in prefixes:
        prefix = item["prefix_text"]
        if not prefix.strip():
            continue
        question = ""
        if "Problem:" in prefix:
            question = prefix.split("Problem:", 1)[-1].strip()

        prompt_ids = verifier.tokenize(prefix)
        gen = greedy_generate_vllm(verifier.llm, prompt_ids, max_tokens=max_tokens)
        raw = verifier.tokenizer.decode(gen["token_ids"], skip_special_tokens=False)
        extracted = extract_handoff_step(raw, question=question)
        reason = classify_empty_reason(
            raw_output=raw,
            extracted_step=extracted,
            prompt_length=len(prompt_ids),
            generated_length=len(gen["token_ids"]),
            finish_reason=gen.get("finish_reason"),
            max_model_len=max_model_len,
        )
        reasons[reason] += 1
        row = {
            **item,
            "raw_output": raw,
            "token_ids": gen["token_ids"],
            "finish_reason": gen.get("finish_reason"),
            "stop_reason": gen.get("stop_reason"),
            "extracted_step": extracted,
            "failure_reason": reason,
            "prompt_length": len(prompt_ids),
            "generated_length": len(gen["token_ids"]),
            "max_tokens": max_tokens,
            "max_model_len": max_model_len,
            "has_think_open": "\x3cthink\x3e" in raw.lower(),
            "has_think_close": "\x3c/think\x3e" in raw.lower(),
            "visible_after_strip": strip_model_thinking(raw)[:500],
        }
        results.append(row)
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n = len(results)
    nonempty = sum(1 for r in results if r["extracted_step"].strip())
    summary = {
        "n_prefixes": n,
        "nonempty_rate": nonempty / n if n else 0,
        "failure_reasons": dict(reasons),
        "target_model": target_model,
        "max_tokens": max_tokens,
        "max_model_len": max_model_len,
    }
    return summary


def write_report(summary: dict[str, Any], report_path: Path) -> None:
    lines = [
        "# Target Step Diagnostic Report",
        "",
        f"- prefixes tested: **{summary['n_prefixes']}**",
        f"- non-empty extracted step rate: **{100*summary['nonempty_rate']:.1f}%**",
        f"- target: `{summary['target_model']}`",
        f"- max_tokens: **{summary['max_tokens']}** | max_model_len: **{summary['max_model_len']}**",
        "",
        "## Failure reason breakdown",
        "",
        "| Reason | Count |",
        "|--------|------:|",
    ]
    for k, v in sorted(summary.get("failure_reasons", {}).items(), key=lambda x: -x[1]):
        lines.append(f"| {k} | {v} |")
    lines.append("")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-dir", default="/mnt/afs/L202500372/reasoning_branch_dataset/outputs/action_study_pilot_v2")
    parser.add_argument("--rollout-steps", default="/mnt/afs/L202500372/reasoning_branch_dataset/outputs/action_study_pilot_v34/rollout_steps.jsonl")
    parser.add_argument("--out-dir", default="/mnt/afs/L202500372/reasoning_branch_dataset/outputs/target_step_diagnostic")
    parser.add_argument("--n-prefixes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument(
        "--target-model",
        default="/mnt/afs/L202500372/specreason/models/DeepSeek-R1-Distill-Qwen-14B",
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    diag_path = out_dir / "target_step_diagnostic.jsonl"
    if diag_path.exists():
        diag_path.unlink()

    prefixes = sample_prefixes(
        Path(args.v2_dir),
        n=args.n_prefixes,
        seed=args.seed,
        rollout_steps=Path(args.rollout_steps) if args.rollout_steps else None,
    )
    summary = run_diagnostic(
        prefixes,
        target_model=args.target_model,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        out_path=diag_path,
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, out_dir / "report.md")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

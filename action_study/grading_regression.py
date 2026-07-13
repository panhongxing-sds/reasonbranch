"""Regression test grading on complete v2 traces (no API)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from reasoning_branch_dataset.action_study.run_utility_scoring import _load_jsonl
from reasoning_branch_dataset.grading import classify_generation_outcome, extract_math_answer, has_boxed_answer, math_equal
from reasoning_branch_dataset.model_utils import build_prompt


def run_grading_regression(
    v2_dir: Path,
    *,
    n_traces: int = 50,
    seed: int = 42,
) -> dict[str, Any]:
    problems = {r["problem_id"]: r for r in _load_jsonl(v2_dir / "problems.jsonl")}
    traces = [r for r in _load_jsonl(v2_dir / "traces.jsonl") if r.get("full_reasoning")]
    rng = random.Random(seed)
    if len(traces) > n_traces:
        traces = rng.sample(traces, n_traces)

    rows: list[dict[str, Any]] = []
    n_boxed = n_extracted = n_scored = n_correct = 0
    for tr in traces:
        pid = tr["problem_id"]
        prob = problems.get(pid, {})
        gold = prob.get("gold_answer", "")
        prompt = build_prompt(prob.get("question", ""))
        full = tr.get("full_reasoning", "")
        reasoning = full[len(prompt) :] if full.startswith(prompt) else full
        outcome = classify_generation_outcome(reasoning, gold, require_marker=True)
        extracted = outcome.get("final_answer") or outcome.get("predicted_answer") or ""
        boxed = has_boxed_answer(reasoning)
        if boxed:
            n_boxed += 1
        if extracted:
            n_extracted += 1
        if outcome.get("is_correct") is not None:
            n_scored += 1
            if outcome.get("is_correct"):
                n_correct += 1
        rows.append(
            {
                "problem_id": pid,
                "has_boxed": boxed,
                "extracted_answer": extracted,
                "gold_answer": gold,
                "is_correct": outcome.get("is_correct"),
                "evaluation_status": outcome.get("evaluation_status"),
                "math_equal_direct": math_equal(extracted, gold) if extracted and gold else None,
            }
        )

    n = len(rows)
    summary = {
        "n_traces": n,
        "boxed_rate": n_boxed / n if n else 0,
        "extracted_rate": n_extracted / n if n else 0,
        "scored_rate": n_scored / n if n else 0,
        "accuracy_among_scored": n_correct / n_scored if n_scored else 0,
        "failures": [r for r in rows if r["evaluation_status"] != "OK"],
    }
    return {"summary": summary, "rows": rows}


def write_report(result: dict[str, Any], path: Path) -> None:
    s = result["summary"]
    lines = [
        "# Grading Regression Report",
        "",
        f"- traces tested: **{s['n_traces']}**",
        f"- has \\boxed{{}}: **{100*s['boxed_rate']:.1f}%**",
        f"- extracted answer: **{100*s['extracted_rate']:.1f}%**",
        f"- scored (is_correct not null): **{100*s['scored_rate']:.1f}%**",
        f"- accuracy (among scored): **{100*s['accuracy_among_scored']:.1f}%**",
        "",
        "## Non-OK traces",
        "",
    ]
    for r in s.get("failures", [])[:20]:
        lines.append(
            f"- `{r['problem_id']}` status={r['evaluation_status']} boxed={r['has_boxed']} extracted={r['extracted_answer']!r}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-dir", default="/mnt/afs/L202500372/reasoning_branch_dataset/outputs/action_study_pilot_v2")
    parser.add_argument("--out-dir", default="/mnt/afs/L202500372/reasoning_branch_dataset/outputs/grading_regression")
    parser.add_argument("--n-traces", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = run_grading_regression(Path(args.v2_dir), n_traces=args.n_traces, seed=args.seed)
    (out / "results.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in result["rows"]) + "\n",
        encoding="utf-8",
    )
    (out / "summary.json").write_text(json.dumps(result["summary"], indent=2), encoding="utf-8")
    write_report(result, out / "report.md")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()

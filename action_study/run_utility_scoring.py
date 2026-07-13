"""V3: SpecReason-style utility scoring on existing 1+4 draft candidates."""

from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from reasoning_branch_dataset.action_study.oracle_labels import classify_oracle, summarize_oracle_table
from reasoning_branch_dataset.action_study.specreason_scorer import first_reasoning_step, score_step_vllm
from reasoning_branch_dataset.action_study.step_extraction import extract_next_substantive_step
from reasoning_branch_dataset.action_study.target_verifier import build_target_verifier, model_slug
from reasoning_branch_dataset.model_utils import build_prompt


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _done_ids(path: Path) -> set[str]:
    return {r["prefix_id"] for r in _load_jsonl(path) if "prefix_id" in r}


def _reasoning_prefix(prefix_text: str, question: str) -> str:
    prompt = build_prompt(question)
    if prefix_text.startswith(prompt):
        return prefix_text[len(prompt) :]
    return prefix_text


def load_candidate_tasks(data_dir: Path, *, admission_only: bool = True) -> list[dict[str, Any]]:
    admission = {r["prefix_id"]: r for r in _load_jsonl(data_dir / "prefix_admission.jsonl")}
    prefixes = {r["prefix_id"]: r for r in _load_jsonl(data_dir / "prefixes.jsonl")}
    problems = {r["problem_id"]: r for r in _load_jsonl(data_dir / "problems.jsonl")}

    by_prefix: dict[str, dict[str, Any]] = defaultdict(lambda: {"branches": []})
    for row in _load_jsonl(data_dir / "actions.jsonl"):
        pid = row["prefix_id"]
        if admission_only and not admission.get(pid, {}).get("admission_main"):
            continue
        if row["action_type"] == "continue" and row.get("sample_id", 0) == 0:
            by_prefix[pid]["continue"] = row
        elif row["action_type"] == "branch":
            by_prefix[pid]["branches"].append(row)

    tasks: list[dict[str, Any]] = []
    for prefix_id, bundle in by_prefix.items():
        if "continue" not in bundle or len(bundle["branches"]) < 4:
            continue
        pfx = prefixes.get(prefix_id, {})
        prob = problems.get(pfx.get("problem_id", bundle["continue"]["problem_id"]), {})
        question = prob.get("question", "")
        prefix_text = pfx.get("prefix_text", "")
        reasoning_prefix = _reasoning_prefix(prefix_text, question)
        branches = sorted(bundle["branches"], key=lambda r: int(r.get("sample_id", 0)))[:4]
        tasks.append(
            {
                "prefix_id": prefix_id,
                "problem_id": pfx.get("problem_id", bundle["continue"]["problem_id"]),
                "question": question,
                "reasoning_prefix": reasoning_prefix,
                "continue_continuation": bundle["continue"].get("continuation", ""),
                "branch_continuations": [b.get("continuation", "") for b in branches],
            }
        )
    return tasks


def write_utility_report(
    data_dir: Path,
    out_dir: Path,
    *,
    target_model: str,
    score_method: str = "greedy",
    thresholds: list[int] | None = None,
    report_path: Path | None = None,
) -> Path:
    thresholds = thresholds or [5, 6, 7, 8]
    scores_path = out_dir / f"utility_scores_{model_slug(target_model)}.jsonl"
    rows = [r for r in _load_jsonl(scores_path) if len(r.get("utility_scores", [])) >= 5]
    table = summarize_oracle_table(rows, thresholds)
    summary = {
        "n_scored": len(rows),
        "target_model": target_model,
        "score_method": score_method,
        "oracle_by_tau": table,
    }
    (out_dir / "oracle_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    prefixes = {r["prefix_id"]: r for r in _load_jsonl(data_dir / "prefixes.jsonl")}
    problems = {r["problem_id"]: r for r in _load_jsonl(data_dir / "problems.jsonl")}
    from reasoning_branch_dataset.action_study.report_samples import (
        format_utility_cases_md,
        pick_utility_cases,
    )

    md_lines = [
        "# Pilot v3 — Utility Oracle Report",
        "",
        "> SpecReason-style QwQ utility scoring on v2 admission_main prefixes (1 greedy + 4 branch).",
        "",
        f"- v2 data: `action_study_pilot_v2/`",
        f"- scores: `action_study_pilot_v3/utility_scores_{model_slug(target_model)}.jsonl`",
        f"- target: `{target_model}`",
        f"- scored prefixes: **{len(rows)}**",
        f"- score method: `{score_method}`",
        "",
        "## Oracle table by τ",
        "",
        "| τ | Continue-sufficient | Branch-rescuable | Handoff-required |",
        "|---|--------------------:|-----------------:|-----------------:|",
    ]
    for t in table:
        md_lines.append(
            f"| {t['tau']} | {t['continue_sufficient']} ({t['pct_continue_sufficient']}%) "
            f"| {t['branch_rescuable']} ({t['pct_branch_rescuable']}%) "
            f"| {t['handoff_required']} ({t['pct_handoff_required']}%) |"
        )
    md_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- **Continue-sufficient**: `u_0 ≥ τ` — greedy draft step acceptable.",
            "- **Branch-rescuable**: `u_0 < τ` but `max(u_1..4) ≥ τ` — selective Branch may avoid target generation.",
            "- **Handoff-required**: all candidates below τ — target should take over.",
            "",
            "This is SpecReason-style utility scoring, **not** token-level acceptance replay.",
            "",
            "> **Probe training blocked** until `pilot_v3_audit_report.md` passes length-bias + step-quality checks.",
        ]
    )
    cases = pick_utility_cases(rows, prefixes, problems, tau=7, n_each=2)
    md_lines.extend(format_utility_cases_md(cases))

    report_path = report_path or (out_dir.parent / "pilot_v3_report.md")
    report_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Wrote {scores_path}")
    print(f"Wrote {report_path}")
    return report_path


def run_utility_scoring(
    data_dir: Path,
    out_dir: Path,
    *,
    target_model: str,
    score_method: str = "greedy",
    thresholds: list[int] | None = None,
    resume: bool = True,
    max_prefixes: int | None = None,
    verifier=None,
) -> None:
    thresholds = thresholds or [5, 6, 7, 8]
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / f"utility_scores_{model_slug(target_model)}.jsonl"

    tasks = load_candidate_tasks(data_dir, admission_only=True)
    if max_prefixes is not None:
        tasks = tasks[:max_prefixes]

    done = _done_ids(scores_path) if resume else set()
    own = verifier is None
    if own:
        verifier = build_target_verifier(target_model, engine="vllm", debug=True)
    llm, tokenizer = verifier.llm, verifier.tokenizer

    try:
        for task in tqdm(tasks, desc="utility_score"):
            if task["prefix_id"] in done:
                continue
            scores: list[int] = []
            details: list[dict[str, Any]] = []

            candidates = [("continue", task["continue_continuation"])] + [
                (f"branch_{i}", c) for i, c in enumerate(task["branch_continuations"])
            ]
            for name, cont in candidates:
                try:
                    r = score_step_vllm(
                        llm,
                        tokenizer,
                        task["question"],
                        task["reasoning_prefix"],
                        cont,
                        score_method=score_method,
                    )
                except Exception as exc:
                    r = {
                        "utility_score": None,
                        "score_token": None,
                        "candidate_step": extract_next_substantive_step(cont, question=task["question"])["candidate_step"],
                        "error": str(exc),
                    }
                if r.get("utility_score") is not None:
                    scores.append(int(r["utility_score"]))
                details.append({"candidate": name, **r})

            if len(scores) != len(candidates):
                continue

            row = {
                "prefix_id": task["prefix_id"],
                "problem_id": task["problem_id"],
                "target_model": target_model,
                "utility_scores": scores,
                "u_greedy": scores[0],
                "u_branch": scores[1:],
                "u_best_branch": max(scores[1:]) if len(scores) > 1 else None,
                "u_max": max(scores),
                "candidate_details": details,
            }
            row.update(classify_oracle(scores, tau=7))
            _append_jsonl(scores_path, row)
    finally:
        if own:
            del verifier
            gc.collect()

    write_utility_report(
        data_dir,
        out_dir,
        target_model=target_model,
        score_method=score_method,
        thresholds=thresholds,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="V3 utility scoring on 1+4 candidates")
    parser.add_argument(
        "--data-dir",
        default="/mnt/afs/L202500372/reasoning_branch_dataset/outputs/action_study_pilot_v2",
    )
    parser.add_argument(
        "--out-dir",
        default="/mnt/afs/L202500372/reasoning_branch_dataset/outputs/action_study_pilot_v3",
    )
    parser.add_argument(
        "--target-model",
        default="/mnt/afs/L202500372/specreason/models/QwQ-32B",
    )
    parser.add_argument("--score-method", default="greedy", choices=["greedy", "average"])
    parser.add_argument("--max-prefixes", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    if args.report_only:
        write_utility_report(
            Path(args.data_dir),
            Path(args.out_dir),
            target_model=args.target_model,
            score_method=args.score_method,
        )
        return

    run_utility_scoring(
        Path(args.data_dir),
        Path(args.out_dir),
        target_model=args.target_model,
        score_method=args.score_method,
        resume=not args.no_resume,
        max_prefixes=args.max_prefixes,
    )


if __name__ == "__main__":
    main()

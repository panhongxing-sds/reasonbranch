"""Phase 2b: pairwise independent judge on raw weak Branch-rescuable cases."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from reasoning_branch_dataset.action_study.oracle_labels import classify_oracle
from reasoning_branch_dataset.action_study.pairwise_judge import judge_pairwise_vllm
from reasoning_branch_dataset.action_study.run_utility_scoring import _load_jsonl, load_candidate_tasks
from reasoning_branch_dataset.action_study.target_verifier import build_target_verifier


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _done_ids(path: Path) -> set[str]:
    return {r["prefix_id"] for r in _load_jsonl(path) if "prefix_id" in r}


def run_pairwise_audit(
    v2_dir: Path,
    v3_dir: Path,
    *,
    target_model: str,
    tau: int = 7,
) -> Path:
    scores_path = v3_dir / "utility_scores_QwQ-32B.jsonl"
    out_path = v3_dir / "pairwise_audit_results.jsonl"
    branch_ids = {
        r["prefix_id"]
        for r in _load_jsonl(scores_path)
        if classify_oracle(r.get("utility_scores", []), tau=tau)["oracle_label"]
        in ("weak_branch_rescuable", "branch_rescuable")
    }

    tasks = {t["prefix_id"]: t for t in load_candidate_tasks(v2_dir) if t["prefix_id"] in branch_ids}
    scores_by_id = {r["prefix_id"]: r for r in _load_jsonl(scores_path)}
    done = _done_ids(out_path)

    verifier = build_target_verifier(target_model, engine="vllm", debug=True)
    llm, tokenizer = verifier.llm, verifier.tokenizer

    try:
        for pid in tqdm(sorted(branch_ids), desc="pairwise"):
            if pid in done or pid not in tasks:
                continue
            task = tasks[pid]
            row = scores_by_id[pid]
            details = row.get("candidate_details") or []
            if len(details) < 5:
                continue
            greedy_step = details[0].get("candidate_step") or task["continue_continuation"]
            u0 = int(row["utility_scores"][0])
            best_i = max(range(1, 5), key=lambda i: int(row["utility_scores"][i]))
            branch_step = details[best_i].get("candidate_step") or task["branch_continuations"][best_i - 1]
            u_best = int(row["utility_scores"][best_i])

            judged = judge_pairwise_vllm(
                llm,
                tokenizer,
                task["question"],
                task["reasoning_prefix"],
                greedy_step,
                branch_step,
                seed=hash(pid) % 10_000,
            )
            _append_jsonl(
                out_path,
                {
                    "prefix_id": pid,
                    "u_greedy": u0,
                    "u_best_branch": u_best,
                    "best_branch_index": best_i - 1,
                    "qwQ_weak_branch": True,
                    **judged,
                    "true_branch_rescue": judged["pairwise_verdict"] == "BRANCH_BETTER",
                },
            )
    finally:
        del verifier
        gc.collect()
    return out_path


def summarize_pairwise(results_path: Path) -> dict[str, Any]:
    rows = _load_jsonl(results_path)
    if not rows:
        return {}
    from collections import Counter

    verdicts = Counter(r.get("pairwise_verdict", "UNKNOWN") for r in rows)
    true_branch = sum(1 for r in rows if r.get("true_branch_rescue"))
    n = len(rows)
    return {
        "n_reviewed": n,
        "verdict_counts": dict(verdicts),
        "branch_better_count": verdicts.get("BRANCH_BETTER", 0),
        "equivalent_count": verdicts.get("EQUIVALENT", 0),
        "greedy_better_count": verdicts.get("GREEDY_BETTER", 0),
        "both_reject_count": verdicts.get("BOTH_REJECT", 0),
        "precision_branch_vs_qwq_weak": round(true_branch / n, 4) if n else 0.0,
        "cleaned_branch_rate_of_1548": round(true_branch / 1548, 4),
        "unlock_precision_threshold": 0.70,
        "unlock_min_branch_n": 50,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-dir", default="/mnt/afs/L202500372/reasoning_branch_dataset/outputs/action_study_pilot_v2")
    parser.add_argument("--v3-dir", default="/mnt/afs/L202500372/reasoning_branch_dataset/outputs/action_study_pilot_v3")
    parser.add_argument("--target-model", default="/mnt/afs/L202500372/specreason/models/QwQ-32B")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    v3_dir = Path(args.v3_dir)
    results = v3_dir / "pairwise_audit_results.jsonl"
    if not args.summarize_only:
        run_pairwise_audit(Path(args.v2_dir), v3_dir, target_model=args.target_model)
    summary = summarize_pairwise(results)
    (v3_dir / "pairwise_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

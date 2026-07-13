"""Phase 2a: shuffle-rescore stability on audit sample."""

from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from reasoning_branch_dataset.action_study.oracle_labels import classify_oracle, is_accept
from reasoning_branch_dataset.action_study.run_utility_scoring import (
    _load_jsonl,
    _reasoning_prefix,
    load_candidate_tasks,
)
from reasoning_branch_dataset.action_study.specreason_scorer import score_step_vllm
from reasoning_branch_dataset.action_study.target_verifier import build_target_verifier


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _done_ids(path: Path) -> set[str]:
    return {r["prefix_id"] for r in _load_jsonl(path) if "prefix_id" in r}


def run_shuffle_rescore(
    v2_dir: Path,
    v3_dir: Path,
    *,
    target_model: str,
    sample_path: Path | None = None,
    tau: int = 7,
    seed: int = 42,
) -> Path:
    sample_path = sample_path or (v3_dir / "audit_sample.jsonl")
    out_path = v3_dir / "shuffle_rescore_results.jsonl"
    sample_ids = {r["prefix_id"] for r in _load_jsonl(sample_path)}
    scores_by_id = {r["prefix_id"]: r for r in _load_jsonl(v3_dir / "utility_scores_QwQ-32B.jsonl")}

    tasks = [t for t in load_candidate_tasks(v2_dir) if t["prefix_id"] in sample_ids]
    done = _done_ids(out_path)

    verifier = build_target_verifier(target_model, engine="vllm", debug=True)
    llm, tokenizer = verifier.llm, verifier.tokenizer
    rng = random.Random(seed)

    try:
        for task in tqdm(tasks, desc="shuffle_rescore"):
            pid = task["prefix_id"]
            if pid in done:
                continue
            pass1 = scores_by_id.get(pid, {})
            pass1_scores = pass1.get("utility_scores") or []

            candidates = [task["continue_continuation"]] + task["branch_continuations"]
            names = ["continue"] + [f"branch_{i}" for i in range(4)]
            order = list(range(5))
            rng.shuffle(order)

            pass2_scores: list[int | None] = [None] * 5
            pass2_details: list[dict[str, Any]] = []
            for idx in order:
                name = names[idx]
                cont = candidates[idx]
                r = score_step_vllm(
                    llm,
                    tokenizer,
                    task["question"],
                    task["reasoning_prefix"],
                    cont,
                )
                pass2_scores[idx] = int(r["utility_score"])
                pass2_details.append({"candidate": name, **r})

            p1 = [int(s) for s in pass1_scores]
            p2 = [int(s) for s in pass2_scores if s is not None]
            oracle1 = classify_oracle(p1, tau=tau)["oracle_label"] if len(p1) == 5 else "unknown"
            oracle2 = classify_oracle(p2, tau=tau)["oracle_label"] if len(p2) == 5 else "unknown"

            row = {
                "prefix_id": pid,
                "pass1_scores": p1,
                "pass2_scores": p2,
                "pass2_order": order,
                "oracle_pass1": oracle1,
                "oracle_pass2": oracle2,
                "oracle_stable": oracle1 == oracle2,
                "per_candidate": [
                    {
                        "name": names[i],
                        "u1": p1[i] if i < len(p1) else None,
                        "u2": p2[i],
                        "exact_match": p1[i] == p2[i] if i < len(p1) else None,
                        "accept_match": is_accept(p1[i], tau=tau) == is_accept(p2[i], tau=tau)
                        if i < len(p1)
                        else None,
                    }
                    for i in range(5)
                ],
            }
            _append_jsonl(out_path, row)
    finally:
        del verifier
        gc.collect()
    return out_path


def summarize_shuffle_rescore(results_path: Path, *, tau: int = 7) -> dict[str, Any]:
    rows = _load_jsonl(results_path)
    if not rows:
        return {}
    exact = []
    within1 = []
    accept = []
    oracle_stable = []
    for row in rows:
        for pc in row.get("per_candidate", []):
            u1, u2 = pc.get("u1"), pc.get("u2")
            if u1 is None or u2 is None:
                continue
            exact.append(u1 == u2)
            within1.append(abs(u1 - u2) <= 1)
            accept.append(is_accept(u1, tau=tau) == is_accept(u2, tau=tau))
        oracle_stable.append(row.get("oracle_stable", False))

    n = len(exact)
    return {
        "n_prefixes": len(rows),
        "n_candidates": n,
        "exact_agreement": float(np.mean(exact)) if exact else 0.0,
        "within1_agreement": float(np.mean(within1)) if within1 else 0.0,
        "accept_agreement": float(np.mean(accept)) if accept else 0.0,
        "oracle_action_agreement": float(np.mean(oracle_stable)) if oracle_stable else 0.0,
        "unlock_thresholds": {
            "accept_agreement": 0.90,
            "oracle_action_agreement": 0.85,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-dir", default="/mnt/afs/L202500372/reasoning_branch_dataset/outputs/action_study_pilot_v2")
    parser.add_argument("--v3-dir", default="/mnt/afs/L202500372/reasoning_branch_dataset/outputs/action_study_pilot_v3")
    parser.add_argument("--target-model", default="/mnt/afs/L202500372/specreason/models/QwQ-32B")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    v3_dir = Path(args.v3_dir)
    results = v3_dir / "shuffle_rescore_results.jsonl"
    if not args.summarize_only:
        run_shuffle_rescore(Path(args.v2_dir), v3_dir, target_model=args.target_model)
    summary = summarize_shuffle_rescore(results)
    (v3_dir / "shuffle_rescore_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

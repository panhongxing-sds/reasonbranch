#!/usr/bin/env python3
"""Relabel uncertainty study with v2 framework (parallel API)."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

from reasoning_branch_dataset.action_study.api_labeling import api_workers, label_prefixes_parallel
from reasoning_branch_dataset.action_study.api_validity import ValidityClient
from reasoning_branch_dataset.action_study.diversity import behavior_state, future_system_action, recovery_profile
from reasoning_branch_dataset.grading import grade_math_answer

OUT = Path(os.environ.get("ACTION_STUDY_OUT", "reasoning_branch_dataset/outputs/action_study_uncertainty_v1"))


def load(name: str) -> list[dict]:
    p = OUT / f"{name}.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def write(name: str, rows: list[dict]) -> None:
    with (OUT / f"{name}.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(out_dir: Path | None = None) -> None:
    global OUT
    if out_dir is not None:
        OUT = out_dir
    traces = {t["problem_id"]: t for t in load("traces")}
    prefixes = load("prefixes")
    next_by: dict[str, list] = {}
    for n in load("next_step_samples"):
        next_by.setdefault(n["prefix_id"], []).append(n)

    client = ValidityClient.from_env(cache_path=OUT / "api_cache_v2.jsonl")
    workers = api_workers()
    print(f"API enabled: {client.enabled}, workers: {workers}, prefixes: {len(prefixes)}")

    validity_rows, cluster_rows, new_prefixes = label_prefixes_parallel(
        client, prefixes, traces, next_by, workers=workers
    )

    actions = load("actions")
    gold_by_prob = {t["problem_id"]: t["gold_answer"] for t in load("traces")}

    for a in actions:
        gold = gold_by_prob[a["problem_id"]]
        text = a.get("start_checkpoint", "") + a.get("continuation", "")
        score = grade_math_answer(text, gold, require_marker=True)
        a.update(
            {
                "predicted_answer": score["predicted_answer"],
                "final_answer": score["final_answer"],
                "is_correct": score["is_correct"],
                "correct": score["is_correct"],
                "evaluation_status": score["evaluation_status"],
                "evaluation_error": score.get("evaluation_error"),
            }
        )

    grouped_actions: dict[tuple, list] = defaultdict(list)
    for a in actions:
        grouped_actions[(a["problem_id"], a["prefix_id"], a["action_type"])].append(a)

    outcomes = []
    for p in new_prefixes:
        pid = p["prefix_id"]
        prob_id = p["problem_id"]
        cont_rows = grouped_actions.get((prob_id, pid, "continue"), [])
        branch_rows = grouped_actions.get((prob_id, pid, "branch"), [])
        cont = cont_rows[0] if cont_rows else {}
        ok_br = [r for r in branch_rows if r.get("evaluation_status") == "OK"]
        correct = sum(1 for r in ok_br if r.get("is_correct") == 1)
        evaluated = len(ok_br)
        rec = recovery_profile(correct, evaluated) if evaluated else "UNKNOWN"
        b_state = behavior_state(
            prefix_validity=p["prefix_validity"],
            prefix_substantiveness=p["prefix_substantiveness"],
            strategy_diversity=p["strategy_diversity"],
            recovery_profile=rec,
            continue_correct=cont.get("is_correct"),
            branch_pass_at_k=int(correct > 0) if evaluated else None,
        )
        p["behavior_state"] = b_state
        p["recovery_profile"] = rec
        p["future_system_action"] = future_system_action(b_state)

        for a in actions:
            if a["prefix_id"] == pid:
                a["state_bucket"] = b_state

        if cont:
            oracle = cont["is_correct"] if cont.get("evaluation_status") == "OK" else None
            outcomes.append(
                {
                    "problem_id": prob_id,
                    "prefix_id": pid,
                    "state_bucket": p["state_bucket"],
                    "behavior_state": b_state,
                    "future_system_action": future_system_action(b_state),
                    "operation": "continue",
                    "oracle_recoverable": oracle,
                    "draft_generated_tokens": cont.get("generated_tokens", 0),
                    "evaluation_status": cont.get("evaluation_status", "OK"),
                    "pass_at_k": oracle,
                    "is_correct": oracle,
                    "num_tokens": cont.get("generated_tokens", 0),
                }
            )
        outcomes.append(
            {
                "problem_id": prob_id,
                "prefix_id": pid,
                "state_bucket": p["state_bucket"],
                "behavior_state": b_state,
                "future_system_action": future_system_action(b_state),
                "operation": "branch",
                "oracle_recoverable": int(correct > 0) if evaluated else None,
                "branch_correct_count": correct if evaluated else None,
                "branch_accuracy_at_k": (correct / evaluated) if evaluated else None,
                "branch_evaluated_count": evaluated,
                "branch_evaluation_errors": len(branch_rows) - evaluated,
                "recovery_profile": rec,
                "draft_generated_tokens": sum(r.get("generated_tokens", 0) for r in branch_rows),
                "evaluation_status": "ERROR" if not ok_br else "OK",
                "pass_at_k": int(correct > 0) if evaluated else None,
                "is_correct": int(correct > 0) if evaluated else None,
                "num_tokens": sum(r.get("generated_tokens", 0) for r in branch_rows),
            }
        )

    write("prefixes", new_prefixes)
    write("validity_labels", validity_rows)
    write("cluster_labels", cluster_rows)
    write("actions", actions)
    write("outcome_results", outcomes)
    write("action_results", outcomes)

    from collections import Counter

    print("behavior_state", Counter(p.get("behavior_state") for p in new_prefixes))
    print("substantiveness", Counter(p.get("prefix_substantiveness") for p in new_prefixes))
    print("strategy", Counter(p.get("strategy_diversity") for p in new_prefixes))
    print("done", len(new_prefixes), "prefixes", len(outcomes), "outcomes")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()
    if args.output_dir:
        main(args.output_dir)
    else:
        main()

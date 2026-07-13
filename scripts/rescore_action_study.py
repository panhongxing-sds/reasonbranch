#!/usr/bin/env python3
"""Re-score existing action study outputs with the fixed grader."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reasoning_branch_dataset.grading import grade_math_answer


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def rescore(out_dir: Path) -> dict:
    traces = _load_jsonl(out_dir / "traces.jsonl")
    actions = _load_jsonl(out_dir / "actions.jsonl")
    action_results = _load_jsonl(out_dir / "action_results.jsonl")

    gold_by_problem = {t["problem_id"]: t["gold_answer"] for t in traces}

    trace_updates = 0
    for row in traces:
        gold = row["gold_answer"]
        score = grade_math_answer(row["full_reasoning"], gold, require_marker=False)
        old = row.get("is_correct")
        row["predicted_answer"] = score["predicted_answer"]
        row["is_correct"] = score["is_correct"]
        row["evaluation_status"] = score["evaluation_status"]
        row["evaluation_error"] = score["evaluation_error"]
        if old != row["is_correct"]:
            trace_updates += 1

    action_updates = 0
    for row in actions:
        gold = gold_by_problem.get(row["problem_id"], "")
        full_text = row.get("start_checkpoint", "") + row.get("continuation", "")
        score = grade_math_answer(full_text, gold, require_marker=True)
        old = row.get("is_correct")
        row["predicted_answer"] = score["predicted_answer"]
        row["final_answer"] = score["final_answer"]
        row["is_correct"] = score["is_correct"]
        row["correct"] = score["is_correct"]
        row["evaluation_status"] = score["evaluation_status"]
        row["evaluation_error"] = score["evaluation_error"]
        if old != row["is_correct"]:
            action_updates += 1

    # Rebuild action_results from actions
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in actions:
        key = (row["problem_id"], row["prefix_id"], row["action_type"])
        grouped.setdefault(key, []).append(row)

    new_results = []
    for (problem_id, prefix_id, action_type), rows in grouped.items():
        state = rows[0].get("state_bucket", "UNCLEAR")
        if action_type == "continue":
            r0 = rows[0]
            new_results.append(
                {
                    "problem_id": problem_id,
                    "prefix_id": prefix_id,
                    "state_bucket": state,
                    "action": "continue",
                    "oracle_recoverable": r0.get("is_correct"),
                    "draft_generated_tokens": r0.get("generated_tokens", 0),
                    "discarded_prefix_tokens": 0,
                    "action_start": "current_prefix",
                    "evaluation_status": r0.get("evaluation_status", "OK"),
                    "pass_at_k": r0.get("is_correct"),
                    "is_correct": r0.get("is_correct"),
                    "num_tokens": r0.get("generated_tokens", 0),
                    "debug_latency_sec": r0.get("latency_ms", 0) / 1000.0,
                }
            )
        else:
            ok = [r for r in rows if r.get("evaluation_status") == "OK"]
            pass_at_k = None
            if ok:
                pass_at_k = int(any(r.get("is_correct") == 1 for r in ok))
            discarded = 0
            if action_type == "rollback" and rows:
                # approximate from first row if available in prefix metadata later
                discarded = rows[0].get("discarded_prefix_tokens", 0)
            new_results.append(
                {
                    "problem_id": problem_id,
                    "prefix_id": prefix_id,
                    "state_bucket": state,
                    "action": action_type,
                    "oracle_recoverable": pass_at_k,
                    "draft_generated_tokens": sum(r.get("generated_tokens", 0) for r in rows),
                    "discarded_prefix_tokens": discarded,
                    "action_start": (
                        "previous_checkpoint" if action_type == "rollback" else "current_prefix"
                    ),
                    "evaluation_status": "ERROR" if not ok else "OK",
                    "pass_at_k": pass_at_k,
                    "oracle_branch_recoverable": pass_at_k if action_type == "branch" else None,
                    "oracle_rollback_recoverable": pass_at_k if action_type == "rollback" else None,
                    "is_correct": pass_at_k,
                    "num_tokens": sum(r.get("generated_tokens", 0) for r in rows),
                    "debug_latency_sec": sum(r.get("latency_ms", 0) for r in rows) / 1000.0,
                }
            )

    _write_jsonl(out_dir / "traces.rescored.jsonl", traces)
    _write_jsonl(out_dir / "actions.rescored.jsonl", actions)
    _write_jsonl(out_dir / "action_results.rescored.jsonl", new_results)

    return {
        "trace_updates": trace_updates,
        "action_updates": action_updates,
        "prefixes": len({r["prefix_id"] for r in new_results}),
        "results": new_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = rescore(args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

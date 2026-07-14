"""Re-derive prefix admission from final traces/actions (ignore stale admission_pass)."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from reasoning_branch_dataset.action_study.admission import trace_is_complete


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _rollout_ok(row: dict[str, Any]) -> bool:
    pred = row.get("predicted_answer")
    return row.get("evaluation_status") == "OK" and pred not in ("", None)


def trace_eligible(trace: dict[str, Any], *, problem: dict[str, Any] | None = None) -> bool:
    if problem and not problem.get("input_complete", True):
        return False
    if problem and problem.get("requires_visual_input") and not problem.get("input_complete", True):
        return False
    if not trace_is_complete(trace):
        return False
    if trace.get("evaluation_status") != "OK":
        return False
    if not trace.get("has_final_answer", trace.get("evaluation_status") == "OK"):
        return False
    return True


def derive_prefix_admission_row(
    *,
    prefix: dict[str, Any],
    trace: dict[str, Any] | None,
    problem: dict[str, Any] | None,
    continue_action: dict[str, Any] | None,
    branch_actions: list[dict[str, Any]],
    expected_branches: int = 4,
) -> dict[str, Any]:
    trace_ok = trace_eligible(trace or {}, problem=problem)
    substantive = prefix.get("prefix_substantiveness") == "SUBSTANTIVE"
    continue_ok = continue_action is not None and _rollout_ok(continue_action)

    branch_expected = expected_branches
    branch_valid = [b for b in branch_actions if _rollout_ok(b)]
    branch_valid_count = len(branch_valid)
    branch_correct_count = sum(1 for b in branch_valid if b.get("is_correct") == 1)
    branch_complete = branch_valid_count == branch_expected

    admission_main = trace_ok and substantive and continue_ok and branch_complete
    admission_partial = trace_ok and substantive and continue_ok and branch_valid_count >= 3

    exclusion_reason: str | None = None
    if problem and not problem.get("input_complete", True):
        exclusion_reason = "excluded_missing_context"
    elif not trace_ok:
        exclusion_reason = "excluded_truncated_trace"
    elif not substantive:
        exclusion_reason = "excluded_non_substantive"
    elif not continue_ok:
        exclusion_reason = "excluded_action_incomplete"
    elif not branch_complete:
        exclusion_reason = "excluded_action_incomplete"

    return {
        "prefix_id": prefix["prefix_id"],
        "problem_id": prefix["problem_id"],
        "base_trace_status": (trace or {}).get("evaluation_status"),
        "base_trace_complete": trace_ok,
        "trace_eligible": trace_ok,
        "continue_eligible": continue_ok,
        "branch_expected": branch_expected,
        "branch_valid_count": branch_valid_count,
        "branch_correct_count": branch_correct_count,
        "branch_complete": branch_complete,
        "admission_main": admission_main,
        "admission_partial": admission_partial,
        "admission_exclusion_reason": exclusion_reason,
        "include_in_main_experiment": admission_main,
    }


def recompute_admission(data_dir: Path, *, expected_branches: int = 4) -> dict[str, Any]:
    prefixes = _load_jsonl(data_dir / "prefixes.jsonl")
    traces = {r["problem_id"]: r for r in _load_jsonl(data_dir / "traces.jsonl")}
    problems = {r["problem_id"]: r for r in _load_jsonl(data_dir / "problems.jsonl")}

    actions = _load_jsonl(data_dir / "actions.jsonl")
    continue_by_pfx: dict[str, dict[str, Any]] = {}
    branch_by_pfx: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for action in actions:
        pid = action["prefix_id"]
        if action.get("action_type") == "continue":
            continue_by_pfx[pid] = action
        elif action.get("action_type") == "branch":
            branch_by_pfx[pid].append(action)

    derived_rows: list[dict[str, Any]] = []
    updated_prefixes: list[dict[str, Any]] = []
    for prefix in prefixes:
        pid = prefix["prefix_id"]
        prob_id = prefix["problem_id"]
        row = derive_prefix_admission_row(
            prefix=prefix,
            trace=traces.get(prob_id),
            problem=problems.get(prob_id),
            continue_action=continue_by_pfx.get(pid),
            branch_actions=branch_by_pfx.get(pid, []),
            expected_branches=expected_branches,
        )
        derived_rows.append(row)
        merged = {**prefix, **row}
        # overwrite stale pipeline fields
        merged["admission_pass"] = row["admission_main"]
        merged["admission_reason"] = row["admission_exclusion_reason"]
        updated_prefixes.append(merged)

    _write_jsonl(data_dir / "prefix_admission.jsonl", derived_rows)
    _write_jsonl(data_dir / "prefixes.jsonl", updated_prefixes)

    branch_hist = Counter(r["branch_valid_count"] for r in derived_rows)
    summary = {
        "total_prefixes": len(prefixes),
        "complete_trace_prefixes": sum(1 for r in derived_rows if r["base_trace_complete"]),
        "substantive_prefixes": sum(
            1 for p in prefixes if p.get("prefix_substantiveness") == "SUBSTANTIVE"
        ),
        "continue_ok": sum(1 for r in derived_rows if r["continue_eligible"]),
        "branch_4of4_ok": branch_hist.get(expected_branches, 0),
        "branch_partial": sum(
            1 for r in derived_rows if 0 < r["branch_valid_count"] < expected_branches
        ),
        "admission_main": sum(1 for r in derived_rows if r["admission_main"]),
        "admission_partial": sum(1 for r in derived_rows if r["admission_partial"]),
        "excluded_missing_context": sum(
            1 for r in derived_rows if r["admission_exclusion_reason"] == "excluded_missing_context"
        ),
        "excluded_truncated_trace": sum(
            1 for r in derived_rows if r["admission_exclusion_reason"] == "excluded_truncated_trace"
        ),
        "excluded_action_incomplete": sum(
            1 for r in derived_rows if r["admission_exclusion_reason"] == "excluded_action_incomplete"
        ),
        "excluded_non_substantive": sum(
            1 for r in derived_rows if r["admission_exclusion_reason"] == "excluded_non_substantive"
        ),
        "branch_valid_histogram": {str(k): v for k, v in sorted(branch_hist.items())},
        "unique_complete_trace_problems": len(
            {r["problem_id"] for r in derived_rows if r["base_trace_complete"]}
        ),
        "excluded_problems_missing_figure": sum(
            1 for _ in _load_jsonl(data_dir / "excluded_problems.jsonl")
        ),
        "traces_ok": sum(1 for t in traces.values() if t.get("evaluation_status") == "OK"),
        "traces_truncated": sum(1 for t in traces.values() if t.get("evaluation_status") == "TRUNCATED"),
    }

    (data_dir / "admission_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Re-derive prefix admission from final artifacts")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--expected-branches", type=int, default=4)
    args = parser.parse_args()
    summary = recompute_admission(args.data_dir, expected_branches=args.expected_branches)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

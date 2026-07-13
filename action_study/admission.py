"""Pilot v2 admission gates for prefix/action study."""

from __future__ import annotations

from typing import Any


def trace_is_complete(trace_row: dict[str, Any]) -> bool:
    status = trace_row.get("evaluation_status")
    finish = trace_row.get("finish_reason")
    if finish == "length":
        return False
    if status == "TRUNCATED":
        return False
    if status == "OK":
        return True
    return bool(trace_row.get("has_final_answer"))


def compute_branch_gain(
    continue_correct: int | None,
    branch_pass_at_k: int | None,
) -> int | float | None:
    if continue_correct is None or branch_pass_at_k is None:
        return None
    return int(branch_pass_at_k) - int(continue_correct)


def prefix_admission_pass(
    *,
    problem_row: dict[str, Any],
    trace_row: dict[str, Any],
    continue_result: dict[str, Any],
    branch_result: dict[str, Any],
    min_branch_evaluated: int = 2,
) -> tuple[bool, str | None]:
    if not problem_row.get("input_complete", True):
        return False, problem_row.get("exclusion_reason") or "missing_input"
    if not trace_is_complete(trace_row):
        return False, "incomplete_trace"
    if trace_row.get("evaluation_status") != "OK":
        return False, f"trace_status_{trace_row.get('evaluation_status', 'unknown')}"

    cont_status = continue_result.get("evaluation_status")
    if cont_status != "OK":
        return False, f"continue_{cont_status}"

    branch_eval = branch_result.get("branch_evaluated_count") or 0
    if branch_eval < min_branch_evaluated:
        return False, "insufficient_branch_rollouts"

    return True, None

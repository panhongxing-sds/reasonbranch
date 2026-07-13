"""Execute Continue / Branch / Rollback actions for each prefix."""

from __future__ import annotations

import time
from typing import Any

from reasoning_branch_dataset.action_study.diversity import recovery_profile
from reasoning_branch_dataset.action_study.prefix_extract import StudyPrefix
from reasoning_branch_dataset.grading import classify_generation_outcome


def _flatten_action_rows(
    result: dict[str, Any],
    *,
    problem_id: str,
    prefix_id: str,
    state_bucket: str,
    start_checkpoint: str,
    temperature: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    action = result["action"]
    if action == "continue":
        rows.append(
            {
                "problem_id": problem_id,
                "prefix_id": prefix_id,
                "state_bucket": state_bucket,
                "action_type": "continue",
                "sample_id": 0,
                "start_checkpoint": start_checkpoint,
                "continuation": result.get("response_text", ""),
                "predicted_answer": result.get("predicted_answer", result.get("final_answer", "")),
                "final_answer": result.get("final_answer", result.get("predicted_answer", "")),
                "is_correct": result.get("is_correct"),
                "correct": result.get("is_correct"),
                "evaluation_status": result.get("evaluation_status", "OK"),
                "evaluation_error": result.get("evaluation_error"),
                "finish_reason": result.get("finish_reason"),
                "has_boxed_answer": result.get("has_boxed_answer"),
                "generated_tokens": result.get("num_tokens", 0),
                "draft_generated_tokens": result.get("num_tokens", 0),
                "latency_ms": round(result.get("latency_sec", 0.0) * 1000, 2),
                "debug_latency_ms": round(result.get("latency_sec", 0.0) * 1000, 2),
                "seed": None,
                "temperature": 0.0,
            }
        )
        return rows

    for br in result.get("branches", []):
        rows.append(
            {
                "problem_id": problem_id,
                "prefix_id": prefix_id,
                "state_bucket": state_bucket,
                "action_type": action,
                "sample_id": int(br.get("branch_id", "0").split("_")[-1].lstrip("br") or 0)
                if "_" in str(br.get("branch_id", ""))
                else 0,
                "start_checkpoint": start_checkpoint,
                "continuation": br.get("response_text", ""),
                "predicted_answer": br.get("predicted_answer", br.get("final_answer", "")),
                "final_answer": br.get("final_answer", br.get("predicted_answer", "")),
                "is_correct": br.get("is_correct"),
                "correct": br.get("is_correct"),
                "evaluation_status": br.get("evaluation_status", "OK"),
                "evaluation_error": br.get("evaluation_error"),
                "finish_reason": br.get("finish_reason"),
                "has_boxed_answer": br.get("has_boxed_answer"),
                "generated_tokens": br.get("num_tokens", 0),
                "draft_generated_tokens": br.get("num_tokens", 0),
                "latency_ms": round(br.get("latency_sec", result.get("latency_sec", 0.0)) * 1000, 2),
                "debug_latency_ms": round(br.get("latency_sec", result.get("latency_sec", 0.0)) * 1000, 2),
                "seed": br.get("seed"),
                "temperature": temperature,
            }
        )
    return rows


def _branch_metrics(branch_rows: list[dict[str, Any]], k: int) -> dict[str, Any]:
    ok = [b for b in branch_rows if b.get("evaluation_status", "OK") == "OK"]
    err_count = len(branch_rows) - len(ok)
    if not ok:
        return {
            "pass_at_k": None,
            "branch_correct_count": None,
            "branch_accuracy_at_k": None,
            "branch_evaluated_count": 0,
            "branch_evaluation_errors": err_count,
            "recovery_profile": None,
        }
    correct = sum(1 for b in ok if b.get("is_correct") == 1)
    evaluated = len(ok)
    return {
        "pass_at_k": int(correct > 0),
        "branch_correct_count": correct,
        "branch_accuracy_at_k": correct / evaluated,
        "branch_evaluated_count": evaluated,
        "branch_evaluation_errors": err_count,
        "recovery_profile": recovery_profile(correct, evaluated),
    }


def _pass_at_k(branches: list[dict[str, Any]]) -> int | None:
    ok = [b for b in branches if b.get("evaluation_status", "OK") == "OK"]
    if not ok:
        return None
    return int(any(b.get("is_correct") == 1 for b in ok))


def run_continue(
    engine,
    prompt: str,
    prefix: StudyPrefix,
    gold_answer: str,
    *,
    max_tokens: int,
    retry_max_tokens: int = 0,
) -> dict[str, Any]:
    full_prefix = prompt + prefix.reasoning_prefix
    t0 = time.perf_counter()
    out = engine.generate_full_continuations(
        full_prefix,
        k=1,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        retry_max_tokens=retry_max_tokens,
    )[0]
    latency = time.perf_counter() - t0
    score = classify_generation_outcome(
        full_prefix + out["text"],
        gold_answer,
        finish_reason=out.get("finish_reason"),
        require_marker=True,
    )
    return {
        "action": "continue",
        "prefix_id": prefix.prefix_id,
        "is_correct": score["is_correct"],
        "oracle_recoverable": score["is_correct"],
        "predicted_answer": score.get("predicted_answer", score.get("final_answer", "")),
        "final_answer": score.get("final_answer", ""),
        "evaluation_status": score.get("evaluation_status", "OK"),
        "evaluation_error": score.get("evaluation_error"),
        "has_boxed_answer": score.get("has_boxed_answer"),
        "finish_reason": out.get("finish_reason"),
        "num_tokens": out["num_tokens"],
        "generated_tokens": out["num_tokens"],
        "discarded_prefix_tokens": 0,
        "action_start": "current_prefix",
        "latency_sec": latency,
        "response_text": out["text"],
    }


def run_branch(
    engine,
    prompt: str,
    prefix: StudyPrefix,
    gold_answer: str,
    *,
    k: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    retry_max_tokens: int = 0,
) -> dict[str, Any]:
    full_prefix = prompt + prefix.reasoning_prefix
    t0 = time.perf_counter()
    outs = engine.generate_full_continuations(
        full_prefix,
        k=k,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        retry_max_tokens=retry_max_tokens,
    )
    batched_latency = time.perf_counter() - t0
    branch_rows = []
    total_tokens = 0
    serial_latency = 0.0
    for i, out in enumerate(outs):
        score = classify_generation_outcome(
            full_prefix + out["text"],
            gold_answer,
            finish_reason=out.get("finish_reason"),
            require_marker=True,
        )
        branch_rows.append(
            {
                "branch_id": f"{prefix.prefix_id}_b{i}",
                "is_correct": score["is_correct"],
                "predicted_answer": score.get("predicted_answer", score.get("final_answer", "")),
                "final_answer": score.get("final_answer", ""),
                "evaluation_status": score.get("evaluation_status", "OK"),
                "evaluation_error": score.get("evaluation_error"),
                "has_boxed_answer": score.get("has_boxed_answer"),
                "finish_reason": out.get("finish_reason"),
                "num_tokens": out["num_tokens"],
                "generated_tokens": out["num_tokens"],
                "latency_sec": out.get("latency_sec", batched_latency / max(k, 1)),
                "seed": i,
                "response_text": out["text"],
            }
        )
        total_tokens += out["num_tokens"]
        serial_latency += out.get("latency_sec", batched_latency / max(k, 1))
    pass_at_k = _pass_at_k(branch_rows)
    metrics = _branch_metrics(branch_rows, k)
    return {
        "action": "branch",
        "prefix_id": prefix.prefix_id,
        "pass_at_k": metrics["pass_at_k"],
        "oracle_recoverable": metrics["pass_at_k"],
        "oracle_branch_recoverable": metrics["pass_at_k"],
        "branch_correct_count": metrics["branch_correct_count"],
        "branch_accuracy_at_k": metrics["branch_accuracy_at_k"],
        "branch_evaluated_count": metrics["branch_evaluated_count"],
        "branch_evaluation_errors": metrics["branch_evaluation_errors"],
        "recovery_profile": metrics["recovery_profile"],
        "num_branches": k,
        "total_tokens": total_tokens,
        "discarded_prefix_tokens": 0,
        "action_start": "current_prefix",
        "latency_sec": batched_latency,
        "latency_parallel_sec": batched_latency,
        "latency_serial_est": serial_latency,
        "branches": branch_rows,
    }


def run_rollback(
    engine,
    prompt: str,
    prefix: StudyPrefix,
    gold_answer: str,
    *,
    k: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    retry_max_tokens: int = 0,
) -> dict[str, Any] | None:
    if not prefix.previous_checkpoint:
        return None
    full_prefix = prompt + prefix.previous_checkpoint
    wasted_tokens = max(0, len(prefix.reasoning_prefix) - len(prefix.previous_checkpoint))
    t0 = time.perf_counter()
    outs = engine.generate_full_continuations(
        full_prefix,
        k=k,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        retry_max_tokens=retry_max_tokens,
    )
    batched_latency = time.perf_counter() - t0
    branch_rows = []
    total_tokens = 0
    serial_latency = 0.0
    for i, out in enumerate(outs):
        score = classify_generation_outcome(
            full_prefix + out["text"],
            gold_answer,
            finish_reason=out.get("finish_reason"),
            require_marker=True,
        )
        branch_rows.append(
            {
                "branch_id": f"{prefix.prefix_id}_r{i}",
                "is_correct": score["is_correct"],
                "predicted_answer": score.get("predicted_answer", score.get("final_answer", "")),
                "final_answer": score.get("final_answer", ""),
                "evaluation_status": score.get("evaluation_status", "OK"),
                "evaluation_error": score.get("evaluation_error"),
                "has_boxed_answer": score.get("has_boxed_answer"),
                "finish_reason": out.get("finish_reason"),
                "num_tokens": out["num_tokens"],
                "latency_sec": out.get("latency_sec", batched_latency / max(k, 1)),
                "seed": i,
                "response_text": out["text"],
            }
        )
        total_tokens += out["num_tokens"]
        serial_latency += out.get("latency_sec", batched_latency / max(k, 1))
    pass_at_k = _pass_at_k(branch_rows)
    return {
        "action": "rollback",
        "prefix_id": prefix.prefix_id,
        "rollback_checkpoint": prefix.previous_checkpoint,
        "pass_at_k": pass_at_k,
        "oracle_recoverable": pass_at_k,
        "oracle_rollback_recoverable": pass_at_k,
        "num_branches": k,
        "total_tokens": total_tokens,
        "wasted_tokens": wasted_tokens,
        "discarded_prefix_tokens": wasted_tokens,
        "action_start": "previous_checkpoint",
        "latency_sec": batched_latency,
        "latency_parallel_sec": batched_latency,
        "latency_serial_est": serial_latency,
        "branches": branch_rows,
    }


def action_rows_from_results(
    results: list[dict[str, Any] | None],
    *,
    problem_id: str,
    prefix_id: str,
    state_bucket: str,
    continue_checkpoint: str,
    rollback_checkpoint: str,
    temperature: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        if result is None:
            continue
        checkpoint = continue_checkpoint if result["action"] != "rollback" else rollback_checkpoint
        rows.extend(
            _flatten_action_rows(
                result,
                problem_id=problem_id,
                prefix_id=prefix_id,
                state_bucket=state_bucket,
                start_checkpoint=checkpoint,
                temperature=0.0 if result["action"] == "continue" else temperature,
            )
        )
    return rows

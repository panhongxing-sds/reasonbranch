"""Technical failure codes — never counted as strategy Handoff."""

from __future__ import annotations

TECHNICAL_FAILURES = frozenset(
    {
        "ORACLE_API_ERROR",
        "TARGET_GENERATION_ERROR",
        "TARGET_EMPTY_STEP",  # legacy alias
        "STEP_EXTRACTION_ERROR",
        "PREFIX_UNCHANGED",
    }
)

STRATEGY_ACTIONS = frozenset({"CONTINUE", "BRANCH", "HANDOFF"})


def is_technical_failure(reason: str) -> bool:
    return reason in TECHNICAL_FAILURES


def valid_for_comparison(termination_reason: str) -> bool:
    return termination_reason not in TECHNICAL_FAILURES

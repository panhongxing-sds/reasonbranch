"""Target replay dataset schema and artifact audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Fields required before target can replay acceptance offline.
PREFIX_REPLAY_FIELDS = (
    "prefix_id",
    "problem_id",
    "prefix_text",
)

OPTIONAL_PREFIX_FIELDS = (
    "prefix_token_ids",
    "entropy",
    "topk_token_ids",
    "topk_probs",
)

CONTINUE_REPLAY_FIELDS = (
    "prefix_id",
    "continuation",
    "generated_tokens",
)

OPTIONAL_CONTINUE_FIELDS = (
    "continuation_token_ids",
    "finish_reason",
    "temperature",
)

BRANCH_REPLAY_FIELDS = (
    "prefix_id",
    "sample_id",
    "continuation",
    "generated_tokens",
)

OPTIONAL_BRANCH_FIELDS = (
    "continuation_token_ids",
    "seed",
    "temperature",
    "finish_reason",
)

# Filled after target replay — the true Branch oracle labels.
TARGET_REPLAY_OUTPUT_FIELDS = (
    "prefix_id",
    "target_accepted_length_continue",
    "target_accepted_length_branch_max",
    "target_acceptance_gain",
    "target_accept_ratio_continue",
    "target_accept_ratio_branch_max",
    "first_reject_position_continue",
    "first_reject_position_branch_best",
    "target_selected_branch",
    "all_branches_rejected_early",
    "target_model",
    "replay_config",
)


def audit_replay_readiness(data_dir: Path, *, expected_branches: int = 4) -> dict[str, Any]:
    """Check whether artifacts are sufficient for future target replay."""
    prefixes = _load(data_dir / "prefixes.jsonl")
    actions = _load(data_dir / "actions.jsonl")
    hidden_exists = (data_dir / "hidden.safetensors").exists()

    continue_by: dict[str, dict] = {}
    branch_by: dict[str, list[dict]] = {}
    for a in actions:
        pid = a["prefix_id"]
        if a.get("action_type") == "continue":
            continue_by[pid] = a
        elif a.get("action_type") == "branch":
            branch_by.setdefault(pid, []).append(a)

    ready_text = 0
    missing_continue = 0
    missing_branch = 0
    has_token_ids = 0
    for p in prefixes:
        pid = p["prefix_id"]
        if not p.get("prefix_text"):
            continue
        c = continue_by.get(pid)
        branches = branch_by.get(pid, [])
        if not c or not c.get("continuation"):
            missing_continue += 1
            continue
        if len(branches) < expected_branches:
            missing_branch += 1
            continue
        ready_text += 1
        if c.get("continuation_token_ids") or all(b.get("continuation_token_ids") for b in branches):
            has_token_ids += 1

    return {
        "total_prefixes": len(prefixes),
        "replay_ready_text_only": ready_text,
        "missing_continue": missing_continue,
        "incomplete_branch_set": missing_branch,
        "has_continuation_token_ids": has_token_ids,
        "hidden_safetensors": hidden_exists,
        "target_labels_present": (data_dir / "target_replay_results.jsonl").exists(),
        "gaps": [
            "continuation_token_ids not in actions.jsonl (text-only replay possible)",
            "prefix_token_ids not in prefixes.jsonl",
            "target_accepted_length_* requires target model replay",
        ],
    }


def _load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Audit target replay artifact readiness")
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    report = audit_replay_readiness(args.data_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

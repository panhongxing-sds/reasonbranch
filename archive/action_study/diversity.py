"""Strategy-level diversity, recovery profiles, and behavior-state mapping."""

from __future__ import annotations

import math
import re
from collections import Counter

# Legacy buckets (v1) — kept for normalize_legacy_state
STATE_STABLE = "Stable"
STATE_FUTURE_DIVERSE = "Future-diverse"
STATE_CURRENT_UNRELIABLE = "Current-unreliable"
STATE_UNCLEAR = "UNCLEAR"
STATE_API_ERROR = "API_ERROR"

# v2 behavior states
BEHAVIOR_STABLE = "Stable"
BEHAVIOR_DECISION_SENSITIVE = "Decision-sensitive"
BEHAVIOR_CORRUPTED_RECOVERABLE = "Corrupted-recoverable"
BEHAVIOR_CORRUPTED_STUCK = "Corrupted-stuck"
BEHAVIOR_EXCLUDED = "Excluded"
BEHAVIOR_UNCLEAR = "Unclear"

STRATEGY_ONE = "ONE_STRATEGY"
STRATEGY_MULTIPLE = "MULTIPLE_GENUINE_STRATEGIES"

RECOVERY_ALL_SUCCEED = "ALL_SUCCEED"
RECOVERY_MOST_SUCCEED = "MOST_SUCCEED"
RECOVERY_MIXED = "MIXED"
RECOVERY_RARE_SUCCESS = "RARE_SUCCESS"
RECOVERY_ALL_FAIL = "ALL_FAIL"

FUTURE_SYSTEM_ACTION = {
    BEHAVIOR_STABLE: "Continue",
    BEHAVIOR_DECISION_SENSITIVE: "Branch",
    BEHAVIOR_CORRUPTED_RECOVERABLE: "Branch-or-Continue",
    BEHAVIOR_CORRUPTED_STUCK: "Handoff",
    BEHAVIOR_EXCLUDED: "Exclude",
    BEHAVIOR_UNCLEAR: "Unclear",
    STATE_STABLE: "Continue",
    STATE_FUTURE_DIVERSE: "Branch",
    STATE_CURRENT_UNRELIABLE: "Handoff",
    STATE_UNCLEAR: "Unclear",
    STATE_API_ERROR: "API_ERROR",
}

# Conservative keyword families — only count as different strategies if distinct families appear
_STRATEGY_FAMILIES = (
    ("index_substitution", ("n = j + k", "change of variable", "index substitution", "reindex", "diagonal")),
    ("fix_outer", ("fix j", "fix k", "for fixed j", "for fixed k", "inner sum")),
    ("generating", ("generating function", "convolution", "cauchy")),
    ("geometric", ("geometry", "triangle", "coordinate", "angle")),
    ("induction", ("induction", "inductive", "base case")),
    ("enumeration", ("enumerate", "list all", "check each", "brute")),
    ("algebraic", ("factor", "expand", "simplify", "substitute")),
)


def _strategy_families(text: str) -> set[str]:
    low = text.lower()
    found: set[str] = set()
    for name, keys in _STRATEGY_FAMILIES:
        if any(k in low for k in keys):
            found.add(name)
    return found


def recovery_profile(correct_count: int, evaluated_count: int) -> str:
    if evaluated_count <= 0:
        return "UNKNOWN"
    k = evaluated_count
    c = correct_count
    if c == k:
        return RECOVERY_ALL_SUCCEED
    if c == k - 1 and k >= 2:
        return RECOVERY_MOST_SUCCEED
    if c >= 2:
        return RECOVERY_MIXED
    if c == 1:
        return RECOVERY_RARE_SUCCESS
    return RECOVERY_ALL_FAIL


def strategy_diversity_label(
    next_steps: list[str],
    *,
    api_clusters: list[int] | None = None,
    api_num_clusters: int | None = None,
    multiple_genuine: bool | None = None,
) -> dict:
    """Strategy-level diversity. API preferred; heuristic is conservative (anti false-positive)."""
    if not next_steps:
        return {
            "strategy_diversity": STRATEGY_ONE,
            "diversity_entropy": 0.0,
            "num_clusters": 0,
            "diversity_label": "LOW_DIVERSITY",
            "cluster_labels": [],
            "cluster_counts": {},
            "cluster_source": "none",
            "multiple_genuine_strategies": False,
        }

    if api_clusters and len(api_clusters) == len(next_steps):
        labels = [f"cluster_{c}" for c in api_clusters]
        n = int(api_num_clusters or len(set(api_clusters)))
        multi = bool(multiple_genuine) if multiple_genuine is not None else n >= 2
        strategy = STRATEGY_MULTIPLE if multi and n >= 2 else STRATEGY_ONE
        source = "api"
    else:
        labels, strategy, multi = _conservative_heuristic(next_steps)
        n = len(set(labels))
        source = "heuristic_conservative"

    ent, _, _ = _entropy_from_labels(labels)
    counts = Counter(labels)
    return {
        "strategy_diversity": strategy,
        "diversity_entropy": ent,
        "num_clusters": n,
        "diversity_label": "HIGH_DIVERSITY" if strategy == STRATEGY_MULTIPLE else "LOW_DIVERSITY",
        "cluster_labels": labels,
        "cluster_counts": dict(counts),
        "cluster_source": source,
        "multiple_genuine_strategies": strategy == STRATEGY_MULTIPLE,
    }


def _conservative_heuristic(next_steps: list[str]) -> tuple[list[str], str, bool]:
    """Default ONE_STRATEGY unless clearly different strategy families appear."""
    families_per_step = [_strategy_families(s) for s in next_steps]
    union: set[str] = set()
    for fam in families_per_step:
        union |= fam
    if len(union) >= 2:
        labels = [next(iter(fam)) if fam else "unknown" for fam in families_per_step]
        return labels, STRATEGY_MULTIPLE, True
    return ["strategy_0"] * len(next_steps), STRATEGY_ONE, False


def _entropy_from_labels(labels: list[str]) -> tuple[float, int, float]:
    counts = Counter(labels)
    total = sum(counts.values())
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log(p + 1e-12)
    n_clusters = len(counts)
    max_share = max(counts.values()) / total
    return ent, n_clusters, max_share


def compute_diversity(
    next_steps: list[str],
    *,
    api_clusters: list[int] | None = None,
    api_num_clusters: int | None = None,
    multiple_genuine: bool | None = None,
) -> dict:
    """Backward-compatible wrapper around strategy_diversity_label."""
    return strategy_diversity_label(
        next_steps,
        api_clusters=api_clusters,
        api_num_clusters=api_num_clusters,
        multiple_genuine=multiple_genuine,
    )


def behavior_state(
    *,
    prefix_validity: str,
    prefix_substantiveness: str,
    strategy_diversity: str,
    recovery_profile: str,
    continue_correct: int | None = None,
    branch_pass_at_k: int | None = None,
) -> str:
    if prefix_substantiveness == "NO_COMMITMENT":
        return BEHAVIOR_EXCLUDED

    v = (prefix_validity or STATE_UNCLEAR).upper()
    if v in {STATE_API_ERROR, "UNCLEAR"}:
        return BEHAVIOR_UNCLEAR
    if v == "NO_COMMITMENT":
        return BEHAVIOR_EXCLUDED

    if v == "INVALID":
        if recovery_profile == RECOVERY_ALL_FAIL:
            return BEHAVIOR_CORRUPTED_STUCK
        if recovery_profile in {
            RECOVERY_MIXED,
            RECOVERY_RARE_SUCCESS,
            RECOVERY_MOST_SUCCEED,
            RECOVERY_ALL_SUCCEED,
        }:
            return BEHAVIOR_CORRUPTED_RECOVERABLE
        return BEHAVIOR_CORRUPTED_STUCK

    if v == "VALID":
        if strategy_diversity == STRATEGY_ONE:
            return BEHAVIOR_STABLE
        # multiple genuine strategies
        if recovery_profile in {RECOVERY_MIXED, RECOVERY_RARE_SUCCESS}:
            return BEHAVIOR_DECISION_SENSITIVE
        if continue_correct == 0 and branch_pass_at_k == 1:
            return BEHAVIOR_DECISION_SENSITIVE
        if recovery_profile in {RECOVERY_ALL_SUCCEED, RECOVERY_MOST_SUCCEED}:
            return BEHAVIOR_STABLE
        return BEHAVIOR_DECISION_SENSITIVE

    return BEHAVIOR_UNCLEAR


def state_bucket(prefix_validity: str, diversity_label: str) -> str:
    """Legacy v1 mapping — prefer behavior_state() for new runs."""
    v = (prefix_validity or STATE_UNCLEAR).upper()
    if v == "API_ERROR":
        return STATE_API_ERROR
    if v == "INVALID":
        return STATE_CURRENT_UNRELIABLE
    if v == "VALID" and diversity_label == "HIGH_DIVERSITY":
        return STATE_FUTURE_DIVERSE
    if v == "VALID" and diversity_label == "LOW_DIVERSITY":
        return STATE_STABLE
    return STATE_UNCLEAR


def future_system_action(state: str) -> str:
    return FUTURE_SYSTEM_ACTION.get(state, "Unclear")


def normalize_legacy_state(state: str) -> str:
    mapping = {
        "Forward-uncertain": STATE_FUTURE_DIVERSE,
        "Backward-uncertain": STATE_CURRENT_UNRELIABLE,
    }
    return mapping.get(state, state)


BEHAVIOR_TABLE_ORDER = [
    (BEHAVIOR_STABLE, "Stable"),
    (BEHAVIOR_DECISION_SENSITIVE, "Decision-sensitive"),
    (BEHAVIOR_CORRUPTED_RECOVERABLE, "Corrupted-recoverable"),
    (BEHAVIOR_CORRUPTED_STUCK, "Corrupted-stuck"),
]

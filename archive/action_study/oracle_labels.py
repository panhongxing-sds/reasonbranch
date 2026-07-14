"""Oracle Continue / Branch / Handoff labels from utility scores."""

from __future__ import annotations

from typing import Any

# Discrete utility buckets observed in QwQ pilot (see audit report).
UTILITY_BUCKETS = {
    "reject": (0, 4),
    "borderline": (5, 6),
    "accept": (7, 9),
}


def utility_category(score: int | None) -> str:
    if score is None:
        return "unknown"
    for name, (lo, hi) in UTILITY_BUCKETS.items():
        if lo <= int(score) <= hi:
            return name
    return "unknown"


def is_accept(score: int | None, *, tau: int = 7) -> bool:
    return score is not None and int(score) >= tau


def classify_data_error(eligible: list[bool] | None) -> bool:
    return eligible is not None and len(eligible) > 0 and not all(eligible)


def classify_oracle(
    scores: list[int],
    *,
    tau: int,
    eligible: list[bool] | None = None,
) -> dict[str, Any]:
    """scores[0]=greedy/continue; scores[1:]=branches."""
    if not scores:
        return {
            "oracle_label": "unknown",
            "tau": tau,
            "u_greedy": None,
            "u_best_branch": None,
            "u_max": None,
        }

    if eligible is not None and len(eligible) == len(scores):
        if not all(eligible):
            # Only score complete substantive candidates
            eff = [s for s, ok in zip(scores, eligible) if ok]
            if not eff:
                return {
                    "oracle_label": "data_error",
                    "oracle_tier": "data_error",
                    "tau": tau,
                    "u_greedy": scores[0] if eligible[0] else None,
                    "u_best_branch": None,
                    "u_max": None,
                    "branch_rescue_gain": None,
                }
            scores = eff

    u0 = scores[0]
    u_br = scores[1:] if len(scores) > 1 else []
    u_best_br = max(u_br) if u_br else -1
    u_max = max(scores)

    if u0 >= tau:
        label = "continue_sufficient"
    elif u_br and max(u_br) >= tau:
        label = "weak_branch_rescuable"
    else:
        label = "handoff_required"

    return {
        "oracle_label": label,
        "oracle_tier": label,
        "tau": tau,
        "u_greedy": u0,
        "u_best_branch": u_best_br if u_br else None,
        "u_max": u_max,
        "branch_rescue_gain": (max(u_br) - u0) if u_br else None,
    }


def classify_oracle_strict(
    scores: list[int],
    *,
    tau_hi: int = 7,
    u0_max: int = 4,
    margin: int = 3,
    eligible: list[bool] | None = None,
) -> dict[str, Any]:
    """Stricter Branch-rescuable: low greedy, high branch, margin, all complete."""
    if eligible is not None and len(eligible) == len(scores):
        if not all(eligible):
            eff = [s for s, ok in zip(scores, eligible) if ok]
            if not eff:
                return {"oracle_label_strict": "data_error", "oracle_tier": "data_error", "tau_hi": tau_hi}
            scores = eff

    u0 = scores[0]
    u_br = scores[1:] if len(scores) > 1 else []
    u_best = max(u_br) if u_br else -1
    delta = u_best - u0 if u_br else 0

    if u0 >= tau_hi:
        label = "continue_sufficient"
        tier = "continue_sufficient"
    elif u0 <= u0_max and u_best >= tau_hi and delta >= margin:
        label = "strong_branch_rescuable"
        tier = "strong_branch_rescuable"
    elif u_br and max(u_br) >= tau_hi:
        label = "weak_branch_rescuable"
        tier = "weak_branch_rescuable"
    else:
        label = "handoff_required"
        tier = "handoff_required"

    return {
        "oracle_label_strict": label,
        "oracle_tier": tier,
        "tau_hi": tau_hi,
        "u0_max": u0_max,
        "margin": margin,
        "utility_margin": delta,
    }


def classify_gpt_step_oracle(
    *,
    prefix_valid: bool | None,
    g_acceptable: bool,
    branch_acceptables: list[bool],
) -> dict[str, Any]:
    """Next-step GPT oracle: Continue / Branch / Handoff / PREFIX_CORRUPTED."""
    if prefix_valid is False:
        action = "prefix_corrupted"
    elif g_acceptable:
        action = "continue"
    elif any(branch_acceptables):
        action = "branch"
    else:
        action = "handoff"
    m = sum(1 for b in branch_acceptables if b)
    return {
        "oracle_action": action,
        "g_acceptable": g_acceptable,
        "branch_acceptables": branch_acceptables,
        "m_acceptable_branches": m,
        "prefix_valid": prefix_valid,
    }


def summarize_gpt_step_oracle(rows: list[dict[str, Any]], *, stable_only: bool = True) -> dict[str, Any]:
    """Aggregate GPT step-oracle labels."""
    if stable_only:
        rows = [r for r in rows if r.get("dual_pass_stable")]
    n = len(rows)
    if n == 0:
        return {"n": 0}
    actions = [r.get("oracle_action", "unknown") for r in rows]
    counts = {
        "continue": sum(1 for a in actions if a == "continue"),
        "branch": sum(1 for a in actions if a == "branch"),
        "handoff": sum(1 for a in actions if a == "handoff"),
        "prefix_corrupted": sum(1 for a in actions if a == "prefix_corrupted"),
        "unknown": sum(1 for a in actions if a not in ("continue", "branch", "handoff", "prefix_corrupted")),
    }
    m_hist: dict[int, int] = {}
    for r in rows:
        m = r.get("m_acceptable_branches")
        if m is not None:
            m_hist[int(m)] = m_hist.get(int(m), 0) + 1
    return {
        "n": n,
        "action_counts": counts,
        "action_rates": {k: round(v / n, 4) for k, v in counts.items()},
        "m_acceptable_branches_hist": dict(sorted(m_hist.items())),
    }


def summarize_oracle_table(rows: list[dict[str, Any]], thresholds: list[int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tau in thresholds:
        labels = [classify_oracle(r["utility_scores"], tau=tau)["oracle_label"] for r in rows]
        n = len(labels)
        if n == 0:
            continue
        counts = {
            "continue_sufficient": sum(1 for x in labels if x == "continue_sufficient"),
            "weak_branch_rescuable": sum(
                1 for x in labels if x in ("weak_branch_rescuable", "branch_rescuable")
            ),
            "handoff_required": sum(1 for x in labels if x == "handoff_required"),
            "data_error": sum(1 for x in labels if x == "data_error"),
        }
        # backward-compat alias
        counts["branch_rescuable"] = counts["weak_branch_rescuable"]
        out.append(
            {
                "tau": tau,
                "n": n,
                **{k: counts[k] for k in counts},
                **{f"pct_{k}": round(100 * counts[k] / n, 2) for k in counts},
            }
        )
        out[-1]["pct_branch_rescuable"] = out[-1].get("pct_weak_branch_rescuable", 0)
    return out

"""Unit tests for V3.6 math / rescue flags (no GPU)."""

from __future__ import annotations

from reasoning_branch_dataset.action_study.v36_analyze import compute_rescue_flags, decide_v36
from reasoning_branch_dataset.action_study.v36_counterfactual import gamma_margin


def test_gamma_margin():
    assert abs(gamma_margin(1.0) - 0.05) < 1e-9
    assert abs(gamma_margin(2.0) - 0.10) < 1e-9  # 5% of 2s
    assert abs(gamma_margin(0.2) - 0.05) < 1e-9  # floor 50ms


def test_safe_rescue_vs_exist():
    row = {
        "branch_oracle_labels": [True, False, True, False],
        "branch_verifier_scores": [-1.0, 2.0, 0.5, -2.0],
        "tau_accept": 0.0,
    }
    # selected = index 1 (score 2.0) which is oracle False → exist True, safe False
    f = compute_rescue_flags(row, 4)
    assert f["exist"] is True
    assert f["accepted"] is True
    assert f["safe"] is False
    assert f["selected_index"] == 1


def test_rescue_flags_none_is_unknown():
    # None labels must NOT be treated as False.
    row = {
        "branch_oracle_labels": [None, None, True, None],
        "branch_verifier_scores": [0.1, 0.2, 2.0, -1.0],
        "tau_accept": 0.0,
    }
    f = compute_rescue_flags(row, 4)
    # selected = index 2 (score 2.0), label True → safe True, exist True
    assert f["selected_index"] == 2
    assert f["exist"] is True
    assert f["safe"] is True
    assert f["oracle_known"] is True


def test_rescue_flags_all_none_unknown():
    row = {
        "branch_oracle_labels": [None, None],
        "branch_verifier_scores": [1.0, 2.0],
        "tau_accept": 0.0,
    }
    f = compute_rescue_flags(row, 2)
    assert f["oracle_known"] is False
    assert f["exist"] is False  # unknown → not asserted true
    assert f["safe"] is False


def test_analyze_nan_safe_and_quadrant():
    from reasoning_branch_dataset.action_study.v36_analyze import analyze_trials, render_report

    rows = [
        {
            "problem_id": "p1",
            "handoff_wall_sec": 1.0,
            "branch_pipeline_sec": {"1": 0.5, "2": 0.6, "4": 0.8},
            "branch_used_fallback": {"1": False, "2": False, "4": True},
            "tau_accept": 0.0,
            "handoff_oracle_label": True,
            "rescue": {
                "1": {"exist": 1.0, "accepted": True, "safe": 1.0, "selected_oracle_label": True},
                "2": {"exist": float("nan"), "accepted": True, "safe": float("nan"), "selected_oracle_label": None},
                "4": {"exist": 0.0, "accepted": True, "safe": 0.0, "selected_oracle_label": False},
            },
        }
    ]
    summ = analyze_trials(rows)
    # NaN safe at K2 must be excluded, not crash.
    assert summ["by_k"]["1"]["rescue"]["safe"] == 1.0
    assert summ["by_k"]["2"]["rescue"]["n_oracle_known"] == 0
    q1 = summ["by_k"]["1"]["quadrant"]
    assert q1["bok_hok"] == 1 and q1["known"] == 1
    # report renders without error even with NaN
    md = render_report(summ)
    assert "quadrant" in md


def test_decide_fixed_handoff():
    by_k = {
        "1": {
            "delta_boot": {"mean_ms": -100, "ci_low_ms": -200, "ci_high_ms": -10},
            "rescue": {"safe": 0.05},
            "delta_ms": {"p_positive": 0.1},
            "profitable_rate": 0.05,
        },
        "4": {
            "delta_boot": {"mean_ms": -50, "ci_low_ms": -80, "ci_high_ms": -5},
            "rescue": {"safe": 0.1},
            "delta_ms": {"p_positive": 0.2},
            "profitable_rate": 0.08,
        },
    }
    d = decide_v36(by_k)
    assert d["decision"] == "fixed_handoff"

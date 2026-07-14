"""Unit tests for V3.5 cost–rescue math (no GPU)."""

from __future__ import annotations

from reasoning_branch_dataset.action_study.cost_rescue import (
    CostSample,
    aggregate_break_even,
    break_even,
    decide_policy,
    expected_branch_cost,
    overall_break_even,
    speedup_vs_handoff,
)


def test_break_even_basic():
    # C_D4 + C_V4 = 0.1 * C_T → r^* = 10%
    assert abs(break_even(0.04, 0.06, 1.0) - 0.1) < 1e-9


def test_expected_branch_cost():
    # r=0.4, C_D+C_V=0.1, C_T=1 → 0.1 + 0.6 = 0.7
    assert abs(expected_branch_cost(0.04, 0.06, 1.0, 0.4) - 0.7) < 1e-9
    assert abs(speedup_vs_handoff(0.04, 0.06, 1.0, 0.4) - (1.0 / 0.7)) < 1e-9


def test_decide_dominated():
    d = decide_policy(r_k=0.99, r_k_star=1.06, k=4)
    assert d.decision == "dominated"


def test_decide_always_branch():
    d = decide_policy(r_k=0.40, r_k_star=0.10, k=4)
    assert d.decision == "always_branch"


def test_heterogeneity_router_signal():
    from reasoning_branch_dataset.action_study.cost_rescue import decide_from_bucket_stars

    h = decide_from_bucket_stars(
        {"s1_64": 1.2, "s65_128": 0.9, "s193_plus": 0.25},
        k=2,
    )
    assert h["decision"] == "train_predictor"


def test_decide_never_branch():
    d = decide_policy(r_k=0.20, r_k_star=0.35, k=4)
    assert d.decision == "never_branch"


def test_decide_boundary():
    d = decide_policy(r_k=0.25, r_k_star=0.20, k=4)
    assert d.decision == "train_predictor"


def test_aggregate_break_even():
    samples = [
        CostSample("short", "short", 100, 40, c_t=1.0, c_d4=0.05, c_v4=0.05),
        CostSample("short", "short", 120, 45, c_t=1.2, c_d4=0.06, c_v4=0.06),
        CostSample("long", "long", 900, 150, c_t=2.0, c_d4=0.2, c_v4=0.3),
    ]
    rows = aggregate_break_even(samples)
    assert len(rows) == 2
    overall = overall_break_even(samples)
    assert overall.n == 3
    assert overall.r4_star is not None
    assert overall.r4_star > 0

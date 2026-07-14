"""Unit tests for the V4.0 conformal abstention gate (no GPU)."""

from __future__ import annotations

import random

from reasoning_branch_dataset.action_study.v40_conformal_gate import (
    calibrate_threshold,
    _cp_upper,
    _binom_cdf,
)


def test_cp_upper_bounds():
    # k=0 closed form: U = 1 - delta**(1/n)
    import math
    assert abs(_cp_upper(0, 100, 0.1) - (1 - 0.1 ** (1 / 100))) < 1e-3
    # more successes -> larger upper bound
    assert _cp_upper(0, 100, 0.1) < _cp_upper(50, 100, 0.1)
    # more samples -> tighter bound at same rate
    assert _cp_upper(5, 100, 0.1) < _cp_upper(1, 20, 0.1)
    assert _cp_upper(0, 0, 0.1) == 1.0
    # CP upper is much tighter than Hoeffding for k=0 (sanity: < 0.07 at n=100)
    assert _cp_upper(0, 100, 0.001) < 0.10


def test_binom_cdf_basic():
    assert abs(_binom_cdf(0, 10, 0.0) - 1.0) < 1e-9
    assert abs(_binom_cdf(10, 10, 0.5) - 1.0) < 1e-9
    assert abs(_binom_cdf(0, 2, 0.5) - 0.25) < 1e-9  # P(X=0)=0.25


def test_calibrate_separable_high_precision():
    # positives score high, negatives low -> a threshold gives ~0 false rate
    rng = random.Random(0)
    scores, labels = [], []
    for _ in range(200):
        y = rng.random() < 0.5
        scores.append(rng.gauss(3.0 if y else -3.0, 0.5))
        labels.append(y)
    res = calibrate_threshold(scores, labels, epsilon=0.10, delta=0.10)
    assert res.tau is not None
    assert res.empirical_false_rate <= 0.10
    assert res.ucb_false_rate <= 0.10
    assert res.coverage > 0.2  # can accept a meaningful chunk


def test_calibrate_impossible_returns_none():
    # score is pure noise vs label -> no threshold can guarantee precision 0.99
    rng = random.Random(1)
    scores = [rng.gauss(0, 1) for _ in range(120)]
    labels = [rng.random() < 0.3 for _ in range(120)]
    res = calibrate_threshold(scores, labels, epsilon=0.01, delta=0.05)
    assert res.tau is None


def test_calibrate_guarantee_is_conservative():
    # UCB must never exceed epsilon for the chosen tau
    rng = random.Random(2)
    scores, labels = [], []
    for _ in range(300):
        y = rng.random() < 0.4
        scores.append(rng.gauss(1.5 if y else -0.5, 1.2))  # overlapping
        labels.append(y)
    res = calibrate_threshold(scores, labels, epsilon=0.15, delta=0.1)
    if res.tau is not None:
        assert res.ucb_false_rate <= 0.15
        assert res.accepted >= 5

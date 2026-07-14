"""Unit tests for layerwise flip metrics (no GPU)."""

from __future__ import annotations

from reasoning_branch_dataset.action_study.sd_audit.layerwise_trajectory import (
    compute_flip_count,
    compute_flip_depth,
    compute_path_speed,
    default_sample_layers,
)


def test_early_resolve():
    # becomes positive at index 1 and stays
    d = [-1.0, 0.5, 1.0, 2.0]
    assert compute_flip_depth(d) == 1
    assert compute_flip_count(d) == 1


def test_late_flip():
    d = [-2.0, -1.0, -0.5, -0.1, 0.2, 1.0]
    assert compute_flip_depth(d) == 4
    assert compute_flip_count(d) == 1


def test_oscillate_ends_positive():
    # last index is the first (and only) point where suffix is all positive
    d = [-1.0, 1.0, -1.0, 1.0]
    assert compute_flip_depth(d) == 3
    assert compute_flip_count(d) == 3


def test_never_positive():
    d = [-1.0, -0.5, -0.1, -2.0]
    assert compute_flip_depth(d) is None
    assert compute_flip_count(d) == 0


def test_path_speed():
    assert compute_path_speed([0.0, 1.0, 3.0]) == 3.0


def test_sample_layers_includes_last():
    layers = default_sample_layers(64, 8)
    assert layers[-1] == 63
    assert len(layers) == 8

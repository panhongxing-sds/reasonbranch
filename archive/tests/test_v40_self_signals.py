"""Unit tests for V4.0 draft self-signal parsing (no GPU).

Verifies teacher-forcing token alignment: mean/min/last logprob are read at the
correct candidate positions, entropy/margin use the top-k distribution, and the
prefix positions are excluded.
"""

from __future__ import annotations

import math

from reasoning_branch_dataset.action_study.v40_self_signals import (
    parse_teacher_forcing_signals,
    _entropy_from_logprobs,
    _repetition_rate,
)


class _LP:
    """Mock vLLM Logprob object."""

    def __init__(self, logprob: float) -> None:
        self.logprob = logprob


def _cell(actual_id: int, actual_lp: float, others: dict[int, float]) -> dict[int, _LP]:
    d = {tid: _LP(lp) for tid, lp in others.items()}
    d[actual_id] = _LP(actual_lp)
    return d


def test_alignment_excludes_prefix_and_reads_actual_token():
    # full_ids = [prefix..., cand...]; cand_start=2 → candidate positions 2,3.
    full_ids = [10, 11, 20, 21]
    # position 0 is always None in vLLM prompt_logprobs
    prompt_logprobs = [
        None,
        _cell(11, -0.1, {99: -5.0}),  # prefix position (must be excluded)
        _cell(20, -0.5, {7: -2.0, 8: -3.0}),  # cand token 1
        _cell(21, -1.5, {7: -0.2, 8: -0.9}),  # cand token 2
    ]
    sig = parse_teacher_forcing_signals(prompt_logprobs, full_ids, cand_start=2, candidate="xy")
    assert sig.n_tokens == 2
    assert abs(sig.mean_logprob - (-1.0)) < 1e-9  # mean(-0.5, -1.5)
    assert abs(sig.min_logprob - (-1.5)) < 1e-9
    assert abs(sig.last_logprob - (-1.5)) < 1e-9
    assert abs(sig.perplexity - math.exp(1.0)) < 1e-6


def test_margin_uses_topk_distribution():
    full_ids = [10, 20]
    # at cand position 1: top1=-0.2 (actual), top2=-0.9 → margin 0.7
    prompt_logprobs = [None, _cell(20, -0.2, {7: -0.9, 8: -3.0})]
    sig = parse_teacher_forcing_signals(prompt_logprobs, full_ids, cand_start=1, candidate="z")
    assert abs(sig.mean_margin - 0.7) < 1e-9
    assert abs(sig.min_margin - 0.7) < 1e-9


def test_actual_token_outside_topk_still_read():
    # actual token has very low logprob and is the only guaranteed inclusion.
    full_ids = [10, 20]
    prompt_logprobs = [None, _cell(20, -12.0, {7: -0.2, 8: -0.5})]
    sig = parse_teacher_forcing_signals(prompt_logprobs, full_ids, cand_start=1, candidate="z")
    assert abs(sig.min_logprob - (-12.0)) < 1e-9


def test_empty_candidate_is_degenerate():
    sig = parse_teacher_forcing_signals([None], [10], cand_start=1, candidate="")
    assert sig.n_tokens == 0
    assert sig.mean_logprob == -20.0


def test_entropy_uniform_vs_peaked():
    # peaked distribution → low entropy; uniform → higher entropy
    peaked = _entropy_from_logprobs([math.log(0.97), math.log(0.02), math.log(0.01)])
    uniform = _entropy_from_logprobs([math.log(1 / 3)] * 3)
    assert uniform > peaked
    assert peaked >= 0.0


def test_repetition_rate():
    assert _repetition_rate([1, 2, 3, 4]) == 0.0  # all distinct bigrams
    assert _repetition_rate([1, 1, 1, 1]) > 0.0  # (1,1) repeats
    assert _repetition_rate([5]) == 0.0  # <2 tokens

"""Greedy token extraction helpers for HF / vLLM logprob alignment."""

from __future__ import annotations

from typing import Any


def hf_greedy_token(logits, pos: int) -> int:
    """logits[pos] predicts token at pos+1 in the concatenated sequence."""
    return int(logits[pos].argmax().item())


def vllm_greedy_token(lp_map: dict[int, Any] | None) -> int | None:
    if not lp_map:
        return None
    return int(max(lp_map.items(), key=lambda kv: kv[1].logprob)[0])


def vllm_candidate_rank(lp_map: dict[int, Any] | None, token_id: int) -> int | None:
    """1 = highest logprob among returned entries."""
    if not lp_map:
        return None
    ranked = sorted(lp_map.items(), key=lambda kv: kv[1].logprob, reverse=True)
    for i, (tid, _) in enumerate(ranked, start=1):
        if int(tid) == int(token_id):
            return i
    return None


def token_accepted_greedy(candidate_id: int, greedy_id: int | None) -> bool:
    return greedy_id is not None and int(candidate_id) == int(greedy_id)

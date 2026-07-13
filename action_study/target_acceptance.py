"""Greedy target acceptance length for speculative draft verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from reasoning_branch_dataset.action_study.logprob_align import hf_greedy_token


@dataclass
class AcceptanceResult:
    accepted_length: int
    draft_length: int
    accept_ratio: float
    first_reject_position: int | None
    greedy_matches: list[bool]

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted_length": self.accepted_length,
            "draft_length": self.draft_length,
            "accept_ratio": self.accept_ratio,
            "first_reject_position": self.first_reject_position,
        }


def tokenize_text(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def truncate_token_ids(token_ids: list[int], gamma: int) -> list[int]:
    return token_ids[:gamma] if gamma > 0 else []


def greedy_acceptance_length_from_ids(
    model,
    prefix_ids: list[int],
    draft_ids: list[int],
    *,
    device: str = "cuda",
) -> AcceptanceResult:
    """HF: logits[i-1] predicts input_ids[i]."""
    if not draft_ids:
        return AcceptanceResult(0, 0, 0.0, None, [])

    full_ids = prefix_ids + draft_ids
    input_ids = torch.tensor([full_ids], device=device, dtype=torch.long)
    with torch.inference_mode():
        out = model(input_ids=input_ids)
        logits = out.logits[0]

    matches: list[bool] = []
    accepted = 0
    first_reject: int | None = None
    base = len(prefix_ids)
    for j, draft_tok in enumerate(draft_ids):
        greedy_tok = hf_greedy_token(logits, base - 1 + j)
        ok = int(greedy_tok) == int(draft_tok)
        matches.append(ok)
        if ok:
            accepted += 1
        elif first_reject is None:
            first_reject = j
            break

    draft_len = len(draft_ids)
    return AcceptanceResult(
        accepted_length=accepted,
        draft_length=draft_len,
        accept_ratio=accepted / draft_len,
        first_reject_position=first_reject,
        greedy_matches=matches,
    )


def greedy_acceptance_length(
    model,
    prefix_ids: list[int],
    draft_ids: list[int],
    *,
    device: str = "cuda",
) -> AcceptanceResult:
    """Count draft tokens matching target greedy predictions (HF path)."""
    return greedy_acceptance_length_from_ids(model, prefix_ids, draft_ids, device=device)


def verify_trace_acceptance(
    model,
    tokenizer,
    prompt_text: str,
    trace_text: str,
    *,
    device: str = "cuda",
) -> dict[str, Any]:
    """Verify draft trace token-by-token; return last accepted position in trace tokens."""
    prompt_ids = tokenize_text(tokenizer, prompt_text)
    trace_ids = tokenize_text(tokenizer, trace_text)
    if not trace_ids:
        return {
            "trace_token_count": 0,
            "trace_accepted_length": 0,
            "trace_accept_ratio": 0.0,
            "first_reject_position": None,
        }

    result = greedy_acceptance_length(model, prompt_ids, trace_ids, device=device)
    return {
        "trace_token_count": len(trace_ids),
        "trace_accepted_length": result.accepted_length,
        "trace_accept_ratio": result.accept_ratio,
        "first_reject_position": result.first_reject_position,
        "trace_fully_accepted": result.accepted_length == len(trace_ids),
    }


def prefix_reachability(
    *,
    trace_accepted_length: int,
    prefix_reasoning_token_len: int,
) -> dict[str, Any]:
    fully = prefix_reasoning_token_len <= trace_accepted_length
    return {
        "prefix_reasoning_token_len": prefix_reasoning_token_len,
        "prefix_fully_target_accepted": fully,
        "last_target_accepted_position": trace_accepted_length,
        "distance_from_accepted_checkpoint": max(0, prefix_reasoning_token_len - trace_accepted_length),
    }

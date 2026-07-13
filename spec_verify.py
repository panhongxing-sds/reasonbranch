"""Token-level speculative decoding verification logs."""

from __future__ import annotations

import time
from typing import Any

import torch

from reasoning_branch_dataset.model_utils import kl_divergence


@torch.no_grad()
def run_speculative_round(
    draft_model,
    target_model,
    tokenizer,
    prompt: str,
    *,
    gamma: int = 4,
    max_new_tokens: int = 128,
) -> list[dict[str, Any]]:
    """Simple draft-verify loop over a continuation segment."""
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(draft_model.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(draft_model.device)

    logs: list[dict[str, Any]] = []
    generated = 0
    round_id = 0

    while generated < max_new_tokens:
        t0 = time.perf_counter()
        draft_out = draft_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=gamma,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=True,
        )
        draft_latency = time.perf_counter() - t0

        prompt_len = input_ids.shape[1]
        draft_ids = draft_out.sequences[0, prompt_len:]
        if draft_ids.numel() == 0:
            break

        verify_input = torch.cat([input_ids, draft_ids.unsqueeze(0)], dim=1)
        verify_mask = None
        if attention_mask is not None:
            ext = torch.ones((1, draft_ids.shape[0]), device=attention_mask.device, dtype=attention_mask.dtype)
            verify_mask = torch.cat([attention_mask, ext], dim=1)

        t1 = time.perf_counter()
        target_logits = target_model(
            input_ids=verify_input,
            attention_mask=verify_mask,
        ).logits
        target_latency = time.perf_counter() - t1

        accepted = 0
        first_reject = None
        rejected_tokens: list[int] = []
        accepted_tokens: list[int] = []
        kl_vals: list[float] = []

        for i in range(draft_ids.shape[0]):
            pos = prompt_len + i - 1
            if pos < 0:
                pos = prompt_len - 1
            draft_score = draft_out.scores[i][0]
            target_score = target_logits[0, prompt_len + i - 1]
            kl_vals.append(kl_divergence(draft_score, target_score))

            draft_lp = torch.log_softmax(draft_score.float(), dim=-1)
            target_lp = torch.log_softmax(target_score.float(), dim=-1)
            draft_tid = int(draft_ids[i])
            if torch.exp(target_lp[draft_tid]) >= torch.exp(draft_lp[draft_tid]):
                accepted += 1
                accepted_tokens.append(draft_tid)
            else:
                first_reject = i
                rejected_tokens = draft_ids[i:].tolist()
                break

        if first_reject is None:
            accepted = draft_ids.shape[0]
            accepted_tokens = draft_ids.tolist()

        tau = accepted
        accept_ratio = tau / max(gamma, 1)
        recover_tid = None
        if first_reject is not None:
            recover_pos = prompt_len + first_reject - 1
            recover_tid = int(target_logits[0, recover_pos].argmax().item())

        logs.append(
            {
                "round_id": round_id,
                "gamma": gamma,
                "accepted_length": tau,
                "accept_ratio": accept_ratio,
                "first_reject_position": first_reject,
                "draft_tokens": draft_ids.tolist(),
                "accepted_tokens": accepted_tokens,
                "rejected_tokens": rejected_tokens,
                "target_recover_token": recover_tid,
                "target_draft_KL": float(sum(kl_vals) / max(len(kl_vals), 1)),
                "latency_draft": draft_latency,
                "latency_target_verify": target_latency,
            }
        )

        if first_reject is None:
            input_ids = torch.cat([input_ids, draft_ids.unsqueeze(0)], dim=1)
            if attention_mask is not None:
                ext = torch.ones((1, draft_ids.shape[0]), device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([attention_mask, ext], dim=1)
            generated += draft_ids.shape[0]
        else:
            recover = torch.tensor([recover_tid], device=input_ids.device, dtype=draft_ids.dtype)
            keep = draft_ids[:accepted]
            new_tokens = torch.cat([keep, recover], dim=0).unsqueeze(0)
            input_ids = torch.cat([input_ids, new_tokens], dim=1)
            if attention_mask is not None:
                ext = torch.ones((1, new_tokens.shape[1]), device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([attention_mask, ext], dim=1)
            generated += new_tokens.shape[1]

        round_id += 1
        if tokenizer.eos_token_id in input_ids[0, prompt_len:].tolist():
            break

    return logs

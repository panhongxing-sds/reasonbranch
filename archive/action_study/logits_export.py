"""Extract prefix-cut logits features (top-10, entropy, margin)."""

from __future__ import annotations

import json
from typing import Any

import torch


@torch.no_grad()
def extract_prefix_logits(model, tokenizer, prefix_text: str, *, topk: int = 10) -> dict[str, Any]:
    inputs = tokenizer(prefix_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    pos = input_ids.shape[1] - 1
    score = logits[0, pos].float()
    log_probs = torch.log_softmax(score, dim=-1)
    probs = log_probs.exp()

    k = min(topk, log_probs.numel())
    topv, topi = torch.topk(log_probs, k=k)
    topk_ids = topi.tolist()
    topk_probs = topv.exp().tolist()
    entropy = float(-(probs * log_probs).sum().item())
    top1 = float(topk_probs[0]) if topk_probs else 0.0
    top2 = float(topk_probs[1]) if len(topk_probs) > 1 else 0.0

    return {
        "entropy": entropy,
        "top1_prob": top1,
        "top2_prob": top2,
        "margin": top1 - top2,
        "topk_token_ids": json.dumps(topk_ids),
        "topk_probs": json.dumps(topk_probs),
    }

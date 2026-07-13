"""Model loading and trace generation utilities."""

from __future__ import annotations

import math
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from reasoning_branch_dataset.grading import extract_math_answer, math_equal


def build_prompt(question: str) -> str:
    return (
        "Solve the following math problem efficiently and clearly. "
        "Please reason step by step, separate logical reasoning steps with "
        "two newline characters (\\n\\n), and put your final answer within \\boxed{}.\n"
        f"Problem: {question}"
    )


def load_model_and_tokenizer(model_path: str, device: str, dtype: str):
    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = None
    last_err: Exception | None = None
    loader_fns: list = []
    try:
        from transformers import AutoModelForImageTextToText

        loader_fns.append(AutoModelForImageTextToText.from_pretrained)
    except ImportError:
        pass
    loader_fns.append(AutoModelForCausalLM.from_pretrained)

    for loader_fn in loader_fns:
        try:
            model = loader_fn(model_path, dtype=torch_dtype, trust_remote_code=True)
            break
        except Exception as exc:
            last_err = exc
    if model is None:
        raise RuntimeError(f"Failed to load model from {model_path}") from last_err

    if device.startswith("cuda") and torch.cuda.is_available():
        model = model.to(device)
    else:
        model = model.to("cpu")
    model.eval()
    return model, tokenizer


def _entropy_from_logprobs(log_probs: torch.Tensor) -> float:
    p = log_probs.exp()
    return float(-(p * log_probs).sum().item())


def build_token_trace(gen_ids: torch.Tensor, scores: tuple, tokenizer, topk_save: int = 5) -> tuple[list[dict], list[str]]:
    trace: list[dict] = []
    token_texts: list[str] = []
    for pos, (tid, score) in enumerate(zip(gen_ids.tolist(), scores)):
        logits = score[0].float() if score.dim() > 1 else score.float()
        log_probs = torch.log_softmax(logits, dim=-1)
        tid = int(tid)
        tok = tokenizer.decode([tid])
        token_texts.append(tok)

        topv, topi = torch.topk(log_probs, k=min(topk_save, log_probs.numel()))
        topk_ids = topi.tolist()
        topk_probs = topv.exp().tolist()
        top1 = float(topk_probs[0]) if topk_probs else 0.0
        top2 = float(topk_probs[1]) if len(topk_probs) > 1 else 0.0

        trace.append(
            {
                "pos": pos,
                "token": tok,
                "token_id": tid,
                "entropy": _entropy_from_logprobs(log_probs),
                "margin_top2": float(topv[0] - topv[1]) if topv.numel() >= 2 else 0.0,
                "top1_prob": top1,
                "top2_prob": top2,
                "topk_token_ids": topk_ids,
                "topk_probs": topk_probs,
            }
        )
    return trace, token_texts


@torch.no_grad()
def generate_with_trace(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    device: str,
    topk_save: int = 5,
) -> dict[str, Any]:
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)
    prompt_len = input_ids.shape[1]
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=pad_id,
        return_dict_in_generate=True,
        output_scores=True,
        use_cache=True,
    )

    gen_ids = out.sequences[0, prompt_len:]
    response_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    trace, token_texts = build_token_trace(gen_ids, out.scores, tokenizer, topk_save=topk_save)
    return {
        "response_text": response_text,
        "token_ids": gen_ids.tolist(),
        "token_texts": token_texts,
        "token_trace": trace,
        "prompt_len": prompt_len,
        "full_input_ids": out.sequences[0].tolist(),
    }


@torch.no_grad()
def generate_continuation(
    model,
    tokenizer,
    prefix_text: str,
    *,
    max_new_tokens: int,
    temperature: float = 0.7,
    top_p: float = 0.95,
    do_sample: bool = True,
    forced_first_token_id: int | None = None,
    stop_at_paragraph: bool = False,
) -> dict[str, Any]:
    inputs = tokenizer(prefix_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)
    prompt_len = input_ids.shape[1]
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    if forced_first_token_id is not None:
        first = torch.tensor([[forced_first_token_id]], device=model.device)
        input_ids = torch.cat([input_ids, first], dim=1)
        if attention_mask is not None:
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), device=model.device, dtype=attention_mask.dtype)],
                dim=1,
            )
        prompt_len = input_ids.shape[1]
        max_new_tokens = max(1, max_new_tokens - 1)

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": pad_id,
        "return_dict_in_generate": True,
        "output_scores": False,
        "use_cache": True,
    }
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        gen_kwargs.update(do_sample=False)

    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **gen_kwargs,
    )
    gen_ids = out.sequences[0, prompt_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=False)

    stop_reason = "max_tokens"
    if stop_at_paragraph and "\n\n" in text:
        cut = text.find("\n\n") + 2
        text = text[:cut]
        gen_ids = tokenizer.encode(text, add_special_tokens=False)
        stop_reason = "paragraph"

    return {
        "continuation_text": text,
        "continuation_token_ids": gen_ids if isinstance(gen_ids, list) else gen_ids.tolist(),
        "stop_reason": stop_reason,
    }


def score_correctness(response_text: str, gold_answer: str) -> tuple[str, bool]:
    pred = extract_math_answer(response_text)
    return pred, math_equal(pred, gold_answer)


def kl_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    p = torch.log_softmax(p_logits.float(), dim=-1)
    q = torch.log_softmax(q_logits.float(), dim=-1)
    p_prob = p.exp()
    return float((p_prob * (p - q)).sum().item())


def js_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    p = torch.log_softmax(p_logits.float(), dim=-1)
    q = torch.log_softmax(q_logits.float(), dim=-1)
    p_prob = p.exp()
    q_prob = q.exp()
    m = 0.5 * (p_prob + q_prob)
    m_log = torch.log(m + 1e-12)
    kl_pm = (p_prob * (p - m_log)).sum()
    kl_qm = (q_prob * (q - m_log)).sum()
    return float((0.5 * (kl_pm + kl_qm)).item())

"""SpecReason-style single-token utility scoring (0-9) for draft reasoning steps."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from reasoning_branch_dataset.action_study.step_extraction import extract_next_substantive_step
from reasoning_branch_dataset.model_utils import build_prompt

EVAL_USER_MSG = (
    "Evaluate the last reasoning step solely based on factual correctness and logical validity. "
    "Ignore style, phrasing, verbosity, Markdown formatting, step length, or how detailed the "
    "explanation is—only judge whether the step is objectively correct and logically follows from "
    "prior steps. Two steps with the same mathematical content must receive the same score even if "
    "one is shorter. Assign a score from 0 to 9."
)
SCORE_ASSISTANT_PREFIX = "<think>I think the quality score is: "


def first_reasoning_step(text: str, *, question: str = "") -> str:
    """Extract the next substantive reasoning step (merges heading-only blocks)."""
    return extract_next_substantive_step(text, question=question)["candidate_step"]


def build_steps_so_far(reasoning_prefix: str, candidate_step: str) -> str:
    prefix = reasoning_prefix.rstrip()
    step = first_reasoning_step(candidate_step)
    if not prefix:
        return step + "\n\n"
    if not step:
        return prefix + "\n\n"
    return prefix.rstrip() + "\n\n" + step + "\n\n"


def build_scoring_messages(
    question: str,
    reasoning_prefix: str,
    candidate_step: str,
    *,
    options: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    steps_so_far_str = build_steps_so_far(reasoning_prefix, candidate_step)
    return [
        {"role": "user", "content": build_prompt(question)},
        {"role": "assistant", "content": f"<think>{steps_so_far_str}"},
        {"role": "user", "content": EVAL_USER_MSG},
        {"role": "assistant", "content": SCORE_ASSISTANT_PREFIX},
    ]


def messages_to_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True,
    )


def parse_digit_score(token: str, top_logprobs: dict[str, float], *, method: str = "greedy") -> int:
    if method == "greedy":
        tok = (token or "").strip()
        return int(tok) if tok.isdigit() else 0
    probs: dict[int, float] = {}
    for tok, lp in top_logprobs.items():
        t = str(tok).strip()
        if t.isdigit() and len(t) == 1:
            probs[int(t)] = float(np.exp(lp))
    if not probs:
        return 0
    total = sum(probs.values())
    return int(round(sum(k * v / total for k, v in probs.items())))


def extract_vllm_digit_logprobs(output) -> tuple[str, dict[str, float]]:
    comp = output.outputs[0]
    token = comp.text or ""
    top: dict[str, float] = {}
    if comp.logprobs and comp.logprobs[0]:
        for tid, lp in comp.logprobs[0].items():
            # vLLM may return token id keys; decode if needed
            if hasattr(lp, "decoded_token"):
                key = lp.decoded_token
            elif hasattr(lp, "token"):
                key = lp.token
            else:
                key = str(tid)
            top[str(key)] = float(lp.logprob)
    return token, top


def truncate_reasoning_prefix(
    tokenizer,
    question: str,
    reasoning_prefix: str,
    candidate_step: str,
    *,
    max_input_tokens: int = 3800,
) -> tuple[str, bool]:
    """Keep tail of reasoning prefix so scoring prompt fits in context window."""
    messages = build_scoring_messages(question, "", candidate_step)
    overhead_prompt = messages_to_prompt(tokenizer, messages)
    overhead = len(tokenizer.encode(overhead_prompt, add_special_tokens=False))
    budget = max(256, max_input_tokens - overhead)
    ids = tokenizer.encode(reasoning_prefix, add_special_tokens=False)
    if len(ids) <= budget:
        return reasoning_prefix, False
    return tokenizer.decode(ids[-budget:], skip_special_tokens=False), True


def score_step_vllm(
    llm,
    tokenizer,
    question: str,
    reasoning_prefix: str,
    candidate_step: str,
    *,
    score_method: str = "greedy",
    max_input_tokens: int = 3800,
) -> dict[str, Any]:
    from vllm import SamplingParams

    trimmed, truncated = truncate_reasoning_prefix(
        tokenizer, question, reasoning_prefix, candidate_step, max_input_tokens=max_input_tokens
    )
    messages = build_scoring_messages(question, trimmed, candidate_step)
    prompt = messages_to_prompt(tokenizer, messages)
    params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=10,
        detokenize=True,
    )
    out = llm.generate([prompt], params)[0]
    token, top = extract_vllm_digit_logprobs(out)
    score = parse_digit_score(token, top, method=score_method)
    return {
        "utility_score": score,
        "score_token": token,
        "candidate_step": first_reasoning_step(candidate_step),
        "top_logprobs": top,
        "prefix_truncated": truncated,
    }

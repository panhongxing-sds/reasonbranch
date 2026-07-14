"""Pairwise comparison judge: greedy vs best branch."""

from __future__ import annotations

import random
import re
from typing import Any

from reasoning_branch_dataset.action_study.specreason_scorer import first_reasoning_step
from reasoning_branch_dataset.model_utils import build_prompt


def _clip_step(text: str, max_chars: int = 1200) -> str:
    text = first_reasoning_step(text or "")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def build_pairwise_messages(
    question: str,
    reasoning_prefix: str,
    greedy_step: str,
    branch_step: str,
    *,
    rng: random.Random,
) -> tuple[list[dict[str, str]], dict[str, str]]:
    g = _clip_step(greedy_step)
    b = _clip_step(branch_step)
    if rng.random() < 0.5:
        mapping = {"A": "greedy", "B": "branch"}
        step_a, step_b = g, b
    else:
        mapping = {"A": "branch", "B": "greedy"}
        step_a, step_b = b, g

    prefix_tail = (reasoning_prefix or "")[-1500:]
    user_msg = (
        "Compare two candidate NEXT steps after the same reasoning prefix.\n"
        "Judge mathematical correctness and valid progress only. "
        "Do NOT reward length, formatting, or verbosity. "
        "If mathematically equivalent, answer EQUIVALENT.\n\n"
        f"**Reasoning prefix (tail)**\n{prefix_tail}\n\n"
        f"**Step A**\n{step_a}\n\n"
        f"**Step B**\n{step_b}\n\n"
        "Reply with exactly one label on the first line:\n"
        "EQUIVALENT\n"
        "STEP_A_BETTER\n"
        "STEP_B_BETTER\n"
        "BOTH_REJECT"
    )
    messages = [
        {"role": "user", "content": build_prompt(question)},
        {"role": "assistant", "content": f"<think>{prefix_tail}"},
        {"role": "user", "content": user_msg},
    ]
    return messages, mapping


def parse_step_ab_label(text: str) -> str:
    first = (text.strip().splitlines() or [""])[0].upper()
    for lab in ("EQUIVALENT", "STEP_A_BETTER", "STEP_B_BETTER", "BOTH_REJECT"):
        if lab in first:
            return lab
    m = re.search(r"(EQUIVALENT|STEP_A_BETTER|STEP_B_BETTER|BOTH_REJECT)", text.upper())
    return m.group(1) if m else "UNKNOWN"


def verdict_from_ab(label: str, mapping: dict[str, str]) -> str:
    if label in ("EQUIVALENT", "BOTH_REJECT", "UNKNOWN"):
        return label
    if label == "STEP_A_BETTER":
        w = mapping.get("A", "")
    elif label == "STEP_B_BETTER":
        w = mapping.get("B", "")
    else:
        return "UNKNOWN"
    if w == "greedy":
        return "GREEDY_BETTER"
    if w == "branch":
        return "BRANCH_BETTER"
    return "UNKNOWN"


def judge_pairwise_vllm(
    llm,
    tokenizer,
    question: str,
    reasoning_prefix: str,
    greedy_step: str,
    branch_step: str,
    *,
    seed: int = 0,
) -> dict[str, Any]:
    from vllm import SamplingParams

    rng = random.Random(seed)
    messages, mapping = build_pairwise_messages(
        question, reasoning_prefix, greedy_step, branch_step, rng=rng
    )
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    params = SamplingParams(temperature=0.0, max_tokens=64)
    out = llm.generate([prompt], params)[0]
    raw = out.outputs[0].text or ""
    ab = parse_step_ab_label(raw)
    verdict = verdict_from_ab(ab, mapping)
    return {
        "pairwise_ab_label": ab,
        "pairwise_verdict": verdict,
        "mapping": mapping,
        "raw_response": raw.strip(),
    }

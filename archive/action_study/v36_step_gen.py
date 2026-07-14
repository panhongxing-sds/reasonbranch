"""V3.6 step generation with explicit boundary status."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Literal

from reasoning_branch_dataset.action_study.step_extraction import (
    STEP_END_RE,
    extract_next_substantive_step,
    strip_model_thinking,
)

StepStatus = Literal[
    "EXPLICIT_STEP_END",
    "FALLBACK_PARAGRAPH",
    "MAX_TOKEN_TRUNCATED",
    "MALFORMED_STEP",
    "EMPTY_STEP",
]

STEP_INSTRUCTION = (
    "Continue the reasoning with exactly one complete next step. "
    "End the step with <STEP_END> on its own."
)


def _cuda_sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


@dataclass
class StepGenResult:
    text: str
    token_ids: list[int]
    num_tokens: int
    latency_sec: float
    status: StepStatus
    finish_reason: str | None
    raw_text: str


def classify_step_status(raw: str, *, finish_reason: str | None, max_tokens: int, n_tokens: int) -> StepStatus:
    text = strip_model_thinking(raw or "")
    if not text.strip():
        return "EMPTY_STEP"
    if STEP_END_RE.search(raw or "") or STEP_END_RE.search(text):
        return "EXPLICIT_STEP_END"
    if finish_reason == "length" or n_tokens >= max_tokens:
        return "MAX_TOKEN_TRUNCATED"
    if "\n\n" in text or text.count("\n") >= 1:
        return "FALLBACK_PARAGRAPH"
    # Single block without boundary marker
    if len(text.strip()) < 20:
        return "MALFORMED_STEP"
    return "FALLBACK_PARAGRAPH"


def normalize_step_text(raw: str) -> str:
    text = strip_model_thinking(raw or "")
    text = STEP_END_RE.sub("", text).strip()
    # Prefer first paragraph block if multiple
    parts = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    if not parts:
        return ""
    return parts[0]


def is_main_analysis_status(status: StepStatus) -> bool:
    return status in ("EXPLICIT_STEP_END", "FALLBACK_PARAGRAPH")


def generate_one_step_vllm(
    llm: Any,
    tokenizer: Any,
    prefix_text: str,
    *,
    max_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    n: int = 1,
    seed: int | None = None,
    append_instruction: bool = True,
) -> list[StepGenResult]:
    """Generate n complete next-step candidates from prefix.

    `seed` is passed to vLLM SamplingParams so pool sampling is reproducible and
    NOT tied to the engine-global seed (which otherwise makes identical prompts
    deterministic and collapses n>1 diversity in degenerate cases).
    """
    from vllm import SamplingParams

    prompt = prefix_text
    if append_instruction and not prefix_text.rstrip().endswith("STEP_END>"):
        # Soft reminder only once at the end of accepted prefix region.
        pass  # keep prefix exact; instruction baked into system prompt elsewhere if needed

    params = SamplingParams(
        n=n,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        stop=["<STEP_END>", "</STEP_END>", "\n\n"],
        include_stop_str_in_output=True,
        detokenize=True,
    )
    prefix_ids = tokenizer.encode(prompt, add_special_tokens=False)
    _cuda_sync()
    t0 = time.perf_counter()
    outs = llm.generate([{"prompt_token_ids": prefix_ids}], params)
    _cuda_sync()
    wall = time.perf_counter() - t0

    results: list[StepGenResult] = []
    for comp in outs[0].outputs:
        raw = comp.text or ""
        tids = list(comp.token_ids)
        status = classify_step_status(
            raw,
            finish_reason=getattr(comp, "finish_reason", None),
            max_tokens=max_tokens,
            n_tokens=len(tids),
        )
        text = normalize_step_text(raw)
        if not text and status != "EMPTY_STEP":
            status = "EMPTY_STEP"
        results.append(
            StepGenResult(
                text=text,
                token_ids=tids,
                num_tokens=len(tids),
                latency_sec=wall / max(n, 1),
                status=status,
                finish_reason=getattr(comp, "finish_reason", None),
                raw_text=raw,
            )
        )
    # Wall-clock for the whole n-parallel call should be attributed once to caller;
    # also stash full wall on first result via attribute convention.
    if results:
        results[0].latency_sec = wall  # full batch wall for n>1
        for r in results[1:]:
            r.latency_sec = wall
    return results


def extract_status_from_continuation(continuation: str, *, question: str = "") -> dict[str, Any]:
    """Reuse extract_next_substantive_step + status tagging."""
    raw = continuation or ""
    status = classify_step_status(raw, finish_reason=None, max_tokens=10**9, n_tokens=0)
    ext = extract_next_substantive_step(raw, question=question)
    return {"status": status, "extraction": ext, "text": normalize_step_text(raw)}

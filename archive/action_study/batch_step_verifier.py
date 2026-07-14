"""32B batch verification of K complete reasoning steps (V3.5a).

Design goals for Cost Gate:
- shared prompt stem is byte-identical across K candidates (prefix-cache friendly)
- judge decode is 1–2 tokens only (ACCEPT / REJECT)
- wall-clock timing uses CUDA synchronize when available
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from reasoning_branch_dataset.action_study.local_step_verifier import parse_accept_reject

_CLIP_Q = 800
_CLIP_P = 1000
_CLIP_C = 600

# Shared stem ends right before the candidate body so K requests share one KV prefix.
VERIFY_STEM = """You are a strict math reasoning step verifier.

Problem:
{question}

Current reasoning prefix:
{prefix_tail}

Candidate next step:
"""

VERIFY_SUFFIX = """{candidate}

Answer with exactly one word: ACCEPT or REJECT."""


def _clip(text: str, n: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[: n - 3] + "..."


def build_verify_stem(*, question: str, prefix_text: str) -> str:
    return VERIFY_STEM.format(
        question=_clip(question, _CLIP_Q),
        prefix_tail=_clip(prefix_text, _CLIP_P),
    )


def build_verify_prompt(*, question: str, prefix_text: str, candidate: str) -> str:
    stem = build_verify_stem(question=question, prefix_text=prefix_text)
    return stem + VERIFY_SUFFIX.format(candidate=_clip(candidate, _CLIP_C))


def _cuda_sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


@dataclass
class BatchVerifyResult:
    acceptable: list[bool | None]
    raw_outputs: list[str]
    latency_sec: float
    n_candidates: int
    parsed_rate: float
    shared_stem_tokens: int = 0
    prompt_tokens: list[int] | None = None
    output_tokens: list[int] | None = None

    def any_accepted(self) -> bool:
        return any(a is True for a in self.acceptable)

    def first_accepted_index(self) -> int | None:
        for i, a in enumerate(self.acceptable):
            if a is True:
                return i
        return None


class BatchStepVerifier:
    """Batch ACCEPT/REJECT over complete candidate steps (vLLM target)."""

    def __init__(
        self,
        llm: Any,
        tokenizer: Any,
        *,
        max_tokens: int = 2,
        temperature: float = 0.0,
    ) -> None:
        self.llm = llm
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.temperature = temperature

    def verify_batch(
        self,
        *,
        question: str,
        prefix_text: str,
        candidates: list[str],
    ) -> BatchVerifyResult:
        from vllm import SamplingParams

        if not candidates:
            return BatchVerifyResult([], [], 0.0, 0, 0.0)

        stem = build_verify_stem(question=question, prefix_text=prefix_text)
        stem_ids = self.tokenizer.encode(stem, add_special_tokens=False)
        prompts: list[dict[str, list[int]]] = []
        prompt_lens: list[int] = []
        for c in candidates:
            full = stem + VERIFY_SUFFIX.format(candidate=_clip(c, _CLIP_C))
            ids = self.tokenizer.encode(full, add_special_tokens=False)
            # Sanity: stem must be a true prefix of each prompt for cache hits.
            if ids[: len(stem_ids)] != stem_ids:
                # Fall back to re-encoding full text only (should be rare).
                pass
            prompts.append({"prompt_token_ids": ids})
            prompt_lens.append(len(ids))

        params = SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=1.0,
            stop=["\n", " "],
            detokenize=True,
        )
        _cuda_sync()
        t0 = time.perf_counter()
        outs = self.llm.generate(prompts, params)
        _cuda_sync()
        latency = time.perf_counter() - t0

        raws: list[str] = []
        accepts: list[bool | None] = []
        out_toks: list[int] = []
        for out in outs:
            tids = list(out.outputs[0].token_ids)
            out_toks.append(len(tids))
            text = self.tokenizer.decode(tids, skip_special_tokens=True)
            raws.append(text.strip())
            accepts.append(parse_accept_reject(text))

        parsed = sum(1 for a in accepts if a is not None)
        return BatchVerifyResult(
            acceptable=accepts,
            raw_outputs=raws,
            latency_sec=latency,
            n_candidates=len(candidates),
            parsed_rate=parsed / len(candidates),
            shared_stem_tokens=len(stem_ids),
            prompt_tokens=prompt_lens,
            output_tokens=out_toks,
        )

    def verify_one(
        self,
        *,
        question: str,
        prefix_text: str,
        candidate: str,
    ) -> dict[str, Any]:
        res = self.verify_batch(
            question=question,
            prefix_text=prefix_text,
            candidates=[candidate],
        )
        return {
            "acceptable": res.acceptable[0] if res.acceptable else None,
            "raw": res.raw_outputs[0] if res.raw_outputs else "",
            "latency_sec": res.latency_sec,
            "parsed": res.acceptable[0] is not None if res.acceptable else False,
            "output_tokens": (res.output_tokens or [None])[0],
        }


def score_accept_reject_text(text: str) -> bool | None:
    return parse_accept_reject(text)


def extract_accept_token(text: str) -> str | None:
    t = (text or "").strip().upper()
    m = re.search(r"\b(ACCEPT|REJECT)\b", t)
    return m.group(1) if m else None

"""V3.6 online step verifier: single-token Accept/Reject logits (no prose).

Qwen tokenizer note:
  - 'ACCEPT' is 1 token, 'REJECT' is 2 tokens → unsafe for fair next-token scoring.
  - We score ' Accept' vs ' Reject' (both 1 token) after a shared stem ending in
    'Judgment:'. Semantic labels remain ACCEPT/REJECT.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

VERIFY_STEM = """You are a strict math reasoning step verifier.

Problem:
{question}

Current reasoning prefix:
{prefix_tail}

Candidate next step:
"""

VERIFY_TAIL = """{candidate}

Is the candidate mathematically correct, consistent with the problem and prefix,
substantive, and safe to append?

Judgment:"""


def _clip(text: str, n: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[: n - 3] + "..."


def _cuda_sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


@dataclass
class LabelTokens:
    accept_token_id: int
    reject_token_id: int
    accept_str: str
    reject_str: str


def resolve_label_tokens(tokenizer: Any) -> LabelTokens:
    accept_str, reject_str = " Accept", " Reject"
    a = tokenizer.encode(accept_str, add_special_tokens=False)
    r = tokenizer.encode(reject_str, add_special_tokens=False)
    if len(a) != 1 or len(r) != 1:
        raise RuntimeError(
            f"Expected single-token labels, got Accept={a} Reject={r}. "
            "Pick a different label pair for this tokenizer."
        )
    return LabelTokens(a[0], r[0], accept_str, reject_str)


@dataclass
class VerifyScore:
    score: float  # logP(Accept) - logP(Reject)
    logp_accept: float
    logp_reject: float
    accepted: bool | None  # None until threshold applied


@dataclass
class BatchLogitVerifyResult:
    scores: list[VerifyScore]
    latency_sec: float
    shared_stem_tokens: int
    prompt_tokens: list[int]
    tau_accept: float | None = None

    def selected_index(self) -> int | None:
        if not self.scores:
            return None
        return max(range(len(self.scores)), key=lambda i: self.scores[i].score)

    def any_accepted(self, tau: float) -> bool:
        return any(s.score >= tau for s in self.scores)


class LogitStepVerifier:
    """Batch prefill + next-token Accept/Reject logit scoring."""

    def __init__(
        self,
        llm: Any,
        tokenizer: Any,
        *,
        clip_q: int = 800,
        clip_p: int = 1000,
        clip_c: int = 600,
        prompt_logprobs: int = 5,
    ) -> None:
        self.llm = llm
        self.tokenizer = tokenizer
        self.clip_q = clip_q
        self.clip_p = clip_p
        self.clip_c = clip_c
        self.prompt_logprobs = prompt_logprobs
        self.labels = resolve_label_tokens(tokenizer)
        self.gen_logprobs = 20  # vLLM max allowed sample logprobs

    def build_prompt(self, *, question: str, prefix_text: str, candidate: str) -> str:
        stem = VERIFY_STEM.format(
            question=_clip(question, self.clip_q),
            prefix_tail=_clip(prefix_text, self.clip_p),
        )
        return stem + VERIFY_TAIL.format(candidate=_clip(candidate, self.clip_c))

    def build_stem(self, *, question: str, prefix_text: str) -> str:
        return VERIFY_STEM.format(
            question=_clip(question, self.clip_q),
            prefix_tail=_clip(prefix_text, self.clip_p),
        )

    def score_batch(
        self,
        *,
        question: str,
        prefix_text: str,
        candidates: list[str],
        tau_accept: float | None = None,
    ) -> BatchLogitVerifyResult:
        """Score K candidates via prompt_logprobs of Accept/Reject (fair pair)."""
        from vllm import SamplingParams

        if not candidates:
            return BatchLogitVerifyResult([], 0.0, 0, [])

        stem = self.build_stem(question=question, prefix_text=prefix_text)
        stem_ids = self.tokenizer.encode(stem, add_special_tokens=False)
        prompts: list[dict[str, list[int]]] = []
        prompt_lens: list[int] = []
        aid = self.labels.accept_token_id
        rid = self.labels.reject_token_id
        for c in candidates:
            full = self.build_prompt(question=question, prefix_text=prefix_text, candidate=c)
            base = self.tokenizer.encode(full, add_special_tokens=False)
            prompts.append({"prompt_token_ids": base + [aid]})
            prompts.append({"prompt_token_ids": base + [rid]})
            prompt_lens.append(len(base))

        # max_tokens=1 still runs decode; prompt_logprobs scores the appended label token.
        params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            prompt_logprobs=1,
            detokenize=False,
        )
        _cuda_sync()
        t0 = time.perf_counter()
        outs = self.llm.generate(prompts, params)
        _cuda_sync()
        latency = time.perf_counter() - t0

        scores: list[VerifyScore] = []
        for i in range(len(candidates)):
            lp_accept = self._prompt_token_logprob(outs[2 * i], aid)
            lp_reject = self._prompt_token_logprob(outs[2 * i + 1], rid)
            score = lp_accept - lp_reject
            accepted = None if tau_accept is None else (score >= tau_accept)
            scores.append(
                VerifyScore(
                    score=score,
                    logp_accept=lp_accept,
                    logp_reject=lp_reject,
                    accepted=accepted,
                )
            )

        return BatchLogitVerifyResult(
            scores=scores,
            latency_sec=latency,
            shared_stem_tokens=len(stem_ids),
            prompt_tokens=prompt_lens,
            tau_accept=tau_accept,
        )

    @staticmethod
    def _prompt_token_logprob(out: Any, token_id: int) -> float:
        """Logprob of the last prompt token (the Accept/Reject label)."""
        pl = getattr(out, "prompt_logprobs", None)
        if not pl:
            return -20.0
        last = pl[-1]
        if last is None:
            return -20.0
        if token_id in last:
            v = last[token_id]
            return float(v.logprob if hasattr(v, "logprob") else v)
        # Fallback: any single entry (should be the scored token)
        if len(last) == 1:
            v = next(iter(last.values()))
            return float(v.logprob if hasattr(v, "logprob") else v)
        return -20.0

    def score_greedy(
        self,
        *,
        question: str,
        prefix_text: str,
        greedy_step: str,
        tau_accept: float,
    ) -> VerifyScore:
        res = self.score_batch(
            question=question,
            prefix_text=prefix_text,
            candidates=[greedy_step],
            tau_accept=tau_accept,
        )
        return res.scores[0]

"""V4.0 draft-intrinsic self-confidence signals (near-zero cost).

The V3.6 finding: the 32B target, prompted as an Accept/Reject step verifier,
has ~0 discriminative power on the real distribution of 1.5B-generated steps
(max precision 7%, AUC~0.5), even though it separates hand-built obvious
right/wrong (AUC 0.97). The V4.0 hypothesis: the *draft itself* exposes cheap
intrinsic signals (teacher-forced token logprobs, predictive entropy/margin, a
self-eval token) that predict whether its own step is oracle-acceptable.

This module extracts those signals by teacher-forcing each candidate step
through the draft model and reading `prompt_logprobs`. Cost is a single prefill
per candidate (no extra generation), so it is essentially free relative to the
draft step that already produced the candidate.

Design: the vLLM call and the numeric parsing are separated so the alignment
logic (`parse_teacher_forcing_signals`) is unit-testable without a live model.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Sequence


def _lp_value(v: Any) -> float:
    """Accept either a vLLM Logprob object (.logprob) or a raw float."""
    if v is None:
        return float("-inf")
    return float(v.logprob if hasattr(v, "logprob") else v)


def _entropy_from_logprobs(logprobs: Sequence[float]) -> float:
    """Shannon entropy (nats) of the renormalized top-k logprob distribution."""
    lps = [lp for lp in logprobs if lp != float("-inf")]
    if not lps:
        return 0.0
    m = max(lps)
    probs = [math.exp(lp - m) for lp in lps]
    s = sum(probs)
    if s <= 0:
        return 0.0
    probs = [p / s for p in probs]
    return float(-sum(p * math.log(p + 1e-12) for p in probs))


@dataclass
class SelfSignals:
    """All signals are oriented so that HIGHER = draft more confident/fluent,
    except entropy/perplexity where higher = less confident. The analyzer flips
    signs as needed when computing AUC."""

    n_tokens: int
    mean_logprob: float
    min_logprob: float
    last_logprob: float
    perplexity: float
    mean_entropy: float
    max_entropy: float
    mean_margin: float
    min_margin: float
    self_eval_logit: float  # logP(yes) - logP(no); NaN if unavailable
    repetition_rate: float
    char_len: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _degenerate_signals(candidate: str) -> SelfSignals:
    return SelfSignals(
        n_tokens=0,
        mean_logprob=-20.0,
        min_logprob=-20.0,
        last_logprob=-20.0,
        perplexity=float("inf"),
        mean_entropy=0.0,
        max_entropy=0.0,
        mean_margin=0.0,
        min_margin=0.0,
        self_eval_logit=float("nan"),
        repetition_rate=0.0,
        char_len=len(candidate or ""),
    )


def _repetition_rate(token_ids: Sequence[int]) -> float:
    """1 - (distinct bigrams / total bigrams). 0 for <2 tokens."""
    if len(token_ids) < 2:
        return 0.0
    bigrams = list(zip(token_ids[:-1], token_ids[1:]))
    return 1.0 - (len(set(bigrams)) / len(bigrams))


def parse_teacher_forcing_signals(
    prompt_logprobs: Sequence[Any],
    full_ids: Sequence[int],
    cand_start: int,
    candidate: str,
) -> SelfSignals:
    """Parse draft self-signals from teacher-forcing prompt_logprobs.

    `prompt_logprobs[j]` (for j>=1) is a dict {token_id: Logprob|float} giving,
    for the actual token `full_ids[j]`, its logprob under the draft, plus the
    top-k alternatives at that position. `prompt_logprobs[0]` is None.

    Candidate token positions are `range(cand_start, len(full_ids))`.
    """
    cand_positions = range(cand_start, len(full_ids))
    tok_logprobs: list[float] = []
    entropies: list[float] = []
    margins: list[float] = []
    for j in cand_positions:
        if j <= 0 or j >= len(prompt_logprobs):
            continue
        cell = prompt_logprobs[j]
        if not cell:
            continue
        actual_id = full_ids[j]
        # Actual-token logprob (vLLM always includes the realized token).
        actual = cell.get(actual_id) if hasattr(cell, "get") else None
        tok_lp = _lp_value(actual) if actual is not None else min(
            (_lp_value(v) for v in cell.values()), default=-20.0
        )
        tok_logprobs.append(tok_lp)
        # Predictive distribution over the top-k at this position.
        dist = sorted((_lp_value(v) for v in cell.values()), reverse=True)
        entropies.append(_entropy_from_logprobs(dist))
        if len(dist) >= 2:
            margins.append(dist[0] - dist[1])
        else:
            margins.append(0.0)

    if not tok_logprobs:
        return _degenerate_signals(candidate)

    mean_lp = sum(tok_logprobs) / len(tok_logprobs)
    return SelfSignals(
        n_tokens=len(tok_logprobs),
        mean_logprob=mean_lp,
        min_logprob=min(tok_logprobs),
        last_logprob=tok_logprobs[-1],
        perplexity=math.exp(-mean_lp) if mean_lp > -50 else float("inf"),
        mean_entropy=sum(entropies) / len(entropies) if entropies else 0.0,
        max_entropy=max(entropies) if entropies else 0.0,
        mean_margin=sum(margins) / len(margins) if margins else 0.0,
        min_margin=min(margins) if margins else 0.0,
        self_eval_logit=float("nan"),
        repetition_rate=_repetition_rate(list(full_ids[cand_start:])),
        char_len=len(candidate or ""),
    )


def _resolve_pair(tokenizer: Any, options: list[tuple[str, str]]) -> tuple[int, int] | None:
    """Return (yes_id, no_id) for the first option pair that is single-token each."""
    for yes_str, no_str in options:
        y = tokenizer.encode(yes_str, add_special_tokens=False)
        n = tokenizer.encode(no_str, add_special_tokens=False)
        if len(y) == 1 and len(n) == 1:
            return y[0], n[0]
    return None


SELF_EVAL_SUFFIX = (
    "\n\nQuestion: Is the reasoning step above correct and a useful, substantive "
    "next step toward solving the problem? Answer with a single word, yes or no.\n"
    "Answer:"
)


class DraftSelfSignalExtractor:
    """Extracts self-confidence signals by teacher-forcing candidates through the draft."""

    def __init__(
        self,
        llm: Any,
        tokenizer: Any,
        *,
        topk: int = 20,
        max_len: int = 4000,
        self_eval: bool = True,
    ) -> None:
        self.llm = llm
        self.tokenizer = tokenizer
        self.topk = topk
        self.max_len = max_len
        self.self_eval = self_eval
        self._yes_no = _resolve_pair(
            tokenizer,
            [(" yes", " no"), (" Yes", " No"), ("yes", "no"), ("Yes", "No")],
        ) if self_eval else None

    def _clip_prefix(self, prefix_ids: list[int], reserve: int) -> list[int]:
        budget = self.max_len - reserve
        if budget <= 0:
            return prefix_ids[-1:]
        if len(prefix_ids) <= budget:
            return prefix_ids
        return prefix_ids[-budget:]  # keep most recent context

    def extract_batch(self, items: list[dict[str, str]]) -> list[SelfSignals]:
        """items: [{prefix_text, candidate}]. Returns aligned SelfSignals list."""
        from vllm import SamplingParams

        # Plan per-item sequences: teacher-forcing (tf), and optional self-eval yes/no.
        plan: list[dict[str, Any]] = []
        prompts: list[dict[str, list[int]]] = []
        for it in items:
            prefix = it.get("prefix_text", "")
            cand = it.get("candidate", "")
            cand_ids = self.tokenizer.encode(cand, add_special_tokens=False)
            if not cand_ids:
                plan.append({"empty": True, "candidate": cand})
                continue
            prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
            prefix_ids = self._clip_prefix(prefix_ids, reserve=len(cand_ids) + 8)
            full = prefix_ids + cand_ids
            cand_start = len(prefix_ids)
            entry: dict[str, Any] = {
                "empty": False,
                "candidate": cand,
                "full": full,
                "cand_start": cand_start,
                "tf_idx": len(prompts),
            }
            prompts.append({"prompt_token_ids": full})
            if self.self_eval and self._yes_no is not None:
                yes_id, no_id = self._yes_no
                eval_ids = self.tokenizer.encode(prefix + cand + SELF_EVAL_SUFFIX, add_special_tokens=False)
                eval_ids = self._clip_prefix(eval_ids, reserve=8)
                # Single prefill: append yes_id and read BOTH yes/no logprobs from
                # the top-k predictive distribution at that final position. The no
                # logprob is identical to what a separate `eval+no` prompt would
                # report (same conditioning context), so this is numerically lossless
                # while halving self-eval prefills.
                entry["eval_idx"] = len(prompts)
                prompts.append({"prompt_token_ids": eval_ids + [yes_id]})
                entry["yes_id"] = yes_id
                entry["no_id"] = no_id
            plan.append(entry)

        if not prompts:
            return [_degenerate_signals(p.get("candidate", "")) for p in plan]

        params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            prompt_logprobs=self.topk,
            detokenize=False,
        )
        outs = self.llm.generate(prompts, params)

        results: list[SelfSignals] = []
        for entry in plan:
            if entry.get("empty"):
                results.append(_degenerate_signals(entry.get("candidate", "")))
                continue
            tf_out = outs[entry["tf_idx"]]
            sig = parse_teacher_forcing_signals(
                getattr(tf_out, "prompt_logprobs", None) or [],
                entry["full"],
                entry["cand_start"],
                entry["candidate"],
            )
            if "eval_idx" in entry:
                ev = outs[entry["eval_idx"]]
                lp_yes = LogprobReader.last_prompt_token_logprob(ev, entry["yes_id"])
                lp_no = LogprobReader.last_prompt_token_logprob(ev, entry["no_id"])
                sig.self_eval_logit = lp_yes - lp_no
            results.append(sig)
        return results


class LogprobReader:
    @staticmethod
    def last_prompt_token_logprob(out: Any, token_id: int) -> float:
        pl = getattr(out, "prompt_logprobs", None)
        if not pl:
            return -20.0
        last = pl[-1]
        if not last:
            return -20.0
        if token_id in last:
            return _lp_value(last[token_id])
        if len(last) == 1:
            return _lp_value(next(iter(last.values())))
        return -20.0


# Ordered list of signals and the sign that makes "higher = more likely acceptable"
# a reasonable prior (used by the analyzer to orient AUC without cheating).
SIGNAL_ORIENTATION: dict[str, int] = {
    "mean_logprob": +1,
    "min_logprob": +1,
    "last_logprob": +1,
    "perplexity": -1,
    "mean_entropy": -1,
    "max_entropy": -1,
    "mean_margin": +1,
    "min_margin": +1,
    "self_eval_logit": +1,
    "repetition_rate": -1,
    "n_tokens": +1,
    "char_len": +1,
}


def _timed_extract(extractor: DraftSelfSignalExtractor, items: list[dict[str, str]]) -> tuple[list[SelfSignals], float]:
    t0 = time.perf_counter()
    sigs = extractor.extract_batch(items)
    return sigs, time.perf_counter() - t0

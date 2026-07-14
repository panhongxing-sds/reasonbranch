"""V3.6 one-step counterfactual timing: Direct Handoff vs Branch@K pipeline."""

from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from reasoning_branch_dataset.action_study.logit_step_verifier import LogitStepVerifier
from reasoning_branch_dataset.action_study.v36_step_gen import (
    StepGenResult,
    generate_one_step_vllm,
    is_main_analysis_status,
)


def _cuda_sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def gamma_margin(t_h_sec: float, *, abs_ms: float = 50.0, rel: float = 0.05) -> float:
    return max(abs_ms / 1000.0, rel * t_h_sec)


@dataclass
class PathTiming:
    wall_sec: float
    draft_sec: float | None = None
    verify_sec: float | None = None
    fallback_sec: float | None = None
    step_text: str = ""
    step_tokens: int = 0
    step_status: str = ""
    selected_index: int | None = None
    verifier_scores: list[float] = field(default_factory=list)
    branch_accepted: bool = False
    used_fallback: bool = False


@dataclass
class CounterfactualTrial:
    t_h: PathTiming
    t_b: dict[str, PathTiming]  # keys "1","2","4"
    deltas: dict[str, float]
    gammas: dict[str, float]


class DualResidentSession:
    """1.5B draft + 32B target on one GPU."""

    def __init__(
        self,
        *,
        draft_model: str,
        target_model: str,
        target_quantization: str | None = "awq",
        draft_gpu_util: float = 0.18,
        target_gpu_util: float = 0.70,
        max_model_len: int = 4096,
        verify_tau: float = 0.0,
    ) -> None:
        from reasoning_branch_dataset.action_study.target_verifier import build_target_verifier
        from reasoning_branch_dataset.action_study.vllm_backend import VLLMEngine

        self.target = build_target_verifier(
            target_model,
            engine="vllm",
            gpu_memory_utilization=target_gpu_util,
            max_model_len=max_model_len,
            quantization=target_quantization if (target_quantization and "awq" in target_model.lower()) else None,
            dual_resident=True,
            enable_prefix_caching=True,
        )
        self.draft = VLLMEngine(
            draft_model,
            gpu_memory_utilization=draft_gpu_util,
            max_model_len=max_model_len,
            enable_prefix_caching=True,
        )
        self.verifier = LogitStepVerifier(self.target.llm, self.target.tokenizer)
        self.verify_tau = verify_tau
        self.max_model_len = max_model_len

    @property
    def target_tok(self):
        return self.target.tokenizer

    def warmup(self, prefix: str, *, n: int = 20) -> None:
        for _ in range(n):
            _ = generate_one_step_vllm(
                self.draft.llm, self.draft.llm.get_tokenizer(), prefix, max_tokens=16, n=1
            )
            _ = generate_one_step_vllm(
                self.target.llm, self.target_tok, prefix, max_tokens=16, n=1
            )
            _ = self.verifier.score_batch(
                question="warmup", prefix_text=prefix[-200:], candidates=["warmup step"]
            )


def run_direct_handoff(
    session: DualResidentSession,
    *,
    prefix_text: str,
    max_tokens: int = 256,
) -> PathTiming:
    """Path H: 32B generates one replacement step from prefix (warm)."""
    # Touch prefix on target to establish warm KV (shared with prior greedy verify in real system).
    # In V3.6 collect phase, caller should have just verified greedy on this prefix.
    _cuda_sync()
    t0 = time.perf_counter()
    steps = generate_one_step_vllm(
        session.target.llm,
        session.target_tok,
        prefix_text,
        max_tokens=max_tokens,
        temperature=0.0,
        n=1,
    )
    _cuda_sync()
    wall = time.perf_counter() - t0
    s = steps[0]
    return PathTiming(
        wall_sec=wall,
        step_text=s.text,
        step_tokens=s.num_tokens,
        step_status=s.status,
    )


def draft_branch_pool(
    session: DualResidentSession,
    *,
    prefix_text: str,
    pool_size: int = 4,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    seed: int | None = None,
) -> tuple[list[StepGenResult], float]:
    """Draw ONE fixed branch pool of `pool_size` candidates from the draft model.

    Returns (results, draft_wall_sec). The same pool is later sliced K1⊂K2⊂K4 so
    all K share identical content and scoring is decoupled from timing noise.
    """
    _cuda_sync()
    t0 = time.perf_counter()
    pool = generate_one_step_vllm(
        session.draft.llm,
        session.draft.llm.get_tokenizer(),
        prefix_text,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        n=pool_size,
        seed=seed,
    )
    _cuda_sync()
    draft_sec = time.perf_counter() - t0
    return pool, draft_sec


def run_branch_pipeline(
    session: DualResidentSession,
    *,
    question: str,
    prefix_text: str,
    k: int,
    max_tokens: int = 256,
    tau_accept: float | None = None,
    branch_texts: list[str],
    branch_statuses: list[str] | None = None,
    draft_sec: float = 0.0,
) -> PathTiming:
    """Full Branch@K pipeline wall-clock using a FIXED pre-sampled pool.

    `branch_texts` is the shared pool (len>=k); we slice the first k (nested).
    `draft_sec` is the (shared) draft cost measured once for the whole pool and
    attributed to the pipeline wall so K-comparisons stay fair.
    """
    tau = session.verify_tau if tau_accept is None else tau_accept
    texts = list(branch_texts[:k])
    statuses = list((branch_statuses or ["FALLBACK_PARAGRAPH"] * len(texts))[:k])
    while len(texts) < k:
        texts.append(" ")
        statuses.append("EMPTY_STEP")

    _cuda_sync()
    t0 = time.perf_counter()

    # Stage 2: batch verify (Stage 1 draft cost is supplied via draft_sec)
    t_v0 = time.perf_counter()
    vres = session.verifier.score_batch(
        question=question,
        prefix_text=prefix_text,
        candidates=texts,
        tau_accept=tau,
    )
    verify_sec = time.perf_counter() - t_v0

    scores = [s.score for s in vres.scores]
    k_star = vres.selected_index()
    branch_ok = k_star is not None and scores[k_star] >= tau

    fallback_sec = None
    used_fallback = False
    step_text = ""
    step_tokens = 0
    step_status = ""
    selected_index = k_star if branch_ok else None

    if branch_ok:
        step_text = texts[k_star]
        step_tokens = len(session.target_tok.encode(step_text))
        step_status = statuses[k_star] if k_star < len(statuses) else "FALLBACK_PARAGRAPH"
    else:
        used_fallback = True
        t_f0 = time.perf_counter()
        fb = generate_one_step_vllm(
            session.target.llm,
            session.target_tok,
            prefix_text,
            max_tokens=max_tokens,
            temperature=0.0,
            n=1,
        )[0]
        _cuda_sync()
        fallback_sec = time.perf_counter() - t_f0
        step_text = fb.text
        step_tokens = fb.num_tokens
        step_status = fb.status

    _cuda_sync()
    wall = draft_sec + (time.perf_counter() - t0)
    return PathTiming(
        wall_sec=wall,
        draft_sec=draft_sec,
        verify_sec=verify_sec,
        fallback_sec=fallback_sec,
        step_text=step_text,
        step_tokens=step_tokens,
        step_status=step_status,
        selected_index=selected_index,
        verifier_scores=scores,
        branch_accepted=branch_ok,
        used_fallback=used_fallback,
    )


def run_counterfactual_once(
    session: DualResidentSession,
    *,
    question: str,
    prefix_text: str,
    ks: tuple[int, ...] = (1, 2, 4),
    max_tokens: int = 256,
    path_order: list[str] | None = None,
    rng: random.Random | None = None,
) -> CounterfactualTrial:
    """Pair Handoff and Branch@K on the same prefix; randomize path order."""
    rng = rng or random.Random(0)
    order = path_order or (["H"] + [f"B{k}" for k in ks])
    order = list(order)
    rng.shuffle(order)

    t_h: PathTiming | None = None
    t_b: dict[str, PathTiming] = {}

    # Fresh warm: score a dummy candidate once to touch shared prefix KV on target.
    _ = session.verifier.score_batch(
        question=question, prefix_text=prefix_text, candidates=["."]
    )

    # One fixed pool for all K (nested K1⊂K2⊂K4), decoupling content from timing.
    pool_size = max(ks)
    pool, pool_draft_sec = draft_branch_pool(
        session, prefix_text=prefix_text, pool_size=pool_size, max_tokens=max_tokens
    )
    pool_texts = [p.text or " " for p in pool]
    pool_statuses = [p.status for p in pool]

    for name in order:
        if name == "H":
            t_h = run_direct_handoff(session, prefix_text=prefix_text, max_tokens=max_tokens)
        elif name.startswith("B"):
            k = int(name[1:])
            t_b[str(k)] = run_branch_pipeline(
                session,
                question=question,
                prefix_text=prefix_text,
                k=k,
                max_tokens=max_tokens,
                branch_texts=pool_texts,
                branch_statuses=pool_statuses,
                draft_sec=pool_draft_sec,
            )

    assert t_h is not None
    for k in ks:
        if str(k) not in t_b:
            t_b[str(k)] = run_branch_pipeline(
                session,
                question=question,
                prefix_text=prefix_text,
                k=k,
                max_tokens=max_tokens,
                branch_texts=pool_texts,
                branch_statuses=pool_statuses,
                draft_sec=pool_draft_sec,
            )

    deltas = {str(k): t_h.wall_sec - t_b[str(k)].wall_sec for k in ks}
    gammas = {str(k): gamma_margin(t_h.wall_sec) for k in ks}
    return CounterfactualTrial(t_h=t_h, t_b=t_b, deltas=deltas, gammas=gammas)


def summarize_reps(walls: list[float]) -> dict[str, float | None]:
    if not walls:
        return {"n": 0, "median": None, "mean": None, "p90": None, "std": None, "cv": None}
    xs = sorted(walls)
    n = len(xs)
    mean = sum(xs) / n
    median = xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])
    p90 = xs[min(n - 1, int(math.ceil(0.9 * n) - 1))]
    var = sum((x - mean) ** 2 for x in xs) / max(n - 1, 1)
    std = math.sqrt(var)
    cv = (std / mean) if mean > 1e-9 else None
    return {"n": float(n), "median": median, "mean": mean, "p90": p90, "std": std, "cv": cv}


def timing_to_dict(t: PathTiming) -> dict[str, Any]:
    return asdict(t)

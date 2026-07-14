"""V3.5 cost–rescue math: break-even thresholds and policy decision.

Core inequality (sunk greedy-verify cost ignored):

    C_branch = C_DK + C_VK + (1 - r_K) * C_T
    C_handoff = C_T

Branch is worthwhile iff:

    r_K > (C_DK + C_VK) / C_T  =: r_K^*
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


Decision = Literal["always_branch", "never_branch", "train_predictor", "dominated"]


@dataclass(frozen=True)
class CostSample:
    """One microbenchmark observation (seconds)."""

    prefix_bucket: str
    step_bucket: str
    prefix_tokens: int
    step_tokens: int
    c_t: float
    c_d1: float | None = None
    c_d2: float | None = None
    c_d4: float | None = None
    c_v1: float | None = None
    c_v2: float | None = None
    c_v4: float | None = None
    c_pipe1: float | None = None
    c_pipe2: float | None = None
    c_pipe4: float | None = None


@dataclass(frozen=True)
class BreakEvenRow:
    prefix_bucket: str
    step_bucket: str
    n: int
    c_t: float
    c_d1: float | None
    c_d2: float | None
    c_d4: float | None
    c_v1: float | None
    c_v2: float | None
    c_v4: float | None
    r1_star: float | None
    r2_star: float | None
    r4_star: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RescueRates:
    """Empirical rescue rates conditioned on greedy reject."""

    r1: float | None
    r2: float | None
    r4: float | None
    n_greedy_reject: int
    source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CostRescueDecision:
    k: int
    r_k: float | None
    r_k_star: float | None
    margin: float | None
    decision: Decision
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def break_even(c_dk: float, c_vk: float, c_t: float) -> float:
    """r_K^* = (C_DK + C_VK) / C_T."""
    if c_t <= 0:
        raise ValueError(f"C_T must be positive, got {c_t}")
    return (c_dk + c_vk) / c_t


def expected_branch_cost(c_dk: float, c_vk: float, c_t: float, r_k: float) -> float:
    """C_branch = C_DK + C_VK + (1 - r_K) C_T."""
    return c_dk + c_vk + (1.0 - r_k) * c_t


def speedup_vs_handoff(c_dk: float, c_vk: float, c_t: float, r_k: float) -> float:
    """C_handoff / E[C_branch]. >1 means Branch is faster in expectation."""
    cb = expected_branch_cost(c_dk, c_vk, c_t, r_k)
    if cb <= 0:
        return float("inf")
    return c_t / cb


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def aggregate_break_even(samples: list[CostSample]) -> list[BreakEvenRow]:
    """Group by (prefix_bucket, step_bucket) and compute mean costs + r_K^*."""
    groups: dict[tuple[str, str], list[CostSample]] = {}
    for s in samples:
        groups.setdefault((s.prefix_bucket, s.step_bucket), []).append(s)

    rows: list[BreakEvenRow] = []
    for (pb, sb), grp in sorted(groups.items()):
        c_t = _mean([g.c_t for g in grp if g.c_t is not None])
        c_d1 = _mean([g.c_d1 for g in grp if g.c_d1 is not None])
        c_d2 = _mean([g.c_d2 for g in grp if g.c_d2 is not None])
        c_d4 = _mean([g.c_d4 for g in grp if g.c_d4 is not None])
        c_v1 = _mean([g.c_v1 for g in grp if g.c_v1 is not None])
        c_v2 = _mean([g.c_v2 for g in grp if g.c_v2 is not None])
        c_v4 = _mean([g.c_v4 for g in grp if g.c_v4 is not None])

        def star(cd: float | None, cv: float | None) -> float | None:
            if c_t is None or cd is None or cv is None:
                return None
            return break_even(cd, cv, c_t)

        rows.append(
            BreakEvenRow(
                prefix_bucket=pb,
                step_bucket=sb,
                n=len(grp),
                c_t=c_t or 0.0,
                c_d1=c_d1,
                c_d2=c_d2,
                c_d4=c_d4,
                c_v1=c_v1,
                c_v2=c_v2,
                c_v4=c_v4,
                r1_star=star(c_d1, c_v1),
                r2_star=star(c_d2, c_v2),
                r4_star=star(c_d4, c_v4),
            )
        )
    return rows


def overall_break_even(samples: list[CostSample]) -> BreakEvenRow:
    """Single overall row (all buckets pooled)."""
    if not samples:
        return BreakEvenRow(
            prefix_bucket="all",
            step_bucket="all",
            n=0,
            c_t=0.0,
            c_d1=None,
            c_d2=None,
            c_d4=None,
            c_v1=None,
            c_v2=None,
            c_v4=None,
            r1_star=None,
            r2_star=None,
            r4_star=None,
        )
    pooled = [
        CostSample(
            prefix_bucket="all",
            step_bucket="all",
            prefix_tokens=s.prefix_tokens,
            step_tokens=s.step_tokens,
            c_t=s.c_t,
            c_d1=s.c_d1,
            c_d2=s.c_d2,
            c_d4=s.c_d4,
            c_v1=s.c_v1,
            c_v2=s.c_v2,
            c_v4=s.c_v4,
        )
        for s in samples
    ]
    return aggregate_break_even(pooled)[0]


def decide_policy(
    *,
    r_k: float | None,
    r_k_star: float | None,
    k: int = 4,
    always_margin: float = 0.10,
    never_margin: float = 0.05,
) -> CostRescueDecision:
    """Map (r_K, r_K^*) to a first-version policy recommendation.

    Special case: if r_K^* >= 1.0, Branch@K is strictly dominated by Handoff
    even at perfect rescue — no router can help for that K.
    """
    if r_k_star is not None and r_k_star >= 1.0:
        return CostRescueDecision(
            k=k,
            r_k=r_k,
            r_k_star=r_k_star,
            margin=(None if r_k is None else r_k - r_k_star),
            decision="dominated",
            rationale=(
                f"r_{k}^*={r_k_star:.1%} ≥ 100% → Branch@{k} strictly dominated by "
                f"Handoff even at r=100% (current impl/length/hardware)"
            ),
        )

    if r_k is None or r_k_star is None:
        return CostRescueDecision(
            k=k,
            r_k=r_k,
            r_k_star=r_k_star,
            margin=None,
            decision="train_predictor",
            rationale="missing r_K or r_K^* — run Experiment A+B before deciding",
        )

    margin = r_k - r_k_star
    if margin >= always_margin:
        return CostRescueDecision(
            k=k,
            r_k=r_k,
            r_k_star=r_k_star,
            margin=margin,
            decision="always_branch",
            rationale=(
                f"r_{k}={r_k:.1%} ≫ r_{k}^*={r_k_star:.1%} "
                f"(margin={margin:.1%}) → Fixed Branch@{k}, no router"
            ),
        )
    if margin <= -never_margin:
        return CostRescueDecision(
            k=k,
            r_k=r_k,
            r_k_star=r_k_star,
            margin=margin,
            decision="never_branch",
            rationale=(
                f"r_{k}={r_k:.1%} ≪ r_{k}^*={r_k_star:.1%} "
                f"(margin={margin:.1%}) → skip fixed Branch@{k}; try smaller K or SpecReason"
            ),
        )
    return CostRescueDecision(
        k=k,
        r_k=r_k,
        r_k_star=r_k_star,
        margin=margin,
        decision="train_predictor",
        rationale=(
            f"r_{k}={r_k:.1%} ≈ r_{k}^*={r_k_star:.1%} "
            f"(margin={margin:.1%}) → near break-even"
        ),
    )


def decide_from_bucket_stars(
    bucket_stars: dict[str, float | None],
    *,
    k: int,
    r_k: float | None = None,
    dominated_thresh: float = 1.0,
    always_thresh: float = 0.35,
    never_thresh: float = 0.70,
) -> dict[str, Any]:
    """Recommend using per-bucket r_K^* heterogeneity (Cost Gate A only).

    Router is valuable only when some buckets are below break-even-ish and
    others are not — i.e. state-dependent action value.
    """
    vals = [(b, v) for b, v in bucket_stars.items() if v is not None]
    if not vals:
        return {
            "k": k,
            "decision": "insufficient_data",
            "rationale": "no bucket r_K^* available",
        }

    stars = [v for _, v in vals]
    n_dom = sum(1 for v in stars if v >= dominated_thresh)
    n_cheap = sum(1 for v in stars if v <= always_thresh)
    n_expensive = sum(1 for v in stars if v >= never_thresh)
    spread = max(stars) - min(stars)

    if n_dom == len(stars):
        decision = "dominated"
        rationale = (
            f"all buckets have r_{k}^*≥100% → Branch@{k} dominated under current impl"
        )
    elif n_cheap == len(stars) and r_k is not None and r_k >= always_thresh:
        decision = "always_branch"
        rationale = f"all buckets cheap (r_{k}^*≤{always_thresh:.0%}) and r_{k} high"
    elif n_expensive == len(stars):
        decision = "never_branch"
        rationale = (
            f"all buckets expensive (r_{k}^*≥{never_thresh:.0%}) → SpecReason/Handoff"
        )
    elif spread >= 0.25 and n_cheap >= 1 and n_expensive >= 1:
        decision = "train_predictor"
        rationale = (
            f"state heterogeneity: r_{k}^* spread={spread:.1%} "
            f"(cheap={n_cheap}, expensive={n_expensive}) → router only if "
            f"P(rescue|s,K) crosses r_K^*(s)"
        )
    else:
        decision = "needs_rescue_gate"
        rationale = (
            f"mixed/mid r_{k}^* (min={min(stars):.1%}, max={max(stars):.1%}); "
            f"finish Experiment B before locking"
        )

    return {
        "k": k,
        "decision": decision,
        "rationale": rationale,
        "bucket_stars": {b: v for b, v in vals},
        "spread": spread,
        "n_dominated": n_dom,
        "n_cheap": n_cheap,
        "n_expensive": n_expensive,
    }


# Provisional V3.3 rates (4B draft + GPT oracle) — NOT final 1.5B+32B numbers.
# Used only as a prior until Experiment B finishes on the final stack.
V33_PROVISIONAL_RESCUE = RescueRates(
    r1=0.265,
    r2=0.365,
    r4=0.446,
    n_greedy_reject=166,
    source="v3.3_gpt_step_oracle_provisional_4B_draft",
)


def length_bucket(n_tokens: int, *, short: int = 256, medium: int = 768) -> str:
    if n_tokens <= short:
        return "short"
    if n_tokens <= medium:
        return "medium"
    return "long"


def prefix_length_bucket(n_tokens: int) -> str:
    if n_tokens < 512:
        return "p0_512"
    if n_tokens < 1024:
        return "p512_1024"
    if n_tokens < 2048:
        return "p1024_2048"
    return "p2048_plus"


def step_length_bucket(n_tokens: int) -> str:
    if n_tokens <= 64:
        return "s1_64"
    if n_tokens <= 128:
        return "s65_128"
    if n_tokens <= 192:
        return "s129_192"
    return "s193_plus"


def percentile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    idx = (len(ys) - 1) * (p / 100.0)
    lo = int(idx)
    hi = min(lo + 1, len(ys) - 1)
    frac = idx - lo
    return ys[lo] * (1 - frac) + ys[hi] * frac


def summarize_latencies(xs: list[float]) -> dict[str, float | None]:
    if not xs:
        return {"n": 0, "mean": None, "median": None, "p90": None, "p95": None}
    return {
        "n": float(len(xs)),
        "mean": sum(xs) / len(xs),
        "median": percentile(xs, 50),
        "p90": percentile(xs, 90),
        "p95": percentile(xs, 95),
    }

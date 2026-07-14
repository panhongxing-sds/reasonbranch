"""V4.0 conformal abstention gate over draft self-confidence.

Decision per reasoning step: ACCEPT the draft step (skip the target) if a fused
draft-confidence score s >= tau, else ABSTAIN -> handoff to the 32B target.

We give a distribution-free finite-sample guarantee on the accepted set:

    P( oracle_acceptable = 0 | accepted )  <=  epsilon        (w.p. >= 1 - delta)

i.e. precision among accepted steps >= 1 - epsilon. This uses Learn-then-Test
(Angelopoulos et al.) with a Hoeffding upper confidence bound on the false-
acceptance rate and fixed-sequence testing (thresholds tested high->low) so no
Bonferroni penalty is paid and FWER is controlled at delta.

The fusion model and the calibration are fit on problem-disjoint splits so the
guarantee is not inflated by prefix/problem leakage.

Pure numpy; no sklearn.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reasoning_branch_dataset.action_study.v40_derisk_analyze import (
    _as_float,
    _load_jsonl,
    average_precision,
    roc_auc,
)

# Draft-only feature set for the fused scorer (self_eval_logit is the strongest;
# the rest add marginal robustness). Verifier score is intentionally excluded so
# the gate is zero-target-cost.
DRAFT_FEATURES = [
    "self_eval_logit",
    "mean_margin",
    "max_entropy",
    "mean_entropy",
    "mean_logprob",
    "min_logprob",
    "repetition_rate",
]


def _feat(row: dict[str, Any], name: str) -> float:
    return _as_float((row.get("self_signals") or {}).get(name))


@dataclass
class FusionModel:
    """Standardized logistic fusion, or single-signal passthrough."""

    mode: str  # "logistic" | "single"
    features: list[str]
    weights: list[float] = field(default_factory=list)
    bias: float = 0.0
    mu: list[float] = field(default_factory=list)
    sd: list[float] = field(default_factory=list)
    single_signal: str = "self_eval_logit"

    def score(self, row: dict[str, Any]) -> float:
        if self.mode == "single":
            return _feat(row, self.single_signal)
        x = [_feat(row, f) for f in self.features]
        z = self.bias
        for j, v in enumerate(x):
            vv = v if math.isfinite(v) else self.mu[j]
            z += self.weights[j] * ((vv - self.mu[j]) / (self.sd[j] or 1.0))
        return 1.0 / (1.0 + math.exp(-z))

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "features": self.features,
            "weights": self.weights,
            "bias": self.bias,
            "mu": self.mu,
            "sd": self.sd,
            "single_signal": self.single_signal,
        }


def fit_fusion(rows: list[dict[str, Any]], *, mode: str = "logistic", features: list[str] | None = None,
               single_signal: str = "self_eval_logit", epochs: int = 600, lr: float = 0.15,
               l2: float = 1e-3) -> FusionModel:
    features = features or DRAFT_FEATURES
    if mode == "single":
        return FusionModel(mode="single", features=[single_signal], single_signal=single_signal)

    import numpy as np

    X = np.array([[_feat(r, f) for f in features] for r in rows], dtype=float)
    y = np.array([1.0 if r["oracle_label"] else 0.0 for r in rows], dtype=float)
    for j in range(X.shape[1]):
        col = X[:, j]
        med = float(np.median(col[np.isfinite(col)])) if np.isfinite(col).any() else 0.0
        col[~np.isfinite(col)] = med
        X[:, j] = col
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    n, d = Xs.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(epochs):
        p = 1.0 / (1.0 + np.exp(-(Xs @ w + b)))
        gw = Xs.T @ (p - y) / n + l2 * w
        gb = float((p - y).mean())
        w -= lr * gw
        b -= lr * gb
    return FusionModel(
        mode="logistic", features=features, weights=w.tolist(), bias=float(b),
        mu=mu.tolist(), sd=sd.tolist(), single_signal=single_signal,
    )


def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(Bin(n, p) <= k) in log-space (stable for large n)."""
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if k >= n else 0.0
    lp = math.log(p)
    lq = math.log1p(-p)
    s = 0.0
    for i in range(0, k + 1):
        logc = math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
        s += math.exp(logc + i * lp + (n - i) * lq)
    return min(1.0, s)


def _cp_upper(k: int, n: int, delta: float) -> float:
    """Clopper-Pearson exact upper (1-delta) bound on a Bernoulli rate given k/n.

    Solves P(Bin(n, U) <= k) = delta by bisection (CDF is decreasing in U).
    Much tighter than Hoeffding for small k (e.g. k=0 -> U = 1 - delta**(1/n))."""
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _binom_cdf(k, n, mid) > delta:
            lo = mid  # true rate could be higher
        else:
            hi = mid
    return hi


@dataclass
class CalibrationResult:
    tau: float | None
    epsilon: float
    delta: float
    n_cal: int
    accepted: int
    coverage: float
    empirical_false_rate: float
    ucb_false_rate: float
    grid: list[dict[str, Any]]


def calibrate_threshold(
    scores: list[float], labels: list[bool], *, epsilon: float, delta: float,
    n_grid: int = 100, min_accept: int = 5
) -> CalibrationResult:
    """Bonferroni Learn-then-Test: over a grid of thresholds, keep those whose
    Hoeffding UCB on P(y=0 | s>=tau) is <= epsilon at per-test level delta/G,
    then return the LOWEST such tau (max coverage). Bonferroni over the grid
    controls FWER at delta with no monotonicity assumption."""
    pairs = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    n = len(pairs)
    uniq = sorted({s for s, _ in pairs}, reverse=True)
    if len(uniq) > n_grid:
        step = len(uniq) / n_grid
        uniq = [uniq[int(i * step)] for i in range(n_grid)]
    g = max(1, len(uniq))
    delta_per = delta / g

    grid: list[dict[str, Any]] = []
    passing: list[dict[str, Any]] = []
    for tau in uniq:
        acc = [(s, y) for s, y in pairs if s >= tau]
        n_acc = len(acc)
        k_false = sum(1 for _, y in acc if not y)
        r_hat = k_false / n_acc if n_acc else 0.0
        ucb = _cp_upper(k_false, n_acc, delta_per)
        passed = (n_acc >= min_accept) and (ucb <= epsilon)
        entry = {
            "tau": tau, "n_accept": n_acc, "false": k_false,
            "empirical_false_rate": r_hat, "ucb_false_rate": ucb,
            "coverage": n_acc / n if n else 0.0, "passed": passed,
        }
        grid.append(entry)
        if passed:
            passing.append(entry)

    if not passing:
        return CalibrationResult(None, epsilon, delta, n, 0, 0.0, float("nan"), float("nan"), grid)
    # max coverage among guaranteed-safe thresholds
    best = max(passing, key=lambda e: e["coverage"])
    return CalibrationResult(
        tau=best["tau"], epsilon=epsilon, delta=delta, n_cal=n,
        accepted=best["n_accept"], coverage=best["coverage"],
        empirical_false_rate=best["empirical_false_rate"],
        ucb_false_rate=best["ucb_false_rate"], grid=grid,
    )


@dataclass
class ConformalGate:
    fusion: FusionModel
    tau: float
    epsilon: float
    delta: float

    def score(self, row: dict[str, Any]) -> float:
        return self.fusion.score(row)

    def accept(self, row: dict[str, Any]) -> bool:
        return self.score(row) >= self.tau

    def to_dict(self) -> dict[str, Any]:
        return {"fusion": self.fusion.to_dict(), "tau": self.tau,
                "epsilon": self.epsilon, "delta": self.delta}


# ---- evaluation on a grouped train/cal/test split -------------------------

def _group_split(rows: list[dict[str, Any]], seed: int = 0) -> tuple[list, list, list]:
    import random

    groups = sorted({r["problem_id"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(groups)
    n = len(groups)
    n_tr = max(1, int(0.4 * n))
    n_ca = max(1, int(0.3 * n))
    tr = set(groups[:n_tr])
    ca = set(groups[n_tr:n_tr + n_ca])
    te = set(groups[n_tr + n_ca:])
    split = lambda S: [r for r in rows if r["problem_id"] in S]
    return split(tr), split(ca), split(te)


def evaluate(rows: list[dict[str, Any]], *, epsilon: float, delta: float, mode: str, seed: int = 0) -> dict[str, Any]:
    tr, ca, te = _group_split(rows, seed=seed)
    fusion = fit_fusion(tr if mode == "logistic" else rows, mode=mode)

    cal_scores = [fusion.score(r) for r in ca]
    cal_labels = [bool(r["oracle_label"]) for r in ca]
    cal = calibrate_threshold(cal_scores, cal_labels, epsilon=epsilon, delta=delta)

    result: dict[str, Any] = {
        "mode": mode, "epsilon": epsilon, "delta": delta, "seed": seed,
        "n_train": len(tr), "n_cal": len(ca), "n_test": len(te),
        "tau": cal.tau, "cal_coverage": cal.coverage,
        "cal_ucb_false_rate": cal.ucb_false_rate,
    }
    if cal.tau is None:
        result["status"] = "no_threshold_meets_guarantee"
        return result

    te_scores = [fusion.score(r) for r in te]
    te_labels = [bool(r["oracle_label"]) for r in te]
    acc = [(s, y) for s, y in zip(te_scores, te_labels) if s >= cal.tau]
    n_acc = len(acc)
    tp = sum(1 for _, y in acc if y)
    test_precision = tp / n_acc if n_acc else float("nan")
    result.update({
        "status": "ok",
        "test_auc": roc_auc(te_scores, te_labels),
        "test_ap": average_precision(te_scores, te_labels),
        "test_coverage": n_acc / len(te) if te else 0.0,
        "test_precision_among_accepted": test_precision,
        "test_false_rate_among_accepted": (1 - test_precision) if n_acc else float("nan"),
        "target_precision": 1 - epsilon,
        "guarantee_held": (n_acc == 0) or (test_precision >= 1 - epsilon),
    })
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="V4.0 conformal abstention gate calibration")
    p.add_argument("--candidates", default="/root/autodl-tmp/reasonbranch/outputs/action_study_v40_derisk/candidates.jsonl")
    p.add_argument("--out-dir", default="/root/autodl-tmp/reasonbranch/outputs/action_study_v40_derisk")
    p.add_argument("--epsilon", type=float, default=0.10)
    p.add_argument("--delta", type=float, default=0.10)
    p.add_argument("--mode", default="logistic", choices=["logistic", "single"])
    p.add_argument("--seeds", type=int, default=5, help="repeat over grouped splits and average")
    args = p.parse_args()

    rows = _load_jsonl(Path(args.candidates))
    if not rows:
        raise SystemExit(f"No candidates at {args.candidates}")

    runs = [evaluate(rows, epsilon=args.epsilon, delta=args.delta, mode=args.mode, seed=s)
            for s in range(args.seeds)]
    ok = [r for r in runs if r.get("status") == "ok"]
    summary = {
        "epsilon": args.epsilon, "delta": args.delta, "mode": args.mode,
        "n_seeds": args.seeds, "n_with_threshold": len(ok),
        "runs": runs,
    }
    if ok:
        import statistics as st
        summary["mean_test_coverage"] = st.fmean([r["test_coverage"] for r in ok])
        summary["mean_test_precision"] = st.fmean(
            [r["test_precision_among_accepted"] for r in ok if math.isfinite(r["test_precision_among_accepted"])] or [float("nan")]
        )
        summary["guarantee_held_rate"] = st.fmean([1.0 if r["guarantee_held"] else 0.0 for r in ok])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "conformal_gate.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

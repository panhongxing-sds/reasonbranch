"""Analyze LVD: M1–M4 predictors of next-cycle acceptance; kill-gate vs M3.

M1: A, gamma
M2: M1 + H_D, maxp_D, m_D
M3: M2 + H_T, m_T, KL
M4: M3 + flip_depth, flip_count, path_speed, late_resolve_frac

Kill gate: M4 must significantly beat M3 on held-out next-A prediction (RMSE / R^2).
Also matched-margin analysis: early vs late resolve at similar m_T.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path
from typing import Any

import numpy as np


FEATURE_SETS = {
    "M1": ["A", "gamma"],
    "M2": ["A", "gamma", "H_D", "maxp_D", "m_D"],
    "M3": ["A", "gamma", "H_D", "maxp_D", "m_D", "H_T", "m_T", "KL"],
    "M4": [
        "A", "gamma", "H_D", "maxp_D", "m_D", "H_T", "m_T", "KL",
        "flip_depth", "flip_count", "path_speed", "late_resolve_frac",
    ],
}


def _load(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return [r for r in rows if r.get("A_next") is not None]


def _xy(rows: list[dict], feats: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[float(r[f]) for f in feats] for r in rows], dtype=np.float64)
    y = np.array([float(r["A_next"]) for r in rows], dtype=np.float64)
    # impute nan
    for j in range(X.shape[1]):
        col = X[:, j]
        med = np.nanmedian(col)
        col[~np.isfinite(col)] = med if np.isfinite(med) else 0.0
        X[:, j] = col
    return X, y


def _fit_ridge(X: np.ndarray, y: np.ndarray, l2: float = 1e-2) -> tuple[np.ndarray, float]:
    n, d = X.shape
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    # [Xs | 1]
    A = np.concatenate([Xs, np.ones((n, 1))], axis=1)
    reg = l2 * np.eye(d + 1)
    reg[-1, -1] = 0.0
    w = np.linalg.solve(A.T @ A + reg, A.T @ y)
    return w, 0.0  # bias folded into w[-1]; store mu/sd via closure in predict


def _predict(X: np.ndarray, w: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    Xs = (X - mu) / sd
    A = np.concatenate([Xs, np.ones((Xs.shape[0], 1))], axis=1)
    return A @ w


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - y
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def oof_eval(rows: list[dict], feats: list[str], *, n_folds: int = 5) -> dict[str, float]:
    X, y = _xy(rows, feats)
    n = len(y)
    # group by prompt_idx
    groups = np.array([r["prompt_idx"] for r in rows])
    uniq = sorted(set(groups.tolist()))
    fold_of = {g: i % n_folds for i, g in enumerate(uniq)}
    pred = np.full(n, np.nan)
    for fold in range(n_folds):
        tr = np.array([fold_of[g] != fold for g in groups])
        te = ~tr
        if tr.sum() < 5 or te.sum() < 1:
            continue
        Xtr, ytr = X[tr], y[tr]
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0)
        sd[sd == 0] = 1.0
        Xs = (Xtr - mu) / sd
        A = np.concatenate([Xs, np.ones((Xs.shape[0], 1))], axis=1)
        d = A.shape[1]
        reg = 1e-2 * np.eye(d)
        reg[-1, -1] = 0.0
        w = np.linalg.solve(A.T @ A + reg, A.T @ ytr)
        pred[te] = _predict(X[te], w, mu, sd)
    valid = np.isfinite(pred)
    return _metrics(y[valid], pred[valid])


def matched_margin_analysis(rows: list[dict], n_layers: int = 64) -> dict[str, Any]:
    """Among cycles with rejection, compare early vs late resolve at similar m_T."""
    rej = [r for r in rows if r.get("n_rejected", 0) > 0]
    if len(rej) < 20:
        return {"n": len(rej), "status": "too_few"}
    # early: flip_depth < 0.4 L; late: flip_depth >= 0.75 L
    early = [r for r in rej if r["flip_depth"] < 0.4 * n_layers]
    late = [r for r in rej if r["flip_depth"] >= 0.75 * n_layers]
    # match on m_T bins
    def mean_Anext(xs):
        return st.fmean([r["A_next"] for r in xs]) if xs else float("nan")

    # quantile-bin matching: for each early, find late with closest m_T
    pairs = []
    for e in early:
        if not late:
            break
        best = min(late, key=lambda x: abs(x["m_T"] - e["m_T"]))
        if abs(best["m_T"] - e["m_T"]) < 1.0:  # similar margin
            pairs.append((e, best))
    if not pairs:
        return {
            "n_early": len(early), "n_late": len(late),
            "early_A_next": mean_Anext(early), "late_A_next": mean_Anext(late),
            "n_matched_pairs": 0,
        }
    early_A = st.fmean([p[0]["A_next"] for p in pairs])
    late_A = st.fmean([p[1]["A_next"] for p in pairs])
    return {
        "n_early": len(early), "n_late": len(late),
        "n_matched_pairs": len(pairs),
        "matched_early_A_next": early_A,
        "matched_late_A_next": late_A,
        "matched_delta": early_A - late_A,
        "early_mean_m_T": st.fmean([p[0]["m_T"] for p in pairs]),
        "late_mean_m_T": st.fmean([p[1]["m_T"] for p in pairs]),
    }


def analyze(rows: list[dict], n_layers: int = 64) -> dict[str, Any]:
    results = {}
    for name, feats in FEATURE_SETS.items():
        results[name] = oof_eval(rows, feats)
    m3, m4 = results["M3"], results["M4"]
    # relative RMSE improvement
    rmse_gain = (m3["rmse"] - m4["rmse"]) / m3["rmse"] if m3["rmse"] > 0 else 0.0
    r2_gain = m4["r2"] - m3["r2"]
    # kill gate: need meaningful lift — rmse_gain >= 5% OR r2_gain >= 0.03
    passed = (rmse_gain >= 0.05) or (r2_gain >= 0.03)
    matched = matched_margin_analysis(rows, n_layers=n_layers)
    return {
        "n": len(rows),
        "models": results,
        "rmse_gain_M4_vs_M3": rmse_gain,
        "r2_gain_M4_vs_M3": r2_gain,
        "matched_margin": matched,
        "decision": "PASS" if passed else "FAIL",
        "kill_gate": "M4 RMSE gain>=5% OR R2 gain>=0.03 vs M3 (GroupKFold OOF)",
    }


def render(summary: dict) -> str:
    L = ["# Layerwise Verification Trajectory — Kill-Gate Report\n"]
    L.append(f"- Decision: **{summary['decision']}**")
    L.append(f"- N cycles: {summary['n']}")
    L.append(f"- M4 vs M3 RMSE gain: **{summary['rmse_gain_M4_vs_M3']*100:.1f}%**")
    L.append(f"- M4 vs M3 R² gain: **{summary['r2_gain_M4_vs_M3']:+.3f}**\n")
    L.append("## Predictor comparison (OOF next acceptance length)\n")
    L.append("| model | RMSE | MAE | R² |")
    L.append("|---|--:|--:|--:|")
    for name in ["M1", "M2", "M3", "M4"]:
        m = summary["models"][name]
        L.append(f"| {name} | {m['rmse']:.3f} | {m['mae']:.3f} | {m['r2']:.3f} |")
    mm = summary.get("matched_margin") or {}
    L.append("\n## Matched-margin early vs late resolve\n")
    L.append("```")
    L.append(json.dumps(mm, indent=2))
    L.append("```\n")
    if summary["decision"] == "PASS":
        L.append("Layerwise trajectory adds information beyond final target logits. Continue.")
    else:
        L.append(
            "Kill gate failed: resolution depth / flip count do not significantly improve "
            "next-cycle acceptance prediction over final-logit features (M3). Stop."
        )
    return "\n".join(L) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cycles", default="/root/autodl-tmp/reasonbranch/outputs/lvd_cycles.jsonl")
    p.add_argument("--out-dir", default="/root/autodl-tmp/reasonbranch/outputs")
    p.add_argument("--n-layers", type=int, default=64)
    args = p.parse_args()
    rows = _load(Path(args.cycles))
    if len(rows) < 30:
        raise SystemExit(f"Need more cycles, got {len(rows)}")
    summary = analyze(rows, n_layers=args.n_layers)
    out = Path(args.out_dir)
    (out / "lvd_report.json").write_text(json.dumps(summary, indent=2))
    report = render(summary)
    (out / "lvd_report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()

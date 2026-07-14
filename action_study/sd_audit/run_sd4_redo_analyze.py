"""SD4 (Layerwise Verification Trajectory) — corrected re-analysis.

Fixes vs first LVD audit:
  1. Faithful logit-lens deltas (model's own final norm), so the last-layer
     margin agrees with the true target margin (old audit used raw LayerNorm,
     sign-inconsistent -> noisy features -> spurious "no signal").
  2. Decision-resolution depth for BOTH accept and reject (full-vocab logit-lens
     argmax == drafted token), not only the y-vs-d delta which is 0 for accepts.

Tests (offline, from tokens.jsonl):
  T1. Asymmetry: reject decisions resolve early, accept decisions resolve late?
  T2. Hard negatives: among rejected tokens, do draft-CONFIDENT rejections
      (the ones a draft-confidence gate would wrongly accept) resolve LATER than
      easy rejections? -> representational account of the verification gap.
  T3. Information content: does trajectory depth improve OOF prediction of
      `accepted` beyond draft-time signals + final target margin?
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC of score for label==1 vs label==0 (rank-based, tie-safe)."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks for ties
    allv = np.concatenate([pos, neg])
    sidx = np.argsort(allv)
    sv = allv[sidx]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            avg = (ranks[sidx[i]] + ranks[sidx[j]]) / 2.0
            for k in range(i, j + 1):
                ranks[sidx[k]] = avg
        i = j + 1
    rpos = ranks[: len(pos)].sum()
    return float((rpos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def mannwhitney_p(a: np.ndarray, b: np.ndarray) -> float:
    """Normal-approx two-sided Mann-Whitney p (large samples)."""
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return float("nan")
    allv = np.concatenate([a, b])
    order = np.argsort(allv)
    ranks = np.empty(len(allv), dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            ranks[order[i : j + 1]] = avg
        i = j + 1
    r1 = ranks[:n1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    if sigma == 0:
        return float("nan")
    z = (u1 - mu) / sigma
    return float(math.erfc(abs(z) / math.sqrt(2)))


def _logistic_oof(X: np.ndarray, y: np.ndarray, folds: int = 5, l2: float = 1.0) -> float:
    """Simple L2 logistic regression, k-fold OOF AUC. Standardize features."""
    n = len(y)
    rng = np.random.RandomState(0)
    idx = rng.permutation(n)
    fold_id = np.zeros(n, dtype=int)
    for i, ii in enumerate(idx):
        fold_id[ii] = i % folds
    oof = np.zeros(n)
    for f in range(folds):
        tr = fold_id != f
        te = fold_id == f
        Xtr, Xte = X[tr], X[te]
        ytr = y[tr]
        mu = Xtr.mean(0)
        sd = Xtr.std(0) + 1e-8
        Xtr = (Xtr - mu) / sd
        Xte = (Xte - mu) / sd
        Xtr = np.hstack([Xtr, np.ones((len(Xtr), 1))])
        Xte = np.hstack([Xte, np.ones((len(Xte), 1))])
        w = np.zeros(Xtr.shape[1])
        for _ in range(300):
            p = 1.0 / (1.0 + np.exp(-Xtr @ w))
            g = Xtr.T @ (p - ytr) / len(ytr) + l2 * w / len(ytr)
            W = p * (1 - p)
            H = Xtr.T @ (Xtr * W[:, None]) / len(ytr) + l2 * np.eye(Xtr.shape[1]) / len(ytr)
            try:
                step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                step = g
            w -= step
            if np.max(np.abs(step)) < 1e-6:
                break
        oof[te] = 1.0 / (1.0 + np.exp(-Xte @ w))
    return auc(oof, y)


def _ridge_oof(X: np.ndarray, y: np.ndarray, folds: int = 5, l2: float = 1.0) -> dict[str, float]:
    n = len(y)
    rng = np.random.RandomState(0)
    fold_id = rng.permutation(n) % folds
    pred = np.zeros(n)
    for f in range(folds):
        tr, te = fold_id != f, fold_id == f
        Xtr, Xte, ytr = X[tr], X[te], y[tr]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        Xtr = np.hstack([(Xtr - mu) / sd, np.ones((tr.sum(), 1))])
        Xte = np.hstack([(Xte - mu) / sd, np.ones((te.sum(), 1))])
        A = Xtr.T @ Xtr + l2 * np.eye(Xtr.shape[1])
        w = np.linalg.solve(A, Xtr.T @ ytr)
        pred[te] = Xte @ w
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    ss_res = np.sum((pred - y) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
    return {"rmse": rmse, "r2": float(1 - ss_res / ss_tot)}


def _next_cycle_eval(samples: list[tuple]) -> dict[str, Any]:
    if len(samples) < 20:
        return {"n": len(samples), "note": "too few samples"}
    draft = np.array([s[0] for s in samples])
    final = np.array([s[1] for s in samples])
    traj = np.array([s[2] for s in samples])
    y = np.array([s[3] for s in samples])
    return {
        "n": len(samples),
        "M_draft": _ridge_oof(draft, y),
        "M_draft_final": _ridge_oof(np.hstack([draft, final]), y),
        "M_draft_final_traj": _ridge_oof(np.hstack([draft, final, traj]), y),
    }


def analyze(path: Path) -> dict[str, Any]:
    rows = load_rows(path)
    n_layers = max(r["layer_ids"][-1] for r in rows) + 1
    acc = [r for r in rows if r["accepted"]]
    rej = [r for r in rows if not r["accepted"]]

    def depth_frac(r) -> float:
        d = r["dec_depth"]
        return d / n_layers

    out: dict[str, Any] = {
        "n": len(rows), "n_acc": len(acc), "n_rej": len(rej),
        "accept_rate": len(acc) / len(rows), "n_layers": n_layers,
        "layer_ids": rows[0]["layer_ids"],
    }

    # T1 asymmetry
    out["T1_asymmetry"] = {
        "acc_dec_depth_frac_mean": float(np.mean([depth_frac(r) for r in acc])),
        "rej_dec_depth_frac_mean": float(np.mean([depth_frac(r) for r in rej])),
        "acc_dec_depth_frac_median": float(np.median([depth_frac(r) for r in acc])),
        "rej_dec_depth_frac_median": float(np.median([depth_frac(r) for r in rej])),
        "mw_p": mannwhitney_p(
            np.array([depth_frac(r) for r in acc]),
            np.array([depth_frac(r) for r in rej]),
        ),
    }

    # T2 hard vs easy negatives (draft-confident rejections)
    conf = np.array([r["draft_top1_prob"] for r in rej])
    thr = np.median(conf)
    hard = [r for r in rej if r["draft_top1_prob"] >= thr]  # gate would accept these
    easy = [r for r in rej if r["draft_top1_prob"] < thr]
    def feats(g, key):
        return np.array([g_i[key] for g_i in g], dtype=float)
    t2: dict[str, Any] = {"conf_median": float(thr), "n_hard": len(hard), "n_easy": len(easy)}
    for key in ["dec_depth", "flip_depth", "flip_count", "path_speed", "target_final_margin"]:
        h = feats(hard, key); e = feats(easy, key)
        t2[key] = {
            "hard_mean": float(np.mean(h)), "easy_mean": float(np.mean(e)),
            "mw_p": mannwhitney_p(h, e),
            # AUC: does feature separate hard(1) from easy(0)?
            "auc_hard_vs_easy": auc(
                np.concatenate([h, e]),
                np.concatenate([np.ones(len(h)), np.zeros(len(e))]),
            ),
        }
    out["T2_hard_vs_easy"] = t2

    # T3 (leak-free): does the CURRENT cycle's target trajectory predict the
    # NEXT cycle's acceptance length, beyond draft signals + current final margin?
    # Trajectory features for accepted tokens are forced to 0 by construction, so
    # predicting the CURRENT `accepted` from them would be circular; we therefore
    # predict the FUTURE (next cycle) which no feature can trivially encode.
    per_cycle: dict[tuple, dict[str, float]] = {}
    for r in rows:
        key = (r["prompt_idx"], r["cycle"])
        c = per_cycle.setdefault(key, {"acc_len": 0, "rej_dec_depth": [], "rej_path_speed": [],
                                       "rej_flip_count": [], "draft_top1": [], "draft_ent": [],
                                       "final_margin": [], "_counting": True})
        c["draft_top1"].append(r["draft_top1_prob"])
        c["draft_ent"].append(r["draft_entropy"])
        c["final_margin"].append(r["target_final_margin"])
        if not r["accepted"]:
            c["rej_dec_depth"].append(r["dec_depth"])
            c["rej_path_speed"].append(r["path_speed"])
            c["rej_flip_count"].append(r["flip_count"])
        if c["_counting"] and r["accepted"]:
            c["acc_len"] += 1
        elif c["_counting"]:
            c["_counting"] = False

    def m(x):
        return float(np.mean(x)) if x else 0.0
    samples = []  # (draft_feats, final_feat, traj_feats, next_acc_len)
    for (pi, cyc), c in per_cycle.items():
        nxt = per_cycle.get((pi, cyc + 1))
        if nxt is None:
            continue
        draft_f = [m(c["draft_top1"]), m(c["draft_ent"]), float(c["acc_len"])]
        final_f = [m(c["final_margin"])]
        traj_f = [m(c["rej_dec_depth"]), m(c["rej_path_speed"]), m(c["rej_flip_count"])]
        samples.append((draft_f, final_f, traj_f, float(nxt["acc_len"])))
    out["T3_next_cycle"] = _next_cycle_eval(samples)

    # Early-exit feasibility: decision agreement if truncated at each sampled layer
    layer_ids = rows[0]["layer_ids"]
    ee = {}
    for li_idx, li in enumerate(layer_ids):
        agree = np.mean([1.0 if (r["dec_match"][li_idx] == r["accepted"]) else 0.0 for r in rows])
        ee[str(li)] = float(agree)
    out["early_exit_agreement_by_layer"] = ee
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", default="/root/autodl-tmp/reasonbranch/outputs/vsignal/tokens.jsonl")
    p.add_argument("--out", default="/root/autodl-tmp/reasonbranch/outputs/vsignal/sd4_redo.json")
    args = p.parse_args()
    res = analyze(Path(args.tokens))
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

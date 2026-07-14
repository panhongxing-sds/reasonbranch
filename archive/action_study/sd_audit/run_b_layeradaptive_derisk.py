"""Route B de-risk: Layer-Adaptive Early-Reject Verification (compute-bound regime).

Question: in a compute-bound (large-batch serving) regime, verification cost ~
sum over positions of layers computed. T1 says rejects resolve early. Can we, at a
SHALLOW probe layer, detect the FIRST reject in a draft block and prune the
(discarded) tail from the deep layers -- saving target FLOPs at ~zero accuracy loss?

The prior audit used a single-layer raw logit-lens margin threshold (weak). Here we
give the idea its best shot:
  1) ORACLE ceiling: a prophet that knows the true first-reject index r; it keeps
     positions 0..r at full depth (accepts + the correction position need the full
     target dist) and drops the tail entirely. This upper-bounds ANY early-reject
     method's compute saving.
  2) LEARNED shallow probe: numpy logistic regression on shallow target features
     (multi-layer logit-lens margin + faithful-lens delta + dec_match) PLUS the
     free draft self-signals. Trained on half the cycles, evaluated on the other
     half by driving the pruning simulation with the classifier score.

Kill gate: learned probe compute_saving at ~zero accept-loss must exceed ~10%.

Correctness note: pruning positions strictly after the committed first reject never
changes emitted tokens vs standard greedy SD. A FALSE early stop (firing before the
true reject) only shortens acceptance -> efficiency loss, not correctness. So the
right axis is compute_saving at bounded accept-loss.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

DRAFT_KEYS = ["draft_logprob_self", "draft_top1_prob", "draft_entropy", "draft_margin"]


def load(path: Path):
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    layer_ids = rows[0]["layer_ids"]
    n_layers = layer_ids[-1] + 1
    groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        groups[(r["prompt_idx"], r["cycle"])].append(r)
    cycles = []
    for key, toks in groups.items():
        toks.sort(key=lambda r: r["pos_in_cycle"])
        gamma = len(toks)
        r_idx = gamma
        for i, t in enumerate(toks):
            if not t["accepted"]:
                r_idx = i
                break
        cycles.append({"key": key, "gamma": gamma, "r": r_idx, "toks": toks})
    return cycles, layer_ids, n_layers


def feat_token(t, m):
    """Shallow features using the first m sampled layers + free draft signals."""
    f = []
    f += list(np.array(t["top1_minus_d"][:m], dtype=float))
    f += list(np.array(t["deltas"][:m], dtype=float))
    f += [1.0 if b else 0.0 for b in t["dec_match"][:m]]
    f += [float(t[k]) for k in DRAFT_KEYS]
    return f


def standardize_fit(X):
    mu = X.mean(0)
    sd = X.std(0) + 1e-6
    return mu, sd


def logreg_train(X, y, iters=400, lr=0.5, l2=1e-3):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    pos = max(y.sum(), 1)
    neg = max(n - y.sum(), 1)
    cw = np.where(y == 1, n / (2 * pos), n / (2 * neg))  # balance classes
    for _ in range(iters):
        z = X @ w + b
        p = 1 / (1 + np.exp(-z))
        g = (p - y) * cw
        gw = X.T @ g / n + l2 * w
        gb = g.mean()
        w -= lr * gw
        b -= lr * gb
    return w, b


def predict(X, w, b):
    return 1 / (1 + np.exp(-(X @ w + b)))


def auc(y, s):
    order = np.argsort(s)
    y = y[order]
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = np.arange(1, len(y) + 1)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def simulate(test_cycles, scores, layer_ids, n_layers, m, thr):
    """scores: dict token-id(index in cycle) -> P(reject). Fire at first pos >= thr."""
    Lc = layer_ids[m - 1] + 1
    base_compute = tot_compute = 0.0
    base_commit = tot_commit = 0
    acc_loss = 0
    n_fire = 0
    for c in test_cycles:
        gamma, r = c["gamma"], c["r"]
        base_commit += r + 1
        base_compute += gamma * n_layers
        sc = c["scores"]
        fired = np.where(sc >= thr)[0]
        p = int(fired[0]) if len(fired) else None
        if p is None:
            tot_commit += r + 1
            tot_compute += gamma * n_layers
            continue
        n_fire += 1
        # all gamma run Lc layers; positions 0..p run the remaining deep layers.
        tot_compute += gamma * Lc + (p + 1) * (n_layers - Lc)
        if r <= p:
            tot_commit += r + 1
        else:
            tot_commit += p + 1
            acc_loss += (r + 1) - (p + 1)
    return {
        "probe_layer": Lc - 1,
        "thr": round(float(thr), 4),
        "compute_saving": 1 - tot_compute / base_compute,
        "speedup": (tot_commit / tot_compute) / (base_commit / base_compute),
        "commit_ratio": tot_commit / base_commit,
        "fire_rate": n_fire / len(test_cycles),
        "acc_loss_per_cycle": acc_loss / len(test_cycles),
    }


def oracle_ceiling(cycles, n_layers):
    base = sum(c["gamma"] for c in cycles) * n_layers
    keep = sum((c["r"] + 1) for c in cycles) * n_layers  # 0..r full depth, tail dropped
    return 1 - keep / base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="/root/autodl-tmp/reasonbranch/outputs/vsignal/tokens.jsonl")
    ap.add_argument("--out", default="/root/autodl-tmp/reasonbranch/outputs/vsignal/b_layeradaptive.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cycles, layer_ids, n_layers = load(Path(args.tokens))
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(cycles))
    half = len(cycles) // 2
    train_c = [cycles[i] for i in idx[:half]]
    test_c = [cycles[i] for i in idx[half:]]

    orc = oracle_ceiling(cycles, n_layers)

    results = {}
    for m in range(1, len(layer_ids) + 1):
        # build train token matrix
        Xtr, ytr = [], []
        for c in train_c:
            for t in c["toks"]:
                Xtr.append(feat_token(t, m))
                ytr.append(0 if t["accepted"] else 1)
        Xtr = np.array(Xtr, float)
        ytr = np.array(ytr, float)
        mu, sd = standardize_fit(Xtr)
        Xtrn = (Xtr - mu) / sd
        w, b = logreg_train(Xtrn, ytr)

        # test token AUC
        Xte, yte = [], []
        for c in test_c:
            cs = []
            for t in c["toks"]:
                x = (np.array(feat_token(t, m), float) - mu) / sd
                s = float(predict(x[None, :], w, b)[0])
                cs.append(s)
                Xte.append(x)
                yte.append(0 if t["accepted"] else 1)
            c["scores"] = np.array(cs)
        yte = np.array(yte, float)
        Xte = np.array(Xte, float)
        ste = predict(Xte, w, b)
        a = auc(yte, ste)

        # sweep thresholds, keep pareto: best compute_saving under acc-loss budgets
        thrs = np.unique(np.quantile(ste, np.linspace(0.5, 0.999, 40)))
        sims = [simulate(test_c, None, layer_ids, n_layers, m, th) for th in thrs]
        def best_under(bud):
            cand = [s for s in sims if s["acc_loss_per_cycle"] <= bud]
            return max(cand, key=lambda s: s["compute_saving"]) if cand else None
        results[f"m{m}_L{layer_ids[m-1]}"] = {
            "auc_reject": round(a, 4),
            "best@accloss=0.00": best_under(1e-9),
            "best@accloss=0.02": best_under(0.02),
            "best@accloss=0.10": best_under(0.10),
        }

    out = {
        "n_cycles": len(cycles),
        "n_layers": n_layers,
        "accept_len_mean": float(np.mean([c["r"] for c in cycles])),
        "oracle_compute_saving_ceiling": round(orc, 4),
        "probes": results,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

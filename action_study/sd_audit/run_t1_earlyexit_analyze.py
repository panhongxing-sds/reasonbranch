"""T1 -> mechanism: early-reject-pruned verification, net-speedup simulation.

T1 finding: rejections resolve in shallow target layers, acceptances only at the
top. Exploit: run the target block forward layer-by-layer; at a shallow probe
layer, use the graded logit-lens margin (best_token - drafted_token) to detect a
confident reject at position p*, then PRUNE all positions > p* for the remaining
layers (they are discarded by standard SD anyway).

Correctness: pruning positions after the first *committed* reject never changes
emitted tokens vs standard greedy SD; a FALSE early reject only ends the block
early (shorter acceptance) -> costs efficiency, not correctness. So the right
metric is throughput = committed_tokens / layer-compute. We sweep (probe layer,
threshold) and report the best net speedup vs full-depth verification.

Idealized FLOP model: block cost ~ sum over kept positions of layers computed
(uniform per-layer cost; ignores attention's position scaling -> conservative-ish
upper bound). Real gains need variable-length batched execution.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def load_cycles(path: Path):
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    n_layers = max(r["layer_ids"][-1] for r in rows) + 1
    layer_ids = rows[0]["layer_ids"]
    groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        groups[(r["prompt_idx"], r["cycle"])].append(r)
    cycles = []
    for key, toks in groups.items():
        toks.sort(key=lambda r: r["pos_in_cycle"])
        gamma = len(toks)
        # first reject index r (accept_len); gamma if all accepted
        r_idx = gamma
        for i, t in enumerate(toks):
            if not t["accepted"]:
                r_idx = i
                break
        margins = np.array([t["top1_minus_d"] for t in toks])  # (gamma, n_sampled)
        cycles.append({"gamma": gamma, "r": r_idx, "margins": margins})
    return cycles, layer_ids, n_layers


def simulate(cycles, layer_ids, n_layers, probe_idx: int, theta: float):
    Lc = layer_ids[probe_idx] + 1  # layers computed at probe
    tot_commit = 0
    tot_compute = 0.0
    base_commit = 0
    base_compute = 0.0
    acc_loss = 0
    n_fire = 0
    for c in cycles:
        gamma, r, M = c["gamma"], c["r"], c["margins"]
        # baseline
        base_commit += r + 1  # r accepts + 1 correction/bonus
        base_compute += gamma * n_layers
        # find first fired position at probe layer
        fired = np.where(M[:, probe_idx] >= theta)[0]
        p = int(fired[0]) if len(fired) else None
        if p is None:
            tot_commit += r + 1
            tot_compute += gamma * n_layers
            continue
        n_fire += 1
        # compute: all gamma run Lc layers; positions 0..p run remaining
        tot_compute += gamma * Lc + (p + 1) * (n_layers - Lc)
        if r <= p:
            # true reject within computed region -> identical to baseline commit
            tot_commit += r + 1
        else:
            # false early end at p -> commit p+1 accepts, no correction
            tot_commit += (p + 1)
            acc_loss += (r + 1) - (p + 1)
    base_tp = base_commit / base_compute
    sch_tp = tot_commit / tot_compute
    return {
        "probe_layer": Lc - 1, "theta": theta,
        "speedup": sch_tp / base_tp,
        "compute_saving": 1 - tot_compute / base_compute,
        "commit_ratio": tot_commit / base_commit,
        "fire_rate": n_fire / len(cycles),
        "acc_loss_per_cycle": acc_loss / len(cycles),
    }


def analyze(path: Path) -> dict[str, Any]:
    cycles, layer_ids, n_layers = load_cycles(path)
    accept_lens = np.array([c["r"] for c in cycles])
    thetas = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0]
    results = []
    for pi in range(len(layer_ids)):
        for th in thetas:
            results.append(simulate(cycles, layer_ids, n_layers, pi, th))
    # best by speedup
    best = max(results, key=lambda r: r["speedup"])
    # best "safe" at several acc-loss budgets (loss = extra tokens re-drafted later)
    def best_within(budget):
        cand = [r for r in results if r["acc_loss_per_cycle"] <= budget]
        return max(cand, key=lambda r: r["compute_saving"]) if cand else None
    return {
        "n_cycles": len(cycles), "n_layers": n_layers, "layer_ids": layer_ids,
        "accept_len_mean": float(accept_lens.mean()),
        "accept_len_hist": {str(k): int((accept_lens == k).sum()) for k in range(int(accept_lens.max()) + 1)},
        "best_overall_by_speedup": best,
        "best_saving_at_accloss": {
            "0.0": best_within(1e-9),
            "0.01": best_within(0.01),
            "0.05": best_within(0.05),
            "0.10": best_within(0.10),
            "0.25": best_within(0.25),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", default="/root/autodl-tmp/reasonbranch/outputs/vsignal/tokens.jsonl")
    p.add_argument("--out", default="/root/autodl-tmp/reasonbranch/outputs/vsignal/t1_earlyexit.json")
    args = p.parse_args()
    res = analyze(Path(args.tokens))
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

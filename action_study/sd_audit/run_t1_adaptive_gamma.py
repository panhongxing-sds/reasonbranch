"""T1/confidence -> mechanism: entropy-adaptive draft length (dynamic gamma).

Finding: the draft's own entropy at position j predicts whether position j+1 will
be REJECTED (AUC ~0.60) — i.e., the draft can sense when it is about to leave the
target's acceptance region. Use it to STOP drafting early, so the target verifies
fewer doomed tokens (saves the expensive 32B verify positions).

Decisive fairness test: an adaptive policy must beat the BEST FIXED gamma, else
one would just tune gamma. Metric = target-verify positions per committed token
(the real cost driver; lower is better). We also report tokens/cycle.

Simulated from collected cycles (each has gamma draft tokens with draft_entropy
and accept flags). Assumes: target verifies L drafted positions in one pass
(cost ~ L); a block commits min(r,L) accepts (+1 correction if reject r < L).
Stopping earlier than the true accept boundary re-drafts the remainder later,
which the per-committed-token metric charges correctly.
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
    groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        groups[(r["prompt_idx"], r["cycle"])].append(r)
    cycles = []
    for toks in groups.values():
        toks.sort(key=lambda r: r["pos_in_cycle"])
        gamma = len(toks)
        r_idx = gamma
        for i, t in enumerate(toks):
            if not t["accepted"]:
                r_idx = i
                break
        cycles.append({
            "gamma": gamma, "r": r_idx,
            "entropy": np.array([t["draft_entropy"] for t in toks]),
        })
    return cycles


def sim_fixed(cycles, gcap: int):
    verify = 0
    commit = 0
    for c in cycles:
        r = min(c["r"], gcap)
        L = gcap
        verify += L
        if c["r"] < gcap:
            commit += r + 1  # reject inside drafted region -> r accepts + correction
        else:
            commit += min(c["r"], gcap)  # all drafted accepted (r>=gcap) -> gcap accepts
    return verify, commit


def sim_adaptive(cycles, theta: float, gcap: int = 8, gmin: int = 1):
    verify = 0
    commit = 0
    for c in cycles:
        ent = c["entropy"][:gcap]
        # stop drafting AFTER first position j with entropy>theta -> length j+1
        stop = gcap
        for j in range(len(ent)):
            if ent[j] > theta and (j + 1) >= gmin:
                stop = j + 1
                break
        L = min(stop, gcap)
        verify += L
        if c["r"] < L:
            commit += c["r"] + 1
        else:
            commit += L
    return verify, commit


def _cost_models(verify: float, commit: float, cycles_n: int) -> dict[str, float]:
    """Two SD cost regimes, cost per committed token (lower=better):
      - memory_bound: one target forward per block, cost ~ #passes (gamma-free).
      - compute_bound: cost ~ #verified positions (large-batch)."""
    return {
        "mem_bound_passes_per_token": cycles_n / commit,
        "compute_bound_verify_per_token": verify / commit,
        "tokens_per_cycle": commit / cycles_n,
    }


def analyze(path: Path) -> dict[str, Any]:
    cycles = load_cycles(path)
    n = len(cycles)
    gcap = max(c["gamma"] for c in cycles)
    fixed = {}
    for g in range(1, gcap + 1):
        v, c = sim_fixed(cycles, g)
        fixed[g] = _cost_models(v, c, n)
    best_fixed_mem = min(fixed, key=lambda g: fixed[g]["mem_bound_passes_per_token"])
    best_fixed_cmp = min(fixed, key=lambda g: fixed[g]["compute_bound_verify_per_token"])

    ent_all = np.concatenate([c["entropy"] for c in cycles])
    thetas = list(np.quantile(ent_all, np.linspace(0.3, 0.98, 16)))
    adaptive = []
    for th in thetas:
        v, c = sim_adaptive(cycles, th, gcap=gcap)
        row = {"theta": float(th)}
        row.update(_cost_models(v, c, n))
        adaptive.append(row)
    best_adapt_mem = min(adaptive, key=lambda r: r["mem_bound_passes_per_token"])
    best_adapt_cmp = min(adaptive, key=lambda r: r["compute_bound_verify_per_token"])

    return {
        "n_cycles": n, "gcap": gcap,
        "fixed_gamma": {str(g): fixed[g] for g in fixed},
        "MEMORY_BOUND (single-request, real)": {
            "best_fixed_gamma": best_fixed_mem,
            "best_fixed_passes_per_token": fixed[best_fixed_mem]["mem_bound_passes_per_token"],
            "best_adaptive_passes_per_token": best_adapt_mem["mem_bound_passes_per_token"],
            "adaptive_speedup_vs_best_fixed": fixed[best_fixed_mem]["mem_bound_passes_per_token"] / best_adapt_mem["mem_bound_passes_per_token"],
        },
        "COMPUTE_BOUND (large-batch)": {
            "best_fixed_gamma": best_fixed_cmp,
            "best_fixed_verify_per_token": fixed[best_fixed_cmp]["compute_bound_verify_per_token"],
            "best_adaptive_verify_per_token": best_adapt_cmp["compute_bound_verify_per_token"],
            "adaptive_speedup_vs_best_fixed": fixed[best_fixed_cmp]["compute_bound_verify_per_token"] / best_adapt_cmp["compute_bound_verify_per_token"],
        },
        "adaptive_grid": adaptive,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", default="/root/autodl-tmp/reasonbranch/outputs/vsignal/tokens.jsonl")
    p.add_argument("--out", default="/root/autodl-tmp/reasonbranch/outputs/vsignal/t1_adaptive_gamma.json")
    args = p.parse_args()
    res = analyze(Path(args.tokens))
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "adaptive_grid"}, indent=2))


if __name__ == "__main__":
    main()

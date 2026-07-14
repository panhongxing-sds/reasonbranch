"""Analyze residual-stream geometry vs the ICML'26 'semantic fixed point' claim,
on DeepSeek-R1-Distill-Qwen-32B (a reasoning-distilled model).

Questions:
  Q1. Plain-decode prediction stabilization: at which depth does the logit-lens
      argmax equal the FINAL argmax? (If only at the top -> depth early-exit for
      generation is infeasible on this model, unlike LLaMA-2.)
  Q2. Geometric fixed point: does the normalized update norm vanish and the
      consecutive-update cosine stabilize early? Sweep thresholds; report the
      earliest layer the oral's criterion fires and whether prediction is settled.
  Q3. Accept vs reject: do they differ in stabilization / convergence depth?
  Q4. Achievable depth (=memory-bandwidth) saving at >=98% prediction fidelity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load(path: Path):
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def analyze(path: Path) -> dict[str, Any]:
    rows = load(path)
    self_layers = rows[0]["self_layers"]
    n_layers = len(rows[0]["upd_norm"])
    acc = np.array([1 if r["accepted"] else 0 for r in rows])

    # Q1: self-prediction stabilization
    SP = np.array([r["selfpred_match"] for r in rows])  # (N, len(self_layers))
    frac_match = SP.mean(0)
    # stabilization depth: first self-layer after which all matches are 1
    def stab_depth(row):
        m = row["selfpred_match"]
        for i in range(len(m)):
            if all(m[j] == 1 for j in range(i, len(m))):
                return self_layers[i]
        return n_layers  # never stabilized before top sampled layer
    sd = np.array([stab_depth(r) for r in rows])

    # Q2: geometric fixed point. Normalized update norm and cosine.
    NORM = np.array([r["upd_norm_rel"] for r in rows])  # (N, n_layers)
    COS = np.array([r["upd_cos"] for r in rows])
    # oral criterion: exit at first layer l where norm<tau_n AND cos>tau_c and stays
    def geo_exit(nrow, crow, tn, tc):
        for l in range(len(nrow)):
            if all((nrow[k] < tn and crow[k] > tc) for k in range(l, len(nrow))):
                return l + 1  # layers computed
        return n_layers
    grid = []
    for tn in [0.05, 0.1, 0.15, 0.2, 0.3]:
        for tc in [0.0, 0.3, 0.5, 0.7, 0.9]:
            exits = np.array([geo_exit(NORM[i], COS[i], tn, tc) for i in range(len(rows))])
            grid.append({"tau_norm": tn, "tau_cos": tc,
                         "mean_exit_layer": float(exits.mean()),
                         "frac_exit_before_top": float((exits < n_layers).mean())})

    # Q4: depth-saving at prediction fidelity. Truncate at layer L (uniform),
    # prediction correct if selfpred_match at the largest self_layer <= L is 1.
    fidelity_by_L = {}
    for i, L in enumerate(self_layers):
        # predict using layer L: correct if selfpred_match[i]==1
        fidelity = float(SP[:, i].mean())
        saving = 1 - (L + 1) / n_layers
        fidelity_by_L[str(L)] = {"fidelity": fidelity, "depth_saving": saving}

    return {
        "n": len(rows), "n_layers": n_layers, "self_layers": self_layers,
        "Q1_selfpred": {
            "frac_match_by_layer": {str(self_layers[i]): float(frac_match[i]) for i in range(len(self_layers))},
            "stab_depth_mean": float(sd.mean()), "stab_depth_median": float(np.median(sd)),
            "stab_depth_frac_at_top": float((sd >= n_layers).mean()),
        },
        "Q2_geometric_grid": grid,
        "Q3_accept_vs_reject": {
            "stab_depth_acc_mean": float(sd[acc == 1].mean()),
            "stab_depth_rej_mean": float(sd[acc == 0].mean()),
            "norm_last8_acc_mean": float(NORM[acc == 1][:, -8:].mean()),
            "norm_last8_rej_mean": float(NORM[acc == 0][:, -8:].mean()),
        },
        "Q4_depth_saving_vs_fidelity": fidelity_by_L,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--geom", default="/root/autodl-tmp/reasonbranch/outputs/vsignal/geom.jsonl")
    p.add_argument("--out", default="/root/autodl-tmp/reasonbranch/outputs/vsignal/geom_analysis.json")
    args = p.parse_args()
    res = analyze(Path(args.geom))
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

"""Method de-risk: is confidence-gated ADAPTIVE draft length viable?

Lever (drafter side, memory-bound): the expensive op is each TARGET forward. If a
draft block will keep being accepted (easy run), drafting deeper collapses many
target forwards into one. If it will be rejected fast (hard block), draft short.
Net win requires that a CHEAP draft-side signal separates "will be accepted" from
"will be rejected" -- per token AND per block.

Data: tokens.jsonl (1.5B greedy draft, gamma=8, 32B verify). Real accept pattern.

Tests:
  1) per-token: AUC of draft self-signals for predicting accepted.
  2) per-position accept rate conditioned on confidence bucket (does gating separate?).
  3) block bimodality: accept-length histogram; fraction of blocks that saturate cap.
  4) idealized adaptive-length simulation (memory-bound cost = #target forwards):
     compare fixed-gamma vs a confidence-gated stop rule, on target-forwards-per-
     committed-token. Kill gate: adaptive must cut target forwards >10% at equal
     committed tokens (i.e. not lose acceptance).
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

P = "/root/autodl-tmp/reasonbranch/outputs/vsignal/tokens.jsonl"
SIGS = ["draft_top1_prob", "draft_margin", "draft_logprob_self", "draft_entropy"]


def auc(y, s):
    order = np.argsort(s); y = y[order]
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return float("nan")
    ranks = np.arange(1, len(y) + 1)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    rows = [json.loads(l) for l in open(P) if l.strip()]
    acc = np.array([1 if r["accepted"] else 0 for r in rows])
    print(f"n_tokens={len(rows)}  accept_rate={acc.mean():.3f}")

    # 1) per-token AUC (higher confidence -> more likely accepted; entropy flipped)
    print("\n[1] per-token AUC for predicting ACCEPT:")
    for k in SIGS:
        s = np.array([r[k] for r in rows], float)
        if k == "draft_entropy":
            s = -s
        print(f"   {k:20} AUC={auc(acc, s):.3f}")

    # 2) accept rate by confidence bucket x position
    print("\n[2] P(accept) by draft_top1_prob quintile (all positions):")
    tp = np.array([r["draft_top1_prob"] for r in rows], float)
    qs = np.quantile(tp, [0.2, 0.4, 0.6, 0.8])
    buck = np.digitize(tp, qs)
    for b in range(5):
        m = buck == b
        print(f"   q{b}: top1_prob in [{tp[m].min():.2f},{tp[m].max():.2f}]  "
              f"P(accept)={acc[m].mean():.3f}  n={m.sum()}")

    # 3) block structure
    groups = defaultdict(list)
    for r in rows:
        groups[(r["prompt_idx"], r["cycle"])].append(r)
    accept_len = []
    for toks in groups.values():
        toks.sort(key=lambda r: r["pos_in_cycle"])
        L = 0
        for t in toks:
            if t["accepted"]:
                L += 1
            else:
                break
        accept_len.append(L)
    accept_len = np.array(accept_len)
    gamma = max(len(t) for t in groups.values())
    hist = {int(k): int((accept_len == k).sum()) for k in range(gamma + 1)}
    sat = (accept_len == gamma).mean()
    print(f"\n[3] blocks={len(accept_len)} gamma={gamma} mean_accept_len={accept_len.mean():.2f}")
    print(f"   accept_len_hist={hist}")
    print(f"   saturated (==gamma, would extend): {sat:.1%}")

    # 4) adaptive-length sim (memory-bound: cost = #target forwards).
    # Fixed-gamma g: each block drafts g, verify once, commit accept_len(capped g)+1.
    # Adaptive: draft token-by-token, STOP drafting when top1_prob < theta (predicting
    # imminent reject) -> shorter blocks on hard runs (fewer wasted), but the KEY win
    # is EXTENDING easy runs. Since data caps at gamma=8 we can only measure the
    # within-cap effect: does gating raise committed-per-forward vs fixed same avg len?
    print("\n[4] adaptive stop-rule vs fixed gamma (within-cap, memory-bound=target fwds):")
    # per-block per-token top1_prob and accepted
    blocks = []
    for toks in groups.values():
        toks.sort(key=lambda r: r["pos_in_cycle"])
        blocks.append([(t["draft_top1_prob"], int(t["accepted"])) for t in toks])

    def sim_fixed(g):
        fwd = 0; commit = 0
        for b in blocks:
            fwd += 1
            L = 0
            for i in range(min(g, len(b))):
                if b[i][1]:
                    L += 1
                else:
                    break
            commit += L + 1  # + bonus token
        return commit, fwd

    def sim_adaptive(theta, cap):
        fwd = 0; commit = 0; drafted = 0
        for b in blocks:
            fwd += 1
            L = 0
            for i in range(min(cap, len(b))):
                drafted += 1
                # stop drafting further if confidence below theta (we still verify
                # what we drafted; this only limits how far we draft)
                if b[i][1]:
                    L += 1
                else:
                    break
                if i + 1 < len(b) and b[i + 1][0] < theta:
                    break
            commit += L + 1
        return commit, fwd, drafted

    for g in [2, 4, 6, 8]:
        c, f = sim_fixed(g)
        print(f"   fixed g={g}: commit/fwd={c/f:.3f}  (tokens per target forward)")
    print("   -- adaptive only limits drafting; cannot extend beyond cap=8 (data-capped) --")
    for th in [0.1, 0.2, 0.3, 0.5]:
        c, f, d = sim_adaptive(th, 8)
        print(f"   adaptive theta={th}: commit/fwd={c/f:.3f}  draft/fwd={d/f:.2f}")


if __name__ == "__main__":
    main()

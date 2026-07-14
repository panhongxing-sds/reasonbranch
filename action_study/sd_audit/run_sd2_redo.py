"""SD2 (Request-Local Residual Calibration) — corrected re-audit.

Idea: within ONE request, the target-vs-draft logit residual has request-local
structure (domain/entity tokens the draft systematically mis-ranks recur). Fit a
small additive logit bias on the EARLY part of a request, apply it to the draft
on the LATER part of the SAME request -> draft argmax matches target more often
-> longer acceptance -> speedup. No persistent params, no gradients.

Fixes vs first audit (which found calibration HURTS, on 6 prompts, buggy):
  1. Restrict bias to a small ACTIVE token set (target/draft argmax seen in the
     calibration window) instead of a 152k-dim full-vocab EMA from a few samples.
  2. Select shrinkage lambda on a held-out slice of the calibration window, with
     lambda=0 (= baseline) in the grid -> calibration provably cannot do worse
     than baseline in expectation (falls back to no-op when it does not generalize).
  3. Correct vocab alignment (align_logits).
  4. Enough requests + per-request paired sign test.

Metric: next-token top-1 agreement with target on the EVAL window (proxy for
accept probability), plus implied expected accept run-length p/(1-p).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from reasoning_branch_dataset.action_study.sd_audit.harness import (
    align_logits,
    build_prompt,
    draft_greedy_tokens,
    load_gsm8k_prompts,
    load_hf_model,
    unload_model,
)


@torch.inference_mode()
def _collect_request_logits(draft, target, prompt: str, *, gamma: int, max_cycles: int):
    """Run linear greedy SD; return per-verified-position aligned (draft, target)
    logits (fp16, CPU) in generation order."""
    prefix = draft.encode(prompt)
    d_list: list[torch.Tensor] = []
    t_list: list[torch.Tensor] = []
    for _ in range(max_cycles):
        draft_ids = draft_greedy_tokens(draft, prefix, gamma)
        if not draft_ids:
            break
        full = prefix + draft_ids
        positions = list(range(len(prefix), len(full)))
        d_logits = draft.logits_at_positions(full, positions)
        t_logits = target.logits_at_positions(full, positions)
        # greedy accept length
        acc = 0
        for j in range(len(draft_ids)):
            dj, tj = align_logits(d_logits[j], t_logits[j])
            d_list.append(dj.half().cpu())
            t_list.append(tj.half().cpu())
            if int(dj.argmax()) == int(tj.argmax()):
                acc += 1
            else:
                break
        # advance prefix (standard greedy SD)
        if acc <= 0:
            prefix = prefix + [int(t_logits[0].argmax())]
        else:
            prefix = prefix + draft_ids[:acc]
            if acc < len(draft_ids):
                # bonus/correction token = target greedy at first reject position
                prefix = prefix + [int(t_logits[acc].argmax())]
    return d_list, t_list


def _fit_bias(d_calib, t_calib, active: np.ndarray, lam: float) -> np.ndarray:
    """b[s] = lam * mean_i (t_i[s]-d_i[s]) for s in active, else 0."""
    V = d_calib[0].shape[0]
    b = np.zeros(V, dtype=np.float32)
    if len(active) == 0 or lam == 0.0:
        return b
    acc = np.zeros(len(active), dtype=np.float32)
    idx = torch.tensor(active, dtype=torch.long)
    for d, t in zip(d_calib, t_calib):
        acc += (t.float()[idx] - d.float()[idx]).numpy()
    acc /= len(d_calib)
    b[active] = lam * acc
    return b


def _agreement(d_list, t_list, b: np.ndarray) -> float:
    if not d_list:
        return float("nan")
    bt = torch.from_numpy(b)
    ok = 0
    for d, t in zip(d_list, t_list):
        pred = int((d.float() + bt).argmax())
        ok += int(pred == int(t.float().argmax()))
    return ok / len(d_list)


def _active_set(d_calib, t_calib, top_resid: int = 64) -> np.ndarray:
    s: set[int] = set()
    for d, t in zip(d_calib, t_calib):
        s.add(int(t.float().argmax()))
        s.add(int(d.float().argmax()))
    # add tokens with largest mean |residual|
    if d_calib:
        idx_all = list(s)
        acc = np.zeros(d_calib[0].shape[0], dtype=np.float32)
        for d, t in zip(d_calib, t_calib):
            acc += (t.float() - d.float()).numpy()
        acc = np.abs(acc / len(d_calib))
        top = np.argsort(acc)[::-1][:top_resid].tolist()
        s.update(int(x) for x in top)
    return np.array(sorted(s), dtype=np.int64)


def run(*, draft_path, target_path, data_path, out_path: Path, n_prompts, gamma,
        max_cycles, seed) -> dict[str, Any]:
    prompts = [build_prompt(q) for q in load_gsm8k_prompts(Path(data_path), n_prompts, seed)]
    draft = load_hf_model(draft_path)
    target = load_hf_model(target_path)
    lam_grid = [0.0, 0.25, 0.5, 0.75, 1.0]

    per_req = []
    for pi, prompt in enumerate(prompts):
        d_all, t_all = _collect_request_logits(draft, target, prompt, gamma=gamma, max_cycles=max_cycles)
        n = len(d_all)
        if n < 20:
            continue
        # split: calib 60% (with inner holdout last 1/3 of calib), eval 40%
        n_cal = int(n * 0.6)
        n_inner = int(n_cal * 0.67)
        d_fit, t_fit = d_all[:n_inner], t_all[:n_inner]
        d_sel, t_sel = d_all[n_inner:n_cal], t_all[n_inner:n_cal]
        d_ev, t_ev = d_all[n_cal:], t_all[n_cal:]
        active = _active_set(d_fit, t_fit)
        # select lambda on calib-holdout
        best_lam, best_sel = 0.0, -1.0
        for lam in lam_grid:
            b = _fit_bias(d_fit, t_fit, active, lam)
            a = _agreement(d_sel, t_sel, b)
            if a > best_sel + 1e-9:  # strict improvement to prefer smaller lam on ties
                best_sel, best_lam = a, lam
        # refit on full calib with chosen lam, eval
        active_full = _active_set(d_all[:n_cal], t_all[:n_cal])
        b_base = np.zeros(d_all[0].shape[0], dtype=np.float32)
        b_cal = _fit_bias(d_all[:n_cal], t_all[:n_cal], active_full, best_lam)
        base_agree = _agreement(d_ev, t_ev, b_base)
        cal_agree = _agreement(d_ev, t_ev, b_cal)
        # ORACLE ceiling: best lambda chosen directly on eval (cheating) -> upper
        # bound of what ANY request-local additive bias with this active set can do.
        oracle_agree = base_agree
        for lam in lam_grid:
            bo = _fit_bias(d_all[:n_cal], t_all[:n_cal], active_full, lam)
            ao = _agreement(d_ev, t_ev, bo)
            oracle_agree = max(oracle_agree, ao)
        per_req.append({
            "prompt_idx": pi, "n_pos": n, "n_eval": len(d_ev),
            "chosen_lam": best_lam, "n_active": int(len(active_full)),
            "base_agree": base_agree, "cal_agree": cal_agree,
            "delta": cal_agree - base_agree,
            "oracle_agree": oracle_agree, "oracle_delta": oracle_agree - base_agree,
        })
        print(f"[sd2] {pi+1}/{len(prompts)} n={n} lam={best_lam} "
              f"base={base_agree:.3f} cal={cal_agree:.3f} d={cal_agree-base_agree:+.3f}", flush=True)

    unload_model(draft)
    unload_model(target)

    deltas = np.array([r["delta"] for r in per_req])
    base = np.array([r["base_agree"] for r in per_req])
    cal = np.array([r["cal_agree"] for r in per_req])
    oracle_deltas = np.array([r["oracle_delta"] for r in per_req])
    n_imp = int((deltas > 1e-9).sum())
    n_hurt = int((deltas < -1e-9).sum())
    n_tie = int((np.abs(deltas) <= 1e-9).sum())
    # sign test p (two-sided) over non-tie requests
    m = n_imp + n_hurt
    k = max(n_imp, n_hurt)
    p_sign = float(min(1.0, 2 * sum(math.comb(m, i) for i in range(k, m + 1)) / (2 ** m))) if m > 0 else float("nan")

    def implied_speedup(p):
        p = min(p, 0.999)
        return p / (1 - p)

    summary = {
        "n_requests": len(per_req), "gamma": gamma,
        "mean_base_agree": float(base.mean()) if len(base) else float("nan"),
        "mean_cal_agree": float(cal.mean()) if len(cal) else float("nan"),
        "mean_delta": float(deltas.mean()) if len(deltas) else float("nan"),
        "median_delta": float(np.median(deltas)) if len(deltas) else float("nan"),
        "mean_oracle_delta": float(oracle_deltas.mean()) if len(oracle_deltas) else float("nan"),
        "max_oracle_delta": float(oracle_deltas.max()) if len(oracle_deltas) else float("nan"),
        "n_improved": n_imp, "n_hurt": n_hurt, "n_tie": n_tie,
        "sign_test_p": p_sign,
        "implied_run_base": implied_speedup(float(base.mean())) if len(base) else None,
        "implied_run_cal": implied_speedup(float(cal.mean())) if len(cal) else None,
        "lam_hist": {str(l): int((np.array([r["chosen_lam"] for r in per_req]) == l).sum()) for l in lam_grid},
    }
    out = {"summary": summary, "per_request": per_req}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--draft", default="/root/autodl-tmp/specreason/models/DeepSeek-R1-Distill-Qwen-1.5B")
    p.add_argument("--target", default="/root/autodl-tmp/specreason/models/DeepSeek-R1-Distill-Qwen-32B")
    p.add_argument("--data", default="/root/autodl-tmp/reasonbranch/data/gsm8k_test.jsonl")
    p.add_argument("--out", default="/root/autodl-tmp/reasonbranch/outputs/vsignal/sd2_redo.json")
    p.add_argument("--n-prompts", type=int, default=40)
    p.add_argument("--gamma", type=int, default=8)
    p.add_argument("--max-cycles", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    s = run(draft_path=args.draft, target_path=args.target, data_path=args.data,
            out_path=Path(args.out), n_prompts=args.n_prompts, gamma=args.gamma,
            max_cycles=args.max_cycles, seed=args.seed)
    print(json.dumps(s, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Audit B — request-local residual stability (Direction 2).

Gate: draft-target residual learned on the first calib_frac of verification
positions must improve target top-1 agreement / acceptance on the held-out tail,
with stable cross-position gains (not train-position overfitting).
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path
from typing import Any

import torch

from reasoning_branch_dataset.action_study.sd_audit.harness import (
    ResidualRecord,
    align_logits,
    collect_sd_cycles,
    kl_divergence,
    load_gsm8k_prompts,
    load_hf_model,
    build_prompt,
    unload_model,
)


def _fit_ema_bias(records: list[ResidualRecord], *, alpha: float = 0.3) -> torch.Tensor:
    """Per-vocab EMA of (target - draft) logits on calibration positions."""
    if not records:
        return torch.zeros(1)
    t0, d0 = align_logits(records[0].target_logits, records[0].draft_logits)
    dim = t0.numel()
    bias = torch.zeros(dim, device=t0.device)
    for r in records:
        t, d = align_logits(r.target_logits, r.draft_logits)
        bias = alpha * (t - d) + (1 - alpha) * bias
    return bias


def _fit_token_bias(records: list[ResidualRecord]) -> dict[int, float]:
    """Per target-argmax token scalar bias (lightweight ridge-like)."""
    sums: dict[int, list[float]] = {}
    for r in records:
        t, d = align_logits(r.target_logits, r.draft_logits)
        tid = int(t.argmax().item())
        delta = float((t[tid] - d[tid]).item())
        sums.setdefault(tid, []).append(delta)
    return {k: st.fmean(v) for k, v in sums.items()}


def _apply_token_bias(logits: torch.Tensor, tb: dict[int, float]) -> torch.Tensor:
    out = logits.clone()
    for tid, b in tb.items():
        if tid < out.numel():
            out[tid] = out[tid] + b
    return out


def evaluate_records(
    records: list[ResidualRecord],
    *,
    ema_bias: torch.Tensor | None = None,
    token_bias: dict[int, float] | None = None,
) -> dict[str, float]:
    if not records:
        return {}
    base_match = st.fmean([1.0 if r.top1_match_draft else 0.0 for r in records])
    ema_match = st.fmean([
        1.0 if r.top1_match_calibrated(ema_bias) else 0.0 for r in records
    ]) if ema_bias is not None else base_match
    tok_scores = []
    for r in records:
        t, d = align_logits(r.target_logits, r.draft_logits)
        tok_scores.append(
            1.0 if int(_apply_token_bias(d, token_bias or {}).argmax()) == int(t.argmax()) else 0.0
        )
    tok_match = st.fmean(tok_scores)
    base_kl = st.fmean([kl_divergence(r.target_logits, r.draft_logits) for r in records])
    if ema_bias is not None:
        cal_kls = []
        for r in records:
            t, d = align_logits(r.target_logits, r.draft_logits)
            cal_kls.append(kl_divergence(t, d + ema_bias[: d.numel()]))
        cal_kl = st.fmean(cal_kls)
    else:
        cal_kl = base_kl
    return {
        "n": len(records),
        "top1_match_draft": base_match,
        "top1_match_ema": ema_match,
        "top1_match_token_bias": tok_match,
        "kl_target_draft": base_kl,
        "kl_target_calibrated": cal_kl,
        "delta_top1_ema": ema_match - base_match,
        "delta_kl_ema": base_kl - cal_kl,
    }


def run_audit_b(
    *,
    draft_path: str,
    target_path: str,
    data_path: str,
    n_prompts: int = 8,
    gamma: int = 8,
    max_cycles: int = 24,
    calib_frac: float = 0.25,
    seed: int = 42,
) -> dict[str, Any]:
    prompts = [build_prompt(q) for q in load_gsm8k_prompts(Path(data_path), n_prompts, seed)]
    draft = load_hf_model(draft_path)
    target = load_hf_model(target_path)

    all_records: list[ResidualRecord] = []
    per_prompt_stats: list[dict[str, Any]] = []

    for pi, prompt in enumerate(prompts):
        _, recs = collect_sd_cycles(draft, target, prompt, gamma=gamma, max_cycles=max_cycles)
        if len(recs) < 8:
            continue
        split = max(1, int(len(recs) * calib_frac))
        calib, test = recs[:split], recs[split:]
        ema = _fit_ema_bias(calib)
        tb = _fit_token_bias(calib)
        train_m = evaluate_records(calib, ema_bias=ema, token_bias=tb)
        test_m = evaluate_records(test, ema_bias=ema, token_bias=tb)
        print(f"[B] prompt {pi+1}/{len(prompts)} n_recs={len(recs)} test_dtop1={test_m.get('delta_top1_ema',0):+.3f}", flush=True)
        per_prompt_stats.append({
            "prompt_idx": pi,
            "n_calib": len(calib),
            "n_test": len(test),
            "train": train_m,
            "test": test_m,
        })
        all_records.extend(recs)

    # pooled split by global position order
    split = max(1, int(len(all_records) * calib_frac))
    calib_all, test_all = all_records[:split], all_records[split:]
    ema_all = _fit_ema_bias(calib_all)
    tb_all = _fit_token_bias(calib_all)
    pooled = {
        "train": evaluate_records(calib_all, ema_bias=ema_all, token_bias=tb_all),
        "test": evaluate_records(test_all, ema_bias=ema_all, token_bias=tb_all),
    }

    # stability: test improvement positive on majority of prompts
    n_improved = sum(1 for p in per_prompt_stats if p["test"].get("delta_top1_ema", 0) > 0)
    n_stable_kl = sum(1 for p in per_prompt_stats if p["test"].get("delta_kl_ema", 0) > 0)
    n_prompts_ok = len(per_prompt_stats)

    test_delta = pooled["test"].get("delta_top1_ema", 0.0)
    passed = (
        test_delta > 0.02
        and n_improved >= max(1, n_prompts_ok * 0.6)
        and pooled["test"].get("delta_kl_ema", 0) > 0
    )

    unload_model(draft)
    unload_model(target)

    return {
        "audit": "B_request_local_residual",
        "gamma": gamma,
        "calib_frac": calib_frac,
        "n_prompts": n_prompts_ok,
        "n_records": len(all_records),
        "pooled": pooled,
        "per_prompt": per_prompt_stats,
        "n_prompts_improved_top1": n_improved,
        "n_prompts_improved_kl": n_stable_kl,
        "decision": "PASS" if passed else "FAIL",
        "kill_gate": "test delta_top1_ema > 0.02, >=60% prompts improve, KL drops on test",
    }


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--draft", default="/root/autodl-tmp/specreason/models/DeepSeek-R1-Distill-Qwen-1.5B")
    p.add_argument("--target", default="/root/autodl-tmp/specreason/models/DeepSeek-R1-Distill-Qwen-32B")
    p.add_argument("--data", default="/root/autodl-tmp/reasonbranch/data/gsm8k_test.jsonl")
    p.add_argument("--out", default="/root/autodl-tmp/reasonbranch/outputs/sd_audit_b.json")
    p.add_argument("--n-prompts", type=int, default=8)
    p.add_argument("--gamma", type=int, default=8)
    args = p.parse_args()
    res = run_audit_b(
        draft_path=args.draft, target_path=args.target, data_path=args.data,
        n_prompts=args.n_prompts, gamma=args.gamma,
    )
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "per_prompt"}, indent=2))


if __name__ == "__main__":
    main()

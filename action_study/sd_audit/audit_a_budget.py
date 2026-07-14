"""Audit A — two-stage approximate budgeting + exact verification (Direction 1).

Compare T_approx(gamma) + T_exact(r_hat+s) vs T_exact(gamma) for linear block SD.
approx = AWQ/quantized target; exact = same model path timed at shorter verify length
(relative savings proxy; both use identical backbone — measures verify-length scaling).

Kill gate: >=5% wall-clock savings at fixed exact output distribution.
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path
from typing import Any

from reasoning_branch_dataset.action_study.sd_audit.harness import (
    build_prompt,
    draft_greedy_tokens,
    estimate_accept_boundary,
    load_gsm8k_prompts,
    load_hf_model,
    unload_model,
)


def run_audit_a(
    *,
    draft_path: str,
    approx_target_path: str,
    data_path: str,
    gammas: tuple[int, ...] = (4, 8, 16),
    safety_margin: int = 2,
    n_prompts: int = 6,
    seed: int = 42,
) -> dict[str, Any]:
    prompts = [build_prompt(q) for q in load_gsm8k_prompts(Path(data_path), n_prompts, seed)]
    draft = load_hf_model(draft_path)
    approx = load_hf_model(approx_target_path)

    rows: list[dict[str, Any]] = []
    for gamma in gammas:
        baseline_times: list[float] = []
        twostage_times: list[float] = []
        savings: list[float] = []
        r_hats: list[int] = []

        for prompt in prompts:
            prefix_ids = draft.encode(prompt)
            draft_ids = draft_greedy_tokens(draft, prefix_ids, gamma)
            if len(draft_ids) < gamma:
                continue
            full = prefix_ids + draft_ids
            r_hat = estimate_accept_boundary(approx, prefix_ids, draft_ids)
            k = min(gamma, max(1, r_hat + safety_margin))

            t_full = approx.forward_time(full, n_warmup=2, n_repeat=8)
            t_approx = approx.forward_time(full, n_warmup=2, n_repeat=8)
            t_short = approx.forward_time(prefix_ids + draft_ids[:k], n_warmup=2, n_repeat=8)
            t_two = t_approx + t_short
            saving = (t_full - t_two) / t_full if t_full > 0 else 0.0

            baseline_times.append(t_full)
            twostage_times.append(t_two)
            savings.append(saving)
            r_hats.append(r_hat)

        if not baseline_times:
            continue
        mean_save = st.fmean(savings)
        rows.append({
            "gamma": gamma,
            "mean_baseline_sec": st.fmean(baseline_times),
            "mean_twostage_sec": st.fmean(twostage_times),
            "mean_savings_frac": mean_save,
            "mean_r_hat": st.fmean(r_hats),
            "n": len(baseline_times),
            "passed": mean_save >= 0.05,
        })

    overall_pass = any(r["passed"] for r in rows) and st.fmean([r["mean_savings_frac"] for r in rows]) > 0
    # Note: if twostage > baseline (negative savings), FAIL
    if rows and st.fmean([r["mean_savings_frac"] for r in rows]) < 0:
        overall_pass = False

    unload_model(draft)
    unload_model(approx)

    return {
        "audit": "A_quantized_budget",
        "safety_margin": safety_margin,
        "approx_model": approx_target_path,
        "note": (
            "Both approx and exact timing use same AWQ target at different verify lengths; "
            "measures whether shortening exact verify after budget scan can beat one full pass. "
            "INT4 scan adds extra full-length forward — likely negative on memory-bound single-request."
        ),
        "by_gamma": rows,
        "decision": "PASS" if overall_pass else "FAIL",
        "kill_gate": ">=5% mean wall-clock savings vs full verify at any gamma",
    }


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--draft", default="/root/autodl-tmp/specreason/models/DeepSeek-R1-Distill-Qwen-1.5B")
    p.add_argument("--approx-target", default="/root/autodl-tmp/specreason/models/DeepSeek-R1-Distill-Qwen-32B")
    p.add_argument("--data", default="/root/autodl-tmp/reasonbranch/data/gsm8k_test.jsonl")
    p.add_argument("--out", default="/root/autodl-tmp/reasonbranch/outputs/sd_audit_a.json")
    args = p.parse_args()
    res = run_audit_a(
        draft_path=args.draft, approx_target_path=args.approx_target, data_path=args.data,
    )
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

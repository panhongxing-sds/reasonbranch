"""Audit C — LM-head share of target verification (Direction 3).

Profile rho_head = T_lm_head / T_verify on target forward passes.
Kill gate: rho_head < 10% => do not pursue certified candidate-only projection.
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path
from typing import Any

from reasoning_branch_dataset.action_study.sd_audit.harness import (
    build_prompt,
    draft_greedy_tokens,
    load_gsm8k_prompts,
    load_hf_model,
    unload_model,
)


def run_audit_c(
    *,
    draft_path: str,
    target_path: str,
    data_path: str,
    gammas: tuple[int, ...] = (4, 8, 16),
    n_prompts: int = 4,
    seed: int = 42,
) -> dict[str, Any]:
    prompts = [build_prompt(q) for q in load_gsm8k_prompts(Path(data_path), n_prompts, seed)]
    draft = load_hf_model(draft_path)
    target = load_hf_model(target_path)

    profiles: list[dict[str, float]] = []
    for prompt in prompts:
        prefix_ids = draft.encode(prompt)
        for gamma in gammas:
            draft_ids = draft_greedy_tokens(draft, prefix_ids, gamma)
            full = prefix_ids + draft_ids
            prof = target.profile_lm_head(full, n_repeat=12)
            prof["gamma"] = gamma
            profiles.append(prof)

    if not profiles:
        unload_model(draft)
        unload_model(target)
        return {"audit": "C_lm_head_profile", "decision": "INCONCLUSIVE"}

    mean_rho = st.fmean([p["rho_head"] for p in profiles])
    mean_lm_ms = st.fmean([p["lm_head_ms"] for p in profiles])
    mean_total_ms = st.fmean([p["total_ms"] for p in profiles])

    # Also profile draft for comparison
    draft_profiles: list[dict[str, float]] = []
    for prompt in prompts[:2]:
        prefix_ids = draft.encode(prompt)
        for gamma in gammas:
            draft_ids = draft_greedy_tokens(draft, prefix_ids, gamma)
            prof = draft.profile_lm_head(prefix_ids + draft_ids, n_repeat=12)
            prof["gamma"] = gamma
            draft_profiles.append(prof)
    draft_rho = st.fmean([p["rho_head"] for p in draft_profiles]) if draft_profiles else 0.0

    passed = mean_rho >= 0.10  # continue only if head is >=10% of verify

    unload_model(draft)
    unload_model(target)

    return {
        "audit": "C_lm_head_profile",
        "target_model": target_path,
        "draft_model": draft_path,
        "target_mean_rho_head": mean_rho,
        "target_mean_lm_head_ms": mean_lm_ms,
        "target_mean_total_ms": mean_total_ms,
        "draft_mean_rho_head": draft_rho,
        "profiles": profiles,
        "draft_profiles": draft_profiles,
        "decision": "CONTINUE" if passed else "KILL",
        "kill_gate": "rho_head >= 10% to pursue candidate-only projection; else stop",
    }


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--draft", default="/root/autodl-tmp/specreason/models/DeepSeek-R1-Distill-Qwen-1.5B")
    p.add_argument("--target", default="/root/autodl-tmp/specreason/models/DeepSeek-R1-Distill-Qwen-32B")
    p.add_argument("--data", default="/root/autodl-tmp/reasonbranch/data/gsm8k_test.jsonl")
    p.add_argument("--out", default="/root/autodl-tmp/reasonbranch/outputs/sd_audit_c.json")
    args = p.parse_args()
    res = run_audit_c(
        draft_path=args.draft, target_path=args.target, data_path=args.data,
    )
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k not in ("profiles", "draft_profiles")}, indent=2))


if __name__ == "__main__":
    main()

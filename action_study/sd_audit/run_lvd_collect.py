"""Collect linear-SD verification cycles with Layerwise Verification Trajectory features.

Each cycle stores:
  A_c (accept length), gamma, draft entropy/margin, target final margin/KL,
  resolution depth / flip count / path speed (aggregated over rejected tokens),
  and next-cycle A_{c+1} for supervised prediction.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path
from typing import Any

import torch

from reasoning_branch_dataset.action_study.sd_audit.harness import (
    align_logits,
    build_prompt,
    draft_greedy_tokens,
    load_gsm8k_prompts,
    load_hf_model,
    unload_model,
)
from reasoning_branch_dataset.action_study.sd_audit.layerwise_trajectory import (
    LayerwiseVerifier,
    default_sample_layers,
)


def _entropy_from_logits(logits: torch.Tensor, topk: int = 64) -> float:
    vals = torch.topk(logits.float(), k=min(topk, logits.numel())).values
    p = torch.softmax(vals, dim=-1)
    return float((-p * (p + 1e-12).log()).sum().item())


def _margin(logits: torch.Tensor) -> float:
    t = torch.topk(logits.float(), k=2).values
    return float((t[0] - t[1]).item())


def _kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    p, q = align_logits(p_logits.float(), q_logits.float())
    k = min(128, p.numel())
    vals, idx = torch.topk(p, k)
    ps = torch.softmax(vals, dim=-1)
    ql = torch.log_softmax(q[idx], dim=-1)
    return float((ps * (ps.clamp_min(1e-12).log() - ql)).sum().item())


@torch.inference_mode()
def draft_pos_logits(draft, full_ids: list[int], positions: list[int]) -> torch.Tensor:
    inp = torch.tensor([full_ids], device=draft.device, dtype=torch.long)
    logits = draft.model(inp).logits[0]
    return torch.stack([logits[p - 1] for p in positions])


def aggregate_traj(trajs, n_layers: int) -> dict[str, float]:
    # Sequential greedy accept length
    accept_len = 0
    for t in trajs:
        if t.accepted:
            accept_len += 1
        else:
            break
    rejected = [t for t in trajs if not t.accepted]
    accepted = [t for t in trajs if t.accepted]
    if rejected:
        depths = [t.flip_depth if t.flip_depth is not None else n_layers for t in rejected]
        flips = [t.flip_count for t in rejected]
        speeds = [t.path_speed for t in rejected]
        margins = [t.final_margin for t in rejected]
        mean_depth = st.fmean(depths)
        mean_flip = st.fmean(flips)
        mean_speed = st.fmean(speeds)
        mean_rej_margin = st.fmean(margins)
        late_frac = st.fmean(
            [1.0 if (t.flip_depth if t.flip_depth is not None else n_layers) >= 0.75 * n_layers else 0.0
             for t in rejected]
        )
    else:
        mean_depth = mean_flip = mean_speed = mean_rej_margin = late_frac = 0.0
    return {
        "n_rejected": len(rejected),
        "n_accepted": len(accepted),
        "mean_flip_depth": mean_depth,
        "mean_flip_count": mean_flip,
        "mean_path_speed": mean_speed,
        "mean_reject_margin": mean_rej_margin,
        "late_resolve_frac": late_frac,
        "accept_len": accept_len,
    }


def collect(
    *,
    draft_path: str,
    target_path: str,
    data_path: str,
    out_path: Path,
    n_prompts: int = 20,
    gamma: int = 8,
    max_cycles: int = 20,
    seed: int = 42,
) -> dict[str, Any]:
    prompts = [build_prompt(q) for q in load_gsm8k_prompts(Path(data_path), n_prompts, seed)]
    draft = load_hf_model(draft_path)
    target = load_hf_model(target_path)
    verifier = LayerwiseVerifier(target.model, device=target.device)
    n_layers = verifier.n_layers

    rows: list[dict[str, Any]] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    for pi, prompt in enumerate(prompts):
        prefix = draft.encode(prompt)
        cycle_feats: list[dict[str, Any]] = []
        for cyc in range(max_cycles):
            draft_ids = draft_greedy_tokens(draft, prefix, gamma)
            if not draft_ids:
                break
            trajs = verifier.trajectories_for_draft(prefix, draft_ids)
            agg = aggregate_traj(trajs, n_layers)

            # draft / target position stats on first reject or last accepted
            full = prefix + draft_ids
            positions = list(range(len(prefix), len(full)))
            d_logits = draft_pos_logits(draft, full, positions)
            # target final logits already computed inside trajectories; recompute cheaply for KL
            inp = torch.tensor([full], device=target.device, dtype=torch.long)
            t_logits = target.model(inp).logits[0]
            # aggregate over draft span positions
            H_D = st.fmean([_entropy_from_logits(d_logits[j]) for j in range(len(draft_ids))])
            m_D = st.fmean([_margin(d_logits[j]) for j in range(len(draft_ids))])
            maxp_D = st.fmean([
                float(torch.softmax(d_logits[j].float(), dim=-1).max().item()) for j in range(len(draft_ids))
            ])
            H_T = st.fmean([_entropy_from_logits(t_logits[positions[j] - 1]) for j in range(len(draft_ids))])
            m_T = st.fmean([_margin(t_logits[positions[j] - 1]) for j in range(len(draft_ids))])
            kl = st.fmean([
                _kl(t_logits[positions[j] - 1], d_logits[j]) for j in range(len(draft_ids))
            ])

            A = agg["accept_len"]
            feat = {
                "prompt_idx": pi,
                "cycle": cyc,
                "gamma": gamma,
                "A": A,
                "H_D": H_D,
                "maxp_D": maxp_D,
                "m_D": m_D,
                "H_T": H_T,
                "m_T": m_T,
                "KL": kl,
                "flip_depth": agg["mean_flip_depth"],
                "flip_count": agg["mean_flip_count"],
                "path_speed": agg["mean_path_speed"],
                "late_resolve_frac": agg["late_resolve_frac"],
                "n_rejected": agg["n_rejected"],
                "mean_reject_margin": agg["mean_reject_margin"],
            }
            cycle_feats.append(feat)

            # advance prefix like greedy SD
            if A <= 0:
                y = int(t_logits[len(prefix) - 1].argmax().item())
                prefix = prefix + [y]
            else:
                prefix = prefix + draft_ids[:A]
                if A < len(draft_ids):
                    # bonus
                    y = int(t_logits[len(prefix) - 1].argmax().item())
                    prefix = prefix + [y]

        # attach A_{c+1}
        for i, feat in enumerate(cycle_feats):
            feat["A_next"] = cycle_feats[i + 1]["A"] if i + 1 < len(cycle_feats) else None
            if feat["A_next"] is None:
                continue
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(feat, ensure_ascii=False) + "\n")
            rows.append(feat)
        print(f"[lvd] prompt {pi+1}/{len(prompts)} cycles={len(cycle_feats)} rows={len(rows)}", flush=True)

    unload_model(draft)
    unload_model(target)
    return {"n_rows": len(rows), "n_prompts": n_prompts, "gamma": gamma, "n_layers": n_layers,
            "sample_layers": verifier.sample_layers}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--draft", default="/root/autodl-tmp/specreason/models/DeepSeek-R1-Distill-Qwen-1.5B")
    p.add_argument("--target", default="/root/autodl-tmp/specreason/models/DeepSeek-R1-Distill-Qwen-32B")
    p.add_argument("--data", default="/root/autodl-tmp/reasonbranch/data/gsm8k_test.jsonl")
    p.add_argument("--out", default="/root/autodl-tmp/reasonbranch/outputs/lvd_cycles.jsonl")
    p.add_argument("--n-prompts", type=int, default=16)
    p.add_argument("--gamma", type=int, default=8)
    p.add_argument("--max-cycles", type=int, default=16)
    args = p.parse_args()
    meta = collect(
        draft_path=args.draft, target_path=args.target, data_path=args.data,
        out_path=Path(args.out), n_prompts=args.n_prompts, gamma=args.gamma,
        max_cycles=args.max_cycles,
    )
    Path(args.out).with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()

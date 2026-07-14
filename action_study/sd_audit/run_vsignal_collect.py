"""Unified verification-signal collector (per-token) for SD2/SD4 re-analysis.

For each draft token in a linear greedy-SD run, dump ONE row with:
  - draft self-confidence: draft_logprob(of its own token), draft_entropy, draft_margin, draft_top1_prob
  - target verdict: accepted (draft==target argmax), target_final_margin (y vs d, or top1-top2 if accepted)
  - layerwise trajectory (rejected tokens): flip_depth (layer), flip_count, path_speed, deltas
  - residual (for SD2): target_logit(top) - draft_logit(same id) summary
  - context: prompt_idx, cycle, pos_in_cycle, gen_index (global decode position)

Design goal: enough per-token signal to test, offline and cheaply, the REAL
hypotheses that my first crude audits skipped:
  SD4: are draft-confident-but-rejected tokens ("hard negatives") resolved LATE
       by the target, i.e. does resolution depth separate hard vs easy negatives
       BEYOND the final-layer margin?
  SD2: is the per-(target-argmax-token) residual stable within a request?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from reasoning_branch_dataset.action_study.sd_audit.harness import (
    build_prompt,
    draft_greedy_tokens,
    load_gsm8k_prompts,
    load_hf_model,
    unload_model,
)
from reasoning_branch_dataset.action_study.sd_audit.layerwise_trajectory import (
    LayerwiseVerifier,
)


@torch.inference_mode()
def _draft_span_logits(draft, full_ids: list[int], positions: list[int]) -> torch.Tensor:
    inp = torch.tensor([full_ids], device=draft.device, dtype=torch.long)
    logits = draft.model(inp).logits[0]
    return torch.stack([logits[p - 1] for p in positions])


def _entropy(logits: torch.Tensor, topk: int = 64) -> float:
    vals = torch.topk(logits.float(), k=min(topk, logits.numel())).values
    p = torch.softmax(vals, dim=-1)
    return float((-p * (p + 1e-12).log()).sum().item())


def _margin(logits: torch.Tensor) -> float:
    t = torch.topk(logits.float(), k=2).values
    return float((t[0] - t[1]).item())


def collect(
    *,
    draft_path: str,
    target_path: str,
    data_path: str,
    out_path: Path,
    n_prompts: int,
    gamma: int,
    max_cycles: int,
    seed: int,
) -> dict[str, Any]:
    prompts = [build_prompt(q) for q in load_gsm8k_prompts(Path(data_path), n_prompts, seed)]
    draft = load_hf_model(draft_path)
    target = load_hf_model(target_path)
    verifier = LayerwiseVerifier(target.model, device=target.device)
    n_layers = verifier.n_layers
    Wt = target.model.lm_head.weight  # (V,H)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    n_rows = 0

    for pi, prompt in enumerate(prompts):
        prefix = draft.encode(prompt)
        gen_index = 0
        for cyc in range(max_cycles):
            draft_ids = draft_greedy_tokens(draft, prefix, gamma)
            if not draft_ids:
                break
            trajs = verifier.trajectories_for_draft(prefix, draft_ids)
            full = prefix + draft_ids
            positions = list(range(len(prefix), len(full)))
            d_logits = _draft_span_logits(draft, full, positions)

            rows = []
            accept_len = 0
            counting = True
            for j, t in enumerate(trajs):
                dl = d_logits[j].float()
                d_id = int(draft_ids[j])
                d_lp = float(F.log_softmax(dl, dim=-1)[d_id].item())
                d_top1p = float(torch.softmax(dl, dim=-1).max().item())
                y_id = t.target_id
                row = {
                    "prompt_idx": pi, "cycle": cyc, "pos_in_cycle": j,
                    "gen_index": gen_index + j, "gamma": gamma,
                    "draft_id": d_id, "target_id": y_id,
                    "accepted": bool(t.accepted),
                    "draft_logprob_self": d_lp,
                    "draft_top1_prob": d_top1p,
                    "draft_entropy": _entropy(dl),
                    "draft_margin": _margin(dl),
                    "target_final_margin": t.final_margin,
                    "flip_depth": t.flip_depth if t.flip_depth is not None else n_layers,
                    "flip_depth_none": t.flip_depth is None,
                    "flip_count": t.flip_count,
                    "path_speed": t.path_speed,
                    "deltas": [round(x, 4) for x in t.deltas],
                    "layer_ids": t.layer_ids,
                    "dec_match": t.dec_match,
                    "dec_depth": t.dec_depth if t.dec_depth is not None else n_layers,
                    "dec_depth_none": t.dec_depth is None,
                    "top1_minus_d": [round(x, 4) for x in t.top1_minus_d],
                }
                rows.append(row)
                if counting and t.accepted:
                    accept_len += 1
                elif counting:
                    counting = False

            with out_path.open("a", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    n_rows += 1

            # advance prefix (greedy SD): accept prefix, bonus/replace at reject
            inp = torch.tensor([full], device=target.device, dtype=torch.long)
            t_logits_full = target.model(inp).logits[0]
            if accept_len <= 0:
                y = int(t_logits_full[len(prefix) - 1].argmax().item())
                prefix = prefix + [y]
                gen_index += 1
            else:
                prefix = prefix + draft_ids[:accept_len]
                gen_index += accept_len
                if accept_len < len(draft_ids):
                    y = int(t_logits_full[len(prefix) - 1].argmax().item())
                    prefix = prefix + [y]
                    gen_index += 1
        print(f"[vsig] prompt {pi+1}/{len(prompts)} rows={n_rows}", flush=True)

    unload_model(draft)
    unload_model(target)
    meta = {"n_rows": n_rows, "n_prompts": n_prompts, "gamma": gamma,
            "n_layers": n_layers, "sample_layers": verifier.sample_layers}
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--draft", default="/root/autodl-tmp/specreason/models/DeepSeek-R1-Distill-Qwen-1.5B")
    p.add_argument("--target", default="/root/autodl-tmp/specreason/models/DeepSeek-R1-Distill-Qwen-32B")
    p.add_argument("--data", default="/root/autodl-tmp/reasonbranch/data/gsm8k_test.jsonl")
    p.add_argument("--out", default="/root/autodl-tmp/reasonbranch/outputs/vsignal/tokens.jsonl")
    p.add_argument("--n-prompts", type=int, default=60)
    p.add_argument("--gamma", type=int, default=8)
    p.add_argument("--max-cycles", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    meta = collect(
        draft_path=args.draft, target_path=args.target, data_path=args.data,
        out_path=Path(args.out), n_prompts=args.n_prompts, gamma=args.gamma,
        max_cycles=args.max_cycles, seed=args.seed,
    )
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Collect per-token layerwise GEOMETRY of the residual stream (target model),
to test the 'semantic fixed point' early-exit idea (ICML'26 oral) on our 32B
target, specifically in the speculative-verification setting.

For each draft token position we record, across ALL layers l:
  - upd_norm[l]      = ||x_l - x_{l-1}||            (raw residual-stream update)
  - upd_norm_rel[l]  = ||x_l - x_{l-1}|| / ||x_{l-1}||   (normalized update)
  - upd_cos[l]       = cos(Δx_l, Δx_{l-1})          (direction stability)
plus `accepted` and keys (prompt_idx, cycle, pos_in_cycle) so rows can be JOINED
with tokens.jsonl (same seed/prompts/gamma) for logit-lens dec_depth comparison.

Depth early-exit (skipping upper layers) saves upper-layer WEIGHT LOADING, hence
is a real win even in the memory-bound single-request regime — unlike skipping
tokens within a pass, which we already showed is free/useless.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from reasoning_branch_dataset.action_study.sd_audit.harness import (
    build_prompt,
    draft_greedy_tokens,
    load_gsm8k_prompts,
    load_hf_model,
    unload_model,
)


@torch.inference_mode()
def _target_forward_hidden(target, full_ids: list[int]):
    inp = torch.tensor([full_ids], device=target.device, dtype=torch.long)
    out = target.model(input_ids=inp, output_hidden_states=True)
    return out.logits[0], out.hidden_states  # hidden_states: tuple len (L+1), each (1,seq,H)


@torch.inference_mode()
def _selfpred_match(target, hidden_states, pos: int, y_id: int, sample_layers: list[int]) -> list[int]:
    """At sampled layers, logit-lens (final-norm + lm_head) argmax; 1 if == final
    argmax y_id (plain-decode prediction stabilized), else 0."""
    backbone = getattr(target.model, "model", None) or getattr(target.model, "transformer", None)
    final_norm = getattr(backbone, "norm", None) or getattr(backbone, "final_layernorm", None)
    W = target.model.lm_head.weight
    out = []
    for li in sample_layers:
        h = hidden_states[li + 1][0, pos - 1]  # after layer li
        h_n = final_norm(h.unsqueeze(0)).squeeze(0) if final_norm is not None else h
        am = int((h_n.to(W.dtype) @ W.T).argmax().item())
        out.append(1 if am == y_id else 0)
    return out


def _geom_for_position(hidden_states, pos: int) -> dict[str, list[float]]:
    # residual stream at this position across layers: x_0..x_L
    xs = [hs[0, pos - 1].float() for hs in hidden_states]  # each (H,)
    deltas = [xs[l] - xs[l - 1] for l in range(1, len(xs))]  # Δx_1..Δx_L
    upd_norm = [float(d.norm().item()) for d in deltas]
    upd_norm_rel = [float((deltas[i].norm() / (xs[i].norm() + 1e-6)).item()) for i in range(len(deltas))]
    upd_cos = [1.0]  # first delta has no predecessor; set neutral
    for i in range(1, len(deltas)):
        a, b = deltas[i], deltas[i - 1]
        c = float((torch.dot(a, b) / (a.norm() * b.norm() + 1e-9)).item())
        upd_cos.append(c)
    return {"upd_norm": [round(x, 4) for x in upd_norm],
            "upd_norm_rel": [round(x, 5) for x in upd_norm_rel],
            "upd_cos": [round(x, 4) for x in upd_cos]}


def collect(*, draft_path, target_path, data_path, out_path: Path, n_prompts, gamma, max_cycles, seed):
    prompts = [build_prompt(q) for q in load_gsm8k_prompts(Path(data_path), n_prompts, seed)]
    draft = load_hf_model(draft_path)
    target = load_hf_model(target_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    n_rows = 0
    for pi, prompt in enumerate(prompts):
        prefix = draft.encode(prompt)
        for cyc in range(max_cycles):
            draft_ids = draft_greedy_tokens(draft, prefix, gamma)
            if not draft_ids:
                break
            full = prefix + draft_ids
            t_logits, hidden_states = _target_forward_hidden(target, full)
            base = len(prefix)
            n_layers_local = len(hidden_states) - 1
            self_layers = list(range(0, n_layers_local, 4))  # every 4th layer
            rows = []
            accept_len = 0
            counting = True
            for j, d_id in enumerate(draft_ids):
                pos = base + j
                y_id = int(t_logits[pos - 1].argmax().item())
                accepted = y_id == int(d_id)
                g = _geom_for_position(hidden_states, pos)
                selfpred = _selfpred_match(target, hidden_states, pos, y_id, self_layers)
                row = {"prompt_idx": pi, "cycle": cyc, "pos_in_cycle": j,
                       "accepted": accepted, "self_layers": self_layers,
                       "selfpred_match": selfpred, **g}
                rows.append(row)
                if counting and accepted:
                    accept_len += 1
                elif counting:
                    counting = False
            with out_path.open("a", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    n_rows += 1
            # advance prefix (same greedy SD as vsignal collector -> identical tokens)
            if accept_len <= 0:
                prefix = prefix + [int(t_logits[base - 1].argmax().item())]
            else:
                prefix = prefix + draft_ids[:accept_len]
                if accept_len < len(draft_ids):
                    prefix = prefix + [int(t_logits[base - 1 + accept_len].argmax().item())]
        print(f"[geom] prompt {pi+1}/{len(prompts)} rows={n_rows}", flush=True)
    n_layers = len(hidden_states) - 1
    unload_model(draft)
    unload_model(target)
    meta = {"n_rows": n_rows, "n_prompts": n_prompts, "gamma": gamma, "n_layers": n_layers}
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--draft", default="/root/autodl-tmp/specreason/models/DeepSeek-R1-Distill-Qwen-1.5B")
    p.add_argument("--target", default="/root/autodl-tmp/specreason/models/DeepSeek-R1-Distill-Qwen-32B")
    p.add_argument("--data", default="/root/autodl-tmp/reasonbranch/data/gsm8k_test.jsonl")
    p.add_argument("--out", default="/root/autodl-tmp/reasonbranch/outputs/vsignal/geom.jsonl")
    p.add_argument("--n-prompts", type=int, default=60)
    p.add_argument("--gamma", type=int, default=8)
    p.add_argument("--max-cycles", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    meta = collect(draft_path=args.draft, target_path=args.target, data_path=args.data,
                   out_path=Path(args.out), n_prompts=args.n_prompts, gamma=args.gamma,
                   max_cycles=args.max_cycles, seed=args.seed)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()

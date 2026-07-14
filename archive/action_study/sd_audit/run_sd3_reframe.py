"""SD3 (Certified Candidate-Only Projection) — reframed re-audit.

v1 found rho_head = 1.9% on the 32B target (single request) -> KILL. The claim to
re-test: does the LM-head become a bottleneck in a LARGE-BATCH serving regime
(many concurrent verifications projected together)?

Physics: for a forward over N tokens, body ~ N * layers * (attn+mlp), head ~ N*H*V.
Both scale linearly in N (= batch*seq), so rho_head should be ~batch-invariant.
This script measures it directly by sweeping batch size, for target and drafter.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from reasoning_branch_dataset.action_study.sd_audit.harness import (
    build_prompt,
    load_gsm8k_prompts,
    load_hf_model,
    unload_model,
)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.inference_mode()
def profile_batched(bundle, token_ids: list[int], batch: int, n_repeat: int = 8) -> dict[str, float]:
    model = bundle.model
    backbone = getattr(model, "model", None) or getattr(model, "transformer", None)
    lm_head = model.lm_head
    inp = torch.tensor([token_ids] * batch, device=bundle.device, dtype=torch.long)

    hidden_store: list[torch.Tensor] = []

    def hook(_m, _i, out):
        h = getattr(out, "last_hidden_state", None)
        if h is None:
            h = out[0] if isinstance(out, (tuple, list)) else out
        hidden_store.append(h)

    h_handle = backbone.register_forward_hook(hook)
    try:
        for _ in range(2):
            hidden_store.clear()
            _ = model(input_ids=inp)
        _sync()
        # full forward
        t0 = time.perf_counter()
        for _ in range(n_repeat):
            _ = model(input_ids=inp)
        _sync()
        full_ms = (time.perf_counter() - t0) / n_repeat * 1000
        # isolated lm_head over all positions
        h = hidden_store[-1]
        _sync()
        t1 = time.perf_counter()
        for _ in range(n_repeat):
            _ = lm_head(h)
        _sync()
        lm_ms = (time.perf_counter() - t1) / n_repeat * 1000
        return {
            "batch": batch, "seq": len(token_ids), "n_tokens": batch * len(token_ids),
            "full_ms": full_ms, "lm_head_ms": lm_ms,
            "rho_head": lm_ms / full_ms if full_ms > 0 else 0.0,
        }
    finally:
        h_handle.remove()


def run(*, draft_path, target_path, data_path, out_path: Path, seq_len, batches, seed) -> dict[str, Any]:
    prompts = [build_prompt(q) for q in load_gsm8k_prompts(Path(data_path), 1, seed)]
    ids = None
    res: dict[str, Any] = {"seq_len": seq_len, "batches": batches}

    target = load_hf_model(target_path)
    ids = target.encode(prompts[0])
    if len(ids) < seq_len:
        ids = (ids * (seq_len // len(ids) + 1))
    ids = ids[:seq_len]
    tgt = []
    for b in batches:
        try:
            tgt.append(profile_batched(target, ids, b))
            print(f"[sd3][target] batch={b} rho_head={tgt[-1]['rho_head']:.4f} full={tgt[-1]['full_ms']:.1f}ms", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"[sd3][target] batch={b} OOM", flush=True)
            torch.cuda.empty_cache()
            break
    res["target"] = {"vocab": target.model.config.vocab_size, "hidden": target.model.config.hidden_size, "profiles": tgt}
    unload_model(target)

    draft = load_hf_model(draft_path)
    dft = []
    for b in batches:
        try:
            dft.append(profile_batched(draft, ids, b))
            print(f"[sd3][draft]  batch={b} rho_head={dft[-1]['rho_head']:.4f} full={dft[-1]['full_ms']:.1f}ms", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"[sd3][draft]  batch={b} OOM", flush=True)
            torch.cuda.empty_cache()
            break
    res["draft"] = {"vocab": draft.model.config.vocab_size, "hidden": draft.model.config.hidden_size, "profiles": dft}
    unload_model(draft)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2))
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--draft", default="/root/autodl-tmp/specreason/models/DeepSeek-R1-Distill-Qwen-1.5B")
    p.add_argument("--target", default="/root/autodl-tmp/specreason/models/DeepSeek-R1-Distill-Qwen-32B")
    p.add_argument("--data", default="/root/autodl-tmp/reasonbranch/data/gsm8k_test.jsonl")
    p.add_argument("--out", default="/root/autodl-tmp/reasonbranch/outputs/vsignal/sd3_reframe.json")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16, 64, 128])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    res = run(draft_path=args.draft, target_path=args.target, data_path=args.data,
              out_path=Path(args.out), seq_len=args.seq_len, batches=args.batches, seed=args.seed)
    print(json.dumps({"target": res["target"]["profiles"], "draft": res["draft"]["profiles"]}, indent=2))


if __name__ == "__main__":
    main()

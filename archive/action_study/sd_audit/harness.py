"""Linear speculative decoding audit harness (B→A→C).

Shared utilities for three verification-centric SD audits on a simple
linear (non-tree) block schedule with gamma in {4, 8, 16}.
"""

from __future__ import annotations

import gc
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def load_gsm8k_prompts(path: Path, n: int, seed: int) -> list[str]:
    rows = _load_jsonl(path)
    qs = [r["question"] for r in rows if r.get("question")]
    rng = random.Random(seed)
    if len(qs) > n:
        qs = rng.sample(qs, n)
    return qs


def build_prompt(question: str) -> str:
    return (
        "Solve the following math problem efficiently and clearly. "
        "Please reason step by step, and put your final answer within \\boxed{}.\n"
        f"Problem: {question}"
    )


@dataclass
class ModelBundle:
    name: str
    model: Any
    tokenizer: Any
    device: str = "cuda"

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    @torch.inference_mode()
    def logits_at_positions(self, token_ids: list[int], positions: list[int]) -> torch.Tensor:
        """Return logits[pos] predicting token at pos+1 style: logits[i] predicts ids[i+1]."""
        if not token_ids:
            return torch.empty(0, self.model.config.vocab_size, device=self.device)
        inp = torch.tensor([token_ids], device=self.device, dtype=torch.long)
        out = self.model(input_ids=inp)
        logits = out.logits[0]  # (seq, vocab)
        idx = [p - 1 for p in positions if p > 0]
        if not idx:
            return torch.empty(0, logits.shape[-1], device=self.device)
        return logits[idx]

    @torch.inference_mode()
    def forward_time(self, token_ids: list[int], *, n_warmup: int = 1, n_repeat: int = 5) -> float:
        if not token_ids:
            return 0.0
        inp = torch.tensor([token_ids], device=self.device, dtype=torch.long)
        for _ in range(n_warmup):
            _ = self.model(input_ids=inp)
        _sync()
        t0 = time.perf_counter()
        for _ in range(n_repeat):
            _ = self.model(input_ids=inp)
        _sync()
        return (time.perf_counter() - t0) / n_repeat

    @torch.inference_mode()
    def profile_lm_head(self, token_ids: list[int], *, n_repeat: int = 10) -> dict[str, float]:
        """Time transformer body (hidden) vs lm_head on last-position hidden states."""
        if not token_ids:
            return {"total_ms": 0.0, "body_ms": 0.0, "lm_head_ms": 0.0, "rho_head": 0.0}
        inp = torch.tensor([token_ids], device=self.device, dtype=torch.long)
        model = self.model
        # HF causal LM: model.model -> hidden, lm_head projection
        backbone = getattr(model, "model", None) or getattr(model, "transformer", None)
        lm_head = getattr(model, "lm_head", None)
        if backbone is None or lm_head is None:
            total = self.forward_time(token_ids, n_warmup=2, n_repeat=n_repeat) * 1000
            return {"total_ms": total, "body_ms": total, "lm_head_ms": 0.0, "rho_head": 0.0}

        hidden_store: list[torch.Tensor] = []

        def hook(_mod, _inp, out):
            hidden_store.append(out[0] if isinstance(out, tuple) else out)

        handle = backbone.register_forward_hook(hook)
        try:
            for _ in range(2):
                hidden_store.clear()
                _ = model(input_ids=inp)
            _sync()

            # body only (hook captures backbone output)
            hidden_store.clear()
            t0 = time.perf_counter()
            for _ in range(n_repeat):
                hidden_store.clear()
                _ = model(input_ids=inp)
            _sync()
            total_ms = (time.perf_counter() - t0) / n_repeat * 1000

            h = hidden_store[-1][0]  # (seq, hidden)
            # lm_head on all positions (as in verify pass)
            _sync()
            t1 = time.perf_counter()
            for _ in range(n_repeat):
                _ = lm_head(h)
            _sync()
            lm_ms = (time.perf_counter() - t1) / n_repeat * 1000

            # approximate body = total - lm (last full forward minus isolated lm_head)
            _sync()
            t2 = time.perf_counter()
            for _ in range(n_repeat):
                _ = model(input_ids=inp)
            _sync()
            full_ms = (time.perf_counter() - t2) / n_repeat * 1000
            body_ms = max(0.0, full_ms - lm_ms)
            rho = lm_ms / full_ms if full_ms > 0 else 0.0
            return {
                "total_ms": full_ms,
                "body_ms": body_ms,
                "lm_head_ms": lm_ms,
                "rho_head": rho,
                "seq_len": len(token_ids),
            }
        finally:
            handle.remove()


def load_hf_model(path: str, *, device: str = "cuda", dtype: str = "bfloat16") -> ModelBundle:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    torch_dtype = getattr(torch, dtype)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch_dtype, trust_remote_code=True, device_map=device
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch_dtype, trust_remote_code=True)
        model = model.to(device)
    model.eval()
    return ModelBundle(name=Path(path).name, model=model, tokenizer=tok, device=device)


def unload_model(bundle: ModelBundle | None) -> None:
    if bundle is None:
        return
    del bundle.model
    del bundle.tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def draft_greedy_tokens(draft: ModelBundle, prefix_ids: list[int], gamma: int) -> list[int]:
    """Autoregressively draft gamma greedy tokens from prefix."""
    ids = list(prefix_ids)
    for _ in range(gamma):
        if not ids:
            break
        inp = torch.tensor([ids], device=draft.device, dtype=torch.long)
        with torch.inference_mode():
            logits = draft.model(inp).logits[0, -1]
        next_id = int(logits.argmax().item())
        ids.append(next_id)
    return ids[len(prefix_ids):]


def greedy_accept_length(
    target: ModelBundle, prefix_ids: list[int], draft_ids: list[int]
) -> int:
    if not draft_ids:
        return 0
    full = prefix_ids + draft_ids
    inp = torch.tensor([full], device=target.device, dtype=torch.long)
    with torch.inference_mode():
        logits = target.model(inp).logits[0]
    base = len(prefix_ids)
    acc = 0
    for j, dt in enumerate(draft_ids):
        greedy = int(logits[base - 1 + j].argmax().item())
        if greedy == dt:
            acc += 1
        else:
            break
    return acc


def estimate_accept_boundary(approx: ModelBundle, prefix_ids: list[int], draft_ids: list[int]) -> int:
    """Longest prefix where draft token matches approx-model greedy (budget predictor)."""
    if not draft_ids:
        return 0
    full = prefix_ids + draft_ids
    inp = torch.tensor([full], device=approx.device, dtype=torch.long)
    with torch.inference_mode():
        logits = approx.model(inp).logits[0]
    base = len(prefix_ids)
    r = 0
    for j, dt in enumerate(draft_ids):
        greedy = int(logits[base - 1 + j].argmax().item())
        if greedy == dt:
            r = j + 1
        else:
            break
    return r


def align_logits(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Truncate to shared vocab size (draft/target tokenizers may differ slightly)."""
    n = min(a.numel(), b.numel())
    return a[:n], b[:n]


def kl_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor, topk: int = 512) -> float:
    """KL(p||q) using top-k support from p for stability."""
    p, q = align_logits(p_logits.float(), q_logits.float())
    k = min(topk, p.numel())
    vals, idx = torch.topk(p, k)
    p_sub = torch.softmax(vals, dim=-1)
    q_sub = torch.log_softmax(q[idx], dim=-1)
    return float(torch.sum(p_sub * (torch.log(p_sub + 1e-12) - q_sub)).item())


@dataclass
class ResidualRecord:
    cycle_idx: int
    pos_in_cycle: int
    draft_logits: torch.Tensor
    target_logits: torch.Tensor

    @property
    def top1_match_draft(self) -> bool:
        t, d = align_logits(self.target_logits, self.draft_logits)
        return int(d.argmax()) == int(t.argmax())

    def top1_match_calibrated(self, bias: torch.Tensor) -> bool:
        t, d = align_logits(self.target_logits, self.draft_logits)
        b = bias[: d.numel()]
        return int((d + b).argmax()) == int(t.argmax())


def collect_sd_cycles(
    draft: ModelBundle,
    target: ModelBundle,
    prompt: str,
    *,
    gamma: int,
    max_cycles: int,
) -> tuple[list[int], list[ResidualRecord]]:
    """Run linear greedy SD; collect per-position draft/target logits at verify steps."""
    prefix_ids = draft.encode(prompt)
    records: list[ResidualRecord] = []
    generated: list[int] = []

    for cycle in range(max_cycles):
        draft_ids = draft_greedy_tokens(draft, prefix_ids, gamma)
        if not draft_ids:
            break
        full = prefix_ids + draft_ids
        positions = list(range(len(prefix_ids), len(full)))
        d_logits = draft.logits_at_positions(full, positions)
        t_logits = target.logits_at_positions(full, positions)
        for j in range(len(draft_ids)):
            records.append(
                ResidualRecord(cycle, j, d_logits[j].clone(), t_logits[j].clone())
            )
        acc = greedy_accept_length(target, prefix_ids, draft_ids)
        if acc <= 0:
            # accept target greedy at reject + append one target token
            inp = torch.tensor([prefix_ids + draft_ids[:1]], device=target.device, dtype=torch.long)
            with torch.inference_mode():
                logits = target.model(inp).logits[0, len(prefix_ids) - 1]
            prefix_ids = prefix_ids + [int(logits.argmax().item())]
            generated.append(prefix_ids[-1])
        else:
            prefix_ids = prefix_ids + draft_ids[:acc]
            generated.extend(draft_ids[:acc])
            if acc < len(draft_ids):
                # bonus token from target at first reject
                inp = torch.tensor([prefix_ids], device=target.device, dtype=torch.long)
                with torch.inference_mode():
                    logits = target.model(inp).logits[0, -1]
                bonus = int(logits.argmax().item())
                prefix_ids.append(bonus)
                generated.append(bonus)
        if len(generated) >= max_cycles * gamma:
            break
    return generated, records

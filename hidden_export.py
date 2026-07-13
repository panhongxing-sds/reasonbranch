"""Export prefix hidden states to safetensors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file


def _resolve_text_model(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    if hasattr(model, "language_model"):
        lm = model.language_model
        if hasattr(lm, "model"):
            return lm.model
        return lm
    return model


def _get_hidden_states(model, input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    text_model = _resolve_text_model(model)
    out = text_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    # tuple: (embed, layer1, ..., layerN); each entry is [batch, seq, hidden]
    stacked = torch.stack(out.hidden_states[1:], dim=0)
    if stacked.dim() == 4 and stacked.shape[1] == 1:
        stacked = stacked.squeeze(1)
    return stacked


def _pool_hidden(
    hidden_layers: torch.Tensor,
    token_index: int,
    step_start: int,
    layer_indices: tuple[int, ...],
) -> dict[str, dict[str, torch.Tensor]]:
    """
    hidden_layers: [num_layers, seq_len, hidden_dim]
    layer_indices are 1-based transformer block indices.
    """
    seq_len = hidden_layers.shape[1]
    token_index = min(max(token_index, 0), seq_len - 1)
    step_start = min(max(step_start, 0), seq_len - 1)

    local_start = max(0, token_index - 3)
    local_slice = hidden_layers[:, local_start : token_index + 1].mean(dim=1)
    last_vec = hidden_layers[:, token_index]
    step_slice = hidden_layers[:, step_start : token_index + 1].mean(dim=1)

    out: dict[str, dict[str, torch.Tensor]] = {}
    num_layers = hidden_layers.shape[0]
    for layer_idx in layer_indices:
        li = min(max(layer_idx - 1, 0), num_layers - 1)
        out[str(layer_idx)] = {
            "last": last_vec[li].detach().cpu().to(torch.float16),
            "step_mean": step_slice[li].detach().cpu().to(torch.float16),
            "local4": local_slice[li].detach().cpu().to(torch.float16),
        }
    return out


@torch.no_grad()
def extract_prefix_hidden(
    model,
    tokenizer,
    prefix_text: str,
    token_index: int,
    step_index: int,
    layer_indices: tuple[int, ...],
) -> dict[str, dict[str, torch.Tensor]]:
    inputs = tokenizer(prefix_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    hidden_layers = _get_hidden_states(model, input_ids, attention_mask)
    prompt_len = input_ids.shape[1]
    # token_index is relative to generated reasoning; map to absolute position in prefix
    abs_token_index = prompt_len - 1
    if token_index >= 0:
        abs_token_index = min(prompt_len - 1, max(0, prompt_len - (1 + max(0, 0))))

    # Re-encode reasoning-only to align token_index with generation stream
    # prefix_text already includes prompt+generated prefix; last token is cut point.
    abs_token_index = input_ids.shape[1] - 1

    # step boundary: approximate via counting \n\n in prefix tail
    reasoning_part = prefix_text
    step_start_char = 0
    parts = reasoning_part.split("\n\n")
    if step_index > 0 and len(parts) > step_index:
        step_start_char = len("\n\n".join(parts[:step_index])) + (2 if step_index > 0 else 0)
    step_prefix = prefix_text[:step_start_char] if step_start_char > 0 else prefix_text[: max(0, len(prefix_text) - 1)]
    step_inputs = tokenizer(step_prefix, return_tensors="pt")
    step_start = max(0, step_inputs["input_ids"].shape[1] - 1)

    return _pool_hidden(hidden_layers, abs_token_index, step_start, layer_indices)


def save_hidden_batch(
    hidden_store: dict[str, torch.Tensor],
    prefix_id: str,
    source: str,
    pooled: dict[str, dict[str, torch.Tensor]],
) -> None:
    for layer, pools in pooled.items():
        for pool_name, vec in pools.items():
            key = f"{prefix_id}/{source}/layer{layer}/{pool_name}"
            hidden_store[key] = vec


def flush_hidden_store(path, hidden_store: dict[str, torch.Tensor]) -> None:
    if not hidden_store:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = dict(hidden_store)
    if path.exists():
        from safetensors.torch import load_file

        merged = {**load_file(str(path)), **merged}
    save_file(merged, str(path))

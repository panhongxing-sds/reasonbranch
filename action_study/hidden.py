"""Draft-only hidden + logits export for action study (HF lazy load)."""

from __future__ import annotations

from typing import Any

import torch

from reasoning_branch_dataset.hidden_export import extract_prefix_hidden, flush_hidden_store, save_hidden_batch
from reasoning_branch_dataset.action_study.logits_export import extract_prefix_logits


def _resolve_layers(model, layers: tuple[int, ...]) -> tuple[int, ...]:
    text_model = model
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        n = len(model.model.layers)
    elif hasattr(model, "language_model"):
        lm = model.language_model
        inner = lm.model if hasattr(lm, "model") else lm
        n = len(inner.layers) if hasattr(inner, "layers") else 32
    else:
        n = 32
    out: list[int] = []
    for li in layers:
        if li < 0:
            out.append(n + li + 1)
        else:
            out.append(li)
    return tuple(out)


class DraftHiddenExporter:
    def __init__(
        self,
        model_path: str,
        layers: tuple[int, ...],
        *,
        device: str = "cuda",
        dtype: str = "bfloat16",
        topk_logits: int = 10,
    ):
        self.model_path = model_path
        self.layers = layers
        self.device = device
        self.dtype = dtype
        self.topk_logits = topk_logits
        self._model = None
        self._tokenizer = None
        self._resolved_layers: tuple[int, ...] | None = None
        self.store: dict[str, torch.Tensor] = {}

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from reasoning_branch_dataset.model_utils import load_model_and_tokenizer

            self._model, self._tokenizer = load_model_and_tokenizer(self.model_path, self.device, self.dtype)
            self._resolved_layers = _resolve_layers(self._model, self.layers)

    @torch.no_grad()
    def export(self, prefix_id: str, prefix_text: str, step_index: int) -> dict[str, Any]:
        self._ensure_loaded()
        assert self._resolved_layers is not None
        pooled = extract_prefix_hidden(
            self._model,
            self._tokenizer,
            prefix_text,
            token_index=-1,
            step_index=step_index,
            layer_indices=self._resolved_layers,
        )
        save_hidden_batch(self.store, prefix_id, "draft", pooled)
        logits = extract_prefix_logits(self._model, self._tokenizer, prefix_text, topk=self.topk_logits)
        return {"prefix_id": prefix_id, "layers": list(self._resolved_layers), **logits}

    def flush(self, path) -> None:
        flush_hidden_store(path, self.store)

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

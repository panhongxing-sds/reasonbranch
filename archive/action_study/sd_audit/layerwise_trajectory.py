"""Target Decision Resolution Depth / Layerwise Verification Trajectory.

For a draft token d_t vs target final token y_t at position t, sample intermediate
layers ell and compute:

    delta_t^(ell) = <LN(h_t^(ell)), e_{y_t} - e_{d_t}>

Then:
  - flip_depth  = first layer after which all subsequent deltas stay positive
                  (target stably prefers y over d); None if never stable
  - flip_count  = number of sign changes along the trajectory
  - path_speed  = sum |delta_ell - delta_{ell-1}|

Final-layer margin alone cannot distinguish early-resolve vs late-flip cases.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F


@dataclass
class LayerTrajectory:
    draft_id: int
    target_id: int
    accepted: bool
    layer_ids: list[int]
    deltas: list[float]
    flip_depth: int | None
    flip_count: int
    path_speed: float
    final_margin: float  # target logit(y) - logit(d) at last layer (via LM head)
    # Early-exit verification signal: at each sampled layer, does the logit-lens
    # full-vocab argmax equal the drafted token d? The verify decision is
    # "accept d" == (argmax == d). dec_depth = earliest sampled layer after which
    # this decision matches the final (full-depth) decision and stays matched.
    dec_match: list[bool] | None = None
    dec_depth: int | None = None
    # Per-layer logit-lens margin (best-token logit - drafted-token logit). This
    # is a graded early-reject signal defined for BOTH accept and reject: large &
    # early -> confident reject; decays to ~0 -> heading to accept.
    top1_minus_d: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_flip_depth(deltas: Sequence[float]) -> int | None:
    """Smallest ell such that delta[j] > 0 for all j >= ell. None if never holds."""
    n = len(deltas)
    for i in range(n):
        if all(deltas[j] > 0 for j in range(i, n)):
            return i
    return None


def compute_flip_count(deltas: Sequence[float]) -> int:
    if len(deltas) < 2:
        return 0
    cnt = 0
    for i in range(1, len(deltas)):
        a, b = deltas[i - 1], deltas[i]
        if a == 0 or b == 0:
            if (a > 0) != (b > 0) and not (a == 0 and b == 0):
                cnt += 1
        elif (a > 0) != (b > 0):
            cnt += 1
    return cnt


def compute_path_speed(deltas: Sequence[float]) -> float:
    if len(deltas) < 2:
        return 0.0
    return float(sum(abs(deltas[i] - deltas[i - 1]) for i in range(1, len(deltas))))


def default_sample_layers(n_layers: int, n_samples: int = 8) -> list[int]:
    """Evenly spaced layer indices including near-final."""
    if n_layers <= n_samples:
        return list(range(n_layers))
    # 1-based depth feel: sample across depth, always include last
    idxs = [int(round(i * (n_layers - 1) / (n_samples - 1))) for i in range(n_samples)]
    idxs[-1] = n_layers - 1
    return sorted(set(idxs))


class LayerwiseVerifier:
    """Hook selected transformer layers; compute draft-vs-target deltas via LM-head rows."""

    def __init__(
        self,
        model: Any,
        *,
        sample_layers: list[int] | None = None,
        device: str = "cuda",
    ) -> None:
        self.model = model
        self.device = device
        backbone = getattr(model, "model", None) or getattr(model, "transformer", None)
        if backbone is None or not hasattr(backbone, "layers"):
            raise RuntimeError("Expected HF CausalLM with .model.layers")
        self.backbone = backbone
        self.n_layers = len(backbone.layers)
        self.sample_layers = sample_layers or default_sample_layers(self.n_layers)
        self.lm_head = model.lm_head
        # Final norm (RMSNorm for Qwen/Llama). Needed for a faithful logit-lens:
        # projecting a raw decoder-layer hidden state without the model's own
        # final norm produces sign-inconsistent deltas (the last layer can even
        # disagree with the real final margin). We reuse the model's norm module.
        self.final_norm = getattr(backbone, "norm", None) or getattr(backbone, "final_layernorm", None)
        # Prefer tied embed if present
        embed = backbone.embed_tokens.weight
        self.embed_weight = embed  # (V, H)
        self._handles: list[Any] = []
        self._cache: dict[int, torch.Tensor] = {}

    def _clear(self) -> None:
        self._cache.clear()

    def _install_hooks(self) -> None:
        self._remove_hooks()
        self._clear()

        def make_hook(layer_idx: int):
            def hook(_mod, _inp, out):
                h = out[0] if isinstance(out, tuple) else out
                # keep only last-batch row tensor on GPU; detach
                self._cache[layer_idx] = h.detach()
            return hook

        for li in self.sample_layers:
            h = self.backbone.layers[li].register_forward_hook(make_hook(li))
            self._handles.append(h)

    def _remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    @torch.inference_mode()
    def trajectories_for_draft(
        self,
        prefix_ids: list[int],
        draft_ids: list[int],
    ) -> list[LayerTrajectory]:
        """One forward on prefix+draft; for each draft position compare d vs target greedy y."""
        if not draft_ids:
            return []
        self._install_hooks()
        try:
            full = prefix_ids + draft_ids
            inp = torch.tensor([full], device=self.device, dtype=torch.long)
            out = self.model(input_ids=inp)
            final_logits = out.logits[0]  # (seq, V)
            base = len(prefix_ids)
            # LM-head weight rows (or embed if tied)
            W = self.lm_head.weight  # (V, H)

            results: list[LayerTrajectory] = []
            for j, d_id in enumerate(draft_ids):
                pos = base + j  # token index of draft token in `full`
                # logits[pos-1] predicts token at pos
                pred_logits = final_logits[pos - 1]
                y_id = int(pred_logits.argmax().item())
                accepted = y_id == int(d_id)
                if accepted:
                    top2 = torch.topk(pred_logits, k=2).values
                    final_margin = float((top2[0] - top2[1]).item())
                else:
                    final_margin = float((pred_logits[y_id] - pred_logits[int(d_id)]).item())

                w_y = W[y_id].float()
                w_d = W[int(d_id)].float()
                deltas: list[float] = []
                dec_match: list[bool] = []
                top1_minus_d: list[float] = []
                for li in self.sample_layers:
                    h = self._cache[li][0, pos - 1]
                    # Faithful logit-lens: apply the model's own final norm, then
                    # project onto the y and d LM-head rows. At the last decoder
                    # layer this equals the true final logit margin.
                    if self.final_norm is not None:
                        h_n = self.final_norm(h.unsqueeze(0)).squeeze(0).float()
                    else:
                        h_n = F.layer_norm(h.float(), (h.shape[-1],))
                    deltas.append(float((torch.dot(h_n, w_y) - torch.dot(h_n, w_d)).item()))
                    # Full-vocab logit-lens at this layer
                    ll = h_n.to(W.dtype) @ W.T
                    ll_top1_val, ll_argmax = ll.max(dim=-1)
                    dec_match.append(int(ll_argmax.item()) == int(d_id))
                    top1_minus_d.append(float((ll_top1_val.float() - ll[int(d_id)].float()).item()))

                # decision stability depth: earliest layer after which dec_match
                # equals its final value (== accepted) and stays there.
                final_dec = dec_match[-1]
                dec_depth_idx = None
                for i in range(len(dec_match)):
                    if all(dec_match[j] == final_dec for j in range(i, len(dec_match))):
                        dec_depth_idx = i
                        break
                dec_depth = None if dec_depth_idx is None else self.sample_layers[dec_depth_idx]

                if accepted:
                    flip_depth, flip_count, path_speed = 0, 0, 0.0
                else:
                    flip_depth = compute_flip_depth(deltas)
                    flip_count = compute_flip_count(deltas)
                    path_speed = compute_path_speed(deltas)

                # map flip_depth index -> actual layer id
                flip_layer = None if flip_depth is None else self.sample_layers[flip_depth]

                results.append(
                    LayerTrajectory(
                        draft_id=int(d_id),
                        target_id=y_id,
                        accepted=accepted,
                        layer_ids=list(self.sample_layers),
                        deltas=deltas,
                        flip_depth=flip_layer,  # actual layer id, or None
                        flip_count=flip_count,
                        path_speed=path_speed,
                        final_margin=final_margin,
                        dec_match=dec_match,
                        dec_depth=dec_depth,
                        top1_minus_d=top1_minus_d,
                    )
                )
            return results
        finally:
            self._remove_hooks()
            self._clear()

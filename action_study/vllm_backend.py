"""vLLM generation backend for action-matching study."""

from __future__ import annotations

import math
import time
from typing import Any

from reasoning_branch_dataset.grading import grade_math_answer


def _topk_entropy(logprobs: list[float]) -> float:
    if not logprobs:
        return 0.0
    m = max(logprobs)
    probs = [math.exp(lp - m) for lp in logprobs]
    s = sum(probs)
    if s <= 0:
        return 0.0
    probs = [p / s for p in probs]
    return float(-sum(p * math.log(p + 1e-12) for p in probs))


class VLLMEngine:
    def __init__(
        self,
        model_path: str,
        *,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 8192,
        dtype: str = "bfloat16",
        enable_prefix_caching: bool = False,
        enforce_eager: bool = True,
    ) -> None:
        import os

        from vllm import LLM

        # Avoid slow/failing FlashInfer JIT on Qwen3.5 GDN layers.
        os.environ.setdefault("VLLM_GDN_PREFILL_BACKEND", "triton")
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        os.environ.pop("VLLM_USE_V1", None)

        llm_kwargs: dict[str, Any] = {
            "model": model_path,
            "dtype": dtype,
            "trust_remote_code": True,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "enforce_eager": enforce_eager,
            "enable_prefix_caching": enable_prefix_caching,
            "additional_config": {"gdn_prefill_backend": "triton"},
        }
        if "Qwen3.5" in model_path or "qwen3.5" in model_path.lower():
            llm_kwargs["language_model_only"] = True

        self.model_path = model_path
        self.llm = LLM(**llm_kwargs)

    def _sample(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        n: int = 1,
        stop: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        from vllm import SamplingParams

        params = SamplingParams(
            n=n,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop or [],
            logprobs=5,
        )
        t0 = time.perf_counter()
        outputs = self.llm.generate([prompt], params)
        latency = time.perf_counter() - t0
        comps = outputs[0].outputs
        rows = []
        for comp in comps:
            token_ids = list(comp.token_ids)
            text = comp.text
            lp_map = comp.logprobs[0] if comp.logprobs else {}
            top_lps = sorted((v.logprob for v in lp_map.values()), reverse=True)[:5] if lp_map else []
            rows.append(
                {
                    "text": text,
                    "token_ids": token_ids,
                    "num_tokens": len(token_ids),
                    "latency_sec": latency / max(n, 1),
                    "entropy": _topk_entropy(top_lps),
                    "top1_prob": float(math.exp(top_lps[0])) if top_lps else 0.0,
                    "margin": float(top_lps[0] - top_lps[1]) if len(top_lps) >= 2 else 0.0,
                    "finish_reason": getattr(comp, "finish_reason", None),
                }
            )
        return rows

    def _continue_if_truncated(
        self,
        prompt: str,
        row: dict[str, Any],
        *,
        extra_max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> dict[str, Any]:
        if row.get("finish_reason") != "length" or extra_max_tokens <= 0:
            return row
        extended = self._sample(
            prompt + row["text"],
            max_tokens=extra_max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=1,
        )[0]
        return {
            "text": row["text"] + extended["text"],
            "token_ids": row["token_ids"] + extended["token_ids"],
            "num_tokens": row["num_tokens"] + extended["num_tokens"],
            "latency_sec": row["latency_sec"] + extended["latency_sec"],
            "entropy": extended.get("entropy", row.get("entropy", 0.0)),
            "top1_prob": extended.get("top1_prob", row.get("top1_prob", 0.0)),
            "margin": extended.get("margin", row.get("margin", 0.0)),
            "finish_reason": extended.get("finish_reason"),
            "continuation_phases": 2,
        }

    def generate_trace(self, prompt: str, *, max_tokens: int) -> dict[str, Any]:
        rows = self._sample(prompt, max_tokens=max_tokens, temperature=0.0, top_p=1.0, n=1)
        row = rows[0]
        return {
            "response_text": row["text"],
            "token_ids": row["token_ids"],
            "num_tokens": row["num_tokens"],
            "latency_sec": row["latency_sec"],
            "finish_reason": row.get("finish_reason"),
        }

    def generate_next_steps(
        self,
        prefix: str,
        *,
        k: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[dict[str, Any]]:
        return self._sample(
            prefix,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=k,
            stop=["\n\n"],
        )

    def generate_full_continuations(
        self,
        prefix: str,
        *,
        k: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        retry_max_tokens: int = 0,
    ) -> list[dict[str, Any]]:
        rows = self._sample(
            prefix,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=k,
        )
        if retry_max_tokens <= 0:
            return rows
        return [
            self._continue_if_truncated(
                prefix,
                row,
                extra_max_tokens=retry_max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            for row in rows
        ]

    @staticmethod
    def score_answer(text: str, gold: str) -> dict[str, Any]:
        return grade_math_answer(text, gold)


class HFEngine:
    """Fallback HF engine when vLLM unavailable."""

    def __init__(self, model_path: str, device: str = "cuda", dtype: str = "bfloat16") -> None:
        from reasoning_branch_dataset.model_utils import load_model_and_tokenizer

        self.model, self.tokenizer = load_model_and_tokenizer(model_path, device, dtype)

    def generate_trace(self, prompt: str, *, max_tokens: int) -> dict[str, Any]:
        from reasoning_branch_dataset.model_utils import generate_with_trace
        import time

        t0 = time.perf_counter()
        out = generate_with_trace(self.model, self.tokenizer, prompt, max_new_tokens=max_tokens, device="cuda")
        finish_reason = "length" if len(out["token_ids"]) >= max_tokens else "stop"
        return {
            "response_text": out["response_text"],
            "token_ids": out["token_ids"],
            "num_tokens": len(out["token_ids"]),
            "latency_sec": time.perf_counter() - t0,
            "finish_reason": finish_reason,
        }

    def generate_next_steps(self, prefix: str, *, k: int, max_tokens: int, temperature: float, top_p: float):
        from reasoning_branch_dataset.model_utils import generate_continuation
        import time

        do_sample = temperature > 0
        rows = []
        for _ in range(k):
            t0 = time.perf_counter()
            out = generate_continuation(
                self.model,
                self.tokenizer,
                prefix,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 1e-5) if do_sample else 1.0,
                top_p=top_p,
                do_sample=do_sample,
                stop_at_paragraph=True,
            )
            rows.append(
                {
                    "text": out["continuation_text"],
                    "token_ids": out["continuation_token_ids"],
                    "num_tokens": len(out["continuation_token_ids"]),
                    "latency_sec": time.perf_counter() - t0,
                    "entropy": 0.0,
                    "top1_prob": 0.0,
                    "margin": 0.0,
                    "finish_reason": "length" if out.get("stop_reason") == "max_tokens" else "stop",
                }
            )
        return rows

    def generate_full_continuations(self, prefix: str, *, k: int, max_tokens: int, temperature: float, top_p: float, retry_max_tokens: int = 0):
        from reasoning_branch_dataset.model_utils import generate_continuation
        import time

        do_sample = temperature > 0
        rows = []
        for _ in range(k):
            t0 = time.perf_counter()
            out = generate_continuation(
                self.model,
                self.tokenizer,
                prefix,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 1e-5) if do_sample else 1.0,
                top_p=top_p,
                do_sample=do_sample,
                stop_at_paragraph=False,
            )
            row = {
                "text": out["continuation_text"],
                "token_ids": out["continuation_token_ids"],
                "num_tokens": len(out["continuation_token_ids"]),
                "latency_sec": time.perf_counter() - t0,
                "entropy": 0.0,
                "top1_prob": 0.0,
                "margin": 0.0,
                "finish_reason": "length" if out.get("stop_reason") == "max_tokens" else "stop",
            }
            if retry_max_tokens > 0 and row["finish_reason"] == "length":
                out2 = generate_continuation(
                    self.model,
                    self.tokenizer,
                    prefix + row["text"],
                    max_new_tokens=retry_max_tokens,
                    temperature=max(temperature, 1e-5) if do_sample else 1.0,
                    top_p=top_p,
                    do_sample=do_sample,
                    stop_at_paragraph=False,
                )
                row["text"] = row["text"] + out2["continuation_text"]
                row["token_ids"] = row["token_ids"] + (
                    out2["continuation_token_ids"]
                    if isinstance(out2["continuation_token_ids"], list)
                    else out2["continuation_token_ids"].tolist()
                )
                row["num_tokens"] = len(row["token_ids"])
                row["finish_reason"] = (
                    "length" if out2.get("stop_reason") == "max_tokens" else "stop"
                )
                row["continuation_phases"] = 2
            rows.append(row)
        return rows

    @staticmethod
    def score_answer(text: str, gold: str) -> dict[str, Any]:
        return VLLMEngine.score_answer(text, gold)


def build_engine(cfg) -> VLLMEngine | HFEngine:
    if cfg.engine == "vllm":
        try:
            return VLLMEngine(cfg.model_path, gpu_memory_utilization=getattr(cfg, "vllm_gpu_util", 0.94))
        except Exception as exc:
            print(f"WARN: vLLM init failed ({exc}), falling back to HF")
    return HFEngine(cfg.model_path)

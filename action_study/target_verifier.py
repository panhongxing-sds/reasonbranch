"""Target model backends for greedy acceptance replay (HF or vLLM)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from reasoning_branch_dataset.action_study.logprob_align import (
    token_accepted_greedy,
    vllm_greedy_token,
)
from reasoning_branch_dataset.action_study.target_acceptance import (
    AcceptanceResult,
    greedy_acceptance_length,
    greedy_acceptance_length_from_ids,
    tokenize_text,
    truncate_token_ids,
    verify_trace_acceptance,
)


class TargetVerifier(Protocol):
    def tokenize(self, text: str) -> list[int]: ...

    def greedy_acceptance(self, prefix_text: str, draft_text: str, *, gamma: int) -> AcceptanceResult: ...

    def greedy_acceptance_ids(self, prefix_ids: list[int], draft_ids: list[int]) -> AcceptanceResult: ...

    def verify_trace(self, prompt_text: str, trace_text: str) -> dict[str, Any]: ...


@dataclass
class HFTargetVerifier:
    model: Any
    tokenizer: Any
    device: str = "cuda"

    def tokenize(self, text: str) -> list[int]:
        return tokenize_text(self.tokenizer, text)

    def greedy_acceptance_ids(self, prefix_ids: list[int], draft_ids: list[int]) -> AcceptanceResult:
        return greedy_acceptance_length_from_ids(self.model, prefix_ids, draft_ids, device=self.device)

    def greedy_acceptance(self, prefix_text: str, draft_text: str, *, gamma: int) -> AcceptanceResult:
        prefix_ids = self.tokenize(prefix_text)
        draft_ids = truncate_token_ids(self.tokenize(draft_text), gamma)
        return self.greedy_acceptance_ids(prefix_ids, draft_ids)

    def verify_trace(self, prompt_text: str, trace_text: str) -> dict[str, Any]:
        return verify_trace_acceptance(
            self.model, self.tokenizer, prompt_text, trace_text, device=self.device
        )


@dataclass
class VLLMTargetVerifier:
    llm: Any
    tokenizer: Any
    prompt_logprobs: int = 20

    def tokenize(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def greedy_acceptance_ids(self, prefix_ids: list[int], draft_ids: list[int]) -> AcceptanceResult:
        from vllm import SamplingParams

        if not draft_ids:
            return AcceptanceResult(0, 0, 0.0, None, [])

        full_ids = prefix_ids + draft_ids
        params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            prompt_logprobs=self.prompt_logprobs,
            detokenize=False,
        )
        out = self.llm.generate([{"prompt_token_ids": full_ids}], params)[0]
        logprob_steps = out.prompt_logprobs or []

        accepted = 0
        first_reject = None
        matches: list[bool] = []
        base = len(prefix_ids)
        for j, draft_tok in enumerate(draft_ids):
            # vLLM: prompt_logprobs[i] scores input_ids[i]; HF logits[i-1] predicts input_ids[i]
            step_idx = base + j
            if step_idx >= len(logprob_steps) or logprob_steps[step_idx] is None:
                first_reject = j
                break
            greedy_tok = vllm_greedy_token(logprob_steps[step_idx])
            ok = token_accepted_greedy(draft_tok, greedy_tok)
            matches.append(ok)
            if ok:
                accepted += 1
            else:
                first_reject = j
                break

        draft_len = len(draft_ids)
        return AcceptanceResult(
            accepted_length=accepted,
            draft_length=draft_len,
            accept_ratio=accepted / draft_len,
            first_reject_position=first_reject,
            greedy_matches=matches,
        )

    def greedy_acceptance(self, prefix_text: str, draft_text: str, *, gamma: int) -> AcceptanceResult:
        prefix_ids = self.tokenize(prefix_text)
        draft_ids = truncate_token_ids(self.tokenize(draft_text), gamma)
        return self.greedy_acceptance_ids(prefix_ids, draft_ids)

    def prompt_logprobs_at(self, full_ids: list[int]) -> list[dict | None]:
        from vllm import SamplingParams

        params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            prompt_logprobs=self.prompt_logprobs,
            detokenize=False,
        )
        out = self.llm.generate([{"prompt_token_ids": full_ids}], params)[0]
        return out.prompt_logprobs or []

    def verify_trace(self, prompt_text: str, trace_text: str) -> dict[str, Any]:
        prompt_ids = self.tokenize(prompt_text)
        trace_ids = self.tokenize(trace_text)
        if not trace_ids:
            return {
                "trace_token_count": 0,
                "trace_accepted_length": 0,
                "trace_accept_ratio": 0.0,
                "first_reject_position": None,
                "trace_fully_accepted": False,
            }
        result = self.greedy_acceptance_ids(prompt_ids, trace_ids)
        return {
            "trace_token_count": len(trace_ids),
            "trace_accepted_length": result.accepted_length,
            "trace_accept_ratio": result.accept_ratio,
            "first_reject_position": result.first_reject_position,
            "trace_fully_accepted": result.accepted_length == len(trace_ids),
        }


def model_slug(model_path: str) -> str:
    return Path(model_path).name.replace("/", "_")


def _vllm_memory_kwargs(
    model_path: str,
    *,
    debug: bool,
    gpu_memory_utilization: float,
    max_model_len: int,
    dual_resident: bool = False,
) -> dict[str, Any]:
    name = Path(model_path).name.lower()
    if "qwq" in name or "32b" in name or "33b" in name:
        # In dual-resident mode honor the caller's fixed share exactly; only
        # bump to a safe floor when QwQ owns the whole card.
        util = gpu_memory_utilization if dual_resident else max(gpu_memory_utilization, 0.92)
        return {
            "gpu_memory_utilization": util,
            "max_model_len": min(max_model_len, 4096),
        }
    if debug and not dual_resident:
        return {"gpu_memory_utilization": 0.75, "max_model_len": max_model_len}
    return {"gpu_memory_utilization": gpu_memory_utilization, "max_model_len": max_model_len}


def build_target_verifier(
    model_path: str,
    *,
    engine: str = "vllm",
    device: str = "cuda",
    dtype: str = "bfloat16",
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 8192,
    debug: bool = False,
    dual_resident: bool = False,
    quantization: str | None = None,
    prompt_logprobs: int = 20,
) -> TargetVerifier:
    if engine == "vllm":
        import os

        from transformers import AutoTokenizer
        from vllm import LLM

        os.environ.setdefault("VLLM_GDN_PREFILL_BACKEND", "triton")
        os.environ.setdefault("VLLM_USE_V1", "0")
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        mem = _vllm_memory_kwargs(
            model_path,
            debug=debug,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dual_resident=dual_resident,
        )
        llm_kwargs: dict[str, Any] = {
            "model": model_path,
            "dtype": dtype,
            "trust_remote_code": True,
            "enforce_eager": True,
            "enable_prefix_caching": False,
            **mem,
        }
        if "Qwen3.5" in Path(model_path).name or "qwen3.5" in model_path.lower():
            llm_kwargs["language_model_only"] = True
        if quantization:
            llm_kwargs["quantization"] = quantization
        llm = LLM(**llm_kwargs)
        return VLLMTargetVerifier(llm=llm, tokenizer=tokenizer, prompt_logprobs=prompt_logprobs)

    from reasoning_branch_dataset.model_utils import load_model_and_tokenizer

    model, tokenizer = load_model_and_tokenizer(model_path, device, dtype)
    model.eval()
    return HFTargetVerifier(model=model, tokenizer=tokenizer, device=device)


def greedy_generate_ids_vllm(llm, prefix_ids: list[int], *, max_tokens: int) -> list[int]:
    return greedy_generate_vllm(llm, prefix_ids, max_tokens=max_tokens)["token_ids"]


def greedy_generate_vllm(llm, prefix_ids: list[int], *, max_tokens: int) -> dict:
    """Greedy decode with metadata for diagnostics."""
    from vllm import SamplingParams

    params = SamplingParams(max_tokens=max_tokens, temperature=0.0, top_p=1.0, detokenize=False)
    out = llm.generate([{"prompt_token_ids": prefix_ids}], params)[0]
    o0 = out.outputs[0]
    return {
        "token_ids": list(o0.token_ids),
        "finish_reason": getattr(o0, "finish_reason", None),
        "stop_reason": getattr(o0, "stop_reason", None),
    }

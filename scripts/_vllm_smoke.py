"""Minimal vLLM smoke test for AutoDL env."""

from __future__ import annotations


def main() -> None:
    from vllm import LLM, SamplingParams

    print("loading...")
    llm = LLM(
        model="/root/autodl-tmp/specreason/models/DeepSeek-R1-Distill-Qwen-1.5B",
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.35,
        max_model_len=2048,
        enforce_eager=True,
    )
    out = llm.generate(["2+2="], SamplingParams(max_tokens=8, temperature=0))
    print("OUT:", repr(out[0].outputs[0].text))
    print("OK")


if __name__ == "__main__":
    main()

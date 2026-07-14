#!/usr/bin/env bash
# EAGLE-3 benchmark: target-only 或 EAGLE-3 head
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

BASE="${BASE:-/root/autodl-tmp/specreason/models/DeepSeek-R1-Distill-Qwen-32B}"
EAGLE="${EAGLE:-/root/autodl-tmp/SpecForge/outputs/r1-qwen-32b-eagle3-math/epoch_6_step_10000}"
N="${N:-15}"
MAX_TOKENS="${MAX_TOKENS:-512}"
SPEC_TOKENS="${SPEC_TOKENS:-5}"

ARGS=(--base "$BASE" --n "$N" --max-tokens "$MAX_TOKENS")
if [[ "${1:-}" != "ar-only" ]]; then
  ARGS+=(--eagle "$EAGLE" --spec-tokens "$SPEC_TOKENS")
fi

exec python3 "$ROOT/eagle3/bench/vllm_eagle_bench.py" "${ARGS[@]}"

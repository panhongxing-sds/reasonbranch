#!/usr/bin/env bash
# Deploy Qwen3.5-4B vLLM for action-matching study (single small model, no target model)
set -euo pipefail
ROOT="/mnt/afs/L202500372"
PORT="${VLLM_PORT:-8010}"
MODEL="${ROOT}/models/Qwen3.5-4B"
LOG="${ROOT}/reasoning_branch_dataset/logs/vllm-action-study-${PORT}.log"

source /tmp/vllm-cu124/bin/activate 2>/dev/null || true
PY="/tmp/vllm-cu124/bin/python"

mkdir -p "${ROOT}/reasoning_branch_dataset/logs"

if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "vLLM already running on :${PORT}"
  exit 0
fi

echo "Starting vLLM Qwen3.5-4B on port ${PORT}..."
nohup "${PY}" -m vllm.entrypoints.cli.main serve "${MODEL}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --served-model-name Qwen3.5-4B \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --max-num-seqs 32 \
  --gpu-memory-utilization "${GPU_UTIL:-0.85}" \
  --reasoning-parser qwen3 \
  --language-model-only \
  > "${LOG}" 2>&1 &

for i in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "vLLM ready: http://127.0.0.1:${PORT}"
    exit 0
  fi
  sleep 5
done

echo "vLLM failed to start. See ${LOG}"
exit 1

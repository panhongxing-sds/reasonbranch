#!/usr/bin/env bash
# Batch collection with API teacher + resume support
set -euo pipefail
ROOT="/mnt/afs/L202500372"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

if [[ -x /tmp/vllm-cu124/bin/python ]]; then
  PY="/tmp/vllm-cu124/bin/python"
else
  PY="$(command -v python3)"
fi

OUT="${ROOT}/reasoning_branch_dataset/outputs/batch_v1"
MATH500_LIMIT="${MATH500_LIMIT:-50}"
AIME_LIMIT="${AIME_LIMIT:-30}"

# API key: export TEACHER_API_KEY=... before running
# Optional domestic node: export TEACHER_BASE_URL=https://endpoint.wendalog.com

mkdir -p "${OUT}"
exec "${PY}" -m reasoning_branch_dataset.pipeline \
  --output-dir "${OUT}" \
  --math500-limit "${MATH500_LIMIT}" \
  --aime-limit "${AIME_LIMIT}" \
  --no-analysis

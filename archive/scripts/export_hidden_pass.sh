#!/usr/bin/env bash
# Export hidden states + logits after vLLM collection pass
set -euo pipefail
ROOT="/mnt/afs/L202500372"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PY="/tmp/vllm-cu124/bin/python"
[[ -x "${PY}" ]] || PY="$(command -v python3)"

DATA_DIR="${1:-${ROOT}/reasoning_branch_dataset/outputs/action_study_v1}"
exec "${PY}" -m reasoning_branch_dataset.action_study.export_hidden_pass --data-dir "${DATA_DIR}"

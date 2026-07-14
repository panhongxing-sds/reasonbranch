#!/usr/bin/env bash
# Shared paths for reasoning_branch_dataset scripts.
# Usage: source "$(dirname "$0")/common.sh"
set -euo pipefail

AFS="${AFS:-/mnt/afs/L202500372}"
REPO_ROOT="${REPO_ROOT:-${AFS}/reasoning_branch_dataset}"
export PYTHONPATH="${AFS}:${PYTHONPATH:-}"
PY="${PY:-/tmp/vllm-cu124/bin/python}"
V2_DIR="${V2_DIR:-${REPO_ROOT}/outputs/action_study_pilot_v2}"
V3_DIR="${V3_DIR:-${REPO_ROOT}/outputs/action_study_pilot_v3}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"

mkdir -p "${LOG_DIR}"

load_teacher_api() {
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/scripts/load_api_env.sh" "${TEACHER_KEYFILE:-${AFS}/key/api}"
}

setup_vllm_env() {
  export VLLM_USE_V1="${VLLM_USE_V1:-0}"
  export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
}

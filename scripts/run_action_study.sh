#!/usr/bin/env bash
# Phase-1 uncertainty study — small model only (Continue + Branch)
set -euo pipefail
ROOT="/mnt/afs/L202500372"
cd "${ROOT}"

source "${ROOT}/bootstrap/max_speed_env.sh"
source "${ROOT}/reasoning_branch_dataset/scripts/load_api_env.sh"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_GDN_PREFILL_BACKEND="${VLLM_GDN_PREFILL_BACKEND:-triton}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

PY="/tmp/vllm-cu124/bin/python"
[[ -x "${PY}" ]] || PY="$(command -v python3)"

"${PY}" -c "import openai" 2>/dev/null || "${PY}" -m pip install -q openai pandas pyarrow

OUT="${OUT:-${ROOT}/reasoning_branch_dataset/outputs/action_study_v1}"
MATH500_LIMIT="${MATH500_LIMIT:-200}"
GSM8K_LIMIT="${GSM8K_LIMIT:-100}"
ENGINE="${ENGINE:-vllm}"
GPU_UTIL="${GPU_UTIL:-0.82}"
EXTRA_ARGS=()

if [[ "${SMOKE:-0}" == "1" ]]; then
  MATH500_LIMIT=1
  GSM8K_LIMIT=0
  ENGINE="vllm"
  OUT="${ROOT}/reasoning_branch_dataset/outputs/action_study_smoke"
  EXTRA_ARGS+=(--no-resume --max-new-tokens 512 --no-hidden --engine vllm)
  export ACTION_STUDY_MAX_MARKERS=1
  export ACTION_STUDY_MAX_PARAGRAPHS=2
fi

mkdir -p "${OUT}" "${ROOT}/reasoning_branch_dataset/logs"
LOG="${ROOT}/reasoning_branch_dataset/logs/action_study_$(date +%Y%m%d_%H%M%S).log"

echo "Starting action study -> ${OUT} (math500=${MATH500_LIMIT}, gsm8k=${GSM8K_LIMIT}, engine=${ENGINE})"
exec "${PY}" -m reasoning_branch_dataset.action_study.pipeline \
  --output-dir "${OUT}" \
  --math500-limit "${MATH500_LIMIT}" \
  --gsm8k-limit "${GSM8K_LIMIT}" \
  --engine "${ENGINE}" \
  "${EXTRA_ARGS[@]}" \
  "$@" 2>&1 | tee "${LOG}"

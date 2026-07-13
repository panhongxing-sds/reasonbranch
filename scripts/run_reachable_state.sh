#!/usr/bin/env bash
# Scheme A: target-prefix reachable-state experiment (100 problems).
set -euo pipefail
AFS=/mnt/afs/L202500372
source "${AFS}/bootstrap/max_speed_env.sh"
export PYTHONPATH="${AFS}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

DATA_DIR="${DATA_DIR:-${AFS}/reasoning_branch_dataset/outputs/action_study_pilot_v2}"
OUT_DIR="${OUT_DIR:-${AFS}/reasoning_branch_dataset/outputs/reachable_state_pilot}"
TARGET_MODEL="${TARGET_MODEL:-${AFS}/specreason/models/QwQ-32B}"
DRAFT_MODEL="${DRAFT_MODEL:-${AFS}/models/Qwen3.5-4B}"
N_PROBLEMS="${N_PROBLEMS:-100}"
GAMMA="${GAMMA:-32}"
PHASES="${PHASES:-all}"
LOG="${LOG:-${AFS}/reasoning_branch_dataset/logs/reachable_state.log}"

mkdir -p "$(dirname "${LOG}")"
cd "${AFS}"

echo "[$(date '+%F %T')] reachable-state phases=${PHASES} n=${N_PROBLEMS} gamma=${GAMMA}" | tee "${LOG}"

pkill -9 -f 'action_study.run_reachable_state' 2>/dev/null || true
# Only kill vLLM if we're starting phase 1 or all (avoid killing mid-run externally)
if [[ "${PHASES}" == "all" || "${PHASES}" == "1" ]]; then
  pkill -9 -f 'vllm.entrypoints|EngineCore' 2>/dev/null || true
  sleep 3
fi

/tmp/vllm-cu124/bin/python -m reasoning_branch_dataset.action_study.run_reachable_state \
  --data-dir "${DATA_DIR}" \
  --out-dir "${OUT_DIR}" \
  --target-model "${TARGET_MODEL}" \
  --draft-model "${DRAFT_MODEL}" \
  --n-problems "${N_PROBLEMS}" \
  --gamma "${GAMMA}" \
  --phases "${PHASES}" \
  2>&1 | tee -a "${LOG}"

echo "[$(date '+%F %T')] done -> ${OUT_DIR}/../reachable_state_report.md" | tee -a "${LOG}"

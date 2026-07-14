#!/usr/bin/env bash
# Target Acceptance Replay — QwQ-32B target via vLLM (no draft re-generation).
set -euo pipefail
AFS=/mnt/afs/L202500372
source "${AFS}/bootstrap/max_speed_env.sh"
export PYTHONPATH="${AFS}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

DATA_DIR="${DATA_DIR:-${AFS}/reasoning_branch_dataset/outputs/action_study_pilot_v2}"
# >=30B target; override with TARGET_MODEL if needed
TARGET_MODEL="${TARGET_MODEL:-${AFS}/specreason/models/QwQ-32B}"
TARGET_ENGINE="${TARGET_ENGINE:-vllm}"
GAMMA="${GAMMA:-64}"
N_SAMPLES="${N_SAMPLES:-300}"
LOG="${LOG:-${AFS}/reasoning_branch_dataset/logs/target_replay.log}"

cd "${AFS}"
echo "[$(date '+%F %T')] target replay n=${N_SAMPLES} gamma=${GAMMA}" | tee "${LOG}"
echo "  target=${TARGET_MODEL} engine=${TARGET_ENGINE}" | tee -a "${LOG}"

pkill -9 -f 'run_target_replay|action_study.run_target_replay' 2>/dev/null || true
pkill -9 -f 'vllm.entrypoints' 2>/dev/null || true
pkill -9 -f 'EngineCore' 2>/dev/null || true
sleep 5

FORCE_REVERIFY="${FORCE_REVERIFY:-1}"
EXTRA=()
if [[ "${FORCE_REVERIFY}" == "1" ]]; then
  EXTRA+=(--force-reverify)
fi

/tmp/vllm-cu124/bin/python -m reasoning_branch_dataset.action_study.run_target_replay \
  --data-dir "${DATA_DIR}" \
  --target-model "${TARGET_MODEL}" \
  --engine "${TARGET_ENGINE}" \
  --gamma "${GAMMA}" \
  --n-samples "${N_SAMPLES}" \
  "${EXTRA[@]}" \
  2>&1 | tee -a "${LOG}"

echo "[$(date '+%F %T')] done -> ${DATA_DIR}/target_replay_report.md" | tee -a "${LOG}"

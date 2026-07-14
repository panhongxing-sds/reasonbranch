#!/usr/bin/env bash
# V3: SpecReason-style utility scoring on admission_main 1+4 candidates.
set -euo pipefail
AFS=/mnt/afs/L202500372
source "${AFS}/bootstrap/max_speed_env.sh"
export PYTHONPATH="${AFS}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

DATA_DIR="${DATA_DIR:-${AFS}/reasoning_branch_dataset/outputs/action_study_pilot_v2}"
OUT_DIR="${OUT_DIR:-${AFS}/reasoning_branch_dataset/outputs/action_study_pilot_v3}"
TARGET_MODEL="${TARGET_MODEL:-${AFS}/specreason/models/QwQ-32B}"
MAX_PREFIXES="${MAX_PREFIXES:-}"
LOG="${LOG:-${AFS}/reasoning_branch_dataset/logs/utility_scoring_v3.log}"

mkdir -p "$(dirname "${LOG}")" "${OUT_DIR}"
cd "${AFS}"

echo "[$(date '+%F %T')] v3 utility scoring" | tee "${LOG}"
echo "  data=${DATA_DIR} out=${OUT_DIR}" | tee -a "${LOG}"

pkill -9 -f 'action_study.run_utility_scoring' 2>/dev/null || true
# Do not pkill vLLM if reachable_state phase3 is running on same GPU — user may share GPU.
# Uncomment next lines for dedicated GPU run:
# pkill -9 -f 'vllm.entrypoints|EngineCore' 2>/dev/null || true
# sleep 3

EXTRA=()
if [[ -n "${MAX_PREFIXES}" ]]; then
  EXTRA+=(--max-prefixes "${MAX_PREFIXES}")
fi

nohup /tmp/vllm-cu124/bin/python -m reasoning_branch_dataset.action_study.run_utility_scoring \
  --data-dir "${DATA_DIR}" \
  --out-dir "${OUT_DIR}" \
  --target-model "${TARGET_MODEL}" \
  "${EXTRA[@]}" \
  >> "${LOG}" 2>&1 &

echo "[$(date '+%F %T')] background pid=$! log=${LOG}" | tee -a "${LOG}"

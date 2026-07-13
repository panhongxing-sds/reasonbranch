#!/usr/bin/env bash
# V3.4 Pilot A: sequential oracle policy rollout (50 problems, 5 policies)
set -euo pipefail
AFS="${AFS:-/mnt/afs/L202500372}"
export PYTHONPATH="${AFS}"
PY="${PY:-/tmp/vllm-cu124/bin/python}"
V2="${V2:-${AFS}/reasoning_branch_dataset/outputs/action_study_pilot_v2}"
OUT="${AFS}/reasoning_branch_dataset/outputs/action_study_pilot_v34"
REPORT="${AFS}/reasoning_branch_dataset/outputs/pilot_v3_4_report.md"
LOG="${AFS}/reasoning_branch_dataset/logs/v3_4_pilot.log"
mkdir -p "$(dirname "$LOG")"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
load_teacher_api
setup_vllm_env

N_PROBLEMS="${N_PROBLEMS:-50}"
MAX_STEPS="${MAX_STEPS:-20}"
BRANCH_SEEDS="${BRANCH_SEEDS:-3}"
EXTRA_FLAGS=()
if [[ "${SKIP_ALWAYS_BRANCH:-0}" == "1" ]]; then
  EXTRA_FLAGS+=(--skip-always-branch)
fi
if [[ "${DUAL_RESIDENT:-1}" == "1" ]]; then
  EXTRA_FLAGS+=(--dual-resident)
fi

echo "[$(date '+%F %T')] V3.4 Pilot A start (n=${N_PROBLEMS} steps=${MAX_STEPS} branch_seeds=${BRANCH_SEEDS})" | tee -a "$LOG"
"$PY" -m reasoning_branch_dataset.action_study.run_v3_4_policy_rollout \
  --v2-dir "$V2" --out-dir "$OUT" --report-path "$REPORT" \
  --n-problems "$N_PROBLEMS" --max-steps "$MAX_STEPS" \
  --branch-seeds "$BRANCH_SEEDS" \
  "${EXTRA_FLAGS[@]}" 2>&1 | tee -a "$LOG"
echo "[$(date '+%F %T')] done -> $REPORT" | tee -a "$LOG"

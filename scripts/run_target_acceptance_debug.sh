#!/usr/bin/env bash
# Greedy acceptance alignment unit tests (4 mandatory checks).
set -euo pipefail
AFS=/mnt/afs/L202500372
source "${AFS}/bootstrap/max_speed_env.sh"
export PYTHONPATH="${AFS}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

LOG="${LOG:-${AFS}/reasoning_branch_dataset/logs/target_acceptance_debug.log}"
mkdir -p "$(dirname "${LOG}")"

cd "${AFS}"
echo "[$(date '+%F %T')] target acceptance debug tests" | tee "${LOG}"

/tmp/vllm-cu124/bin/python -m reasoning_branch_dataset.action_study.target_acceptance_debug \
  2>&1 | tee -a "${LOG}"

echo "[$(date '+%F %T')] done -> outputs/action_study_pilot_v2/target_acceptance_debug_report.md" | tee -a "${LOG}"

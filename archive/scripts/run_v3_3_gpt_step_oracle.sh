#!/usr/bin/env bash
# V3.3: GPT-5.5 per-candidate next-step oracle (1 greedy + 4 branch)
set -euo pipefail
AFS="${AFS:-/mnt/afs/L202500372}"
export PYTHONPATH="${AFS}"
PY="${PY:-/tmp/vllm-cu124/bin/python}"
V2="${V2:-${AFS}/reasoning_branch_dataset/outputs/action_study_pilot_v2}"
V3="${V3:-${AFS}/reasoning_branch_dataset/outputs/action_study_pilot_v3}"
OUT="${AFS}/reasoning_branch_dataset/outputs"
LOG="${AFS}/reasoning_branch_dataset/logs/v3_3_gpt_step_oracle.log"
mkdir -p "$(dirname "$LOG")"

export DS_API_CONCURRENCY_LIMIT="${DS_API_CONCURRENCY_LIMIT:-96}"

source "${AFS}/reasoning_branch_dataset/scripts/load_api_env.sh"
source "${AFS}/bootstrap/max_speed_env.sh" 2>/dev/null || true

FULL_FLAG=()
if [[ "${FULL:-0}" == "1" ]]; then
  FULL_FLAG=(--full)
fi

MAX_PREFIXES_FLAG=()
if [[ -n "${MAX_PREFIXES:-}" ]]; then
  MAX_PREFIXES_FLAG=(--max-prefixes "${MAX_PREFIXES}")
fi

echo "[$(date '+%F %T')] V3.3 GPT-5.5 next-step oracle" | tee -a "$LOG"
"$PY" -m reasoning_branch_dataset.action_study.run_v3_gpt_step_oracle \
  --v2-dir "$V2" --v3-dir "$V3" \
  --report-path "${OUT}/pilot_v3_3_report.md" \
  --max-workers "${DS_API_CONCURRENCY_LIMIT}" \
  "${FULL_FLAG[@]}" "${MAX_PREFIXES_FLAG[@]}" 2>&1 | tee -a "$LOG"

echo "[$(date '+%F %T')] done -> ${OUT}/pilot_v3_3_report.md" | tee -a "$LOG"

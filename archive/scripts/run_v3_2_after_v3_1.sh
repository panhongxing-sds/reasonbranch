#!/usr/bin/env bash
# Wait for V3.1 (QwQ phase2 audit) then run V3.2 GPT-5.5 pairwise oracle
set -euo pipefail
AFS="${AFS:-/mnt/afs/L202500372}"
export PYTHONPATH="${AFS}"
PY="${PY:-/tmp/vllm-cu124/bin/python}"
V2="${V2:-${AFS}/reasoning_branch_dataset/outputs/action_study_pilot_v2}"
V3="${V3:-${AFS}/reasoning_branch_dataset/outputs/action_study_pilot_v3}"
OUT="${AFS}/reasoning_branch_dataset/outputs"
LOG="${AFS}/reasoning_branch_dataset/logs/v3_2_gpt_oracle.log"
PAIRWISE_N="${PAIRWISE_TARGET:-162}"
mkdir -p "$(dirname "$LOG")"

wait_v31() {
  local n=0
  while true; do
    if [[ -f "${OUT}/pilot_v3_audit_phase2_report.md" ]]; then
      n=$(wc -l < "${V3}/pairwise_audit_results.jsonl" 2>/dev/null || echo 0)
      if [[ "${n}" -ge "${PAIRWISE_N}" ]]; then
        echo "[$(date '+%F %T')] V3.1 done (pairwise=${n})" | tee -a "$LOG"
        return 0
      fi
    fi
    if ! pgrep -af 'run_v3_audit_phase2|run_v3_pairwise_audit|run_v3_shuffle_rescore' >/dev/null 2>&1; then
      n=$(wc -l < "${V3}/pairwise_audit_results.jsonl" 2>/dev/null || echo 0)
      if [[ "${n}" -ge "${PAIRWISE_N}" ]] || [[ -f "${OUT}/pilot_v3_audit_phase2_report.md" ]]; then
        echo "[$(date '+%F %T')] V3.1 process ended (pairwise=${n})" | tee -a "$LOG"
        return 0
      fi
    fi
    echo "[$(date '+%F %T')] waiting V3.1..." | tee -a "$LOG"
    sleep 60
  done
}

source "${AFS}/reasoning_branch_dataset/scripts/load_api_env.sh"
source "${AFS}/bootstrap/max_speed_env.sh" 2>/dev/null || true
export DS_API_CONCURRENCY_LIMIT="${DS_API_CONCURRENCY_LIMIT:-96}"

wait_v31

echo "[$(date '+%F %T')] V3.2 GPT-5.5 pairwise oracle" | tee -a "$LOG"
"$PY" -m reasoning_branch_dataset.action_study.run_v3_gpt_oracle \
  --v2-dir "$V2" --v3-dir "$V3" \
  --report-path "${OUT}/pilot_v3_2_report.md" \
  --max-workers "${DS_API_CONCURRENCY_LIMIT}" 2>&1 | tee -a "$LOG"

echo "[$(date '+%F %T')] done -> ${OUT}/pilot_v3_2_report.md" | tee -a "$LOG"

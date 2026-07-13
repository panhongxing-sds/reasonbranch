#!/usr/bin/env bash
# V3.4b — P0 fixes: ORACLE_API_ERROR separation, target handoff extraction, grading.
# Default: SpecReason vs Conditional Branch only (paired core comparison).
set -euo pipefail
AFS="${AFS:-/mnt/afs/L202500372}"
export PYTHONPATH="${AFS}"
PY="${PY:-/tmp/vllm-cu124/bin/python}"
V2="${V2:-${AFS}/reasoning_branch_dataset/outputs/action_study_pilot_v2}"
OUT="${OUT:-${AFS}/reasoning_branch_dataset/outputs/action_study_pilot_v34b}"
REPORT="${REPORT:-${AFS}/reasoning_branch_dataset/outputs/pilot_v3_4b_report.md}"
LOG="${AFS}/reasoning_branch_dataset/logs/v3_4b.log"
mkdir -p "$(dirname "$LOG")" "$OUT"

source "${AFS}/bootstrap/max_speed_env.sh"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
load_teacher_api
setup_vllm_env

pkill -9 -f 'run_v3_4_policy_rollout' 2>/dev/null || true
pkill -9 -f 'EngineCore' 2>/dev/null || true
sleep 5

N_PROBLEMS="${N_PROBLEMS:-30}"
MAX_STEPS="${MAX_STEPS:-20}"
BRANCH_SEEDS="${BRANCH_SEEDS:-1}"
SKIP_POLICIES="${SKIP_POLICIES:-DRAFT_ONLY,TARGET_ONLY,ALWAYS_BRANCH}"

echo "[$(date '+%F %T')] V3.4b (P0 fixes, core policies) n=${N_PROBLEMS} steps=${MAX_STEPS}" | tee "$LOG"
"$PY" -m reasoning_branch_dataset.action_study.run_v3_4_policy_rollout \
  --v2-dir "$V2" --out-dir "$OUT" --report-path "$REPORT" \
  --n-problems "$N_PROBLEMS" --max-steps "$MAX_STEPS" \
  --branch-seeds "$BRANCH_SEEDS" --dual-resident \
  --skip-policies "$SKIP_POLICIES" \
  2>&1 | tee -a "$LOG"
echo "[$(date '+%F %T')] done -> $REPORT" | tee -a "$LOG"

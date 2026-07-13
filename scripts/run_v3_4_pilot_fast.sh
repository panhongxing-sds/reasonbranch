#!/usr/bin/env bash
# V3.4 fast pilot: oracle-only (SpecReason vs Conditional Branch), skip slow baselines
set -euo pipefail
AFS="${AFS:-/mnt/afs/L202500372}"
export PYTHONPATH="${AFS}"
PY="${PY:-/tmp/vllm-cu124/bin/python}"
V2="${V2:-${AFS}/reasoning_branch_dataset/outputs/action_study_pilot_v2}"
OUT="${AFS}/reasoning_branch_dataset/outputs/action_study_pilot_v34"
REPORT="${AFS}/reasoning_branch_dataset/outputs/pilot_v3_4_report.md"
LOG="${AFS}/reasoning_branch_dataset/logs/v3_4_pilot_fast.log"
mkdir -p "$(dirname "$LOG")"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
load_teacher_api
setup_vllm_env
export DS_API_CONCURRENCY_LIMIT="${DS_API_CONCURRENCY_LIMIT:-96}"

# Clean stale GPU processes (spawned EngineCore workers linger otherwise)
pkill -9 -f 'run_v3_4_policy_rollout' 2>/dev/null || true
pkill -9 -f 'VLLM::EngineCore' 2>/dev/null || true
pkill -9 -f 'EngineCore' 2>/dev/null || true
pkill -9 -f 'multiprocessing.spawn' 2>/dev/null || true
# spawn workers may not match by name; kill by GPU pid (dedicated box)
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null || true
sleep 8

N_PROBLEMS="${N_PROBLEMS:-30}"
MAX_STEPS="${MAX_STEPS:-12}"
BRANCH_SEEDS="${BRANCH_SEEDS:-1}"
# Skip slow baselines; core comparison is SPECREASON vs CONDITIONAL_BRANCH
SKIP_POLICIES="${SKIP_POLICIES:-DRAFT_ONLY,TARGET_ONLY,ALWAYS_BRANCH}"

echo "[$(date '+%F %T')] V3.4 FAST oracle-only (n=${N_PROBLEMS} steps=${MAX_STEPS}) dual-resident" | tee "$LOG"
"$PY" -m reasoning_branch_dataset.action_study.run_v3_4_policy_rollout \
  --v2-dir "$V2" --out-dir "$OUT" --report-path "$REPORT" \
  --n-problems "$N_PROBLEMS" --max-steps "$MAX_STEPS" \
  --branch-seeds "$BRANCH_SEEDS" --skip-always-branch \
  --skip-policies "$SKIP_POLICIES" --dual-resident \
  2>&1 | tee -a "$LOG"
echo "[$(date '+%F %T')] done -> $REPORT" | tee -a "$LOG"

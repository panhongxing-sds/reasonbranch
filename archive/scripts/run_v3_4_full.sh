#!/usr/bin/env bash
# V3.4 FULL run: R1-Distill-1.5B (draft) + R1-Distill-32B-AWQ (target), both
# resident on one 80GB card (no per-handoff model swap). Runs all 5 policies.
set -euo pipefail
AFS="${AFS:-/mnt/afs/L202500372}"
export PYTHONPATH="${AFS}"
PY="${PY:-/tmp/vllm-cu124/bin/python}"
V2="${V2:-${AFS}/reasoning_branch_dataset/outputs/action_study_pilot_v2}"
OUT="${AFS}/reasoning_branch_dataset/outputs/action_study_pilot_v34"
REPORT="${AFS}/reasoning_branch_dataset/outputs/pilot_v3_4_report.md"
LOG="${AFS}/reasoning_branch_dataset/logs/v3_4_full.log"
mkdir -p "$(dirname "$LOG")" "$OUT"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
load_teacher_api
setup_vllm_env
export DS_API_CONCURRENCY_LIMIT="${DS_API_CONCURRENCY_LIMIT:-96}"

# Clean stale GPU processes (dedicated box)
pkill -9 -f 'run_v3_4_policy_rollout' 2>/dev/null || true
pkill -9 -f 'EngineCore' 2>/dev/null || true
pkill -9 -f 'multiprocessing.spawn' 2>/dev/null || true
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null || true
sleep 8

# Old summaries were produced with the previous QwQ/4B model pair — archive so
# the resume logic starts fresh for the new model config.
STAMP="$(date '+%Y%m%d_%H%M%S')"
for f in rollout_summaries.jsonl rollout_steps.jsonl; do
  if [[ -s "$OUT/$f" ]]; then
    mv "$OUT/$f" "$OUT/${f%.jsonl}.pre_r1_${STAMP}.jsonl"
  fi
done

N_PROBLEMS="${N_PROBLEMS:-30}"
MAX_STEPS="${MAX_STEPS:-20}"
BRANCH_SEEDS="${BRANCH_SEEDS:-1}"

echo "[$(date '+%F %T')] V3.4 FULL (R1-1.5B + R1-32B-AWQ dual-resident, all policies) n=${N_PROBLEMS} steps=${MAX_STEPS}" | tee "$LOG"
"$PY" -m reasoning_branch_dataset.action_study.run_v3_4_policy_rollout \
  --v2-dir "$V2" --out-dir "$OUT" --report-path "$REPORT" \
  --n-problems "$N_PROBLEMS" --max-steps "$MAX_STEPS" \
  --branch-seeds "$BRANCH_SEEDS" --dual-resident \
  2>&1 | tee -a "$LOG"
echo "[$(date '+%F %T')] done -> $REPORT" | tee -a "$LOG"

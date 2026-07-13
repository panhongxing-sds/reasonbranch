#!/usr/bin/env bash
# Wait for v2 reachable-state phase 3, then launch v3 utility scoring.
set -euo pipefail
AFS=/mnt/afs/L202500372
LOG="${AFS}/reasoning_branch_dataset/logs/v3_after_v2.log"

echo "[$(date '+%F %T')] waiting for v2 reachable_state..." | tee "${LOG}"
while pgrep -f 'action_study.run_reachable_state' >/dev/null 2>&1; do
  V=$(wc -l < "${AFS}/reasoning_branch_dataset/outputs/reachable_state_pilot/verify_results.jsonl" 2>/dev/null || echo 0)
  echo "[$(date '+%F %T')] v2 phase3 verify=${V}/288" | tee -a "${LOG}"
  sleep 30
done
pkill -9 -f 'EngineCore' 2>/dev/null || true
sleep 5
echo "[$(date '+%F %T')] v2 done; starting v3 utility scoring (1548 prefixes)" | tee -a "${LOG}"
bash "${AFS}/reasoning_branch_dataset/scripts/run_utility_scoring_v3.sh"

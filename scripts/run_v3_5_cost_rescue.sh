#!/usr/bin/env bash
# V3.5 — combine latency + rescue into cost–rescue decision.
set -euo pipefail

source /root/autodl-tmp/activate_reasonbranch.sh
export PYTHONPATH="/root/autodl-tmp:${PYTHONPATH:-}"
PY="${PY:-python3}"
REPO_ROOT="${REPO_ROOT:-/root/autodl-tmp/reasonbranch}"

LATENCY="${LATENCY:-${REPO_ROOT}/outputs/action_study_v35_latency/latency_summary.json}"
RESCUE_DIR="${RESCUE_DIR:-${REPO_ROOT}/outputs/action_study_v35_rescue}"
OUT="${OUT:-${REPO_ROOT}/outputs/action_study_v35_cost_rescue}"
mkdir -p "$RESCUE_DIR" "$OUT"

# If no Experiment B yet, write provisional V3.3 rates.
if [[ ! -f "${RESCUE_DIR}/rescue_rates.json" ]]; then
  echo "[v3.5] no rescue_rates.json — writing provisional V3.3 prior"
  "$PY" -m reasoning_branch_dataset.action_study.run_v3_5_rescue_rate \
    --mode provisional --out-dir "$RESCUE_DIR"
fi

"$PY" -m reasoning_branch_dataset.action_study.run_v3_5_cost_rescue \
  --latency-summary "$LATENCY" \
  --rescue-rates "${RESCUE_DIR}/rescue_rates.json" \
  --out-dir "$OUT" \
  --use-provisional-rescue

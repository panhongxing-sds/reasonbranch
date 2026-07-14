#!/usr/bin/env bash
# V3.6 Pilot — one-step counterfactual Cost–Rescue Gate.
# Does NOT auto-start a long GPU job without states; collect first if needed.
set -euo pipefail
source /root/autodl-tmp/activate_reasonbranch.sh
export PYTHONPATH="/root/autodl-tmp:${PYTHONPATH:-}"
PY="${PY:-python3}"
REPO="${REPO_ROOT:-/root/autodl-tmp/reasonbranch}"
OUT="${OUT:-${REPO}/outputs/action_study_v36}"
LOG_DIR="${LOG_DIR:-${REPO}/logs}"
mkdir -p "$OUT" "$LOG_DIR"

STATES="${STATES:-${OUT}/rejected_states.jsonl}"
N_STATES="${N_STATES:-64}"
MAX_STATES="${MAX_STATES:-64}"
N_REPS="${N_REPS:-3}"
N_SEEDS="${N_SEEDS:-3}"
TAU="${TAU_ACCEPT:-0.0}"

echo "[v3.6] OUT=$OUT"

if [[ ! -f "$STATES" ]]; then
  echo "[v3.6] collecting rejected states -> $STATES"
  "$PY" -m reasoning_branch_dataset.action_study.run_v3_6_collect_states \
    --draft-model "$DRAFT_MODEL" \
    --target-model "${TARGET_MODEL_AWQ}" \
    --problems "${REPO}/data/aime_train.jsonl" \
    --out "$STATES" \
    --n-states "$N_STATES" \
    --tau-accept "$TAU" \
    2>&1 | tee "${LOG_DIR}/v3_6_collect.log"
fi

echo "[v3.6] running pilot timing (skip-oracle by default)"
"$PY" -m reasoning_branch_dataset.action_study.run_v3_6_pilot \
  --states "$STATES" \
  --out-dir "$OUT" \
  --draft-model "$DRAFT_MODEL" \
  --target-model "${TARGET_MODEL_AWQ}" \
  --max-states "$MAX_STATES" \
  --n-reps "$N_REPS" \
  --n-seeds "$N_SEEDS" \
  --tau-accept "$TAU" \
  --skip-oracle \
  2>&1 | tee "${LOG_DIR}/v3_6_pilot.log"

echo "[v3.6] done -> ${OUT}/v36_report.md"

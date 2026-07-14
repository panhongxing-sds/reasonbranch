#!/usr/bin/env bash
# V3.5a Formal Cost Gate: K=1/2/4, per-bucket r_K^*, pipeline E2E.
set -euo pipefail
source /root/autodl-tmp/activate_reasonbranch.sh
export PYTHONPATH="/root/autodl-tmp:${PYTHONPATH:-}"
PY="${PY:-python3}"
REPO_ROOT="${REPO_ROOT:-/root/autodl-tmp/reasonbranch}"
OUT="${OUT:-${REPO_ROOT}/outputs/action_study_v35a_cost_gate}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
mkdir -p "$OUT" "$LOG_DIR"
LOG="${LOG_DIR}/v3_5a_cost_gate.log"

N_STATES="${N_STATES:-48}"
N_PROBLEMS="${N_PROBLEMS:-30}"
STEP_MAX_TOKENS="${STEP_MAX_TOKENS:-256}"
REPS="${REPS:-3}"
SEED="${SEED:-42}"

pkill -9 -f 'EngineCore' 2>/dev/null || true
sleep 1

echo "[$(date '+%F %T')] V3.5a Cost Gate n_states=${N_STATES}" | tee "$LOG"
"$PY" -m reasoning_branch_dataset.action_study.run_v3_5a_cost_gate \
  --draft-model "${DRAFT_MODEL}" \
  --target-model "${TARGET_MODEL_AWQ}" \
  --target-quantization awq \
  --problems "${REPO_ROOT}/data/aime_train.jsonl" \
  --out-dir "$OUT" \
  --n-states "$N_STATES" \
  --n-problems "$N_PROBLEMS" \
  --step-max-tokens "$STEP_MAX_TOKENS" \
  --reps "$REPS" \
  --seed "$SEED" \
  --verify-max-tokens 2 \
  --draft-gpu-util "${DRAFT_GPU_UTIL:-0.18}" \
  --target-gpu-util "${TARGET_GPU_UTIL:-0.70}" \
  2>&1 | tee -a "$LOG"
echo "[$(date '+%F %T')] done -> ${OUT}/cost_gate_report.md" | tee -a "$LOG"

#!/usr/bin/env bash
# V3.5 Experiment A — latency microbenchmark (no API).
# Measures C_T, C_D{1,2,4}, C_V{1,2,4} and prints break-even r_K^*.
set -euo pipefail

source /root/autodl-tmp/activate_reasonbranch.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
setup_vllm_env

REPO_ROOT="${REPO_ROOT:-/root/autodl-tmp/reasonbranch}"
export PYTHONPATH="/root/autodl-tmp:${PYTHONPATH:-}"
PY="${PY:-python3}"

DRAFT_MODEL="${DRAFT_MODEL:-/root/autodl-tmp/specreason/models/DeepSeek-R1-Distill-Qwen-1.5B}"
# Prefer AWQ for 32B on a single card; override TARGET_MODEL to full bf16 if desired.
TARGET_MODEL="${TARGET_MODEL_AWQ:-/root/autodl-tmp/specreason/models/DeepSeek-R1-Distill-Qwen-32B-AWQ}"
PROBLEMS="${PROBLEMS:-${REPO_ROOT}/data/aime_train.jsonl}"
OUT="${OUT:-${REPO_ROOT}/outputs/action_study_v35_latency}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
mkdir -p "$OUT" "$LOG_DIR"
LOG="${LOG_DIR}/v3_5_latency_bench.log"

N_STATES="${N_STATES:-48}"
N_PROBLEMS="${N_PROBLEMS:-24}"
STEP_MAX_TOKENS="${STEP_MAX_TOKENS:-128}"
SEED="${SEED:-42}"

# Avoid killing this script itself: only reap leftover EngineCore workers.
pkill -9 -f 'vllm.entrypoints' 2>/dev/null || true
pkill -9 -f 'EngineCore' 2>/dev/null || true
sleep 1

echo "[$(date '+%F %T')] V3.5A latency bench n_states=${N_STATES}" | tee "$LOG"
"$PY" -m reasoning_branch_dataset.action_study.run_v3_5_latency_bench \
  --draft-model "$DRAFT_MODEL" \
  --target-model "$TARGET_MODEL" \
  --target-quantization awq \
  --problems "$PROBLEMS" \
  --out-dir "$OUT" \
  --n-states "$N_STATES" \
  --n-problems "$N_PROBLEMS" \
  --step-max-tokens "$STEP_MAX_TOKENS" \
  --seed "$SEED" \
  2>&1 | tee -a "$LOG"
echo "[$(date '+%F %T')] done -> ${OUT}/cost_rescue_latency_report.md" | tee -a "$LOG"

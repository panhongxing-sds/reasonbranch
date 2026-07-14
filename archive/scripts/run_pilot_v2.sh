#!/usr/bin/env bash
# Phase-1 v2 pilot: DeepScaler only (default 800)
# Artifacts: traces/prefixes/actions JSONL + hidden.safetensors + API labels
set -euo pipefail
AFS=/mnt/afs/L202500372
source "${AFS}/bootstrap/max_speed_env.sh"
max_speed_env_for_4b
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

export PYTHONPATH="${AFS}"
export OUT="${AFS}/reasoning_branch_dataset/outputs/action_study_pilot_v2"
export LOG="${AFS}/reasoning_branch_dataset/logs/pilot_v2.log"
export DEEPSCALER_LIMIT="${DEEPSCALER_LIMIT:-800}"
export DEEPSCALER_OFFSET="${DEEPSCALER_OFFSET:-1000}"
# Auto-resume if checkpoint exists unless FRESH=1
if [[ -f "${OUT}/checkpoints/done_problems.json" ]]; then
  export FRESH="${FRESH:-0}"
else
  export FRESH="${FRESH:-1}"
fi
export DS_API_CONCURRENCY_LIMIT="${DS_API_CONCURRENCY_LIMIT:-96}"
export CONTINUATION_MAX_TOKENS="${CONTINUATION_MAX_TOKENS:-2048}"
export CONTINUATION_RETRY_TOKENS="${CONTINUATION_RETRY_TOKENS:-2048}"

cd "${AFS}"
echo "[$(date '+%F %T')] pilot v2 DeepScaler×${DEEPSCALER_LIMIT} offset=${DEEPSCALER_OFFSET} -> ${OUT}" | tee "${LOG}"
echo "API workers=${DS_API_CONCURRENCY_LIMIT} EXPORT_HIDDEN=${EXPORT_HIDDEN} FRESH=${FRESH}" | tee -a "${LOG}"

pkill -9 -f 'action_study.pipeline.*action_study_pilot_v2' 2>/dev/null || true
pkill -9 -f 'vllm.entrypoints' 2>/dev/null || true
pkill -9 -f 'vllm.*Qwen3.5-4B' 2>/dev/null || true
pkill -9 -f 'EngineCore' 2>/dev/null || true
sleep 5
for i in $(seq 1 12); do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 99999)
  if [[ "${USED}" -lt 5000 ]]; then
    echo "GPU free (${USED} MiB used)" | tee -a "${LOG}"
    break
  fi
  echo "Waiting for GPU (${USED} MiB used)..." | tee -a "${LOG}"
  pkill -9 -f 'vllm|EngineCore' 2>/dev/null || true
  sleep 5
done

if [[ "${FRESH}" == "1" ]]; then
  rm -rf "${OUT}"
fi
mkdir -p "${OUT}"

RESUME_FLAG=()
if [[ "${FRESH}" != "1" && -f "${OUT}/checkpoints/done_problems.json" ]]; then
  echo "Resuming from checkpoint" | tee -a "${LOG}"
else
  RESUME_FLAG=(--no-resume)
fi

# vLLM: fast generation; --no-hidden (HF post-pass exports hidden separately)
/tmp/vllm-cu124/bin/python -m reasoning_branch_dataset.action_study.pipeline \
  --output-dir "${OUT}" \
  --math500-limit 0 \
  --deepscaler-limit "${DEEPSCALER_LIMIT}" \
  --deepscaler-offset "${DEEPSCALER_OFFSET}" \
  --aime-limit 0 \
  --gsm8k-limit 0 \
  --engine vllm \
  --no-api \
  --no-hidden \
  --max-new-tokens 4096 \
  "${RESUME_FLAG[@]}" \
  2>&1 | tee -a "${LOG}"

if [[ "${EXPORT_HIDDEN}" == "1" ]]; then
  echo "[$(date '+%F %T')] HF hidden+logits export (layers -4,-2,-1)" | tee -a "${LOG}"
  /tmp/vllm-cu124/bin/python -m reasoning_branch_dataset.action_study.export_hidden_pass \
    --data-dir "${OUT}" 2>&1 | tee -a "${LOG}"
fi

echo "[$(date '+%F %T')] parallel API backfill (workers=${DS_API_CONCURRENCY_LIMIT})" | tee -a "${LOG}"
source "${AFS}/reasoning_branch_dataset/scripts/load_api_env.sh"
export DS_API_CONCURRENCY_LIMIT
/tmp/vllm-cu124/bin/python "${AFS}/reasoning_branch_dataset/scripts/backfill_uncertainty_study.py" \
  --output-dir "${OUT}" 2>&1 | tee -a "${LOG}"

/tmp/vllm-cu124/bin/python -m reasoning_branch_dataset.action_study.analyze \
  --data-dir "${OUT}" 2>&1 | tee -a "${LOG}"

echo "[$(date '+%F %T')] pilot v2 complete (${DEEPSCALER_LIMIT} DeepScaler)" | tee -a "${LOG}"
echo "Reusable artifacts in ${OUT}:" | tee -a "${LOG}"
echo "  problems/traces/prefixes/next_step_samples/actions/outcome_results *.jsonl" | tee -a "${LOG}"
echo "  hidden.safetensors (prefix hidden last/step_mean/local4 @ layers -4,-2,-1)" | tee -a "${LOG}"
echo "  validity_labels/cluster_labels + api_cache_v2.jsonl" | tee -a "${LOG}"

#!/usr/bin/env bash
set -euo pipefail
ROOT="/mnt/afs/L202500372"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

if [[ -x /tmp/vllm-cu124/bin/python ]]; then
  PY="/tmp/vllm-cu124/bin/python"
elif [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PY="${ROOT}/.venv/bin/python"
else
  PY="$(command -v python3)"
fi

bash "${ROOT}/reasoning_branch_dataset/scripts/setup_env_fast.sh"

ANALYSIS_FLAG="--no-analysis"
if "${PY}" -c "import sklearn,seaborn" 2>/dev/null; then
  ANALYSIS_FLAG=""
fi

exec "${PY}" -m reasoning_branch_dataset.pipeline \
  --output-dir "${ROOT}/reasoning_branch_dataset/outputs/pilot_v1" \
  --math500-limit "${MATH500_LIMIT:-3}" \
  --aime-limit "${AIME_LIMIT:-2}" \
  ${ANALYSIS_FLAG}

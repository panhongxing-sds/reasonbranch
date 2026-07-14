#!/usr/bin/env bash
set -euo pipefail
ROOT="/mnt/afs/L202500372"
PY="$(command -v python3)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
DATA_DIR="${1:-${ROOT}/reasoning_branch_dataset/outputs/action_study_v1}"
exec "${PY}" -m reasoning_branch_dataset.action_study.analyze --data-dir "${DATA_DIR}"

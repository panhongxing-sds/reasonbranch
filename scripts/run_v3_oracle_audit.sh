#!/usr/bin/env bash
set -euo pipefail
AFS="${AFS:-/mnt/afs/L202500372}"
export PYTHONPATH="${AFS}"
PY="${PY:-/tmp/vllm-cu124/bin/python}"
exec "$PY" -m reasoning_branch_dataset.action_study.run_v3_oracle_audit "$@"

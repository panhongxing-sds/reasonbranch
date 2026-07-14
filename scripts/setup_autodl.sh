#!/usr/bin/env bash
# Autodl server one-shot setup for reasonbranch.
set -euo pipefail

AUTODL_ROOT="/root/autodl-tmp"
REPO="${AUTODL_ROOT}/reasonbranch"
export AFS="${AUTODL_ROOT}"
export PYTHONPATH="${AFS}:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# Package symlink (scripts import reasoning_branch_dataset)
ln -sfn "${REPO}" "${AUTODL_ROOT}/reasoning_branch_dataset"

mkdir -p "${AUTODL_ROOT}/specreason/models" "${AUTODL_ROOT}/logs"

# .env for autodl
cat > "${REPO}/.env" <<EOF
HF_ENDPOINT=https://hf-mirror.com
AFS=${AUTODL_ROOT}
DRAFT_MODEL=${AUTODL_ROOT}/specreason/models/DeepSeek-R1-Distill-Qwen-1.5B
TARGET_MODEL=${AUTODL_ROOT}/specreason/models/DeepSeek-R1-Distill-Qwen-32B
EOF

echo "AFS=${AFS}"
echo "DRAFT_MODEL=${AUTODL_ROOT}/specreason/models/DeepSeek-R1-Distill-Qwen-1.5B"
echo "TARGET_MODEL=${AUTODL_ROOT}/specreason/models/DeepSeek-R1-Distill-Qwen-32B"
echo "setup_autodl.sh done"

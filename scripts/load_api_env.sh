#!/usr/bin/env bash
# Load teacher API env from a local key file (never committed).
set -euo pipefail

_AFS="${AFS:-/mnt/afs/L202500372}"
KEYFILE="${1:-${TEACHER_KEYFILE:-${_AFS}/key/api}}"

if [[ ! -f "${KEYFILE}" ]]; then
  echo "WARN: ${KEYFILE} not found — set TEACHER_API_KEY or TEACHER_KEYFILE"
  return 0 2>/dev/null || exit 0
fi

GR_KEY="$(grep -oE 'sk-gr-[a-zA-Z0-9]+' "${KEYFILE}" | tail -1 || true)"
if [[ -z "${GR_KEY}" ]]; then
  GR_KEY="$(grep -oE 'sk-gr-[a-zA-Z0-9]+' "${KEYFILE}" | head -1 || true)"
fi
if [[ -n "${GR_KEY}" ]]; then
  export TEACHER_API_KEY="${TEACHER_API_KEY:-${GR_KEY}}"
  export TEACHER_BASE_URL="${TEACHER_BASE_URL:-https://endpoint.greatrouter.com}"
  export TEACHER_MODEL="${TEACHER_MODEL:-gpt-5.5}"
fi

if [[ -z "${TEACHER_API_KEY:-}" ]]; then
  OR_KEY="$(grep -oE 'sk-or-v1-[a-zA-Z0-9]+' "${KEYFILE}" | head -1 || true)"
  if [[ -n "${OR_KEY}" ]]; then
    export TEACHER_API_KEY="${OR_KEY}"
    export TEACHER_BASE_URL="${TEACHER_BASE_URL:-https://openrouter.ai/api/v1}"
    export TEACHER_MODEL="${TEACHER_MODEL:-openai/gpt-4.1}"
  fi
fi

if [[ -z "${TEACHER_API_KEY:-}" ]]; then
  echo "WARN: no API key parsed from ${KEYFILE}"
else
  echo "API env loaded (model=${TEACHER_MODEL}, base=${TEACHER_BASE_URL})"
fi

#!/usr/bin/env bash
# Reuse /tmp/vllm-cu124 (torch+transformers). No heavy pip unless analysis libs missing.
set -euo pipefail
ROOT="/mnt/afs/L202500372"
PY="/tmp/vllm-cu124/bin/python"

if [[ ! -x "${PY}" ]]; then
  PY="${ROOT}/.venv/bin/python"
  [[ -x "${PY}" ]] || { python3 -m venv "${ROOT}/.venv" --system-site-packages; PY="${ROOT}/.venv/bin/python"; }
fi

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
"${PY}" - <<'PY'
import torch, transformers, sys
print("env ready:", sys.executable)
print("  torch", torch.__version__, "cuda", torch.cuda.is_available())
print("  transformers", transformers.__version__)
PY

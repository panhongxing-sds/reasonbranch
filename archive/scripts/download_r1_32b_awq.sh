#!/usr/bin/env bash
# Download AWQ 4bit DeepSeek-R1-Distill-Qwen-32B (~18GB) via hf-mirror, using
# snapshot_download with resume; retry loop until all index shards present.
set -uo pipefail
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
DEST="${DEST:-/mnt/afs/L202500372/specreason/models/DeepSeek-R1-Distill-Qwen-32B-AWQ}"
REPO="${REPO:-casperhansen/deepseek-r1-distill-qwen-32b-awq}"
PY="${PY:-/tmp/vllm-cu124/bin/python}"
mkdir -p "$DEST"

for attempt in $(seq 1 40); do
  echo "[$(date '+%F %T')] attempt $attempt: snapshot_download $REPO"
  "$PY" - "$REPO" "$DEST" <<'PYEOF'
import os, sys, time
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from huggingface_hub import snapshot_download
repo, dest = sys.argv[1], sys.argv[2]
try:
    snapshot_download(
        repo_id=repo, local_dir=dest, max_workers=4,
        allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.model", "*.txt"],
    )
    print("SNAPSHOT_OK")
except Exception as e:
    print("SNAPSHOT_ERR", repr(e)[:300])
PYEOF
  MISSING=$("$PY" - "$DEST" <<'PYEOF'
import json, sys, os
d = sys.argv[1]
idx = os.path.join(d, "model.safetensors.index.json")
if not os.path.exists(idx):
    print("INDEX_MISSING"); sys.exit(0)
shards = set(json.load(open(idx))["weight_map"].values())
miss = [s for s in shards if not os.path.exists(os.path.join(d, s))]
print("|".join(miss) if miss else "OK")
PYEOF
)
  if [[ "$MISSING" == "OK" ]]; then
    echo "[$(date '+%F %T')] ALL SHARDS PRESENT — complete"
    exit 0
  fi
  echo "[$(date '+%F %T')] still missing: $MISSING ; retry in 5s"
  sleep 5
done
echo "[$(date '+%F %T')] FAILED after retries"; exit 1

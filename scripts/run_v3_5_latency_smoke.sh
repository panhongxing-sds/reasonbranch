#!/usr/bin/env bash
# V3.5 smoke: tiny latency bench (few states) to validate the pipeline.
set -euo pipefail
export N_STATES="${N_STATES:-4}"
export N_PROBLEMS="${N_PROBLEMS:-4}"
export OUT="${OUT:-/root/autodl-tmp/reasonbranch/outputs/action_study_v35_latency_smoke}"
bash "$(dirname "$0")/run_v3_5_latency_bench.sh"

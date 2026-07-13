#!/usr/bin/env bash
# No-API local pipeline: engineering fixes + probe + verifier dataset.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
setup_vllm_env

echo "[1/5] build probe datasets"
"$PY" -m reasoning_branch_dataset.action_study.build_probe_dataset --v3-dir "$V3_DIR" --v2-dir "$V2_DIR"

echo "[2/5] train two-stage probes (GroupKFold)"
"$PY" -m reasoning_branch_dataset.action_study.train_local_probe

echo "[3/5] build verifier candidate dataset"
"$PY" -m reasoning_branch_dataset.action_study.build_verifier_dataset --v3-dir "$V3_DIR" --v2-dir "$V2_DIR"

echo "[4/5] grading regression on v2 traces"
"$PY" -m reasoning_branch_dataset.action_study.grading_regression --v2-dir "$V2_DIR" --n-traces 50

echo "[5/5] target step diagnostic (100 prefixes, needs GPU)"
"$PY" -m reasoning_branch_dataset.action_study.target_step_diagnostic --v2-dir "$V2_DIR" --n-prefixes 100

echo "done -> outputs/{probe_datasets,probe_models,verifier_dataset,grading_regression,target_step_diagnostic}"

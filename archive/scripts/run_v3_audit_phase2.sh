#!/usr/bin/env bash
# Phase 2 V3 audit: shuffle-rescore (200) + pairwise (162 weak Branch) + report
set -euo pipefail
AFS="${AFS:-/mnt/afs/L202500372}"
export PYTHONPATH="${AFS}"
PY="${PY:-/tmp/vllm-cu124/bin/python}"
V2="${V2:-${AFS}/reasoning_branch_dataset/outputs/action_study_pilot_v2}"
V3="${V3:-${AFS}/reasoning_branch_dataset/outputs/action_study_pilot_v3}"
OUT="${AFS}/reasoning_branch_dataset/outputs"
LOG="${AFS}/reasoning_branch_dataset/logs/v3_audit_phase2.log"
mkdir -p "$(dirname "$LOG")"

echo "[$(date '+%F %T')] Phase2 shuffle-rescore" | tee -a "$LOG"
"$PY" -m reasoning_branch_dataset.action_study.run_v3_shuffle_rescore \
  --v2-dir "$V2" --v3-dir "$V3" 2>&1 | tee -a "$LOG"

echo "[$(date '+%F %T')] Phase2 pairwise audit" | tee -a "$LOG"
"$PY" -m reasoning_branch_dataset.action_study.run_v3_pairwise_audit \
  --v2-dir "$V2" --v3-dir "$V3" 2>&1 | tee -a "$LOG"

echo "[$(date '+%F %T')] Phase2 report" | tee -a "$LOG"
"$PY" -m reasoning_branch_dataset.action_study.run_v3_audit_phase2_report \
  --v3-dir "$V3" --report-path "${OUT}/pilot_v3_audit_phase2_report.md" 2>&1 | tee -a "$LOG"

echo "[$(date '+%F %T')] done -> ${OUT}/pilot_v3_audit_phase2_report.md" | tee -a "$LOG"

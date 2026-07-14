"""Load QwQ once, run multiple target-side jobs (reachable verify + utility scoring)."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

from reasoning_branch_dataset.action_study.run_reachable_state import (
    ReachableStateConfig,
    phase3_verify,
    write_report,
)
from reasoning_branch_dataset.action_study.run_utility_scoring import run_utility_scoring
from reasoning_branch_dataset.action_study.target_verifier import build_target_verifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Single QwQ load: reachable p3 + v3 utility")
    parser.add_argument("--reachable-dir", default="/mnt/afs/L202500372/reasoning_branch_dataset/outputs/reachable_state_pilot")
    parser.add_argument("--v2-data-dir", default="/mnt/afs/L202500372/reasoning_branch_dataset/outputs/action_study_pilot_v2")
    parser.add_argument("--v3-out-dir", default="/mnt/afs/L202500372/reasoning_branch_dataset/outputs/action_study_pilot_v3")
    parser.add_argument("--target-model", default="/mnt/afs/L202500372/specreason/models/QwQ-32B")
    parser.add_argument("--skip-reachable", action="store_true")
    parser.add_argument("--skip-utility", action="store_true")
    args = parser.parse_args()

    verifier = build_target_verifier(args.target_model, engine="vllm", debug=True)
    try:
        if not args.skip_reachable:
            cfg = ReachableStateConfig(
                out_dir=Path(args.reachable_dir),
                target_model=args.target_model,
                draft_model="",
            )
            drafts = cfg.out_dir / "drafts.jsonl"
            if drafts.exists():
                phase3_verify(cfg, drafts, resume=True, verifier=verifier)
                write_report(cfg, cfg.out_dir / "verify_results.jsonl")
        if not args.skip_utility:
            run_utility_scoring(
                Path(args.v2_data_dir),
                Path(args.v3_out_dir),
                target_model=args.target_model,
                verifier=verifier,
            )
    finally:
        del verifier
        gc.collect()


if __name__ == "__main__":
    main()

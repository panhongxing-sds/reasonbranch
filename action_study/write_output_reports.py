"""Regenerate all version reports directly under outputs/."""

from __future__ import annotations

import argparse
from pathlib import Path

from reasoning_branch_dataset.action_study.analyze import run_analysis
from reasoning_branch_dataset.action_study.run_reachable_state import ReachableStateConfig, write_report
from reasoning_branch_dataset.action_study.run_v3_oracle_audit import write_audit_report
from reasoning_branch_dataset.action_study.run_v3_gpt_oracle import write_gpt_oracle_report
from reasoning_branch_dataset.action_study.run_v3_gpt_step_oracle import write_step_oracle_report
from reasoning_branch_dataset.action_study.run_utility_scoring import write_utility_report

DEFAULT_ROOT = Path("/mnt/afs/L202500372/reasoning_branch_dataset/outputs")
DEFAULT_TARGET = "/mnt/afs/L202500372/specreason/models/QwQ-32B"


def write_all_reports(
    output_root: Path,
    *,
    target_model: str = DEFAULT_TARGET,
) -> list[Path]:
    v2_dir = output_root / "action_study_pilot_v2"
    v3_dir = output_root / "action_study_pilot_v3"
    reachable_dir = output_root / "reachable_state_pilot"

    paths: list[Path] = []
    paths.append(run_analysis(v2_dir, report_path=output_root / "pilot_v2_report.md"))
    paths.append(
        write_utility_report(
            v2_dir,
            v3_dir,
            target_model=target_model,
            report_path=output_root / "pilot_v3_report.md",
        )
    )
    cfg = ReachableStateConfig(
        out_dir=reachable_dir,
        target_model=target_model,
        draft_model="/mnt/afs/L202500372/models/Qwen3.5-4B",
    )
    paths.append(
        write_report(
            cfg,
            reachable_dir / "verify_results.jsonl",
            report_path=output_root / "reachable_state_report.md",
        )
    )
    paths.append(
        write_audit_report(
            v2_dir,
            v3_dir,
            report_path=output_root / "pilot_v3_audit_report.md",
        )
    )
    gpt_step_report = output_root / "pilot_v3_3_report.md"
    if (v3_dir / "gpt_step_labels.jsonl").exists():
        paths.append(write_step_oracle_report(v3_dir, gpt_step_report))
    gpt_pair_report = output_root / "pilot_v3_2_report.md"
    if (v3_dir / "gpt_pairwise_results.jsonl").exists():
        paths.append(write_gpt_oracle_report(v3_dir, gpt_pair_report))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Write pilot_v2/v3/reachable reports to outputs/")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--target-model", default=DEFAULT_TARGET)
    args = parser.parse_args()
    for path in write_all_reports(args.output_root, target_model=args.target_model):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()

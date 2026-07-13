#!/usr/bin/env python3
"""Post-process labels and run analysis on collected dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from reasoning_branch_dataset.labels import (
    compute_branch_labels,
    compute_rollback_labels,
    merge_labels,
)


def load_table(path: Path) -> pd.DataFrame:
    if path.with_suffix(".jsonl").exists():
        return pd.read_json(path.with_suffix(".jsonl"), lines=True)
    return pd.read_parquet(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-analysis", action="store_true")
    args = parser.parse_args()
    d = args.data_dir

    traces_df = load_table(d / "traces.parquet")
    prefixes_df = load_table(d / "prefixes.parquet")
    token_features_df = load_table(d / "token_features.parquet")
    step_branches_df = load_table(d / "step_branches.parquet")
    verification_df = load_table(d / "verification_logs.parquet")

    # Only label prefixes that were selected for rollout
    if "selected_for_rollout" in prefixes_df.columns:
        selected_prefixes = prefixes_df[prefixes_df["selected_for_rollout"] == 1]
    else:
        selected_prefixes = prefixes_df

    next_step_df = step_branches_df[step_branches_df["branch_mode"] == "next_step"]
    branch_labels = compute_branch_labels(selected_prefixes, next_step_df, traces_df)
    rollback_labels = compute_rollback_labels(verification_df)
    labels_df = merge_labels(branch_labels, rollback_labels, token_features_df)

    labels_df.to_parquet(d / "labels.parquet", index=False)
    print(f"Wrote {d / 'labels.parquet'} ({len(labels_df)} rows)")

    summary = {
        "n_prefixes": len(prefixes_df),
        "branch_rate": float(labels_df["branch_label"].mean()) if len(labels_df) else 0.0,
        "prefix_types": prefixes_df["prefix_type"].value_counts().to_dict(),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.run_analysis:
        from reasoning_branch_dataset.analyze import (
            analysis_api_enrichment,
            analysis_entropy_correlation,
            analysis_hidden_branch_probe,
            analysis_prefix_type_branch_rate,
            analysis_rollback_probe,
            save_analysis_outputs,
        )

        prefix_rates = analysis_prefix_type_branch_rate(labels_df, prefixes_df)
        entropy_stats = analysis_entropy_correlation(labels_df)
        api_stats = analysis_api_enrichment(labels_df, prefixes_df)
        hidden_path = d / "hidden.safetensors"
        branch_probe = (
            analysis_hidden_branch_probe(labels_df, prefixes_df, hidden_path)
            if hidden_path.exists()
            else {}
        )
        rollback_probe = (
            analysis_rollback_probe(labels_df, prefixes_df, hidden_path)
            if hidden_path.exists()
            else {}
        )
        report = save_analysis_outputs(
            d, prefix_rates, entropy_stats, branch_probe, rollback_probe, api_enrichment=api_stats
        )
        print(f"Analysis report: {report}")


if __name__ == "__main__":
    main()

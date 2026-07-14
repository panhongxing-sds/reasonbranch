"""Analyze Phase-1 uncertainty study: behavior state × Continue vs Branch outcomes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from reasoning_branch_dataset.action_study.diversity import (
    BEHAVIOR_TABLE_ORDER,
    STATE_CURRENT_UNRELIABLE,
    STATE_FUTURE_DIVERSE,
    STATE_STABLE,
    STATE_UNCLEAR,
    normalize_legacy_state,
)

STATE_TABLE_ORDER = [
    (STATE_STABLE, "Valid + Low Diversity (legacy)"),
    (STATE_FUTURE_DIVERSE, "Valid + High Diversity (legacy)"),
    (STATE_CURRENT_UNRELIABLE, "Invalid Prefix (legacy)"),
    (STATE_UNCLEAR, "Unclear"),
]


def load_table(data_dir: Path, name: str) -> pd.DataFrame:
    rescored = data_dir / f"{name}.rescored.jsonl"
    j = data_dir / f"{name}.jsonl"
    p = data_dir / f"{name}.parquet"
    if rescored.exists() and name in {"traces", "actions", "action_results", "outcome_results"}:
        return pd.read_json(rescored, lines=True)
    # Prefer jsonl when newer than parquet (avoids stale API_ERROR parquet)
    if j.exists() and p.exists() and j.stat().st_mtime >= p.stat().st_mtime:
        return pd.read_json(j, lines=True)
    if p.exists():
        return pd.read_parquet(p)
    if j.exists():
        return pd.read_json(j, lines=True)
    return pd.DataFrame()


def _outcomes(data_dir: Path) -> pd.DataFrame:
    for name in ("outcome_results", "action_results"):
        df = load_table(data_dir, name)
        if not df.empty:
            return df
    return pd.DataFrame()


def _oracle_col(df: pd.DataFrame) -> pd.Series:
    if "oracle_recoverable" in df.columns:
        return df["oracle_recoverable"]
    if "pass_at_k" in df.columns:
        return df["pass_at_k"]
    return df.get("is_correct", pd.Series(dtype=float))


def _op_col(df: pd.DataFrame) -> pd.Series:
    if "operation" in df.columns:
        return df["operation"]
    return df.get("action", pd.Series(dtype=str))


def _evaluated(df: pd.DataFrame) -> pd.DataFrame:
    if "evaluation_status" not in df.columns:
        return df
    return df[df["evaluation_status"] == "OK"]


def _aggregate_state_rows(
    prefixes: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    state_col: str,
    state_key: str,
    label: str,
) -> dict | None:
    sub_p = prefixes[prefixes[state_col] == state_key]
    if sub_p.empty:
        return None
    pids = set(sub_p["prefix_id"])
    sub_o = outcomes[outcomes["prefix_id"].isin(pids)]
    cont = _evaluated(sub_o[sub_o["_op"] == "continue"])
    branch = _evaluated(sub_o[sub_o["_op"] == "branch"])
    cont_err = len(sub_o[sub_o["_op"] == "continue"]) - len(cont)
    branch_err = len(sub_o[sub_o["_op"] == "branch"]) - len(branch)

    cont_acc = float(cont["_oracle"].mean()) if len(cont) else None
    branch_p4 = float(branch["_oracle"].mean()) if len(branch) else None
    branch_acc = (
        float(branch["branch_accuracy_at_k"].mean())
        if len(branch) and "branch_accuracy_at_k" in branch.columns
        else None
    )
    branch_gain = None
    if cont_acc is not None and branch_p4 is not None:
        branch_gain = branch_p4 - cont_acc

    return {
        "state": label,
        "n_prefixes": len(sub_p),
        "n_continue_evaluated": len(cont),
        "n_branch_evaluated": len(branch),
        "n_continue_errors": cont_err,
        "n_branch_errors": branch_err,
        "continue_accuracy": cont_acc,
        "branch_pass_at_4": branch_p4,
        "branch_accuracy_at_4": branch_acc,
        "branch_gain": branch_gain,
    }


def build_behavior_table(
    data_dir: Path,
    *,
    substantive_only: bool = True,
    admission_col: str = "admission_main",
) -> pd.DataFrame:
    prefixes = load_table(data_dir, "prefixes")
    outcomes = _outcomes(data_dir)
    if prefixes.empty or outcomes.empty:
        return pd.DataFrame()

    prefixes = prefixes.copy()
    outcomes = outcomes.copy()
    if substantive_only and admission_col in prefixes.columns:
        prefixes = prefixes[prefixes[admission_col] == True]  # noqa: E712
    elif substantive_only and "admission_pass" in prefixes.columns:
        prefixes = prefixes[prefixes["admission_pass"] == True]  # noqa: E712
    elif substantive_only and "include_in_main_experiment" in prefixes.columns:
        prefixes = prefixes[prefixes["include_in_main_experiment"] == True]  # noqa: E712
    elif substantive_only and "prefix_substantiveness" in prefixes.columns:
        prefixes = prefixes[prefixes["prefix_substantiveness"] == "SUBSTANTIVE"]

    state_col = "behavior_state" if "behavior_state" in prefixes.columns else "state_bucket"
    outcomes["_oracle"] = _oracle_col(outcomes)
    outcomes["_op"] = _op_col(outcomes)

    rows = []
    for state_key, label in BEHAVIOR_TABLE_ORDER:
        row = _aggregate_state_rows(prefixes, outcomes, state_col=state_col, state_key=state_key, label=label)
        if row:
            rows.append(row)
    return pd.DataFrame(rows)


def build_main_table(data_dir: Path) -> pd.DataFrame:
    """Legacy v1 table — kept for comparison."""
    prefixes = load_table(data_dir, "prefixes")
    outcomes = _outcomes(data_dir)
    if prefixes.empty or outcomes.empty:
        return pd.DataFrame()

    prefixes = prefixes.copy()
    prefixes["state_bucket"] = prefixes["state_bucket"].map(normalize_legacy_state)
    outcomes = outcomes.copy()
    outcomes["state_bucket"] = outcomes["state_bucket"].map(normalize_legacy_state)
    outcomes["_oracle"] = _oracle_col(outcomes)
    outcomes["_op"] = _op_col(outcomes)

    rows = []
    for state_key, label in STATE_TABLE_ORDER:
        row = _aggregate_state_rows(prefixes, outcomes, state_col="state_bucket", state_key=state_key, label=label)
        if row:
            rows.append(row)
    return pd.DataFrame(rows)


def pilot_readiness_checks(behavior_table: pd.DataFrame) -> dict:
    out: dict = {
        "checks": {},
        "ready_for_scale_up": False,
        "ready_for_probe": False,
        "ready_for_target_replay": False,
        "metric_tier": {
            "primary": "target_acceptance_gain (requires target replay)",
            "auxiliary": ["branch_pass_at_4", "branch_gain_correctness", "continue_accuracy"],
        },
    }
    if behavior_table.empty:
        return out

    def _row(label: str) -> pd.Series | None:
        sub = behavior_table[behavior_table["state"] == label]
        return sub.iloc[0] if len(sub) else None

    stable = _row("Stable")
    decision = _row("Decision-sensitive")
    recoverable = _row("Corrupted-recoverable")
    stuck = _row("Corrupted-stuck")

    if stable is not None:
        out["checks"]["stable_continue_approx_branch"] = bool(
            abs(stable["continue_accuracy"] - stable["branch_pass_at_4"]) < 0.15
        )
    if decision is not None and decision.get("branch_gain") is not None:
        out["checks"]["decision_sensitive_correctness_gain_positive"] = bool(
            decision["branch_gain"] > 0.05
        )
        out["checks"]["decision_sensitive_exists"] = decision["n_prefixes"] >= 3
    else:
        out["checks"]["decision_sensitive_correctness_gain_positive"] = False
        out["checks"]["decision_sensitive_exists"] = False

    checks = out["checks"]
    out["ready_for_scale_up"] = bool(
        checks.get("decision_sensitive_exists", False)
        and checks.get("corrupted_recoverable_exists", False)
        and checks.get("corrupted_stuck_exists", False)
    )
    # Probe training requires target acceptance labels, not correctness branch_gain.
    out["ready_for_probe"] = False
    out["ready_for_target_replay"] = out["ready_for_scale_up"]
    return out


def _load_admission_summary(data_dir: Path) -> dict:
    path = data_dir / "admission_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def build_pipeline_map_section(data_dir: Path) -> list[str]:
    """Short context block; full reports live as siblings under outputs/."""
    root = data_dir.parent
    v3_report = root / "pilot_v3_report.md"
    reachable_report = root / "reachable_state_report.md"

    def _ok(path: Path) -> str:
        return "已完成" if path.exists() else "未生成"

    lines = [
        "## 相关报告（同 `outputs/` 目录）",
        "",
        "| 版本 | 报告文件 | 状态 |",
        "|------|----------|------|",
        "| v2（本报告） | `pilot_v2_report.md` | 数据采集 + 正确率 |",
        f"| v3 utility oracle | `pilot_v3_report.md` | {_ok(v3_report)} |",
        f"| reachable-state | `reachable_state_report.md` | {_ok(reachable_report)} |",
        "",
        "### 本报告范围",
        "",
        "- **包含**：4B draft Continue/Branch、终答正确率、behavior state、完整 case。",
        "- **不含**：主指标 G_B（见 `reachable_state_report.md`）；utility oracle（见 `pilot_v3_report.md`）。",
        "- **原始数据**：`action_study_pilot_v2/*.jsonl`（本报告不重复列出）。",
        "",
    ]
    return lines


def run_analysis(data_dir: Path, report_path: Path | None = None) -> Path:
    admission_path = data_dir / "admission_summary.json"
    if not admission_path.exists():
        from reasoning_branch_dataset.action_study.admission_derive import recompute_admission

        recompute_admission(data_dir)

    admission = _load_admission_summary(data_dir)
    behavior_table = build_behavior_table(data_dir, substantive_only=True, admission_col="admission_main")
    behavior_partial = build_behavior_table(data_dir, substantive_only=True, admission_col="admission_partial")
    legacy_table = build_main_table(data_dir)
    verdict = pilot_readiness_checks(behavior_table)

    report = report_path or (data_dir.parent / "pilot_v2_report.md")
    lines = [
        "# Pilot v2 — Uncertainty Study Report",
        "",
        "> 4B draft only. Labels: behavior state + recovery profile (correctness auxiliary).",
        "> **Admission re-derived from final traces/actions (ignores stale pipeline admission_pass).**",
        "",
    ]
    lines.extend(build_pipeline_map_section(data_dir))
    lines.extend(
        [
            "## Admission Summary",
        ]
    )
    if admission:
        lines.append("| metric | count |")
        lines.append("|--------|------:|")
        for key in (
            "total_prefixes",
            "complete_trace_prefixes",
            "substantive_prefixes",
            "continue_ok",
            "branch_4of4_ok",
            "branch_partial",
            "admission_main",
            "admission_partial",
            "excluded_missing_context",
            "excluded_truncated_trace",
            "excluded_action_incomplete",
            "excluded_non_substantive",
            "traces_ok",
            "traces_truncated",
            "excluded_problems_missing_figure",
            "unique_complete_trace_problems",
        ):
            if key in admission:
                lines.append(f"| {key} | {admission[key]} |")
        if admission.get("branch_valid_histogram"):
            lines.append("")
            lines.append("**Branch valid count histogram**")
            lines.append("")
            lines.append("| valid_branches | prefixes |")
            lines.append("|----------------|---------:|")
            for k, v in admission["branch_valid_histogram"].items():
                lines.append(f"| {k}/4 | {v} |")
    else:
        lines.append("_no admission summary_")

    lines.extend(
        [
            "",
            "## Behavior State Table (`admission_main` only)",
            behavior_table.to_markdown(index=False) if not behavior_table.empty else "_no data_",
            "",
            "## Behavior State Table (`admission_partial`, branch≥3/4)",
            behavior_partial.to_markdown(index=False) if not behavior_partial.empty else "_no data_",
            "",
            "## Legacy v1 Table (for comparison)",
            legacy_table.to_markdown(index=False) if not legacy_table.empty else "_no data_",
            "",
            "## Labeling Notes",
            "- Strategy diversity uses API strategy-level clustering (cluster_v2); heuristic is conservative.",
            "- Main table uses `admission_main`: complete trace + substantive + continue OK + branch 4/4.",
            "- Partial table uses `admission_partial`: branch≥3/4 (not standard Pass@4).",
            "- **`branch_pass_at_4` / correctness `branch_gain` are auxiliary only** — not Branch utility oracle.",
            "- Primary Branch metric: `target_acceptance_gain = max_j A_j - A_single` (requires target replay).",
            "- Behavior states are exploratory; see `docs/target_acceptance_framework.md`.",
            "- Truncated base traces produce no prefixes in this pilot.",
            "- `branch_accuracy_at_4` = mean(correct_branches / evaluated_branches); distinct from pass@4.",
            "- Report `n_*_evaluated` and `n_*_errors` — do not compare metrics with mismatched denominators.",
            "",
            "## Pilot Readiness (`admission_main`, exploratory)",
        ]
    )
    for k, v in verdict["checks"].items():
        lines.append(f"- {k}: {v}")
    lines.append(f"- **ready_for_scale_up**: {verdict['ready_for_scale_up']}")
    lines.append(f"- **ready_for_target_replay**: {verdict.get('ready_for_target_replay', False)}")
    lines.append(f"- **ready_for_probe**: {verdict['ready_for_probe']} (requires target acceptance labels)")
    lines.append("")
    lines.append("## Current Status & Next Step")
    lines.append("")
    lines.append("1. **v2 数据采集**：完成（本报告）。")
    lines.append("2. **draft-only target replay**：已尝试，4B draft prefix 不在 QwQ reachable 状态，无有效 G_B。")
    lines.append("3. **reachable-state**：完成 → 同目录 `reachable_state_report.md`。")
    lines.append("4. **v3 utility oracle**：完成 → 同目录 `pilot_v3_report.md`。")
    lines.append("5. **下一步**：先做 V3 oracle audit → `pilot_v3_audit_report.md`；**审计通过前不训练 probe**。")
    lines.append("")
    lines.append("_Correctness `branch_gain` 仅作辅助；主 Branch 信号看 reachable G_B 或 v3 utility oracle。_")

    from reasoning_branch_dataset.action_study.report_samples import (
        format_uncertainty_cases_md,
        pick_uncertainty_cases,
    )

    uncertainty_cases = pick_uncertainty_cases(data_dir, admission_col="admission_main", n_each=2)
    lines.extend(format_uncertainty_cases_md(uncertainty_cases))

    report.write_text("\n".join(lines))

    summary_path = data_dir / "uncertainty_study_summary.json"

    def _json_default(obj: object):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        raise TypeError(type(obj))

    summary_path.write_text(
        json.dumps(
            {
                "admission": admission,
                "behavior_table": behavior_table.to_dict("records"),
                "behavior_table_partial": behavior_partial.to_dict("records"),
                "legacy_table": legacy_table.to_dict("records"),
                "verdict": verdict,
            },
            indent=2,
            default=_json_default,
        )
    )
    if not behavior_table.empty:
        behavior_table.to_csv(data_dir / "uncertainty_behavior_table.csv", index=False)
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    path = run_analysis(args.data_dir)
    print(f"Report: {path}")


if __name__ == "__main__":
    main()

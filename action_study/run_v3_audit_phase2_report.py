"""Write combined Phase 2 audit report (shuffle + pairwise + unlock gates)."""

from __future__ import annotations

import json
from pathlib import Path


UNLOCK = {
    "accept_agreement": 0.90,
    "oracle_action_agreement": 0.85,
    "branch_precision": 0.70,
    "min_cleaned_branch_n": 50,
}


def write_phase2_report(
    v3_dir: Path,
    *,
    report_path: Path,
    n_total: int = 1548,
) -> Path:
    audit_summary = _load(v3_dir / "audit_summary.json")
    shuffle = _load(v3_dir / "shuffle_rescore_summary.json")
    pairwise = _load(v3_dir / "pairwise_audit_summary.json")

    lines = [
        "# V3 Oracle Audit — Phase 2",
        "",
        "> Shuffle-rescore stability + pairwise judge on raw weak Branch cases.",
        "",
        "## Unlock gates (probe training)",
        "",
        "| gate | threshold | current | pass |",
        "|------|-----------|---------|------|",
    ]

    def _row(name: str, thr: float, cur: float | None) -> None:
        ok = cur is not None and cur >= thr
        lines.append(f"| {name} | {thr:.0%} | {cur:.1%} if cur is not None else '—' | {'✓' if ok else '✗'} |")

    # fix f-string in _row
    aa = shuffle.get("accept_agreement")
    oa = shuffle.get("oracle_action_agreement")
    bp = pairwise.get("precision_branch_vs_qwq_weak")
    bn = pairwise.get("branch_better_count", 0)

    lines = [
        "# V3 Oracle Audit — Phase 2",
        "",
        "> Shuffle-rescore stability + pairwise judge on raw weak Branch cases.",
        "",
        "## Oracle tiers (reminder)",
        "",
        "| tier | definition | n (pilot) |",
        "|------|------------|----------:|",
        f"| **Weak Branch** | u₀<τ, max≥τ | ~{audit_summary.get('oracle_raw_tau7', {}).get('branch_rescuable', 162)} |",
        f"| **Strong Branch** | u₀≤4, max≥7, Δ≥3 | {audit_summary.get('strict_counts', {}).get('branch_rescuable', 28)} |",
        f"| **DATA_ERROR** | incomplete candidates | {1548 - audit_summary.get('n_oracle_eligible', 1527)} |",
        "",
        "## Unlock gates (probe training)",
        "",
        "| gate | threshold | current | pass |",
        "|------|-----------|---------|------|",
        f"| Accept/Reject agreement (shuffle) | ≥90% | {_pct(aa)} | {_pass(aa, 0.9)} |",
        f"| Oracle action agreement (shuffle) | ≥85% | {_pct(oa)} | {_pass(oa, 0.85)} |",
        f"| Branch precision (pairwise) | ≥70% | {_pct(bp)} | {_pass(bp, 0.7)} |",
        f"| Cleaned Branch count | ≥50 | {bn} | {'✓' if bn >= 50 else '✗'} |",
        "",
    ]

    if shuffle:
        lines.extend(
            [
                "## 1. Shuffle-rescore stability",
                "",
                f"- prefixes rescored: **{shuffle.get('n_prefixes', 0)}**",
                f"- exact score agreement: **{_pct(shuffle.get('exact_agreement'))}**",
                f"- within-1 agreement: **{_pct(shuffle.get('within1_agreement'))}**",
                f"- accept agreement (u≥7): **{_pct(shuffle.get('accept_agreement'))}**",
                f"- oracle action agreement: **{_pct(shuffle.get('oracle_action_agreement'))}**",
                "",
            ]
        )
    else:
        lines.append("_shuffle-rescore not run yet_\n")

    if pairwise:
        vc = pairwise.get("verdict_counts", {})
        lines.extend(
            [
                "## 2. Pairwise judge (raw weak Branch cases)",
                "",
                f"- reviewed: **{pairwise.get('n_reviewed', 0)}** / ~162",
                f"- BRANCH_BETTER: **{vc.get('BRANCH_BETTER', 0)}** ({_pct(pairwise.get('branch_better_count', 0) / max(1, pairwise.get('n_reviewed', 1)))})",
                f"- EQUIVALENT: **{vc.get('EQUIVALENT', 0)}**",
                f"- GREEDY_BETTER: **{vc.get('GREEDY_BETTER', 0)}**",
                f"- BOTH_REJECT: **{vc.get('BOTH_REJECT', 0)}**",
                "",
                f"- **Precision** (QwQ weak ∧ judge=BRANCH_BETTER): **{_pct(bp)}**",
                f"- **Estimated cleaned Branch rate**: **{_pct(pairwise.get('cleaned_branch_rate_of_1548'))}** of all prefixes",
                "",
                "### Interpretation",
                "",
                "- `EQUIVALENT` on Case 1/2-style pairs → absolute utility oracle noise, not real Branch.",
                "- True Branch = QwQ weak **and** pairwise `BRANCH_BETTER`.",
                "- If cleaned N<50 → train **Continue vs Non-Continue** first, or rare Branch detector.",
                "",
            ]
        )
    else:
        lines.append("_pairwise audit not run yet_\n")

    all_pass = (
        (aa or 0) >= 0.9
        and (oa or 0) >= 0.85
        and (bp or 0) >= 0.7
        and bn >= 50
    )
    lines.extend(
        [
            "## Verdict",
            "",
            f"**Probe training: {'UNLOCKED' if all_pass else 'BLOCKED'}**",
            "",
            "Next if blocked: hardened-prompt rescore on weak Branch + sample Handoff/Continue; "
            "then re-run pairwise on disagreements.",
            "",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _pct(x: float | int | None) -> str:
    if x is None:
        return "—"
    if isinstance(x, int) and x > 1:
        return f"{x}"
    return f"{100 * float(x):.1f}%"


def _pass(x: float | None, thr: float) -> str:
    if x is None:
        return "—"
    return "✓" if x >= thr else "✗"


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--v3-dir", default="/mnt/afs/L202500372/reasoning_branch_dataset/outputs/action_study_pilot_v3")
    p.add_argument("--report-path", default="/mnt/afs/L202500372/reasoning_branch_dataset/outputs/pilot_v3_audit_phase2_report.md")
    args = p.parse_args()
    path = write_phase2_report(Path(args.v3_dir), report_path=Path(args.report_path))
    print(f"Wrote {path}")

"""V3 utility oracle audit: step admission, score calibration, length bias, V2×V3 cross-tab."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from reasoning_branch_dataset.action_study.oracle_labels import (
    classify_oracle,
    classify_oracle_strict,
    summarize_oracle_table,
)
from reasoning_branch_dataset.action_study.step_admission import (
    annotate_candidate_details,
    prefix_oracle_eligibility,
)
from reasoning_branch_dataset.action_study.report_samples import _clip


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _score_histogram(rows: list[dict[str, Any]]) -> dict[int, int]:
    hist: Counter[int] = Counter()
    for row in rows:
        for s in row.get("utility_scores") or []:
            hist[int(s)] += 1
    return dict(sorted(hist.items()))


def _enrich_rows(
    rows: list[dict[str, Any]],
    problems: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        prob = problems.get(row["problem_id"], {})
        question = prob.get("question", "")
        details = annotate_candidate_details(row.get("candidate_details") or [], question=question)
        elig = prefix_oracle_eligibility(details)
        eligible_flags = [d.get("eligible_for_oracle", False) for d in details]
        scores = row.get("utility_scores") or []
        oracle7 = classify_oracle(scores, tau=7, eligible=eligible_flags if len(eligible_flags) == len(scores) else None)
        strict = classify_oracle_strict(scores, eligible=eligible_flags if len(eligible_flags) == len(scores) else None)
        out.append(
            {
                **row,
                "question": question,
                "candidate_details": details,
                **elig,
                "oracle_label": oracle7["oracle_label"],
                "oracle_label_filtered": oracle7["oracle_label"],
                "oracle_tier": strict.get("oracle_tier", oracle7.get("oracle_tier")),
                **strict,
            }
        )
    return out


def _length_bias_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs: list[tuple[int, int]] = []
    high_lens: list[int] = []
    low_lens: list[int] = []
    for row in rows:
        for d in row.get("candidate_details") or []:
            u = d.get("utility_score")
            ln = d.get("step_chars") or len(d.get("candidate_step") or "")
            if u is None:
                continue
            pairs.append((int(u), int(ln)))
            if int(u) >= 7:
                high_lens.append(ln)
            if int(u) < 7:
                low_lens.append(ln)
    if not pairs:
        return {}
    us, lens = zip(*pairs)
    corr = float(np.corrcoef(us, lens)[0, 1]) if len(pairs) > 2 else 0.0
    return {
        "n_candidates": len(pairs),
        "corr_score_chars": corr,
        "mean_chars_u_ge_7": float(np.mean(high_lens)) if high_lens else None,
        "mean_chars_u_lt_7": float(np.mean(low_lens)) if low_lens else None,
    }


def _stratified_sample(
    rows: list[dict[str, Any]],
    *,
    n_continue: int = 80,
    n_branch: int = 80,
    n_handoff: int = 40,
    seed: int = 42,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        lab = r.get("oracle_label", "unknown")
        if lab == "branch_rescuable":
            lab = "weak_branch_rescuable"
        buckets[lab].append(r)
    out: list[dict[str, Any]] = []
    for label, n in (
        ("continue_sufficient", n_continue),
        ("weak_branch_rescuable", n_branch),
        ("handoff_required", n_handoff),
    ):
        pool = buckets.get(label, [])
        rng.shuffle(pool)
        out.extend(pool[:n])
    return out


def _v2_cross_tab(
    enriched: list[dict[str, Any]],
    prefixes: dict[str, dict[str, Any]],
    *,
    label_key: str = "oracle_label_filtered",
) -> list[dict[str, Any]]:
    state_col = "behavior_state" if any("behavior_state" in p for p in prefixes.values()) else "state_bucket"
    tab: dict[str, Counter[str]] = defaultdict(Counter)
    for row in enriched:
        pfx = prefixes.get(row["prefix_id"], {})
        state = pfx.get(state_col) or "Unknown"
        tab[state][row.get(label_key, "unknown")] += 1

    rows_out: list[dict[str, Any]] = []
    for state in sorted(tab):
        c = tab[state]
        n = sum(c.values())
        rows_out.append(
            {
                "behavior_state": state,
                "n": n,
                "continue_sufficient": c.get("continue_sufficient", 0),
                "weak_branch_rescuable": c.get("weak_branch_rescuable", c.get("branch_rescuable", 0)),
                "handoff_required": c.get("handoff_required", 0),
                "data_error": c.get("data_error", 0),
                "pct_branch": round(
                    100 * c.get("weak_branch_rescuable", c.get("branch_rescuable", 0)) / n, 2
                )
                if n
                else 0,
            }
        )
    return rows_out


def _pick_audit_cases(enriched: list[dict[str, Any]], n: int = 6) -> list[dict[str, Any]]:
    """Flag length-sensitive branch rescue and inadmissible handoff."""
    cases: list[dict[str, Any]] = []

    branch_rows = [
        r for r in enriched if r.get("oracle_label") in ("weak_branch_rescuable", "branch_rescuable")
    ]
    for row in sorted(branch_rows, key=lambda r: -(r.get("branch_rescue_gain") or 0)):
        details = row.get("candidate_details") or []
        if len(details) < 5:
            continue
        u0 = details[0].get("utility_score", 0)
        best_i = max(range(1, 5), key=lambda i: details[i].get("utility_score", 0))
        l0 = details[0].get("step_chars", 0)
        lb = details[best_i].get("step_chars", 0)
        if (details[best_i].get("utility_score", 0) - u0) >= 3 and lb > l0 * 1.2:
            cases.append(
                {
                    "audit_flag": "length_sensitive_branch_rescue",
                    "prefix_id": row["prefix_id"],
                    "u_greedy": u0,
                    "u_best_branch": details[best_i].get("utility_score"),
                    "greedy_chars": l0,
                    "best_branch_chars": lb,
                    "greedy_step": _clip(details[0].get("candidate_step", ""), 500),
                    "best_branch_step": _clip(details[best_i].get("candidate_step", ""), 500),
                }
            )
        if len(cases) >= 2:
            break

    inadmissible = [r for r in enriched if not r.get("oracle_eligible") and r.get("oracle_label") == "handoff_required"]
    for row in inadmissible[:2]:
        cases.append(
            {
                "audit_flag": "handoff_with_incomplete_steps",
                "prefix_id": row["prefix_id"],
                "step_qualities": row.get("step_qualities"),
                "question_tail": _clip(row.get("question", ""), 200),
                "candidates": [
                    {
                        "name": d.get("candidate"),
                        "u": d.get("utility_score"),
                        "quality": d.get("step_quality"),
                        "step": _clip(d.get("candidate_step", ""), 300),
                    }
                    for d in (row.get("candidate_details") or [])
                ],
            }
        )
    return cases[:n]


def _format_cases_md(cases: list[dict[str, Any]]) -> list[str]:
    lines = ["", "## Audit Flag Cases", ""]
    for i, c in enumerate(cases, 1):
        lines.append(f"### Audit Case {i}: `{c['audit_flag']}` (`{c['prefix_id']}`)")
        lines.append("")
        if c["audit_flag"] == "length_sensitive_branch_rescue":
            lines.extend(
                [
                    f"- greedy u={c['u_greedy']} ({c['greedy_chars']} chars) vs branch u={c['u_best_branch']} ({c['best_branch_chars']} chars)",
                    "",
                    "**Greedy step:**",
                    "```text",
                    c.get("greedy_step", ""),
                    "```",
                    "",
                    "**Best branch step:**",
                    "```text",
                    c.get("best_branch_step", ""),
                    "```",
                    "",
                ]
            )
        else:
            lines.append(f"- step qualities: `{c.get('step_qualities')}`")
            lines.append("")
            for cand in c.get("candidates", []):
                lines.extend(
                    [
                        f"**{cand['name']}** u={cand['u']} quality={cand['quality']}",
                        "```text",
                        cand.get("step", ""),
                        "```",
                        "",
                    ]
                )
    return lines


def write_audit_report(
    v2_dir: Path,
    v3_dir: Path,
    *,
    report_path: Path,
    sample_seed: int = 42,
) -> Path:
    scores_path = v3_dir / "utility_scores_QwQ-32B.jsonl"
    rows = [r for r in _load_jsonl(scores_path) if len(r.get("utility_scores", [])) >= 5]
    problems = {r["problem_id"]: r for r in _load_jsonl(v2_dir / "problems.jsonl")}
    prefixes = {r["prefix_id"]: r for r in _load_jsonl(v2_dir / "prefixes.jsonl")}

    enriched = _enrich_rows(rows, problems)
    hist = _score_histogram(rows)
    length_stats = _length_bias_stats(enriched)

    # Step quality breakdown (all candidates)
    qual_counter: Counter[str] = Counter()
    for row in enriched:
        for d in row.get("candidate_details") or []:
            qual_counter[d.get("step_quality", "UNKNOWN")] += 1

    n_eligible = sum(1 for r in enriched if r.get("oracle_eligible"))
    n_inadmissible = len(enriched) - n_eligible

    filtered_rows = [r for r in enriched if r.get("oracle_eligible")]
    table_raw = summarize_oracle_table(rows, [5, 6, 7, 8])
    table_filtered = summarize_oracle_table(
        [
            {
                **r,
                "utility_scores": [
                    d["utility_score"]
                    for d in r["candidate_details"]
                    if d.get("eligible_for_oracle")
                ],
            }
            for r in filtered_rows
        ],
        [5, 6, 7, 8],
    )

    strict_labels = [
        classify_oracle_strict(
            r["utility_scores"],
            eligible=[d.get("eligible_for_oracle", False) for d in r["candidate_details"]],
        )["oracle_tier"]
        for r in enriched
        if r.get("oracle_eligible")
    ]
    strict_counts = Counter(strict_labels)

    sample = _stratified_sample(enriched, seed=sample_seed)
    cross = _v2_cross_tab(enriched, prefixes, label_key="oracle_label")
    cross_f = _v2_cross_tab(filtered_rows, prefixes, label_key="oracle_label_filtered")
    audit_cases = _pick_audit_cases(enriched)

    lines = [
        "# V3 Oracle Audit Report",
        "",
        "> **Do not train probe until this audit passes.** Validates utility scorer + step extraction.",
        "",
        f"- scored prefixes: **{len(rows)}**",
        f"- audit sample (stratified): **{len(sample)}** → `action_study_pilot_v3/audit_sample.jsonl`",
        "",
        "## Verdict (preliminary)",
        "",
    ]

    branch_pct_raw = next((t["pct_branch_rescuable"] for t in table_raw if t["tau"] == 7), 0)
    branch_pct_f = next((t["pct_branch_rescuable"] for t in table_filtered if t["tau"] == 7), 0)
    corr = length_stats.get("corr_score_chars", 0)
    lines.extend(
        [
            f"- Weak Branch @ τ=7: **{branch_pct_raw}%** (u₀<τ, max≥τ)",
            f"- After complete-step filter: **{branch_pct_f}%** ({n_eligible}/{len(rows)} prefixes eligible)",
            f"- **Strong Branch** (u₀≤4, max≥7, Δ≥3): **{strict_counts.get('strong_branch_rescuable', 0)}** ({100*strict_counts.get('strong_branch_rescuable',0)/max(1,len(strict_labels)):.1f}%)",
            f"- DATA_ERROR (incomplete candidates): **{n_inadmissible}** prefixes — exclude from Handoff",
            f"- Score–length correlation: **{corr:.3f}** (|r|>0.15 → length bias risk)",
            f"- Inadmissible prefixes (any incomplete step): **{n_inadmissible}** ({100*n_inadmissible/max(1,len(rows)):.1f}%)",
            "",
            "**Recommendation**: probe training blocked until shuffle-rescore stability + manual review of audit cases.",
            "",
            "## 1. Score distribution P(u)",
            "",
            "| u | count | % |",
            "|---|------:|--:|",
        ]
    )
    total_scores = sum(hist.values())
    for u, cnt in hist.items():
        lines.append(f"| {u} | {cnt} | {100*cnt/total_scores:.1f}% |")

    lines.extend(["", "## 2. Step quality (per candidate)", "", "| quality | count | % |", "|---------|------:|--:|"])
    total_q = sum(qual_counter.values())
    for q, cnt in qual_counter.most_common():
        lines.append(f"| {q} | {cnt} | {100*cnt/total_q:.1f}% |")

    lines.extend(["", "## 3. Oracle table — raw vs filtered (complete steps only)", ""])
    lines.append("| τ | raw Continue | raw Branch | raw Handoff | filt Continue | filt Branch | filt Handoff |")
    lines.append("|---|-------------:|-----------:|------------:|--------------:|------------:|-------------:|")
    raw_by_tau = {t["tau"]: t for t in table_raw}
    fil_by_tau = {t["tau"]: t for t in table_filtered}
    for tau in [5, 6, 7, 8]:
        r, f = raw_by_tau.get(tau, {}), fil_by_tau.get(tau, {})
        lines.append(
            f"| {tau} | {r.get('pct_continue_sufficient',0)}% | {r.get('pct_branch_rescuable',0)}% | {r.get('pct_handoff_required',0)}% "
            f"| {f.get('pct_continue_sufficient',0)}% | {f.get('pct_branch_rescuable',0)}% | {f.get('pct_handoff_required',0)}% |"
        )

    lines.extend(
        [
            "",
            "## 4. Length bias",
            "",
            f"- corr(u, step_chars): **{corr:.3f}**",
            f"- mean chars | u≥7: **{length_stats.get('mean_chars_u_ge_7')}**",
            f"- mean chars | u<7: **{length_stats.get('mean_chars_u_lt_7')}**",
            "",
            "## 5. V2 behavior_state × V3 oracle (raw τ=7)",
            "",
            "| behavior_state | n | Continue | Branch | Handoff | %Branch |",
            "|----------------|--:|---------:|-------:|--------:|--------:|",
        ]
    )
    for row in cross:
        lines.append(
            f"| {row['behavior_state']} | {row['n']} | {row['continue_sufficient']} | {row['weak_branch_rescuable']} "
            f"| {row['handoff_required']} | {row['pct_branch']}% |"
        )

    lines.extend(
        [
            "",
            "## 6. V2 × V3 (filtered, complete steps only)",
            "",
            "| behavior_state | n | Continue | Branch | Handoff | %Branch |",
            "|----------------|--:|---------:|-------:|--------:|--------:|",
        ]
    )
    for row in cross_f:
        lines.append(
            f"| {row['behavior_state']} | {row['n']} | {row['continue_sufficient']} | {row['weak_branch_rescuable']} "
            f"| {row['handoff_required']} | {row['pct_branch']}% |"
        )

    ds = [r for r in cross if r["behavior_state"] == "Decision-sensitive"]
    st = [r for r in cross if r["behavior_state"] == "Stable"]
    if ds and st:
        lines.extend(
            [
                "",
                f"- P(V3 Branch | V2 Decision-sensitive) = **{ds[0]['pct_branch']}%**",
                f"- P(V3 Branch | V2 Stable) = **{st[0]['pct_branch']}%**",
                "",
            ]
        )

    lines.extend(
        [
            "## 7. Next checks (not yet run)",
            "",
            "1. **Shuffle rescore stability**: re-score audit sample with QwQ, check score agreement.",
            "2. **Independent judge**: binary ACCEPT/REJECT on Branch-rescuable cases.",
            "3. **Re-score with hardened prompt** (already in specreason_scorer for future runs).",
            "",
        ]
    )
    lines.extend(_format_cases_md(audit_cases))

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    audit_json = v3_dir / "audit_summary.json"
    audit_json.write_text(
        json.dumps(
            {
                "n_scored": len(rows),
                "n_oracle_eligible": n_eligible,
                "score_histogram": hist,
                "step_quality_counts": dict(qual_counter),
                "length_bias": length_stats,
                "oracle_raw_tau7": raw_by_tau.get(7),
                "oracle_filtered_tau7": fil_by_tau.get(7),
                "strict_counts": dict(strict_counts),
                "v2_cross_tab": cross,
                "v2_cross_tab_filtered": cross_f,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    sample_path = v3_dir / "audit_sample.jsonl"
    sample_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sample),
        encoding="utf-8",
    )
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="V3 oracle audit")
    parser.add_argument(
        "--v2-dir",
        default="/mnt/afs/L202500372/reasoning_branch_dataset/outputs/action_study_pilot_v2",
    )
    parser.add_argument(
        "--v3-dir",
        default="/mnt/afs/L202500372/reasoning_branch_dataset/outputs/action_study_pilot_v3",
    )
    parser.add_argument(
        "--report-path",
        default="/mnt/afs/L202500372/reasoning_branch_dataset/outputs/pilot_v3_audit_report.md",
    )
    args = parser.parse_args()
    path = write_audit_report(Path(args.v2_dir), Path(args.v3_dir), report_path=Path(args.report_path))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

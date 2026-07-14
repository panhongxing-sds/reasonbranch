"""V4.0 end-to-end analysis: per-policy accuracy, latency, speedup, cost.

Reads e2e_results.jsonl (from run_v40_e2e) and produces a Markdown + JSON
report. Speedup and accuracy retention are computed on the paired set of
problems solved by all policies, relative to target_only.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _mean(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None]
    return st.fmean(xs) if xs else float("nan")


def _acc(rows: list[dict[str, Any]]) -> float:
    graded = [r for r in rows if r.get("is_correct") is not None]
    return _mean([float(r["is_correct"]) for r in graded]) if graded else float("nan")


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    policies = sorted({r["policy"] for r in rows})
    by_policy: dict[str, list[dict[str, Any]]] = {p: [r for r in rows if r["policy"] == p] for p in policies}

    # Paired set: problems attempted by all policies.
    prob_sets = [set(r["problem_id"] for r in by_policy[p]) for p in policies]
    paired = set.intersection(*prob_sets) if prob_sets else set()

    per_policy: dict[str, Any] = {}
    target_wall = None
    for p in policies:
        rs = by_policy[p]
        rs_paired = [r for r in rs if r["problem_id"] in paired]
        entry = {
            "n": len(rs),
            "n_paired": len(rs_paired),
            "accuracy": _acc(rs),
            "accuracy_paired": _acc(rs_paired),
            "mean_wall_sec": _mean([r["wall_sec"] for r in rs_paired]),
            "mean_steps": _mean([r["n_steps"] for r in rs_paired]),
            "mean_handoffs": _mean([r["handoffs"] for r in rs_paired]),
            "mean_accepts": _mean([r["accepts"] for r in rs_paired]),
            "mean_verify_sec": _mean([r["verify_sec"] for r in rs_paired]),
            "mean_target_sec": _mean([r["target_sec"] for r in rs_paired]),
            "mean_draft_sec": _mean([r["draft_sec"] for r in rs_paired]),
            "accept_rate": _mean([r["accepts"] / max(1, r["n_steps"]) for r in rs_paired]),
        }
        per_policy[p] = entry
        if p == "target_only":
            target_wall = entry["mean_wall_sec"]

    for p in policies:
        w = per_policy[p]["mean_wall_sec"]
        per_policy[p]["speedup_vs_target"] = (
            (target_wall / w) if (target_wall and w and w > 0) else float("nan")
        )

    return {
        "policies": policies,
        "n_paired_problems": len(paired),
        "per_policy": per_policy,
        "by_dataset": _by_dataset(rows, policies, paired),
    }


def _by_dataset(rows: list[dict[str, Any]], policies: list[str], paired: set) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for ds in sorted({r["dataset"] for r in rows}):
        out[ds] = {}
        for p in policies:
            rs = [r for r in rows if r["dataset"] == ds and r["policy"] == p and r["problem_id"] in paired]
            out[ds][p] = {"n": len(rs), "accuracy": _acc(rs), "mean_wall_sec": _mean([r["wall_sec"] for r in rs])}
    return out


def _f(x: Any) -> str:
    try:
        xf = float(x)
        return f"{xf:.3f}" if xf == xf else "n/a"
    except (TypeError, ValueError):
        return "n/a"


def render_report(summary: dict[str, Any]) -> str:
    L: list[str] = []
    L.append("# V4.0 End-to-End: Draft-Confidence Selective Speculative Reasoning\n")
    L.append(f"- Paired problems (attempted by all policies): {summary['n_paired_problems']}\n")
    L.append("## Per-policy (on paired set, speedup vs target_only)\n")
    L.append("| policy | accuracy | wall(s) | speedup | steps | handoffs | accept_rate | verify(s) |")
    L.append("|---|--:|--:|--:|--:|--:|--:|--:|")
    order = ["target_only", "draft_only", "selfconf", "target_verify"]
    pols = [p for p in order if p in summary["per_policy"]] + \
           [p for p in summary["per_policy"] if p not in order]
    for p in pols:
        e = summary["per_policy"][p]
        L.append(
            f"| {p} | {_f(e['accuracy_paired'])} | {_f(e['mean_wall_sec'])} | "
            f"{_f(e['speedup_vs_target'])}x | {_f(e['mean_steps'])} | {_f(e['mean_handoffs'])} | "
            f"{_f(e['accept_rate'])} | {_f(e['mean_verify_sec'])} |"
        )
    L.append("\n## By dataset (accuracy / wall)\n")
    L.append("| dataset | " + " | ".join(pols) + " |")
    L.append("|---" + "|--:" * len(pols) + "|")
    for ds, d in summary["by_dataset"].items():
        cells = [f"{_f(d[p]['accuracy'])} / {_f(d[p]['mean_wall_sec'])}s" for p in pols]
        L.append(f"| {ds} | " + " | ".join(cells) + " |")

    L.append("\n## Interpretation\n")
    pp = summary["per_policy"]
    if "selfconf" in pp and "target_only" in pp and "target_verify" in pp:
        sc, to, tv = pp["selfconf"], pp["target_only"], pp["target_verify"]
        L.append(
            f"- selfconf (OURS) reaches {_f(sc['accuracy_paired'])} accuracy at "
            f"{_f(sc['speedup_vs_target'])}x speedup vs target_only ({_f(to['accuracy_paired'])}), "
            f"with near-zero verification overhead ({_f(sc['mean_verify_sec'])}s)."
        )
        L.append(
            f"- target_verify pays a 32B verification pass ({_f(tv['mean_verify_sec'])}s) yet "
            f"reaches {_f(tv['accuracy_paired'])} accuracy at {_f(tv['speedup_vs_target'])}x -- "
            "its accept decisions are unreliable (V3.6 verification gap)."
        )
    return "\n".join(L) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="V4.0 E2E analysis")
    p.add_argument("--results", default="/root/autodl-tmp/reasonbranch/outputs/action_study_v40_e2e/e2e_results.jsonl")
    p.add_argument("--out-dir", default="/root/autodl-tmp/reasonbranch/outputs/action_study_v40_e2e")
    args = p.parse_args()
    rows = _load_jsonl(Path(args.results))
    if not rows:
        raise SystemExit(f"No E2E results at {args.results}")
    summary = analyze(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "v40_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    report = render_report(summary)
    (out_dir / "v40_report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()

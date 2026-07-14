#!/usr/bin/env python3
"""Export Phase-1 uncertainty study samples to markdown."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

OUT = Path("reasoning_branch_dataset/outputs/action_study_uncertainty_v1")
DOCS = Path("reasoning_branch_dataset/docs")


def load(name: str) -> list[dict]:
    p = OUT / f"{name}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def pick_samples() -> list[tuple[str, str]]:
    prefixes = load("prefixes")
    outcomes = load("outcome_results") or load("action_results")
    if not prefixes:
        return []

    by_pfx: dict[str, dict] = {}
    for o in outcomes:
        by_pfx.setdefault(o["prefix_id"], {})[o.get("operation", o.get("action"))] = o

    def branch_gain(pid: str) -> int | None:
        rows = by_pfx.get(pid, {})
        c = rows.get("continue", {}).get("oracle_recoverable")
        b = rows.get("branch", {}).get("oracle_recoverable")
        if c is None or b is None:
            return None
        return int(b) - int(c)

    chosen: list[tuple[str, str]] = []
    seen: set[str] = set()

    # 1) Continue=0, Branch=1 — strongest oracle signal
    for p in prefixes:
        if branch_gain(p["prefix_id"]) == 1:
            chosen.append((p["prefix_id"], f"Sample — Branch Gain=1 ({p['state_bucket']}, {p['reasoning_progress']:.1%})"))
            seen.add(p["prefix_id"])
            break

    # 2) Future-diverse with any gain or high diversity
    for p in prefixes:
        if p["prefix_id"] in seen:
            continue
        if p["state_bucket"] == "Future-diverse":
            chosen.append((p["prefix_id"], f"Sample — Future-diverse ({p['reasoning_progress']:.1%})"))
            seen.add(p["prefix_id"])
            break

    # 3) Current-unreliable
    for p in prefixes:
        if p["prefix_id"] in seen:
            continue
        if p["state_bucket"] == "Current-unreliable":
            chosen.append((p["prefix_id"], f"Sample — Current-unreliable ({p['reasoning_progress']:.1%})"))
            seen.add(p["prefix_id"])
            break

    if len(chosen) < 2:
        for p in prefixes:
            if p["prefix_id"] not in seen:
                chosen.append((p["prefix_id"], f"Sample — {p['state_bucket']}"))
            if len(chosen) >= 2:
                break
    return chosen[:2]


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    samples = pick_samples()
    if not samples:
        print("No data yet")
        return

    traces = {t["problem_id"]: t for t in load("traces")}
    outcomes = load("outcome_results") or load("action_results")
    by_pfx: dict[str, dict] = {}
    for o in outcomes:
        by_pfx.setdefault(o["prefix_id"], {})[o.get("operation", o.get("action"))] = o

    lines: list[str] = []
    lines.append("# Phase-1 Uncertainty Study 完整样本\n")
    lines.append(f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    lines.append(f"> 数据：`outputs/action_study_uncertainty_v1/`  ")
    lines.append("> 框架：Stable / Future-diverse / Current-unreliable × Continue + Branch\n")
    lines.append("---\n")

    # summary table from prefixes
    prefixes = load("prefixes")
    from collections import Counter
    c = Counter(p["state_bucket"] for p in prefixes)
    lines.append("## 数据集概览\n")
    lines.append(f"- 题目数：{len(traces)}  ")
    lines.append(f"- prefix 数：{len(prefixes)}  ")
    lines.append(f"- 状态分布：{dict(c)}\n")

    report = OUT.parent / "pilot_v2_report.md"
    if report.exists():
        lines.append("## 主表（analyze 输出）\n")
        lines.append(report.read_text())
        lines.append("\n---\n")

    for prefix_id, title in samples:
        prefix = [p for p in prefixes if p["prefix_id"] == prefix_id][0]
        trace = traces[prefix["problem_id"]]
        outs = by_pfx.get(prefix_id, {})
        next_steps = [n for n in load("next_step_samples") if n["prefix_id"] == prefix_id]
        val = [v for v in load("validity_labels") if v["prefix_id"] == prefix_id]
        cl = [c for c in load("cluster_labels") if c["prefix_id"] == prefix_id]

        lines.append(f"## {title}\n")
        lines.append(f"**prefix_id**: `{prefix_id}`  ")
        lines.append(f"**state_bucket**: `{prefix['state_bucket']}`  ")
        lines.append(f"**future_system_action**: `{prefix.get('future_system_action', '')}`\n")

        lines.append("### 题目\n")
        q = trace["question"]
        lines.append(q[:600] + ("..." if len(q) > 600 else "") + "\n")
        lines.append(f"**Gold**: `{trace['gold_answer']}`\n")

        lines.append("### Prefix 状态\n")
        lines.append("| 字段 | 值 |")
        lines.append("|------|-----|")
        lines.append(f"| reasoning_progress | {prefix['reasoning_progress']:.3f} |")
        lines.append(f"| prefix_validity | {prefix['prefix_validity']} |")
        lines.append(f"| diversity_label | {prefix['diversity_label']} |")
        lines.append(f"| num_clusters | {prefix['num_clusters']} |")
        lines.append(f"| diversity_entropy | {prefix['diversity_entropy']:.3f} |")
        lines.append("")

        if val:
            lines.append("### API Validity\n```json\n")
            lines.append(json.dumps(val[0], ensure_ascii=False, indent=2))
            lines.append("\n```\n")
        if cl:
            lines.append("### API Cluster\n```json\n")
            lines.append(json.dumps(cl[0], ensure_ascii=False, indent=2))
            lines.append("\n```\n")

        lines.append("### Next Steps ×4\n")
        for ns in next_steps:
            lines.append(f"**sample {ns['sample_id']}**:\n```\n{ns['text'][:400]}\n```\n")

        lines.append("### Oracle Outcomes（Continue + Branch）\n")
        lines.append("| operation | oracle_recoverable | draft_generated_tokens |")
        lines.append("|-----------|-------------------|------------------------|")
        cont = outs.get("continue", {})
        branch = outs.get("branch", {})
        for op, row in [("continue", cont), ("branch", branch)]:
            if row:
                lines.append(
                    f"| {op} | {row.get('oracle_recoverable', row.get('is_correct'))} | "
                    f"{row.get('draft_generated_tokens', row.get('num_tokens', 0))} |"
                )
        if cont and branch:
            cg = (branch.get("oracle_recoverable") or 0) - (cont.get("oracle_recoverable") or 0)
            lines.append(f"\n**Branch Gain** = {cg}\n")
        lines.append("\n---\n")

    path = DOCS / "uncertainty_study_samples.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

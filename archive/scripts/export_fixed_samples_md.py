#!/usr/bin/env python3
"""Export two fixed action-study samples to markdown."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from reasoning_branch_dataset.action_study.api_validity import ValidityClient
from reasoning_branch_dataset.action_study.diversity import compute_diversity, state_bucket

OUT = Path("reasoning_branch_dataset/outputs/action_study_v1")
DOCS = Path("reasoning_branch_dataset/docs")


def load(name: str, *, rescored: bool = False) -> list[dict]:
    suffix = ".rescored" if rescored and name in {"traces", "actions", "action_results"} else ""
    path = OUT / f"{name}{suffix}.jsonl"
    if rescored and not path.exists():
        path = OUT / f"{name}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def reasoning_only(prefix_text: str) -> str:
    if "</think>" in prefix_text:
        return prefix_text.split("</think>", 1)[1].strip()
    if "Problem:" in prefix_text:
        tail = prefix_text.split("Problem:", 1)[1]
        return tail[tail.find("\n") + 1 :] if "\n" in tail else tail
    return prefix_text


def pick(prefix_id: str) -> tuple[dict, dict, list, list, list, list]:
    trace = [t for t in load("traces", rescored=True) if t["problem_id"] == "math500_0000"][-1]
    prefix = [p for p in load("prefixes") if p["prefix_id"] == prefix_id][0]
    next_steps = [n for n in load("next_step_samples") if n["prefix_id"] == prefix_id]
    actions = [a for a in load("actions", rescored=True) if a["prefix_id"] == prefix_id]
    results = [r for r in load("action_results", rescored=True) if r["prefix_id"] == prefix_id]
    reviews = [r for r in load("api_reviews") if r["prefix_id"] == prefix_id]
    return trace, prefix, next_steps, actions, results, reviews


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    client = ValidityClient.from_env(cache_path=OUT / "api_cache_fixed.jsonl")

    samples = [
        ("math500_0000_p03_paragraph_end", "Sample 1 — prefix p03（MIDDLE，进度 26.9%）"),
        ("math500_0000_p04_paragraph_end", "Sample 2 — prefix p04（LATE，进度 94.1%）"),
    ]

    lines: list[str] = []
    lines.append("# Action Study 修复后完整样本（2 题）\n")
    lines.append(f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    lines.append("> 修复项：① grader ② API 不传 temperature  ")
    lines.append("> 判分来源：`traces.rescored.jsonl` / `actions.rescored.jsonl`  ")
    lines.append("> API：修复后实时调用 gpt-5.5\n")
    lines.append("---\n")

    lines.append("## 全局流程\n\n```\n")
    lines.append(
        "题目 → greedy trace [grade_math_answer]\n"
        "  → prefix 切分\n"
        "  → next_step ×4 + API validity + API cluster\n"
        "  → Continue@1 / Branch@4 / Rollback@4 [grade_math_answer, require \\boxed{}]\n"
    )
    lines.append("```\n\n---\n")

    for prefix_id, title in samples:
        trace, prefix, next_steps, actions, results, reviews = pick(prefix_id)
        question = trace["question"]
        gold = trace["gold_answer"]
        rp = reasoning_only(prefix["prefix_text"])

        val = client.label_prefix(
            prefix_id=prefix_id,
            question=question,
            gold_answer=gold,
            reasoning_prefix=rp,
        )
        step_texts = [n["text"] for n in next_steps]
        cl = client.cluster_next_steps(
            prefix_id=prefix_id,
            question=question,
            reasoning_prefix=rp,
            next_steps=step_texts,
        )
        div = compute_diversity(step_texts, api_clusters=cl.get("clusters") or None)
        bucket = state_bucket(val["prefix_validity"], div["diversity_label"])

        lines.append(f"## {title}\n")
        lines.append("**problem_id**: `math500_0000`  ")
        lines.append(f"**prefix_id**: `{prefix_id}`\n")

        lines.append("### 0. 题目\n")
        lines.append(f"{question}\n")
        lines.append(f"**Gold**: `{gold}`\n")

        lines.append("### 1. Greedy 完整推理\n")
        lines.append("| 字段 | 值 |")
        lines.append("|------|-----|")
        lines.append(f"| predicted_answer | `{trace['predicted_answer']}` |")
        lines.append(f"| is_correct | **{trace['is_correct']}** |")
        lines.append(f"| evaluation_status | `{trace.get('evaluation_status', 'OK')}` |")
        lines.append(f"| token_count | {trace['token_count']} |")
        lines.append(f"| generation_latency | {trace['generation_latency']:.1f}s |")
        lines.append("")
        lines.append("<details><summary>full_reasoning</summary>\n\n```\n")
        lines.append(trace["full_reasoning"])
        lines.append("\n```\n</details>\n")

        lines.append("### 2. Prefix 切分\n")
        lines.append("| 字段 | 值 |")
        lines.append("|------|-----|")
        lines.append(f"| prefix_type | `{prefix['prefix_type']}` |")
        lines.append(f"| reasoning_progress | **{prefix['reasoning_progress']:.3f}** |")
        lines.append(f"| previous_checkpoint | {'有' if prefix['previous_checkpoint'] else '无'} |")
        lines.append("")
        pt = prefix["prefix_text"]
        if "Problem:" in pt:
            pt = pt[pt.index("Problem:") :]
        lines.append("**prefix 末尾：**\n```\n")
        lines.append(pt[-700:])
        lines.append("\n```\n")

        lines.append("### 3. Next steps ×4\n")
        for ns in next_steps:
            lines.append(f"**sample {ns['sample_id']}** ({ns['num_tokens']} tok):\n```\n")
            lines.append(ns["text"])
            lines.append("\n```\n")

        lines.append("### 4. API Validity（修复后）\n```json\n")
        lines.append(json.dumps(val, ensure_ascii=False, indent=2))
        lines.append("\n```\n")

        lines.append("### 5. API Cluster + Diversity\n```json\n")
        lines.append(
            json.dumps(
                {
                    **cl,
                    "diversity_label": div["diversity_label"],
                    "diversity_entropy": div["diversity_entropy"],
                    "num_clusters": div["num_clusters"],
                    "state_bucket": bucket,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        lines.append("\n```\n")

        lines.append("### 6. 三动作结果（rescored）\n")
        lines.append("| action | pass@k | evaluation_status | tokens | latency |")
        lines.append("|--------|--------|-------------------|--------|---------|")
        for ar in results:
            lines.append(
                f"| {ar['action']} | {ar.get('pass_at_k', ar.get('is_correct'))} | "
                f"{ar.get('evaluation_status', 'OK')} | {ar['num_tokens']} | {ar['latency_sec']:.1f}s |"
            )
        lines.append("")

        lines.append("### 7. 动作明细节选\n")
        for act in actions:
            cont = act.get("continuation", "")
            trunc = cont[:600] + ("...(truncated)" if len(cont) > 600 else "")
            lines.append(f"#### {act['action_type']} sample {act['sample_id']}\n")
            lines.append(f"- predicted_answer: `{act.get('predicted_answer', '')}`")
            lines.append(f"- is_correct: {act.get('is_correct')}")
            lines.append(f"- evaluation_status: `{act.get('evaluation_status', 'OK')}`\n")
            lines.append(f"```\n{trunc}\n```\n")

        if reviews:
            lines.append("### 8. API Review（旧 run 触发）\n```json\n")
            lines.append(json.dumps(reviews[0], ensure_ascii=False, indent=2))
            lines.append("\n```\n")

        cont = next((r for r in results if r["action"] == "continue"), {})
        br = next((r for r in results if r["action"] == "branch"), {})
        rb = next((r for r in results if r["action"] == "rollback"), {})
        lines.append("### 9. 小结\n")
        lines.append(f"- validity: **{val['prefix_validity']}** (conf={val.get('confidence', 0)})")
        lines.append(f"- diversity: **{div['diversity_label']}**")
        lines.append(f"- state_bucket: `{bucket}`")
        lines.append(
            f"- Continue@1={cont.get('is_correct')}, "
            f"Branch oracle_recoverable={br.get('pass_at_k')}, "
            f"Rollback oracle_recoverable={rb.get('pass_at_k')}"
        )
        if prefix["reasoning_progress"] > 0.9:
            lines.append("- 接近 TERMINAL，动作选择研究价值低")
        lines.append("\n---\n")

    path = DOCS / "action_study_2_samples_fixed.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

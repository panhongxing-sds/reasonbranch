#!/usr/bin/env python3
"""Export research-valuable action-study samples to markdown."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from reasoning_branch_dataset.action_study.api_validity import ValidityClient
from reasoning_branch_dataset.action_study.diversity import compute_diversity, state_bucket

OUT = Path("reasoning_branch_dataset/outputs/action_study_v1")
DOCS = Path("reasoning_branch_dataset/docs")

SAMPLES = [
    {
        "prefix_id": "math500_0004_p00_paragraph_end",
        "title": "Sample 1 — math500_0004 EARLY（8.6%）Rollback 不可用",
        "why": [
            "推理最早期（progress=0.086），路线尚未展开",
            "Rollback=None：无 previous_checkpoint，只能 Continue / Branch",
            "展示「动作可用性」随 prefix 位置变化",
        ],
    },
    {
        "prefix_id": "math500_0004_p01_paragraph_end",
        "title": "Sample 2 — math500_0004 MIDDLE（45.5%）唯一 Forward-uncertain",
        "why": [
            "当前 18 个 prefix 中唯一的 Forward-uncertain（VALID + HIGH_DIVERSITY）",
            "next step 聚为 2 个语义簇：「逐个算速度」vs「先算 Evelyn」",
            "Continue@1 480 tok vs Branch@4 2082 tok — 精度-成本权衡可研究",
        ],
    },
]


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
    prefix = [p for p in load("prefixes") if p["prefix_id"] == prefix_id][0]
    problem_id = prefix["problem_id"]
    trace = [t for t in load("traces", rescored=True) if t["problem_id"] == problem_id][-1]
    next_steps = [n for n in load("next_step_samples") if n["prefix_id"] == prefix_id]
    actions = [a for a in load("actions", rescored=True) if a["prefix_id"] == prefix_id]
    results = [r for r in load("action_results", rescored=True) if r["prefix_id"] == prefix_id]
    reviews = [r for r in load("api_reviews") if r["prefix_id"] == prefix_id]
    return trace, prefix, next_steps, actions, results, reviews


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    client = ValidityClient.from_env(cache_path=OUT / "api_cache_fixed.jsonl")

    lines: list[str] = []
    lines.append("# Action Study 有研究价值的样本（2 题）\n")
    lines.append(f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    lines.append("> 筛选：从 5 题 / 18 prefix 中按 state_bucket、动作可用性、语义多样性选取  ")
    lines.append("> 判分：rescored grader；API：修复 temperature 后 gpt-5.5\n")
    lines.append("---\n")

    lines.append("## 为什么换样本？\n")
    lines.append("先前 p03/p04（极坐标题）的问题：\n")
    lines.append("- 三种动作全部 pass@k=1，**无动作差异**")
    lines.append("- p04 接近 TERMINAL，**无决策研究价值**\n")
    lines.append("当前 pilot（5 题）修复 grader 后：**尚无 Continue=0 且 Branch=1 的 case**。")
    lines.append("因此选取两类仍有分析价值的样本：\n")
    lines.append("1. **动作可用性差异**（Rollback 不可用）")
    lines.append("2. **不确定性状态差异**（唯一 Forward-uncertain + 语义分支）\n")
    lines.append("---\n")

    for spec in SAMPLES:
        prefix_id = spec["prefix_id"]
        trace, prefix, next_steps, actions, results, reviews = pick(prefix_id)
        question = trace["question"]
        gold = trace["gold_answer"]
        rp = reasoning_only(prefix["prefix_text"])

        val = client.label_prefix(
            prefix_id=prefix_id, question=question, gold_answer=gold, reasoning_prefix=rp
        )
        step_texts = [n["text"] for n in next_steps]
        cl = client.cluster_next_steps(
            prefix_id=prefix_id, question=question, reasoning_prefix=rp, next_steps=step_texts
        )
        div = compute_diversity(step_texts, api_clusters=cl.get("clusters") or None)
        bucket = state_bucket(val["prefix_validity"], div["diversity_label"])

        lines.append(f"## {spec['title']}\n")
        lines.append(f"**prefix_id**: `{prefix_id}`\n")
        lines.append("**选取理由：**\n")
        for w in spec["why"]:
            lines.append(f"- {w}")
        lines.append("")

        lines.append("### 0. 题目\n")
        q = question if len(question) < 500 else question[:500] + "...(truncated)"
        lines.append(f"{q}\n")
        lines.append(f"**Gold**: `{gold}`\n")

        lines.append("### 1. Greedy trace\n")
        lines.append("| 字段 | 值 |")
        lines.append("|------|-----|")
        lines.append(f"| predicted_answer | `{trace['predicted_answer']}` |")
        lines.append(f"| is_correct | **{trace['is_correct']}** |")
        lines.append(f"| token_count | {trace['token_count']} |\n")

        lines.append("### 2. Prefix\n")
        lines.append("| 字段 | 值 |")
        lines.append("|------|-----|")
        lines.append(f"| reasoning_progress | **{prefix['reasoning_progress']:.3f}** |")
        lines.append(f"| previous_checkpoint | {'有（可 Rollback）' if prefix['previous_checkpoint'] else '**无（Rollback 不可用）**'} |")
        lines.append("")
        pt = prefix["prefix_text"]
        if "Problem:" in pt:
            pt = pt[pt.index("Problem:") :]
        lines.append("```\n" + pt[-650:] + "\n```\n")

        lines.append("### 3. Next steps ×4\n")
        for ns in next_steps:
            lines.append(f"**sample {ns['sample_id']}**:\n```\n{ns['text']}\n```\n")

        lines.append("### 4. API Validity\n```json\n")
        lines.append(json.dumps(val, ensure_ascii=False, indent=2))
        lines.append("\n```\n")

        lines.append("### 5. Cluster + state_bucket\n```json\n")
        lines.append(
            json.dumps(
                {**cl, "diversity_label": div["diversity_label"], "state_bucket": bucket},
                ensure_ascii=False,
                indent=2,
            )
        )
        lines.append("\n```\n")

        lines.append("### 6. 三动作 Oracle Outcome（Phase 1 主指标）\n")
        lines.append("| action | oracle_recoverable | draft_generated_tokens | discarded_prefix_tokens | action_start |")
        lines.append("|--------|-------------------|------------------------|-------------------------|--------------|")
        for ar in results:
            lines.append(
                f"| {ar['action']} | {ar.get('oracle_recoverable', ar.get('pass_at_k', ar.get('is_correct')))} | "
                f"{ar.get('draft_generated_tokens', ar.get('num_tokens', 0))} | "
                f"{ar.get('discarded_prefix_tokens', 0)} | "
                f"{ar.get('action_start', 'current_prefix' if ar['action'] != 'rollback' else 'previous_checkpoint')} |"
            )
        lines.append("")
        lines.append("> `debug_latency_sec` 仅作工程排查，**不用于论文延迟结论**。\n")

        lines.append("### 7. 动作明细（各举 1 条）\n")
        shown = set()
        for act in actions:
            if act["action_type"] in shown:
                continue
            shown.add(act["action_type"])
            cont = act.get("continuation", "")[:500]
            lines.append(f"**{act['action_type']}** → `{act.get('predicted_answer','')}` correct={act.get('is_correct')}\n")
            lines.append(f"```\n{cont}\n```\n")

        lines.append("### 8. 研究解读\n")
        cont = next((r for r in results if r["action"] == "continue"), {})
        br = next((r for r in results if r["action"] == "branch"), {})
        rb = next((r for r in results if r["action"] == "rollback"), None)
        lines.append(f"- state_bucket = `{bucket}`（validity={val['prefix_validity']}, diversity={div['diversity_label']}）")
        if not prefix["previous_checkpoint"]:
            lines.append("- **Rollback 动作不可执行**：研究问题退化为 Continue vs Branch")
        if div["diversity_label"] == "HIGH_DIVERSITY":
            lines.append(f"- 语义分支数 = {div['num_clusters']}：适合研究 Branch 并行探索的价值")
        if cont and br and cont.get("draft_generated_tokens", cont.get("num_tokens")):
            ct = cont.get("draft_generated_tokens", cont.get("num_tokens", 0))
            bt = br.get("draft_generated_tokens", br.get("num_tokens", 0))
            lines.append(f"- Branch draft tokens ≈ Continue 的 **{bt/max(ct,1):.1f}×**（计算量参考，非 E2E 延迟）")
        if cont.get("is_correct") == br.get("pass_at_k") == (rb or {}).get("pass_at_k"):
            lines.append("- 当前判分：三动作 oracle recoverability 均为 1 → **无准确率差异**，需更多难题/错误 prefix")
        lines.append("\n---\n")

    lines.append("## 数据局限 & 下一步\n")
    lines.append("| 现状 | 影响 |")
    lines.append("|------|------|")
    lines.append("| 仅完成 5/200 题 | 样本太少，无 INVALID bucket |")
    lines.append("| 修复 grader 后全 pass@k=1 | 尚无动作 outcome 差异 |")
    lines.append("| 无 Continue=0 案例 | 无法验证 rescue 效应 |")
    lines.append("")
    lines.append("**建议**：Phase 0 pilot 刻意纳入 greedy 答错的题 + early/middle checkpoint，才能产生有差异的 oracle action label。\n")

    path = DOCS / "action_study_2_samples_research.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

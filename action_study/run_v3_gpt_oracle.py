"""V3.2: GPT-5.5 offline pairwise oracle on audit sample (~312 prefixes)."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

from reasoning_branch_dataset.action_study.gpt_pairwise_judge import GPTPairwiseClient, is_true_branch_rescue
from reasoning_branch_dataset.action_study.oracle_labels import classify_oracle
from reasoning_branch_dataset.action_study.run_utility_scoring import _load_jsonl, load_candidate_tasks


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _done_ids(path: Path) -> set[str]:
    return {r["prefix_id"] for r in _load_jsonl(path) if "prefix_id" in r}


def build_audit_sample(
    v3_dir: Path,
    *,
    n_continue: int = 50,
    n_handoff: int = 50,
    n_random: int = 50,
    tau: int = 7,
    seed: int = 42,
) -> list[dict[str, Any]]:
    scores = [r for r in _load_jsonl(v3_dir / "utility_scores_QwQ-32B.jsonl") if len(r.get("utility_scores", [])) >= 5]
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in scores:
        lab = classify_oracle(r["utility_scores"], tau=tau)["oracle_label"]
        if lab == "branch_rescuable":
            lab = "weak_branch_rescuable"
        buckets[lab].append(r)

    weak = buckets.get("weak_branch_rescuable", [])
    cont = buckets.get("continue_sufficient", [])
    hand = buckets.get("handoff_required", [])
    rng.shuffle(cont)
    rng.shuffle(hand)

    used = {r["prefix_id"] for r in weak}
    rest = [r for r in scores if r["prefix_id"] not in used]
    rng.shuffle(rest)

    sample: list[dict[str, Any]] = []
    for r in weak:
        sample.append({**r, "sample_bucket": "qwQ_weak_branch"})
    for r in cont[:n_continue]:
        sample.append({**r, "sample_bucket": "control_continue"})
    for r in hand[:n_handoff]:
        sample.append({**r, "sample_bucket": "control_handoff"})
    for r in rest[:n_random]:
        sample.append({**r, "sample_bucket": "control_random"})
    return sample


def _best_branch_index(scores: list[int]) -> int:
    u_br = scores[1:]
    return 1 + max(range(len(u_br)), key=lambda i: u_br[i])


def run_gpt_oracle(
    v2_dir: Path,
    v3_dir: Path,
    *,
    max_workers: int | None = None,
) -> Path:
    import os

    sample_path = v3_dir / "gpt_audit_sample.jsonl"
    if not sample_path.exists():
        sample = build_audit_sample(v3_dir)
        sample_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sample), encoding="utf-8")

    sample = _load_jsonl(sample_path)
    tasks = {t["prefix_id"]: t for t in load_candidate_tasks(v2_dir)}
    out_path = v3_dir / "gpt_pairwise_results.jsonl"
    done = _done_ids(out_path)
    client = GPTPairwiseClient.from_env(cache_path=v3_dir / "gpt_pairwise_cache.jsonl")
    workers = max_workers or int(os.environ.get("DS_API_CONCURRENCY_LIMIT", "48"))

    def _one(row: dict[str, Any]) -> dict[str, Any]:
        pid = row["prefix_id"]
        task = tasks[pid]
        scores = row["utility_scores"]
        bi = _best_branch_index(scores)
        details = row.get("candidate_details") or []
        greedy_step = (details[0].get("candidate_step") if details else None) or task["continue_continuation"]
        branch_step = (
            details[bi].get("candidate_step")
            if bi < len(details)
            else task["branch_continuations"][bi - 1]
        )
        judged = client.judge_dual_pass(
            prefix_id=pid,
            question=task["question"],
            prefix_tail=task["reasoning_prefix"],
            greedy_step=greedy_step,
            branch_step=branch_step,
        )
        qwq_weak = row.get("sample_bucket") == "qwQ_weak_branch" or classify_oracle(scores, tau=7)["oracle_label"] in (
            "weak_branch_rescuable",
            "branch_rescuable",
        )
        return {
            "prefix_id": pid,
            "sample_bucket": row.get("sample_bucket"),
            "qwq_oracle": classify_oracle(scores, tau=7)["oracle_label"],
            "u_greedy": scores[0],
            "u_best_branch": scores[bi],
            "best_branch_index": bi - 1,
            "qwq_weak_branch": qwq_weak,
            **judged,
        }

    pending = [r for r in sample if r["prefix_id"] not in done]
    if not pending:
        return out_path

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, row): row["prefix_id"] for row in pending}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="gpt_pairwise"):
            row = fut.result()
            _append_jsonl(out_path, row)
    return out_path


def summarize_gpt_oracle(v3_dir: Path) -> dict[str, Any]:
    rows = _load_jsonl(v3_dir / "gpt_pairwise_results.jsonl")
    if not rows:
        return {}
    weak = [r for r in rows if r.get("qwq_weak_branch")]
    stable = [r for r in rows if r.get("dual_pass_stable")]
    canon = Counter(r.get("canonical_verdict", "UNKNOWN") for r in stable)
    weak_stable = [r for r in weak if r.get("dual_pass_stable")]
    true_branch = [r for r in weak_stable if r.get("true_branch_rescue")]
    qwq_weak_n = len(weak)
    return {
        "n_total": len(rows),
        "n_qwq_weak_branch": qwq_weak_n,
        "n_dual_pass_stable": len(stable),
        "dual_pass_stable_rate": len(stable) / len(rows) if rows else 0,
        "canonical_counts_stable": dict(canon),
        "n_true_branch_rescue_stable": len(true_branch),
        "precision_gpt_branch_given_qwq_weak": len(true_branch) / qwq_weak_n if qwq_weak_n else 0,
        "cleaned_branch_rate_1548": len(true_branch) / 1548,
        "by_bucket": {
            b: len([r for r in rows if r.get("sample_bucket") == b]) for b in sorted({r.get("sample_bucket") for r in rows})
        },
    }


def write_gpt_oracle_report(v3_dir: Path, report_path: Path) -> Path:
    summary = summarize_gpt_oracle(v3_dir)
    lines = [
        "# V3.2 — GPT-5.5 Pairwise Oracle Report",
        "",
        "> Offline structured judge (NOT latency path). Dual-pass A/B swap required for stable labels.",
        "",
        f"- total reviewed: **{summary.get('n_total', 0)}**",
        f"- QwQ weak Branch pool: **{summary.get('n_qwq_weak_branch', 0)}**",
        f"- dual-pass stable: **{summary.get('n_dual_pass_stable', 0)}** ({100*summary.get('dual_pass_stable_rate',0):.1f}%)",
        "",
        "## Key metrics",
        "",
        f"- P(GPT true Branch | QwQ weak): **{100*summary.get('precision_gpt_branch_given_qwq_weak',0):.1f}%**",
        f"- Stable true Branch count: **{summary.get('n_true_branch_rescue_stable', 0)}**",
        f"- Est. rate over 1548: **{100*summary.get('cleaned_branch_rate_1548',0):.2f}%**",
        "",
        "## Stable canonical verdicts",
        "",
        "| verdict | count |",
        "|---------|------:|",
    ]
    for k, v in sorted((summary.get("canonical_counts_stable") or {}).items(), key=lambda x: -x[1]):
        lines.append(f"| {k} | {v} |")

    lines.extend(
        [
            "",
            "## GPT action mapping (stable only)",
            "",
            "- `GREEDY_ONLY_ACCEPTABLE` / `BOTH_OK_GREEDY_PREFERRED` → **Continue**",
            "- `BRANCH_ONLY_ACCEPTABLE` / `BOTH_OK_BRANCH_PREFERRED` → **Branch**",
            "- `BOTH_ACCEPTABLE_EQUIVALENT` → **Continue** (no Branch cost)",
            "- `BOTH_UNACCEPTABLE` → **Handoff**",
            "",
            "## Probe unlock (v3.2)",
            "",
            "- Require dual-pass stable rate ≥ 85%",
            "- Require stable true Branch N ≥ 50 for 3-way probe; else Continue vs Non-Continue",
            "",
        ]
    )

    # Flag cases
    rows = _load_jsonl(v3_dir / "gpt_pairwise_results.jsonl")
    equiv_weak = [
        r
        for r in rows
        if r.get("qwq_weak_branch") and r.get("dual_pass_stable") and r.get("canonical_verdict") == "BOTH_ACCEPTABLE_EQUIVALENT"
    ][:3]
    if equiv_weak:
        lines.extend(["## Audit cases: QwQ weak but GPT equivalent", ""])
        for i, r in enumerate(equiv_weak, 1):
            lines.append(f"### Case {i}: `{r['prefix_id']}`")
            lines.append(f"- QwQ: u₀={r.get('u_greedy')} u_best={r.get('u_best_branch')}")
            lines.append(f"- GPT reason: {r.get('pass1', {}).get('reason', '')}")
            lines.append("")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (v3_dir / "gpt_oracle_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-dir", default="/mnt/afs/L202500372/reasoning_branch_dataset/outputs/action_study_pilot_v2")
    parser.add_argument("--v3-dir", default="/mnt/afs/L202500372/reasoning_branch_dataset/outputs/action_study_pilot_v3")
    parser.add_argument("--report-path", default="/mnt/afs/L202500372/reasoning_branch_dataset/outputs/pilot_v3_2_report.md")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--max-workers", type=int, default=None)
    args = parser.parse_args()
    v2, v3 = Path(args.v2_dir), Path(args.v3_dir)
    if not args.report_only:
        run_gpt_oracle(v2, v3, max_workers=args.max_workers)
    path = write_gpt_oracle_report(v3, Path(args.report_path))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

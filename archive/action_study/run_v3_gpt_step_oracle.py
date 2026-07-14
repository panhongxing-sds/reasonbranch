"""V3.3: GPT-5.5 local next-step action oracle — full pipeline + report."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from reasoning_branch_dataset.action_study.gpt_step_oracle import (
    BRANCH_KEYS,
    CANDIDATE_KEYS,
    GREEDY_KEY,
    GPTStepOracleClient,
    oracle_action_from_acceptability,
)
from reasoning_branch_dataset.action_study.oracle_labels import classify_oracle
from reasoning_branch_dataset.action_study.run_utility_scoring import _load_jsonl, load_candidate_tasks
from reasoning_branch_dataset.action_study.step_extraction import extract_candidate_bundle


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _done_ids(path: Path) -> set[str]:
    return {r["prefix_id"] for r in _load_jsonl(path) if "prefix_id" in r}


def _load_v2_context(v2_dir: Path) -> dict[str, Any]:
    prefixes = {r["prefix_id"]: r for r in _load_jsonl(v2_dir / "prefixes.jsonl")}
    continue_correct: dict[str, bool] = {}
    branch_correct: dict[str, list[bool]] = defaultdict(lambda: [False] * 4)
    for row in _load_jsonl(v2_dir / "actions.jsonl"):
        pid = row["prefix_id"]
        ok = bool(row.get("is_correct") or row.get("correct"))
        if row.get("action_type") == "continue" and int(row.get("sample_id", 0)) == 0:
            continue_correct[pid] = ok
        elif row.get("action_type") == "branch":
            sid = int(row.get("sample_id", 0))
            if 0 <= sid < 4:
                branch_correct[pid][sid] = ok
    return {"prefixes": prefixes, "continue_correct": continue_correct, "branch_correct": dict(branch_correct)}


def _prepare_record(task: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    bundle = extract_candidate_bundle(
        question=task["question"],
        continue_continuation=task["continue_continuation"],
        branch_continuations=task["branch_continuations"],
    )
    details = bundle["candidate_details"]
    steps = {key: d["candidate_step"] for key, d in zip(CANDIDATE_KEYS, details)}
    qualities = {
        "greedy": details[0]["step_quality"],
        **{f"branch_{i}": details[i + 1]["step_quality"] for i in range(4)},
    }
    completes = [d["eligible_for_oracle"] for d in details]
    pfx = ctx["prefixes"].get(task["prefix_id"], {})
    return {
        "steps": steps,
        "greedy_step": steps[GREEDY_KEY],
        "branch_steps": [steps[k] for k in BRANCH_KEYS],
        "candidate_quality": qualities,
        "greedy_complete": completes[0],
        "branch_completes": completes[1:],
        "all_branches_complete": all(completes[1:]),
        "prefix_text": pfx.get("prefix_text", ""),
        "v2_behavior_state": pfx.get("behavior_state") or pfx.get("state_bucket"),
        "continue_final_correct": ctx["continue_correct"].get(task["prefix_id"]),
        "branch_final_correct": ctx["branch_correct"].get(task["prefix_id"], [False] * 4),
    }


def _finalize_oracle(row: dict[str, Any], prep: dict[str, Any]) -> dict[str, Any]:
    if not row.get("oracle_stable"):
        row["oracle_action"] = "ORACLE_UNSTABLE"
        row["oracle_eligible_for_probe"] = False
        return row

    g_ok = bool(row.get("g_acceptable"))
    b_ok = list(row.get("branch_acceptable") or [False] * 4)
    action = oracle_action_from_acceptability(
        prefix_status=row.get("prefix_status") or "UNCLEAR",
        g_acceptable=g_ok,
        branch_acceptables=b_ok,
        greedy_complete=prep["greedy_complete"],
        branch_completes=prep["branch_completes"],
    )
    row["oracle_action"] = action
    row["greedy_acceptable"] = g_ok
    row["branch_acceptable"] = b_ok
    row["n_acceptable_branches"] = sum(b_ok)
    row["any_branch_acceptable"] = any(b_ok)
    row["oracle_eligible_for_probe"] = action in ("CONTINUE", "BRANCH", "HANDOFF")
    return row


def run_gpt_step_oracle(
    v2_dir: Path,
    v3_dir: Path,
    *,
    max_workers: int | None = None,
    full: bool = True,
    max_prefixes: int | None = None,
) -> Path:
    import os

    all_tasks = load_candidate_tasks(v2_dir, admission_only=True)
    task_map = {t["prefix_id"]: t for t in all_tasks}

    if full:
        tasks = all_tasks
    else:
        sample_path = v3_dir / "gpt_audit_sample.jsonl"
        if not sample_path.exists():
            from reasoning_branch_dataset.action_study.run_v3_gpt_oracle import build_audit_sample

            sample = build_audit_sample(v3_dir)
            sample_path.write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sample),
                encoding="utf-8",
            )
        tasks = [task_map[r["prefix_id"]] for r in _load_jsonl(sample_path) if r["prefix_id"] in task_map]

    if max_prefixes is not None:
        tasks = tasks[:max_prefixes]
    ctx = _load_v2_context(v2_dir)
    v3_scores = {r["prefix_id"]: r for r in _load_jsonl(v3_dir / "utility_scores_QwQ-32B.jsonl")}

    out_path = v3_dir / "gpt_step_labels.jsonl"
    done = _done_ids(out_path)
    client = GPTStepOracleClient.from_env(cache_path=v3_dir / "gpt_step_oracle_cache.jsonl")
    workers = max_workers or int(os.environ.get("DS_API_CONCURRENCY_LIMIT", "96"))

    def _one(task: dict[str, Any]) -> dict[str, Any]:
        pid = task["prefix_id"]
        prep = _prepare_record(task, ctx)
        judged = client.judge_dual_pass(
            prefix_id=pid,
            question=task["question"],
            prefix_tail=task["reasoning_prefix"],
            steps=prep["steps"],
        )
        qwq = v3_scores.get(pid, {})
        scores = qwq.get("utility_scores") or []
        qwq_oracle = classify_oracle(scores, tau=7)["oracle_label"] if scores else "unknown"
        qwq_weak = qwq_oracle in ("weak_branch_rescuable", "branch_rescuable")
        row = {
            "problem_id": task["problem_id"],
            "prefix_id": pid,
            "prefix_status": judged.get("prefix_status"),
            "greedy_step": prep["greedy_step"],
            "branch_steps": prep["branch_steps"],
            "candidate_quality": prep["candidate_quality"],
            "greedy_complete": prep["greedy_complete"],
            "branch_completes": prep["branch_completes"],
            "all_branches_complete": prep["all_branches_complete"],
            "v2_behavior_state": prep["v2_behavior_state"],
            "qwen_utility_scores": scores,
            "qwq_oracle": qwq_oracle,
            "qwq_weak_branch": qwq_weak,
            "continue_final_correct": prep["continue_final_correct"],
            "branch_final_correct": prep["branch_final_correct"],
            "branch_final_correct_count": sum(1 for x in prep["branch_final_correct"] if x),
            **judged,
        }
        return _finalize_oracle(row, prep)

    pending = [t for t in tasks if t["prefix_id"] not in done]
    if not pending:
        return out_path

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, t): t["prefix_id"] for t in pending}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="gpt_step_oracle"):
            _append_jsonl(out_path, fut.result())
    return out_path


def _cluster_bootstrap_ci(
    rows: list[dict[str, Any]],
    *,
    key: str,
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap CI by problem_id cluster for boolean indicator in `key`."""
    by_prob: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_prob[r["problem_id"]].append(r)
    probs = list(by_prob.keys())
    if not probs:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    rates = []
    for _ in range(n_boot):
        sampled = rng.choice(probs, size=len(probs), replace=True)
        sel = [r for p in sampled for r in by_prob[p]]
        rates.append(sum(1 for r in sel if r.get(key)) / len(sel))
    rates_arr = np.array(rates)
    return float(np.mean(rates_arr)), float(np.percentile(rates_arr, 2.5)), float(np.percentile(rates_arr, 97.5))


def _rescue_at_k(rows: list[dict[str, Any]], *, k: int, n_perm: int = 50, seed: int = 0) -> float:
    """Rescue@K among greedy-rejected with complete branches; average over permutations."""
    pool = [
        r
        for r in rows
        if not r.get("g_acceptable")
        and r.get("all_branches_complete")
        and r.get("oracle_action") in ("BRANCH", "HANDOFF", "PARTIAL_BRANCH_EVIDENCE")
    ]
    if not pool:
        return 0.0
    rng = random.Random(seed)
    hits = []
    for r in pool:
        flags = list(r.get("branch_acceptable") or [False] * 4)
        rate_perm = []
        for _ in range(n_perm):
            perm = flags[:]
            rng.shuffle(perm)
            rate_perm.append(any(perm[:k]))
        hits.append(sum(rate_perm) / len(rate_perm))
    return sum(hits) / len(hits)


def summarize_step_oracle(v3_dir: Path, *, n_total_expected: int = 1548) -> dict[str, Any]:
    rows = _load_jsonl(v3_dir / "gpt_step_labels.jsonl")
    if not rows:
        return {}
    stable = [r for r in rows if r.get("oracle_stable")]
    eligible = [r for r in stable if r.get("oracle_eligible_for_probe")]
    unstable = [r for r in rows if not r.get("oracle_stable")]

    greedy_complete = sum(1 for r in rows if r.get("greedy_complete"))
    all_br_complete = sum(1 for r in rows if r.get("all_branches_complete"))
    partial = sum(1 for r in rows if r.get("oracle_action") == "PARTIAL_BRANCH_EVIDENCE")
    data_err = sum(1 for r in rows if r.get("oracle_action") == "DATA_ERROR_GREEDY_INCOMPLETE")

    action_counts = Counter(r.get("oracle_action") for r in eligible)
    n_el = len(eligible)

    # Branch width table (greedy rejected, stable eligible)
    g_reject = [r for r in eligible if not r.get("g_acceptable")]
    m_hist = Counter(r.get("n_acceptable_branches", 0) for r in g_reject)

    # QwQ precision/recall vs GPT Branch
    qwq_weak = [r for r in eligible if r.get("qwq_weak_branch")]
    gpt_branch = [r for r in eligible if r.get("oracle_action") == "BRANCH"]
    tp = sum(1 for r in eligible if r.get("qwq_weak_branch") and r.get("oracle_action") == "BRANCH")
    precision = tp / len(qwq_weak) if qwq_weak else 0.0
    recall = tp / len(gpt_branch) if gpt_branch else 0.0

    # V2 behavior cross-tab
    behavior_tab: dict[str, dict[str, int]] = defaultdict(lambda: Counter())
    for r in eligible:
        st = r.get("v2_behavior_state") or "UNKNOWN"
        behavior_tab[st][r.get("oracle_action", "UNKNOWN")] += 1

    # Final correctness by action
    final_by_action: dict[str, list[bool]] = defaultdict(list)
    for r in eligible:
        act = r.get("oracle_action")
        if r.get("continue_final_correct") is not None:
            final_by_action[act].append(bool(r["continue_final_correct"]))

    # Cluster bootstrap CI for main actions
    action_ci = {}
    for act in ("CONTINUE", "BRANCH", "HANDOFF"):
        ind_rows = [{**r, "_is": r.get("oracle_action") == act} for r in eligible]
        m2, lo2, hi2 = _cluster_bootstrap_ci(ind_rows, key="_is")
        action_ci[act] = {
            "rate": sum(1 for r in eligible if r.get("oracle_action") == act) / n_el if n_el else 0,
            "ci_lo": lo2,
            "ci_hi": hi2,
        }

    rescue = {k: _rescue_at_k(eligible, k=k) for k in (1, 2, 4)}

    return {
        "n_total": len(rows),
        "n_expected": n_total_expected,
        "greedy_complete": greedy_complete,
        "all_branches_complete": all_br_complete,
        "n_oracle_stable": len(stable),
        "oracle_stable_rate": len(stable) / len(rows) if rows else 0,
        "n_oracle_unstable": len(unstable),
        "n_partial_branch_evidence": partial,
        "n_data_error_greedy": data_err,
        "n_eligible_probe": n_el,
        "action_counts": dict(action_counts),
        "action_ci": action_ci,
        "m_hist_g_reject": dict(sorted(m_hist.items())),
        "n_g_reject_eligible": len(g_reject),
        "qwQ_precision": precision,
        "qwQ_recall": recall,
        "n_qwq_weak_eligible": len(qwq_weak),
        "n_gpt_branch_eligible": len(gpt_branch),
        "behavior_tab": {k: dict(v) for k, v in behavior_tab.items()},
        "final_correct_by_action": {
            k: {"n": len(v), "rate": sum(v) / len(v) if v else None} for k, v in final_by_action.items()
        },
        "rescue_at_k": rescue,
        "rq1_branch_exists": sum(1 for r in eligible if r.get("oracle_action") == "BRANCH") > 0,
        "branch_rate_eligible": action_counts.get("BRANCH", 0) / n_el if n_el else 0,
    }


def write_step_oracle_report(v3_dir: Path, report_path: Path) -> Path:
    s = summarize_step_oracle(v3_dir)
    n = s.get("n_total", 0)
    ne = s.get("n_expected", 1548)
    pct = lambda x: f"{100 * x / n:.1f}%" if n else "—"

    lines = [
        "# V3.3 — GPT Local Next-Step Action Oracle Report",
        "",
        "> GPT-5.5 offline oracle: independently assess 1 greedy + 4 branch **next steps** at fixed prefix.",
        "> Dual-pass stability on greedy acceptability and any-branch-exists (E).",
        "",
        f"- prefixes labeled: **{n}** / {ne}",
        f"- prompt version: `gpt_step_oracle_v2`",
        "",
        "## 15.1 Data quality",
        "",
        "| Metric | Count | Rate |",
        "|--------|------:|-----:|",
        f"| Total prefixes | {n} | 100% |",
        f"| Greedy complete | {s.get('greedy_complete', 0)} | {pct(s.get('greedy_complete', 0))} |",
        f"| All branches complete | {s.get('all_branches_complete', 0)} | {pct(s.get('all_branches_complete', 0))} |",
        f"| Action-stable dual pass | {s.get('n_oracle_stable', 0)} | {pct(s.get('n_oracle_stable', 0))} |",
        f"| Oracle unstable | {s.get('n_oracle_unstable', 0)} | {pct(s.get('n_oracle_unstable', 0))} |",
        f"| Partial branch evidence | {s.get('n_partial_branch_evidence', 0)} | {pct(s.get('n_partial_branch_evidence', 0))} |",
        f"| Data error (greedy incomplete) | {s.get('n_data_error_greedy', 0)} | {pct(s.get('n_data_error_greedy', 0))} |",
        f"| **Final eligible prefixes** | **{s.get('n_eligible_probe', 0)}** | **{pct(s.get('n_eligible_probe', 0))}** |",
        "",
        "## 15.2 Oracle action distribution (stable, eligible)",
        "",
        "| Action | Count | Rate | 95% CI (cluster) |",
        "|--------|------:|-----:|-----------------:|",
    ]
    counts = s.get("action_counts", {})
    n_el = s.get("n_eligible_probe", 0) or 1
    for act in ("CONTINUE", "BRANCH", "HANDOFF", "PREFIX_INVALID", "PARTIAL_BRANCH_EVIDENCE"):
        c = counts.get(act, 0)
        ci = s.get("action_ci", {}).get(act, {})
        ci_s = f"[{100*ci.get('ci_lo',0):.1f}%, {100*ci.get('ci_hi',0):.1f}%]" if ci else "—"
        lines.append(f"| {act} | {c} | {100*c/n_el:.1f}% | {ci_s} |")

    lines.extend(
        [
            "",
            "## 15.3 Acceptable branches (greedy rejected only)",
            "",
            "| Acceptable branches | Count | Rate |",
            "|--------------------:|------:|-----:|",
        ]
    )
    g_rej_n = s.get("n_g_reject_eligible", 0) or 1
    for m in range(5):
        c = s.get("m_hist_g_reject", {}).get(m, 0)
        lines.append(f"| {m}/4 | {c} | {100*c/g_rej_n:.1f}% |")

    lines.extend(
        [
            "",
            "## 15.4 Branch width (Rescue@K | G=0)",
            "",
            "| Width K | Rescue rate |",
            "|--------:|------------:|",
        ]
    )
    for k, v in sorted((s.get("rescue_at_k") or {}).items()):
        lines.append(f"| {k} | {100*v:.1f}% |")

    lines.extend(
        [
            "",
            "## 16.1 QwQ weak Branch vs GPT V3.3",
            "",
            f"- Precision P(GPT Branch | QwQ weak): **{100*s.get('qwQ_precision',0):.1f}%**",
            f"- Recall P(QwQ weak | GPT Branch): **{100*s.get('qwQ_recall',0):.1f}%**",
            f"- QwQ weak (eligible): {s.get('n_qwq_weak_eligible', 0)}",
            f"- GPT Branch (eligible): {s.get('n_gpt_branch_eligible', 0)}",
            "",
            "## 16.2 V2 behavior_state × V3.3 action",
            "",
            "| V2 state | N | Continue | Branch | Handoff | %Branch |",
            "|----------|--:|---------:|-------:|--------:|--------:|",
        ]
    )
    for st, acts in sorted((s.get("behavior_tab") or {}).items()):
        tot = sum(acts.values())
        br = acts.get("BRANCH", 0)
        lines.append(
            f"| {st} | {tot} | {acts.get('CONTINUE',0)} | {br} | {acts.get('HANDOFF',0)} | {100*br/tot:.1f}% |"
            if tot
            else f"| {st} | 0 | 0 | 0 | 0 | — |"
        )

    lines.extend(["", "## 16.3 Final answer correctness (auxiliary)", ""])
    for act, info in sorted((s.get("final_correct_by_action") or {}).items()):
        rate = info.get("rate")
        lines.append(f"- **{act}**: n={info.get('n')}, P(final correct)={100*rate:.1f}%" if rate is not None else f"- **{act}**: n=0")

    lines.extend(
        [
            "",
            "## 17 Research questions",
            "",
            f"- **RQ1** Branch-rescuable exists: **{s.get('rq1_branch_exists')}** (Branch count={counts.get('BRANCH',0)})",
            f"- **RQ2** Branch rate (eligible): **{100*s.get('branch_rate_eligible',0):.2f}%**",
            f"- **RQ3** Rescue@4 vs Rescue@1: **{100*(s.get('rescue_at_k') or {}).get(4,0):.1f}%** vs **{100*(s.get('rescue_at_k') or {}).get(1,0):.1f}%**",
            f"- **RQ4** QwQ precision/recall: **{100*s.get('qwQ_precision',0):.1f}%** / **{100*s.get('qwQ_recall',0):.1f}%**",
            "- **RQ5** Probe: blocked until stability ≥85% and N_Branch≥50",
            "",
            "## 18 Probe unlock",
            "",
            f"- Dual-pass stability: **{100*s.get('oracle_stable_rate',0):.1f}%** (need ≥85%)",
            f"- Stable Branch N: **{counts.get('BRANCH',0)}** (need ≥50)",
            "",
            "## Definition",
            "",
            "- **CONTINUE**: G=1",
            "- **BRANCH**: G=0 ∧ ∃k B_k=1",
            "- **HANDOFF**: G=0 ∧ all complete branches rejected",
            "- V3.3 does NOT claim online latency or cascade effects (see V3.4).",
            "",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (v3_dir / "gpt_step_oracle_summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-dir", default="/mnt/afs/L202500372/reasoning_branch_dataset/outputs/action_study_pilot_v2")
    parser.add_argument("--v3-dir", default="/mnt/afs/L202500372/reasoning_branch_dataset/outputs/action_study_pilot_v3")
    parser.add_argument("--report-path", default="/mnt/afs/L202500372/reasoning_branch_dataset/outputs/pilot_v3_3_report.md")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--full", action="store_true", default=True)
    parser.add_argument("--audit-only", action="store_true", help="Use 312 audit sample instead of full 1548")
    parser.add_argument("--max-prefixes", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    args = parser.parse_args()
    v2, v3 = Path(args.v2_dir), Path(args.v3_dir)
    full = not args.audit_only
    if not args.report_only:
        run_gpt_step_oracle(
            v2,
            v3,
            max_workers=args.max_workers,
            full=full,
            max_prefixes=args.max_prefixes,
        )
    path = write_step_oracle_report(v3, Path(args.report_path))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

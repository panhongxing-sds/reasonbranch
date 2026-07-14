"""V3.4 Sequential Oracle Policy Rollout — pilot runner + report."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from reasoning_branch_dataset.action_study.gpt_step_oracle import GPTStepOracleClient
from reasoning_branch_dataset.action_study.sequential_rollout import (
    Policy,
    RolloutConfig,
    ModelSession,
    StepOracle,
    run_rollout,
)
from reasoning_branch_dataset.action_study.run_utility_scoring import _load_jsonl


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _done_rollout_ids(path: Path) -> set[str]:
    return {r["rollout_id"] for r in _load_jsonl(path) if "rollout_id" in r}


def sample_problems(v2_dir: Path, n: int, seed: int) -> list[dict[str, Any]]:
    rows = [r for r in _load_jsonl(v2_dir / "problems.jsonl") if r.get("input_complete", True)]
    rng = random.Random(seed)
    if len(rows) <= n:
        return rows
    return rng.sample(rows, n)


def build_rollout_plan(
    problems: list[dict[str, Any]],
    *,
    branch_seeds: tuple[int, ...] = (1, 2, 3),
    skip_always_branch: bool = False,
    skip_policies: set[str] | None = None,
) -> list[tuple[dict, Policy, int]]:
    skip = {s.upper() for s in (skip_policies or set())}
    plan: list[tuple[dict, Policy, int]] = []
    for prob in problems:
        if "DRAFT_ONLY" not in skip:
            plan.append((prob, Policy.DRAFT_ONLY, 1))
        if "TARGET_ONLY" not in skip:
            plan.append((prob, Policy.TARGET_ONLY, 1))
        if "SPECREASON" not in skip:
            plan.append((prob, Policy.SPECREASON, 1))
        for s in branch_seeds:
            if "CONDITIONAL_BRANCH" not in skip:
                plan.append((prob, Policy.CONDITIONAL_BRANCH, s))
            if not skip_always_branch and "ALWAYS_BRANCH" not in skip:
                plan.append((prob, Policy.ALWAYS_BRANCH, s))
    return plan


def run_pilot(
    v2_dir: Path,
    out_dir: Path,
    *,
    n_problems: int = 50,
    seed: int = 42,
    branch_seeds: tuple[int, ...] = (1, 2, 3),
    skip_always_branch: bool = False,
    skip_policies: set[str] | None = None,
    cfg: RolloutConfig | None = None,
) -> None:
    cfg = cfg or RolloutConfig(gpt_cache_path=out_dir / "gpt_step_oracle_cache.jsonl")
    out_dir.mkdir(parents=True, exist_ok=True)
    problems = sample_problems(v2_dir, n_problems, seed)
    plan = build_rollout_plan(
        problems,
        branch_seeds=branch_seeds,
        skip_always_branch=skip_always_branch,
        skip_policies=skip_policies,
    )
    # Batch by policy to maximize model reuse
    policy_order = {
        Policy.DRAFT_ONLY: 0,
        Policy.TARGET_ONLY: 1,
        Policy.SPECREASON: 2,
        Policy.CONDITIONAL_BRANCH: 3,
        Policy.ALWAYS_BRANCH: 4,
    }
    plan.sort(key=lambda x: (policy_order[x[1]], x[0]["problem_id"], x[2]))

    steps_path = out_dir / "rollout_steps.jsonl"
    summary_path = out_dir / "rollout_summaries.jsonl"
    done = _done_rollout_ids(summary_path)

    oracle_client = GPTStepOracleClient.from_env(cache_path=cfg.gpt_cache_path)
    oracle = StepOracle(oracle_client, max_retries=cfg.oracle_max_retries)
    session = ModelSession(cfg)

    pending = [(p, pol, s) for p, pol, s in plan if f"{p['problem_id']}::{pol.value}::s{s}" not in done]

    current_policy: Policy | None = None
    try:
        # In dual-resident mode both models stay loaded for the whole run, so we
        # load once up front and never unload between policies (zero swap).
        if cfg.dual_resident:
            session.preload_dual()  # may flip cfg.dual_resident=False on OOM
        for prob, policy, rseed in tqdm(pending, desc="v34_rollout"):
            if policy != current_policy:
                current_policy = policy
                if not cfg.dual_resident:
                    session.unload_all()
                    if policy in (Policy.SPECREASON, Policy.CONDITIONAL_BRANCH, Policy.ALWAYS_BRANCH):
                        session.draft()
                    elif policy == Policy.TARGET_ONLY:
                        session.target()
                    elif policy == Policy.DRAFT_ONLY:
                        session.draft()
            needs_oracle = policy in (
                Policy.SPECREASON,
                Policy.CONDITIONAL_BRANCH,
                Policy.ALWAYS_BRANCH,
            )
            step_recs, summary = run_rollout(
                prob,
                policy=policy,
                seed=rseed,
                cfg=cfg,
                oracle=oracle if needs_oracle else None,
                session=session,
            )
            for rec in step_recs:
                _append_jsonl(steps_path, rec.to_dict())
            _append_jsonl(summary_path, summary)
    finally:
        session.unload_all()


def _cluster_bootstrap_mean(
    pairs: list[tuple[str, float]],
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    by_prob: dict[str, list[float]] = defaultdict(list)
    for pid, val in pairs:
        by_prob[pid].append(val)
    probs = list(by_prob.keys())
    if not probs:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sampled = rng.choice(probs, size=len(probs), replace=True)
        vals = [np.mean(by_prob[p]) for p in sampled]
        means.append(float(np.mean(vals)))
    arr = np.array(means)
    return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def _transition_matrix(steps: list[dict[str, Any]], *, policy: str | None = None) -> list[dict[str, float]]:
    trans = Counter()
    by_rollout: dict[str, list[str]] = defaultdict(list)
    for st in steps:
        if policy and st.get("policy") != policy:
            continue
        by_rollout[st["rollout_id"]].append(st.get("action", ""))
    for acts in by_rollout.values():
        for a, b in zip(acts, acts[1:]):
            if a and b:
                trans[(a, b)] += 1
    trans_rows = []
    for a_cur in ("CONTINUE", "BRANCH", "HANDOFF", "ORACLE_API_ERROR"):
        row = {"from": a_cur}
        total = sum(v for (f, _), v in trans.items() if f == a_cur)
        for b in ("CONTINUE", "BRANCH", "HANDOFF", "ORACLE_API_ERROR"):
            c = trans.get((a_cur, b), 0)
            row[b] = c / total if total else 0
        trans_rows.append(row)
    return trans_rows


def summarize_v34(out_dir: Path) -> dict[str, Any]:
    summaries = _load_jsonl(out_dir / "rollout_summaries.jsonl")
    steps = _load_jsonl(out_dir / "rollout_steps.jsonl")
    if not summaries:
        return {}

    by_policy: dict[str, list[dict]] = defaultdict(list)
    for s in summaries:
        by_policy[s["policy"]].append(s)

    policy_stats = {}
    for pol, rows in by_policy.items():
        n = len(rows)
        acc = [r for r in rows if r.get("is_correct") is True]
        policy_stats[pol] = {
            "n": n,
            "accuracy": len(acc) / n if n else 0,
            "avg_continue": np.mean([r.get("n_continue", 0) for r in rows]),
            "avg_branch": np.mean([r.get("n_branch", 0) for r in rows]),
            "avg_handoff": np.mean([r.get("n_handoff", 0) for r in rows]),
            "avg_oracle_api_error": np.mean([r.get("n_oracle_api_error", 0) for r in rows]),
            "avg_target_steps": np.mean([r.get("n_target_steps", 0) for r in rows]),
            "avg_steps": np.mean([r.get("n_steps", 0) for r in rows]),
            "truncated_rate": sum(1 for r in rows if r.get("termination_reason") == "MAX_STEP_TRUNCATED") / n if n else 0,
            "prefix_unchanged_rate": sum(1 for r in rows if r.get("termination_reason") == "PREFIX_UNCHANGED") / n if n else 0,
            "oracle_api_error_rate": sum(1 for r in rows if r.get("termination_reason") == "ORACLE_API_ERROR") / n if n else 0,
            "target_empty_rate": sum(1 for r in rows if r.get("termination_reason") == "TARGET_EMPTY_STEP") / n if n else 0,
            "valid_for_comparison_rate": sum(1 for r in rows if r.get("valid_for_comparison", True)) / n if n else 0,
            "final_answer_rate": sum(1 for r in rows if r.get("termination_reason") == "FINAL_ANSWER") / n if n else 0,
        }

    oracle_steps = [
        st for st in steps
        if st.get("policy") in ("SPECREASON", "CONDITIONAL_BRANCH", "ALWAYS_BRANCH")
    ]
    oracle_api_error_steps = sum(1 for st in oracle_steps if st.get("action") == "ORACLE_API_ERROR")
    oracle_step_total = len(oracle_steps)

    # Transition matrix from steps (all policies)
    trans = Counter()
    by_rollout: dict[str, list[str]] = defaultdict(list)
    for st in steps:
        by_rollout[st["rollout_id"]].append(st.get("action", ""))
    for acts in by_rollout.values():
        for a, b in zip(acts, acts[1:]):
            if a and b:
                trans[(a, b)] += 1

    trans_rows = _transition_matrix(steps)
    trans_spec = _transition_matrix(steps, policy="SPECREASON")
    trans_cond = _transition_matrix(steps, policy="CONDITIONAL_BRANCH")

    # L_B and L_H
    lb_lens, lh_lens = [], []
    for acts in by_rollout.values():
        for i, a in enumerate(acts):
            if a == "BRANCH":
                run = 0
                for b in acts[i + 1 :]:
                    if b == "CONTINUE":
                        run += 1
                    else:
                        break
                lb_lens.append(run)
            if a == "HANDOFF":
                run = 0
                for b in acts[i + 1 :]:
                    if b == "CONTINUE":
                        run += 1
                    else:
                        break
                lh_lens.append(run)

    # Paired SpecReason vs Conditional Branch (seed=1 only for fair compare)
    spec = {r["problem_id"]: r for r in by_policy.get("SPECREASON", [])}
    cond = {r["problem_id"]: r for r in by_policy.get("CONDITIONAL_BRANCH", []) if r.get("seed") == 1}
    paired = []
    paired_valid = []
    for pid in sorted(set(spec) & set(cond)):
        s, c = spec[pid], cond[pid]
        row = {
            "problem_id": pid,
            "delta_handoff": s.get("n_handoff", 0) - c.get("n_handoff", 0),
            "delta_target": s.get("n_target_steps", 0) - c.get("n_target_steps", 0),
            "n_branch_cond": c.get("n_branch", 0),
            "cascade_c": (s.get("n_handoff", 0) - c.get("n_handoff", 0)) / max(c.get("n_branch", 0), 1),
            "spec_valid": s.get("valid_for_comparison", True),
            "cond_valid": c.get("valid_for_comparison", True),
        }
        paired.append(row)
        if row["spec_valid"] and row["cond_valid"]:
            paired_valid.append(row)

    delta_h = [p["delta_handoff"] for p in paired]
    delta_h_valid = [p["delta_handoff"] for p in paired_valid]
    cascade = [p["cascade_c"] for p in paired if p["n_branch_cond"] > 0]

    # Proxy latency (structural)
    T_D, T_V, T_T, T_BK = 1.0, 0.5, 8.0, 3.0
    for pol, st in policy_stats.items():
        st["proxy_latency"] = (
            st["avg_continue"] * (T_D + T_V)
            + st["avg_branch"] * (T_D + T_BK)
            + st["avg_handoff"] * (T_D + T_V + T_T)
        )

    paired_boot = []
    if paired:
        m, lo, hi = _cluster_bootstrap_mean([(p["problem_id"], p["delta_handoff"]) for p in paired])
        paired_boot.append({"metric": "delta_handoff", "mean": m, "ci_lo": lo, "ci_hi": hi})
    if paired_valid:
        m, lo, hi = _cluster_bootstrap_mean([(p["problem_id"], p["delta_handoff"]) for p in paired_valid])
        paired_boot.append({"metric": "delta_handoff_valid_only", "mean": m, "ci_lo": lo, "ci_hi": hi})

    return {
        "n_summaries": len(summaries),
        "n_steps": len(steps),
        "policy_stats": policy_stats,
        "oracle_step_api_error_rate": oracle_api_error_steps / oracle_step_total if oracle_step_total else 0,
        "transition_matrix": trans_rows,
        "transition_matrix_specreason": trans_spec,
        "transition_matrix_cond_branch": trans_cond,
        "p_continue_given_branch": trans.get(("BRANCH", "CONTINUE"), 0)
        / max(sum(v for (f, _), v in trans.items() if f == "BRANCH"), 1),
        "p_handoff_given_branch": trans.get(("BRANCH", "HANDOFF"), 0)
        / max(sum(v for (f, _), v in trans.items() if f == "BRANCH"), 1),
        "p_branch_given_branch": trans.get(("BRANCH", "BRANCH"), 0)
        / max(sum(v for (f, _), v in trans.items() if f == "BRANCH"), 1),
        "mean_L_B": float(np.mean(lb_lens)) if lb_lens else 0,
        "median_L_B": float(np.median(lb_lens)) if lb_lens else 0,
        "p_L_B_ge_1": sum(1 for x in lb_lens if x >= 1) / len(lb_lens) if lb_lens else 0,
        "p_L_B_ge_3": sum(1 for x in lb_lens if x >= 3) / len(lb_lens) if lb_lens else 0,
        "mean_L_H": float(np.mean(lh_lens)) if lh_lens else 0,
        "median_L_H": float(np.median(lh_lens)) if lh_lens else 0,
        "paired_spec_vs_cond": paired,
        "paired_spec_vs_cond_valid": paired_valid,
        "mean_delta_H": float(np.mean(delta_h)) if delta_h else 0,
        "mean_delta_H_valid_only": float(np.mean(delta_h_valid)) if delta_h_valid else 0,
        "mean_cascade_C": float(np.mean(cascade)) if cascade else 0,
        "paired_bootstrap": paired_boot,
        "n_branch_events": sum(1 for st in steps if st.get("action") == "BRANCH"),
    }


def write_v34_report(out_dir: Path, report_path: Path) -> Path:
    s = summarize_v34(out_dir)
    lines = [
        "# V3.4 — Sequential Oracle Policy Rollout Report",
        "",
        "> GPT-5.5-guided sequential policy from problem prompt; actions modify prefix cascade.",
        "",
        f"- rollouts completed: **{s.get('n_summaries', 0)}**",
        f"- step records: **{s.get('n_steps', 0)}**",
        f"- Branch events: **{s.get('n_branch_events', 0)}**",
        f"- Oracle step API error rate: **{100*s.get('oracle_step_api_error_rate',0):.1f}%**",
        "",
        "## 18.1 Policy-level results",
        "",
        "| Policy | N | Accuracy | Avg Continue | Avg Branch | Avg Handoff | Target steps | Proxy latency |",
        "|--------|--:|---------:|-------------:|-----------:|------------:|-------------:|--------------:|",
    ]
    for pol in ("DRAFT_ONLY", "TARGET_ONLY", "SPECREASON", "CONDITIONAL_BRANCH", "ALWAYS_BRANCH"):
        st = s.get("policy_stats", {}).get(pol, {})
        if not st:
            continue
        lines.append(
            f"| {pol} | {st.get('n',0)} | {100*st.get('accuracy',0):.1f}% "
            f"| {st.get('avg_continue',0):.2f} | {st.get('avg_branch',0):.2f} "
            f"| {st.get('avg_handoff',0):.2f} | {st.get('avg_target_steps',0):.2f} "
            f"| {st.get('proxy_latency',0):.1f} |"
        )

    lines.extend(
        [
            "",
            "## 18.2 Cascade metrics",
            "",
            f"- P(Continue | Branch): **{100*s.get('p_continue_given_branch',0):.1f}%**",
            f"- P(Handoff | Branch): **{100*s.get('p_handoff_given_branch',0):.1f}%**",
            f"- P(Branch | Branch): **{100*s.get('p_branch_given_branch',0):.1f}%**",
            f"- Mean L_B (Continue run after Branch): **{s.get('mean_L_B',0):.2f}**",
            f"- Median L_B: **{s.get('median_L_B',0):.1f}**",
            f"- P(L_B≥1): **{100*s.get('p_L_B_ge_1',0):.1f}%**",
            f"- P(L_B≥3): **{100*s.get('p_L_B_ge_3',0):.1f}%**",
            f"- Mean L_H (Continue run after Handoff): **{s.get('mean_L_H',0):.2f}**",
            f"- Mean ΔH (SpecReason − CondBranch handoffs/problem): **{s.get('mean_delta_H',0):.3f}**",
            f"- Mean ΔH (valid paired only): **{s.get('mean_delta_H_valid_only',0):.3f}**",
            f"- Mean cascade C_q (ΔH / N_B): **{s.get('mean_cascade_C',0):.3f}**",
            "",
            "## 18.3 Paired SpecReason vs Conditional Branch (seed=1)",
            "",
        ]
    )
    boot = s.get("paired_bootstrap") or []
    if boot:
        b = boot[0]
        lines.append(
            f"- ΔHandoff mean **{b['mean']:.3f}** (95% CI [{b['ci_lo']:.3f}, {b['ci_hi']:.3f}])"
        )
    lines.extend(
        [
            "",
            "## Action transition matrix",
            "",
            "| From \\ To | Continue | Branch | Handoff |",
            "|-----------|----------|--------|---------|",
        ]
    )
    for row in s.get("transition_matrix", []):
        lines.append(
            f"| {row['from']} | {100*row.get('CONTINUE',0):.1f}% | {100*row.get('BRANCH',0):.1f}% | {100*row.get('HANDOFF',0):.1f}% |"
        )

    lines.extend(
        [
            "",
            "## V3.4 success criteria",
            "",
            f"- Branch events exist: **{s.get('n_branch_events',0) > 0}**",
            f"- E[ΔH] > 0 (fewer handoffs with Branch): **{s.get('mean_delta_H',0) > 0}**",
            "",
            "> GPT oracle is offline only — not included in deployment latency.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "v34_summary.json").write_text(json.dumps(s, indent=2, default=float), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-dir", default="/mnt/afs/L202500372/reasoning_branch_dataset/outputs/action_study_pilot_v2")
    parser.add_argument("--out-dir", default="/mnt/afs/L202500372/reasoning_branch_dataset/outputs/action_study_pilot_v34")
    parser.add_argument("--report-path", default="/mnt/afs/L202500372/reasoning_branch_dataset/outputs/pilot_v3_4_report.md")
    parser.add_argument("--n-problems", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--branch-seeds", type=int, default=3, help="seeds for Branch policies (1..N)")
    parser.add_argument("--skip-always-branch", action="store_true", help="skip Policy E (aux ablation)")
    parser.add_argument("--dual-resident", action="store_true", help="keep draft+target loaded for oracle policies")
    parser.add_argument("--skip-policies", default="", help="comma-separated: DRAFT_ONLY,TARGET_ONLY,...")
    args = parser.parse_args()
    out = Path(args.out_dir)
    branch_seeds = tuple(range(1, args.branch_seeds + 1))
    skip_policies = {s.strip() for s in args.skip_policies.split(",") if s.strip()}
    if not args.report_only:
        import os
        os.environ.setdefault("VLLM_USE_V1", "0")
        cfg = RolloutConfig(
            max_steps=args.max_steps,
            gpt_cache_path=out / "gpt_step_oracle_cache.jsonl",
            dual_resident=args.dual_resident,
        )
        run_pilot(
            Path(args.v2_dir),
            out,
            n_problems=args.n_problems,
            seed=args.seed,
            branch_seeds=branch_seeds,
            skip_always_branch=args.skip_always_branch,
            skip_policies=skip_policies,
            cfg=cfg,
        )
    path = write_v34_report(out, Path(args.report_path))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

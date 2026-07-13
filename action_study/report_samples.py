"""Select and format illustrative full cases for experiment reports."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def _clip(text: str, max_chars: int = 900) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pick_by_key(rows: list[dict], key: str, *, largest: bool = True, n: int = 1) -> list[dict]:
    if not rows:
        return []
    ordered = sorted(rows, key=lambda r: r.get(key, 0), reverse=largest)
    return ordered[:n]


def pick_reachable_cases(
    verify_rows: list[dict[str, Any]],
    drafts_by_id: dict[str, dict[str, Any]],
    problems: dict[str, dict[str, Any]],
    *,
    n_each: int = 1,
) -> list[dict[str, Any]]:
    """Pick branch-win, branch-tie, greedy-only-win style reachable-state cases."""
    rng = random.Random(42)
    cases: list[dict[str, Any]] = []

    wins = [r for r in verify_rows if r.get("branch_gain", 0) > 0]
    ties = [r for r in verify_rows if r.get("branch_gain", 0) == 0]
    big_wins = pick_by_key(wins, "branch_gain", largest=True, n=5)

    for bucket, label in [
        (big_wins, "branch_large_gain"),
        (pick_by_key(wins, "branch_gain", largest=False, n=5), "branch_small_gain"),
        (rng.sample(ties, min(n_each, len(ties))) if ties else [], "branch_no_gain"),
    ]:
        for row in bucket[:n_each]:
            draft = drafts_by_id.get(row["row_id"], {})
            prob = problems.get(row["problem_id"], {})
            branches = row.get("accepted_lengths_branch", [])
            best_i = int(max(range(len(branches)), key=lambda i: branches[i])) if branches else -1
            cases.append(
                {
                    "case_type": label,
                    "row_id": row["row_id"],
                    "problem_id": row["problem_id"],
                    "question": prob.get("question", ""),
                    "checkpoint_token_pos": row.get("checkpoint_token_pos"),
                    "gamma": row.get("gamma"),
                    "a_single": row.get("accepted_length_single"),
                    "a_best4": row.get("accepted_length_best4"),
                    "branch_gain": row.get("branch_gain"),
                    "branch_accept_lengths": branches,
                    "best_branch_index": best_i,
                    "greedy_continuation": draft.get("greedy_continuation", ""),
                    "best_branch_continuation": (
                        draft.get("branch_continuations", [""])[best_i]
                        if best_i >= 0 and draft.get("branch_continuations")
                        else ""
                    ),
                    "prefix_tail": _clip(draft.get("prefix_full", ""), 500),
                }
            )
    return cases


def format_reachable_cases_md(cases: list[dict[str, Any]]) -> list[str]:
    lines = ["", "## Illustrative Cases", ""]
    if not cases:
        lines.append("_no cases selected_")
        return lines

    for i, c in enumerate(cases, start=1):
        lines.extend(
            [
                f"### Case {i}: {c['case_type']} (`{c['row_id']}`)",
                "",
                f"**Problem** ({c['problem_id']}):",
                "",
                f"> {_clip(c.get('question', ''), 400)}",
                "",
                f"- checkpoint: token {c.get('checkpoint_token_pos')} | γ={c.get('gamma')}",
                f"- A_single={c.get('a_single')} | A_best4={c.get('a_best4')} | **G_B={c.get('branch_gain')}**",
                f"- branch accept lengths: `{c.get('branch_accept_lengths')}` (best branch index={c.get('best_branch_index')})",
                "",
                "**Prefix tail (target-reachable context):**",
                "",
                "```text",
                c.get("prefix_tail", ""),
                "```",
                "",
                "**Greedy draft block (first γ tokens):**",
                "",
                "```text",
                _clip(c.get("greedy_continuation", ""), 900),
                "```",
                "",
            ]
        )
        if c.get("branch_gain", 0) > 0 and c.get("best_branch_continuation"):
            lines.extend(
                [
                    "**Best branch draft block:**",
                    "",
                    "```text",
                    _clip(c.get("best_branch_continuation", ""), 900),
                    "```",
                    "",
                ]
            )
    return lines


def _utility_case_row(
    row: dict[str, Any],
    prefixes: dict[str, dict[str, Any]],
    problems: dict[str, dict[str, Any]],
    *,
    tau: int,
    label: str,
) -> dict[str, Any]:
    scores = row.get("utility_scores") or []
    pfx = prefixes.get(row["prefix_id"], {})
    prob = problems.get(row["problem_id"], {})
    details = row.get("candidate_details") or []
    u_branch = scores[1:]
    return {
        "case_type": label,
        "prefix_id": row["prefix_id"],
        "problem_id": row["problem_id"],
        "question": prob.get("question", ""),
        "behavior_state": pfx.get("behavior_state") or pfx.get("state_bucket"),
        "tau": tau,
        "utility_scores": scores,
        "u_greedy": scores[0],
        "u_branch": u_branch,
        "u_best_branch": max(u_branch) if u_branch else None,
        "u_max": max(scores),
        "branch_rescue_gain": (max(u_branch) - scores[0]) if u_branch else 0,
        "oracle_label": label,
        "reasoning_prefix_tail": _clip(pfx.get("prefix_text", ""), 500),
        "candidates": [
            {
                "name": d.get("candidate"),
                "u": d.get("utility_score"),
                "step": _clip(d.get("candidate_step", ""), 700),
            }
            for d in details
        ],
    }


def pick_utility_cases(
    score_rows: list[dict[str, Any]],
    prefixes: dict[str, dict[str, Any]],
    problems: dict[str, dict[str, Any]],
    *,
    tau: int = 7,
    n_each: int = 2,
) -> list[dict[str, Any]]:
    from reasoning_branch_dataset.action_study.oracle_labels import classify_oracle

    labeled: dict[str, list[dict[str, Any]]] = {
        "continue_sufficient": [],
        "weak_branch_rescuable": [],
        "handoff_required": [],
    }
    for row in score_rows:
        scores = row.get("utility_scores") or []
        if len(scores) < 5:
            continue
        label = classify_oracle(scores, tau=tau)["oracle_label"]
        bucket = "weak_branch_rescuable" if label in ("weak_branch_rescuable", "branch_rescuable") else label
        if bucket in labeled:
            labeled[bucket].append(_utility_case_row(row, prefixes, problems, tau=tau, label=bucket))

    branch_rescuable = sorted(
        labeled["weak_branch_rescuable"], key=lambda c: c.get("branch_rescue_gain", 0), reverse=True
    )[:n_each]
    handoff = sorted(labeled["handoff_required"], key=lambda c: c.get("u_max", c["u_greedy"]))[:n_each]
    continue_suff = sorted(labeled["continue_sufficient"], key=lambda c: -c["u_greedy"])[:n_each]

    out: list[dict[str, Any]] = []
    for bucket in (continue_suff, branch_rescuable, handoff):
        out.extend(bucket)
    return out


def format_utility_cases_md(cases: list[dict[str, Any]]) -> list[str]:
    lines = ["", "## Illustrative Cases (τ=7)", ""]
    if not cases:
        lines.append("_no cases selected_")
        return lines

    for i, c in enumerate(cases, start=1):
        lines.extend(
            [
                f"### Case {i}: {c['oracle_label']} (`{c['prefix_id']}`)",
                "",
                f"**Problem** ({c['problem_id']}):",
                "",
                f"> {_clip(c.get('question', ''), 400)}",
                "",
                f"- scores: greedy={c['u_greedy']}, branches={c['u_branch']}, τ={c['tau']}",
                "",
                "**Prefix tail:**",
                "",
                "```text",
                c.get("reasoning_prefix_tail", ""),
                "```",
                "",
            ]
        )
        for cand in c.get("candidates", []):
            lines.extend(
                [
                    f"**{cand['name']}** — utility **{cand['u']}**",
                    "",
                    "```text",
                    cand.get("step", ""),
                    "```",
                    "",
                ]
            )
    return lines


def regenerate_reachable_report(out_dir: Path) -> None:
    from reasoning_branch_dataset.action_study.run_reachable_state import (
        ReachableStateConfig,
        write_report,
    )

    cfg = ReachableStateConfig(
        out_dir=out_dir,
        target_model="",
        draft_model="",
    )
    write_report(
        cfg,
        out_dir / "verify_results.jsonl",
        report_path=out_dir.parent / "reachable_state_report.md",
    )


def regenerate_utility_report(v2_dir: Path, v3_dir: Path, target_model: str) -> None:
    from reasoning_branch_dataset.action_study.run_utility_scoring import write_utility_report

    write_utility_report(
        v2_dir,
        v3_dir,
        target_model=target_model,
        report_path=v3_dir.parent / "pilot_v3_report.md",
    )


def pick_uncertainty_cases(
    data_dir: Path,
    *,
    admission_col: str = "admission_main",
    n_each: int = 2,
) -> list[dict[str, Any]]:
    """Pick correctness-based illustrative cases from pilot v2 actions."""
    prefixes = {r["prefix_id"]: r for r in load_jsonl(data_dir / "prefixes.jsonl")}
    problems = {r["problem_id"]: r for r in load_jsonl(data_dir / "problems.jsonl")}
    admission = {r["prefix_id"]: r for r in load_jsonl(data_dir / "prefix_admission.jsonl")}

    by_prefix: dict[str, dict[str, list[dict]]] = {}
    for row in load_jsonl(data_dir / "actions.jsonl"):
        pid = row["prefix_id"]
        if admission_col == "admission_main" and not admission.get(pid, {}).get("admission_main"):
            continue
        slot = by_prefix.setdefault(pid, {"continue": [], "branch": []})
        if row["action_type"] == "continue":
            slot["continue"].append(row)
        elif row["action_type"] == "branch":
            slot["branch"].append(row)

    decision_sensitive: list[dict[str, Any]] = []
    branch_rescue: list[dict[str, Any]] = []
    continue_wins: list[dict[str, Any]] = []

    for pid, bundle in by_prefix.items():
        cont_rows = bundle.get("continue") or []
        branch_rows = bundle.get("branch") or []
        if not cont_rows or not branch_rows:
            continue
        cont = cont_rows[0]
        cont_ok = bool(cont.get("is_correct") or cont.get("correct"))
        branch_ok = [bool(b.get("is_correct") or b.get("correct")) for b in branch_rows]
        n_branch_ok = sum(branch_ok)
        pfx = prefixes.get(pid, {})
        prob = problems.get(pfx.get("problem_id", cont.get("problem_id")), {})
        base = {
            "prefix_id": pid,
            "problem_id": pfx.get("problem_id", cont.get("problem_id")),
            "question": prob.get("question", ""),
            "behavior_state": pfx.get("behavior_state") or pfx.get("state_bucket"),
            "gold_answer": prob.get("gold_answer"),
            "reasoning_prefix_tail": _clip(pfx.get("prefix_text", ""), 500),
            "continue_correct": cont_ok,
            "branch_correct_count": n_branch_ok,
            "continue_answer": cont.get("predicted_answer") or cont.get("final_answer"),
            "continue_step": _clip(cont.get("continuation", ""), 700),
            "best_branch_step": "",
            "best_branch_answer": None,
        }
        if not cont_ok and n_branch_ok > 0:
            best_i = next(i for i, ok in enumerate(branch_ok) if ok)
            best = branch_rows[best_i]
            case = {
                **base,
                "case_type": "correctness_branch_only_rescue",
                "behavior_state": base.get("behavior_state"),
                "best_branch_step": _clip(best.get("continuation", ""), 700),
                "best_branch_answer": best.get("predicted_answer") or best.get("final_answer"),
            }
            decision_sensitive.append(case)
        elif cont_ok and n_branch_ok == 0:
            continue_wins.append({**base, "case_type": "correctness_continue_sufficient"})
        elif not cont_ok and n_branch_ok == 0:
            branch_rescue.append({**base, "case_type": "correctness_both_fail"})

    out: list[dict[str, Any]] = []
    out.extend(sorted(decision_sensitive, key=lambda c: -c["branch_correct_count"])[:n_each])
    out.extend(continue_wins[:n_each])
    out.extend(branch_rescue[:n_each])
    return out


def format_uncertainty_cases_md(cases: list[dict[str, Any]]) -> list[str]:
    lines = ["", "## Illustrative Cases (correctness auxiliary)", ""]
    if not cases:
        lines.append("_no cases selected_")
        return lines

    for i, c in enumerate(cases, start=1):
        lines.extend(
            [
                f"### Case {i}: {c['case_type']} (`{c['prefix_id']}`)",
                "",
                f"**Problem** ({c['problem_id']}):",
                f"- `behavior_state` = **{c.get('behavior_state')}** (draft-side label)",
                f"- correctness: continue **{c.get('continue_correct')}**, branch **{c.get('branch_correct_count')}/4**",
                "",
                f"> {_clip(c.get('question', ''), 400)}",
                "",
                f"- gold: `{c.get('gold_answer')}` | continue answer: `{c.get('continue_answer')}`",
                "",
                "**Prefix tail:**",
                "",
                "```text",
                c.get("reasoning_prefix_tail", ""),
                "```",
                "",
                "**Continue continuation (first step block):**",
                "",
                "```text",
                c.get("continue_step", ""),
                "```",
                "",
            ]
        )
        if c.get("best_branch_step"):
            lines.extend(
                [
                    f"**Rescuing branch answer `{c.get('best_branch_answer')}`:**",
                    "",
                    "```text",
                    c.get("best_branch_step", ""),
                    "```",
                    "",
                ]
            )
    return lines

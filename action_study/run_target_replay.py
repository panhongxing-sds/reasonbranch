"""Target Acceptance Replay Experiment — offline verification on existing candidates."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from reasoning_branch_dataset.action_study.replay_sample import stratified_sample_prefixes
from reasoning_branch_dataset.action_study.target_verifier import (
    build_target_verifier,
    model_slug,
)
from reasoning_branch_dataset.model_utils import build_prompt


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _reasoning_from_prefix(prefix_text: str, question: str) -> str:
    prompt = build_prompt(question)
    if prefix_text.startswith(prompt):
        return prefix_text[len(prompt) :]
    if "Problem:" in prefix_text:
        idx = prefix_text.find("Problem:")
        body = prefix_text[idx:]
        if "\n" in body:
            return body.split("\n", 1)[1]
    return prefix_text


def _trace_cache_path(data_dir: Path, target_model: str) -> Path:
    return data_dir / f"target_replay_trace_verify_{model_slug(target_model)}.jsonl"


def _build_reachability_cache(
    data_dir: Path,
    verifier,
    *,
    target_model: str,
    force: bool = False,
) -> dict[str, dict[str, Any]]:
    cache_path = _trace_cache_path(data_dir, target_model)
    if cache_path.exists() and not force:
        return {r["problem_id"]: r for r in _load_jsonl(cache_path)}

    traces = _load_jsonl(data_dir / "traces.jsonl")
    problems = {p["problem_id"]: p for p in _load_jsonl(data_dir / "problems.jsonl")}
    prefixes = _load_jsonl(data_dir / "prefixes.jsonl")

    prefix_by_id = {p["prefix_id"]: p for p in prefixes}
    by_problem: dict[str, list[str]] = defaultdict(list)
    for p in prefixes:
        by_problem[p["problem_id"]].append(p["prefix_id"])

    rows: list[dict] = []
    reachable: dict[str, dict] = {}
    for trace in tqdm(traces, desc="trace_verify"):
        prob_id = trace["problem_id"]
        question = trace.get("question") or problems.get(prob_id, {}).get("question", "")
        prompt = build_prompt(question)
        tv = verifier.verify_trace(prompt, trace.get("full_reasoning", ""))
        accepted = tv["trace_accepted_length"]
        reachable_prefixes = []
        for prefix_id in by_problem.get(prob_id, []):
            pfx = prefix_by_id[prefix_id]
            reasoning = _reasoning_from_prefix(pfx.get("prefix_text", ""), question)
            rlen = len(verifier.tokenize(reasoning))
            if rlen <= accepted:
                reachable_prefixes.append(prefix_id)
        row = {
            "problem_id": prob_id,
            "target_model": target_model,
            **tv,
            "n_reachable_prefixes": len(reachable_prefixes),
            "reachable_prefix_ids": reachable_prefixes,
        }
        rows.append(row)
        reachable[prob_id] = row

    cache_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return reachable


def _reachable_prefix_ids(data_dir: Path, reach_cache: dict[str, dict]) -> set[str]:
    out: set[str] = set()
    for row in reach_cache.values():
        out.update(row.get("reachable_prefix_ids", []))
    return out


def _scheme_lengths(row: dict[str, Any]) -> dict[str, int | None]:
    return {
        "greedy_1": row.get("accepted_length_continue"),
        "sample_1": row.get("accepted_length_branch_0"),
        "best_of_2": row.get("accepted_length_best2"),
        "best_of_4": row.get("accepted_length_best4"),
    }


def run_replay(
    data_dir: Path,
    *,
    target_model: str,
    engine: str = "vllm",
    gamma: int = 64,
    n_samples: int = 300,
    seed: int = 42,
    resume: bool = True,
    reachable_only: bool = True,
    force_reverify: bool = False,
    device: str = "cuda",
    dtype: str = "bfloat16",
) -> Path:
    slug = model_slug(target_model)
    out_path = data_dir / f"target_replay_results_{slug}.jsonl"
    done: set[str] = set()
    if resume and out_path.exists():
        done = {r["prefix_id"] for r in _load_jsonl(out_path)}

    prefixes = _load_jsonl(data_dir / "prefixes.jsonl")
    actions = _load_jsonl(data_dir / "actions.jsonl")
    traces = {t["problem_id"]: t for t in _load_jsonl(data_dir / "traces.jsonl")}
    problems = {p["problem_id"]: p for p in _load_jsonl(data_dir / "problems.jsonl")}

    verifier = build_target_verifier(target_model, engine=engine, device=device, dtype=dtype)
    reach_cache = _build_reachability_cache(
        data_dir, verifier, target_model=target_model, force=force_reverify
    )
    reachable_ids = _reachable_prefix_ids(data_dir, reach_cache)

    manifest_path = data_dir / f"target_replay_sample_{slug}.jsonl"
    meta_path = data_dir / f"target_replay_sample_meta_{slug}.json"
    if manifest_path.exists() and resume:
        samples = _load_jsonl(manifest_path)
    else:
        samples = stratified_sample_prefixes(
            prefixes, actions, n=n_samples, seed=seed, reachable_ids=reachable_ids
        )
        manifest_path.write_text(
            "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in samples), encoding="utf-8"
        )
        meta = {
            "n_samples": len(samples),
            "n_problems": len({s["problem_id"] for s in samples}),
            "target_model": target_model,
            "engine": engine,
            "n_reachable_in_pool": len(reachable_ids),
            "n_sampled": len(samples),
            "prefix_ids": [s["prefix_id"] for s in samples],
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    samples = [s for s in samples if s["prefix_id"] not in done]

    action_map: dict[str, dict[str, list[dict]]] = defaultdict(lambda: {"continue": [], "branch": []})
    for a in actions:
        at = a.get("action_type")
        if at in {"continue", "branch"}:
            action_map[a["prefix_id"]][at].append(a)

    if not samples:
        return out_path

    rows_out: list[dict] = []
    for pfx in tqdm(samples, desc="target_replay"):
        pid = pfx["prefix_id"]
        prob_id = pfx["problem_id"]
        trace = traces.get(prob_id, {})
        prob = problems.get(prob_id, {})
        question = prob.get("question") or trace.get("question", "")
        prompt = build_prompt(question)

        trace_verify = reach_cache.get(prob_id, verifier.verify_trace(prompt, trace.get("full_reasoning", "")))
        reasoning = _reasoning_from_prefix(pfx.get("prefix_text", ""), question)
        prefix_reasoning_ids = verifier.tokenize(reasoning)
        reach = prefix_reachability(
            trace_accepted_length=trace_verify["trace_accepted_length"],
            prefix_reasoning_token_len=len(prefix_reasoning_ids),
        )

        if reachable_only and not reach["prefix_fully_target_accepted"]:
            rows_out.append(
                {
                    "prefix_id": pid,
                    "problem_id": prob_id,
                    "gamma": gamma,
                    "target_model": target_model,
                    "target_engine": engine,
                    "skipped": True,
                    "skip_reason": "prefix_not_target_reachable",
                    **trace_verify,
                    **reach,
                }
            )
            continue

        prefix_text = pfx.get("prefix_text", prompt + reasoning)
        amap = action_map.get(pid, {})
        cont_actions = amap.get("continue", [])
        branch_actions = sorted(amap.get("branch", []), key=lambda x: x.get("sample_id", 0))

        cont_len = None
        if cont_actions:
            cont_len = verifier.greedy_acceptance(
                prefix_text, cont_actions[0].get("continuation", ""), gamma=gamma
            ).accepted_length

        branch_lens: list[int] = []
        for ba in branch_actions[:4]:
            branch_lens.append(
                verifier.greedy_acceptance(prefix_text, ba.get("continuation", ""), gamma=gamma).accepted_length
            )
        while len(branch_lens) < 4:
            branch_lens.append(0)

        best2 = max(branch_lens[:2]) if branch_lens else 0
        best4 = max(branch_lens) if branch_lens else 0
        gain_best4 = (best4 - cont_len) if cont_len is not None else None

        rows_out.append(
            {
                "prefix_id": pid,
                "problem_id": prob_id,
                "gamma": gamma,
                "target_model": target_model,
                "target_engine": engine,
                "skipped": False,
                "prefix_validity": pfx.get("prefix_validity"),
                "strategy_diversity": pfx.get("strategy_diversity"),
                "reasoning_progress": pfx.get("reasoning_progress"),
                "behavior_state": pfx.get("behavior_state"),
                "continue_is_correct": cont_actions[0].get("is_correct") if cont_actions else None,
                **trace_verify,
                **reach,
                "accepted_length_continue": cont_len,
                "accepted_length_branch_0": branch_lens[0],
                "accepted_length_branch_1": branch_lens[1],
                "accepted_length_branch_2": branch_lens[2],
                "accepted_length_branch_3": branch_lens[3],
                "accepted_length_best2": best2,
                "accepted_length_best4": best4,
                "branch_acceptance_gain": gain_best4,
                "target_selected_branch": int(np.argmax(branch_lens)) if branch_lens else None,
                "all_branches_rejected_early": best4 < 4,
            }
        )

    _append_jsonl(out_path, rows_out)
    # symlink canonical path for downstream tools
    canonical = data_dir / "target_replay_results.jsonl"
    if canonical.exists() or canonical.is_symlink():
        canonical.unlink(missing_ok=True)
    canonical.symlink_to(out_path.name)
    return out_path


def build_acceptance_tables(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [r for r in _load_jsonl(data_dir / "target_replay_results.jsonl") if not r.get("skipped")]
    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    schemes = {
        "Greedy-1": "accepted_length_continue",
        "Sample-1": "accepted_length_branch_0",
        "Best-of-2": "accepted_length_best2",
        "Best-of-4": "accepted_length_best4",
    }
    main_rows = []
    for name, col in schemes.items():
        vals = [r[col] for r in rows if r.get(col) is not None]
        if not vals:
            continue
        arr = np.array(vals, dtype=float)
        main_rows.append(
            {
                "candidate_scheme": name,
                "n": len(vals),
                "mean_accepted_length": float(arr.mean()),
                "median_accepted_length": float(np.median(arr)),
                "P_A_ge_8": float((arr >= 8).mean()),
                "P_A_ge_16": float((arr >= 16).mean()),
                "P_A_ge_32": float((arr >= 32).mean()),
            }
        )
    main_df = pd.DataFrame(main_rows)

    greedy = np.array([r["accepted_length_continue"] for r in rows if r.get("accepted_length_continue") is not None])
    best4 = np.array([r["accepted_length_best4"] for r in rows if r.get("accepted_length_best4") is not None])
    if len(greedy) and len(best4):
        gain = best4[: len(greedy)] - greedy[: len(best4)]
        summary = {
            "E_best4_minus_greedy": float(gain.mean()),
            "P_best4_gt_greedy": float((gain > 0).mean()),
            "P_gain_ge_8": float((gain >= 8).mean()),
            "P_gain_ge_16": float((gain >= 16).mean()),
        }
        (data_dir / "target_replay_gain_summary.json").write_text(json.dumps(summary, indent=2))

    # uncertainty breakdown
    def _div_bucket(s: str | None) -> str:
        return "high" if s in {"MULTIPLE_GENUINE_STRATEGIES", "HIGH_DIVERSITY"} else "low"

    unc_rows = []
    groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        key = (r.get("prefix_validity", "UNCLEAR"), _div_bucket(r.get("strategy_diversity")))
        groups[key].append(r)

    for (validity, div), items in sorted(groups.items()):
        g = np.array([x["accepted_length_continue"] for x in items], dtype=float)
        b = np.array([x["accepted_length_best4"] for x in items], dtype=float)
        gain = b - g
        unc_rows.append(
            {
                "prefix_signal": f"{validity} + {div} diversity",
                "N": len(items),
                "greedy_acceptance": float(g.mean()),
                "best_of_4_acceptance": float(b.mean()),
                "branch_gain": float(gain.mean()),
                "P_gain_ge_8": float((gain >= 8).mean()),
            }
        )
    unc_df = pd.DataFrame(unc_rows)
    return main_df, unc_df


def write_report(data_dir: Path) -> Path:
    main_df, unc_df = build_acceptance_tables(data_dir)
    all_rows = _load_jsonl(data_dir / "target_replay_results.jsonl")
    n_total = len(all_rows)
    n_skipped = sum(1 for r in all_rows if r.get("skipped"))
    n_eval = n_total - n_skipped

    lines = [
        "# Target Acceptance Replay Report",
        "",
        f"> Evaluated: **{n_eval}** prefixes (skipped unreachable: {n_skipped})",
        "",
    ]
    meta_path = data_dir / "target_replay_sample_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        lines.append(f"> Reachable pool: **{meta.get('n_reachable_in_pool', '?')}** prefixes")
        lines.append(f"> Sampled: **{meta.get('n_sampled', meta.get('n_samples', '?'))}**")
        lines.append("")
    lines.append("## Acceptance Main Table")
    lines.append(main_df.to_markdown(index=False) if not main_df.empty else "_no data_")
    lines.append("")
    gain_path = data_dir / "target_replay_gain_summary.json"
    if gain_path.exists():
        g = json.loads(gain_path.read_text())
        lines.append("## Branch Acceptance Gain (Best-of-4 vs Greedy-1)")
        for k, v in g.items():
            lines.append(f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}")
        lines.append("")

    lines.extend(
        [
            "## Uncertainty Breakdown",
            unc_df.to_markdown(index=False) if not unc_df.empty else "_no data_",
            "",
            "## Notes",
            "- Primary metric: `branch_acceptance_gain = best_of_4 - greedy_1`",
            "- `branch_pass_at_4` / final correctness are auxiliary only",
            "- Only target-reachable prefixes included when `reachable_only=True`",
        ]
    )
    report = data_dir / "target_replay_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    if not main_df.empty:
        main_df.to_csv(data_dir / "target_replay_main_table.csv", index=False)
    if not unc_df.empty:
        unc_df.to_csv(data_dir / "target_replay_uncertainty_table.csv", index=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Target acceptance replay on existing candidates")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--target-model",
        type=str,
        default="/mnt/afs/L202500372/specreason/models/QwQ-32B",
        help="Target verifier model (>=30B recommended)",
    )
    parser.add_argument("--engine", choices=["vllm", "hf"], default="vllm")
    parser.add_argument("--gamma", type=int, default=64)
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--force-reverify", action="store_true")
    parser.add_argument("--include-unreachable", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    if not args.report_only:
        run_replay(
            args.data_dir,
            target_model=args.target_model,
            engine=args.engine,
            gamma=args.gamma,
            n_samples=args.n_samples,
            seed=args.seed,
            resume=not args.no_resume,
            reachable_only=not args.include_unreachable,
            force_reverify=args.force_reverify,
        )
    report = write_report(args.data_dir)
    print(f"Report: {report}")


if __name__ == "__main__":
    main()

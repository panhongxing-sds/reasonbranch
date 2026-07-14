"""Scheme A: target-prefix reachable-state branch gain experiment.

1. QwQ greedy-generates a short target trace per problem.
2. Pick 2-3 checkpoints along that trace (target-reachable by construction).
3. From each checkpoint, 4B drafts: greedy x1 + sample x4 (gamma tokens each).
4. QwQ verifies accepted_length for each draft; report G_B = A_best4 - A_single.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from reasoning_branch_dataset.model_utils import build_prompt


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _free_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _done_ids(path: Path, key: str = "row_id") -> set[str]:
    return {r[key] for r in _load_jsonl(path) if key in r}


@dataclass
class ReachableStateConfig:
    out_dir: Path
    target_model: str
    draft_model: str
    n_problems: int = 100
    gamma: int = 32
    target_trace_tokens: int = 256
    checkpoint_positions: tuple[int, ...] = (32, 96, 192)
    branch_k: int = 4
    branch_temperature: float = 0.7
    branch_top_p: float = 0.95
    seed: int = 42


def sample_problems(data_dir: Path, n: int, seed: int) -> list[dict[str, Any]]:
    rows = [r for r in _load_jsonl(data_dir / "problems.jsonl") if r.get("input_complete", True)]
    if not rows:
        raise FileNotFoundError(f"No problems in {data_dir / 'problems.jsonl'}")
    rng = random.Random(seed)
    if len(rows) <= n:
        return rows
    return rng.sample(rows, n)


def pick_checkpoints(trace_token_count: int, positions: tuple[int, ...]) -> list[int]:
    out = [p for p in positions if 0 < p < trace_token_count]
    if not out and trace_token_count > 8:
        out = [max(8, trace_token_count // 3)]
    return out[:3]


def phase1_target_traces(cfg: ReachableStateConfig, problems: list[dict[str, Any]], *, resume: bool) -> Path:
    from reasoning_branch_dataset.action_study.target_verifier import (
        build_target_verifier,
        greedy_generate_ids_vllm,
    )

    out_path = cfg.out_dir / "target_traces.jsonl"
    done = _done_ids(out_path, "problem_id") if resume else set()
    verifier = build_target_verifier(cfg.target_model, engine="vllm", debug=True)

    try:
        for prob in tqdm(problems, desc="target_traces"):
            pid = prob["problem_id"]
            if pid in done:
                continue
            prompt = build_prompt(prob["question"])
            prefix_ids = verifier.tokenize(prompt)
            gen_ids = greedy_generate_ids_vllm(
                verifier.llm, prefix_ids, max_tokens=cfg.target_trace_tokens
            )
            reasoning_ids = gen_ids
            reasoning_text = verifier.tokenizer.decode(reasoning_ids, skip_special_tokens=False)
            checkpoints = pick_checkpoints(len(reasoning_ids), cfg.checkpoint_positions)
            ck_rows = []
            for pos in checkpoints:
                ck_rows.append(
                    {
                        "checkpoint_token_pos": pos,
                        "reasoning_prefix_text": verifier.tokenizer.decode(
                            reasoning_ids[:pos], skip_special_tokens=False
                        ),
                        "reasoning_prefix_ids": reasoning_ids[:pos],
                    }
                )
            _append_jsonl(
                out_path,
                {
                    "problem_id": pid,
                    "question": prob["question"],
                    "gold_answer": prob.get("gold_answer"),
                    "prompt": prompt,
                    "target_model": cfg.target_model,
                    "trace_token_count": len(reasoning_ids),
                    "trace_text": reasoning_text,
                    "trace_token_ids": reasoning_ids,
                    "checkpoints": ck_rows,
                },
            )
    finally:
        del verifier
        _free_cuda()
    return out_path


def _build_draft_engine(model_path: str):
    import os

    from vllm import LLM, SamplingParams

    os.environ.setdefault("VLLM_GDN_PREFILL_BACKEND", "triton")
    kwargs: dict[str, Any] = {
        "model": model_path,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "gpu_memory_utilization": 0.75,
        "max_model_len": 8192,
        "enforce_eager": True,
        "enable_prefix_caching": False,
    }
    if "Qwen3.5" in Path(model_path).name:
        kwargs["language_model_only"] = True
    llm = LLM(**kwargs)

    def sample(prefix: str, *, k: int, max_tokens: int, temperature: float, top_p: float) -> list[dict]:
        params = SamplingParams(
            n=k,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            detokenize=True,
        )
        out = llm.generate([prefix], params)[0].outputs
        return [
            {
                "text": c.text,
                "token_ids": list(c.token_ids),
                "num_tokens": len(c.token_ids),
            }
            for c in out
        ]

    return llm, sample


def phase2_drafts(cfg: ReachableStateConfig, traces_path: Path, *, resume: bool) -> Path:
    out_path = cfg.out_dir / "drafts.jsonl"
    done = _done_ids(out_path, "row_id") if resume else set()
    llm, sample = _build_draft_engine(cfg.draft_model)

    try:
        for trace in tqdm(_load_jsonl(traces_path), desc="draft_gen"):
            pid = trace["problem_id"]
            prompt = trace["prompt"]
            for ck in trace.get("checkpoints", []):
                pos = ck["checkpoint_token_pos"]
                row_id = f"{pid}@ck{pos}"
                if row_id in done:
                    continue
                prefix_full = prompt + ck["reasoning_prefix_text"]
                greedy = sample(prefix_full, k=1, max_tokens=cfg.gamma, temperature=0.0, top_p=1.0)[0]
                branches = sample(
                    prefix_full,
                    k=cfg.branch_k,
                    max_tokens=cfg.gamma,
                    temperature=cfg.branch_temperature,
                    top_p=cfg.branch_top_p,
                )
                _append_jsonl(
                    out_path,
                    {
                        "row_id": row_id,
                        "problem_id": pid,
                        "checkpoint_token_pos": pos,
                        "prefix_full": prefix_full,
                        "gamma": cfg.gamma,
                        "greedy_continuation": greedy["text"],
                        "greedy_token_ids": greedy["token_ids"],
                        "branch_continuations": [b["text"] for b in branches],
                        "branch_token_ids": [b["token_ids"] for b in branches],
                    },
                )
    finally:
        del llm, sample
        _free_cuda()
    return out_path


def phase3_verify(
    cfg: ReachableStateConfig,
    drafts_path: Path,
    *,
    resume: bool,
    verifier=None,
) -> Path:
    from reasoning_branch_dataset.action_study.target_verifier import build_target_verifier

    out_path = cfg.out_dir / "verify_results.jsonl"
    done = _done_ids(out_path, "row_id") if resume else set()
    own = verifier is None
    if own:
        verifier = build_target_verifier(cfg.target_model, engine="vllm", debug=True)

    try:
        for row in tqdm(_load_jsonl(drafts_path), desc="target_verify"):
            row_id = row["row_id"]
            if row_id in done:
                continue
            prompt = row["prefix_full"]
            # prefix_full already includes prompt; verifier expects prefix_text as full context
            # greedy_acceptance splits via tokenize on prefix_text + draft separately in our API
            # Use prefix as the full string before continuation
            prefix_text = row["prefix_full"]
            a_greedy = verifier.greedy_acceptance(
                prefix_text, row["greedy_continuation"], gamma=cfg.gamma
            ).accepted_length
            branch_as = []
            for cont in row["branch_continuations"]:
                branch_as.append(
                    verifier.greedy_acceptance(prefix_text, cont, gamma=cfg.gamma).accepted_length
                )
            a_best = max(branch_as) if branch_as else 0
            _append_jsonl(
                out_path,
                {
                    "row_id": row_id,
                    "problem_id": row["problem_id"],
                    "checkpoint_token_pos": row["checkpoint_token_pos"],
                    "gamma": cfg.gamma,
                    "accepted_length_single": a_greedy,
                    "accepted_lengths_branch": branch_as,
                    "accepted_length_best4": a_best,
                    "branch_gain": a_best - a_greedy,
                },
            )
    finally:
        if own:
            del verifier
            _free_cuda()
    return out_path


def write_report(cfg: ReachableStateConfig, results_path: Path, report_path: Path | None = None) -> Path:
    rows = _load_jsonl(results_path)
    dest = report_path or (cfg.out_dir.parent / "reachable_state_report.md")
    if not rows:
        dest.write_text("# Reachable-State Experiment\n\n_no results_\n", encoding="utf-8")
        return dest

    gains = np.array([r["branch_gain"] for r in rows], dtype=float)
    singles = np.array([r["accepted_length_single"] for r in rows], dtype=float)
    bests = np.array([r["accepted_length_best4"] for r in rows], dtype=float)
    pos_gain = float(np.mean(gains > 0))
    by_ck: dict[int, list[float]] = {}
    for r in rows:
        by_ck.setdefault(int(r["checkpoint_token_pos"]), []).append(r["branch_gain"])

    lines = [
        "# Reachable-State Report (Scheme A)",
        "",
        "> QwQ greedy trace → checkpoint → 4B draft (γ tokens) → QwQ verify acceptance length.",
        "",
        f"- data: `reachable_state_pilot/`",
        f"- target: `{cfg.target_model}`",
        f"- draft: `{cfg.draft_model}`",
        f"- gamma: {cfg.gamma}",
        f"- n_checkpoints_evaluated: {len(rows)}",
        "",
        "## Summary",
        "",
        f"| metric | value |",
        f"|--------|-------|",
        f"| mean A_single | {singles.mean():.3f} |",
        f"| mean A_best4 | {bests.mean():.3f} |",
        f"| mean G_B | {gains.mean():.3f} |",
        f"| P(G_B > 0) | {pos_gain:.3f} |",
        f"| median G_B | {float(np.median(gains)):.3f} |",
        "",
        "## By checkpoint position",
        "",
        "| checkpoint_tokens | mean G_B | n |",
        "|-------------------|----------|---|",
    ]
    for pos in sorted(by_ck):
        g = np.array(by_ck[pos])
        lines.append(f"| {pos} | {g.mean():.3f} | {len(g)} |")

    drafts_by_id = {r["row_id"]: r for r in _load_jsonl(cfg.out_dir / "drafts.jsonl")}
    problems = {r["problem_id"]: r for r in _load_jsonl(cfg.out_dir / "sampled_problems.jsonl")}
    from reasoning_branch_dataset.action_study.report_samples import (
        format_reachable_cases_md,
        pick_reachable_cases,
    )

    cases = pick_reachable_cases(rows, drafts_by_id, problems, n_each=1)
    lines.extend(format_reachable_cases_md(cases))

    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "n_rows": len(rows),
        "mean_a_single": float(singles.mean()),
        "mean_a_best4": float(bests.mean()),
        "mean_branch_gain": float(gains.mean()),
        "p_gain_positive": pos_gain,
    }
    (cfg.out_dir / "reachable_state_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return dest


def run_experiment(
    cfg: ReachableStateConfig,
    problems: list[dict[str, Any]],
    *,
    resume: bool = True,
    phases: str = "all",
) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    traces_path = cfg.out_dir / "target_traces.jsonl"
    drafts_path = cfg.out_dir / "drafts.jsonl"
    results_path = cfg.out_dir / "verify_results.jsonl"

    if phases in ("all", "1"):
        phase1_target_traces(cfg, problems, resume=resume)
    if phases in ("all", "2"):
        if not traces_path.exists():
            raise FileNotFoundError(f"Missing {traces_path}; run phase 1 first")
        phase2_drafts(cfg, traces_path, resume=resume)
    if phases in ("all", "3"):
        if not drafts_path.exists():
            raise FileNotFoundError(f"Missing {drafts_path}; run phase 2 first")
        phase3_verify(cfg, drafts_path, resume=resume)
    if phases in ("all", "3", "report"):
        write_report(cfg, results_path)
        print(f"Report -> {cfg.out_dir.parent / 'reachable_state_report.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Target-prefix reachable-state experiment")
    parser.add_argument(
        "--data-dir",
        default="/mnt/afs/L202500372/reasoning_branch_dataset/outputs/action_study_pilot_v2",
    )
    parser.add_argument(
        "--out-dir",
        default="/mnt/afs/L202500372/reasoning_branch_dataset/outputs/reachable_state_pilot",
    )
    parser.add_argument(
        "--target-model",
        default="/mnt/afs/L202500372/specreason/models/QwQ-32B",
    )
    parser.add_argument(
        "--draft-model",
        default="/mnt/afs/L202500372/models/Qwen3.5-4B",
    )
    parser.add_argument("--n-problems", type=int, default=100)
    parser.add_argument("--gamma", type=int, default=32)
    parser.add_argument("--target-trace-tokens", type=int, default=256)
    parser.add_argument("--phases", default="all", choices=["all", "1", "2", "3", "report"])
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    cfg = ReachableStateConfig(
        out_dir=Path(args.out_dir),
        target_model=args.target_model,
        draft_model=args.draft_model,
        n_problems=args.n_problems,
        gamma=args.gamma,
        target_trace_tokens=args.target_trace_tokens,
    )
    problems = sample_problems(Path(args.data_dir), cfg.n_problems, cfg.seed)
    _write_jsonl(cfg.out_dir / "sampled_problems.jsonl", problems)
    run_experiment(cfg, problems, resume=not args.no_resume, phases=args.phases)


if __name__ == "__main__":
    main()

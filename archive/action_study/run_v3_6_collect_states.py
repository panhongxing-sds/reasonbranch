"""V3.6: collect greedy-rejected states under final 1.5B+32B stack."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from reasoning_branch_dataset.action_study.v36_counterfactual import DualResidentSession
from reasoning_branch_dataset.action_study.v36_step_gen import generate_one_step_vllm
from reasoning_branch_dataset.model_utils import build_prompt


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_problems(path: Path, n: int, seed: int) -> list[dict[str, Any]]:
    rows = _load_jsonl(path)
    cleaned = []
    for i, r in enumerate(rows):
        q = r.get("problem") or r.get("question") or ""
        if q.strip():
            cleaned.append({"problem_id": str(r.get("id", r.get("problem_id", f"p{i:04d}"))), "question": q.strip()})
    rng = random.Random(seed)
    return cleaned if len(cleaned) <= n else rng.sample(cleaned, n)


def reject_bucket(score: float, tau: float) -> str:
    """Stratify rejection strength relative to the calibrated tau."""
    d = score - tau  # negative for rejects
    if -1.0 <= d < 0.0:
        return "near"
    if -3.0 <= d < -1.0:
        return "medium"
    if d < -3.0:
        return "hard"
    return "accept"  # d >= 0 (not a reject)


def collect_rejected_states(
    session: DualResidentSession,
    problems: list[dict[str, Any]],
    *,
    n_states: int,
    out_path: Path,
    seed: int = 42,
    tau_accept: float = 0.0,
    max_depth: int = 8,
    step_max_tokens: int = 256,
    warm_tokens: tuple[int, ...] = (128, 384, 768, 1280),
    per_problem_cap: int = 4,
    bucket_targets: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Roll draft forward; keep verifier-rejected states, stratified by reject
    strength (near/medium/hard) with a per-problem cap for coverage."""
    rng = random.Random(seed)
    existing = _load_jsonl(out_path)
    done_ids = {r["state_id"] for r in existing}
    collected: list[dict[str, Any]] = []
    idx = 0
    attempts = 0
    max_attempts = max(n_states * 30, 150)
    n_accept = 0
    n_reject = 0
    n_empty = 0
    score_samples: list[float] = []

    # Coverage counters (seed from existing rows so re-runs extend, not restart).
    per_problem: dict[str, int] = {}
    bucket_counts: dict[str, int] = {"near": 0, "medium": 0, "hard": 0}
    for r in existing:
        per_problem[r["problem_id"]] = per_problem.get(r["problem_id"], 0) + 1
        b = r.get("reject_bucket")
        if b in bucket_counts:
            bucket_counts[b] += 1
    if bucket_targets is None:
        # roughly balanced across three buckets
        each = max(1, n_states // 3)
        bucket_targets = {"near": each, "medium": each, "hard": n_states - 2 * each}

    while len(done_ids) < n_states and attempts < max_attempts:
        attempts += 1
        prob = problems[idx % len(problems)]
        idx += 1
        prompt = build_prompt(prob["question"])
        # Build a multi-step accepted prefix by appending verifier-accepted greedy steps,
        # then capture the first rejected greedy at some depth.
        prefix = prompt + "\n"
        depth = 0
        target_warm = warm_tokens[attempts % len(warm_tokens)]
        # Warm generate a long draft trace and cut into a prefix of ~target_warm tokens.
        warm = session.draft.llm  # noqa — using generate_one_step iteratively is slow;
        from reasoning_branch_dataset.action_study.vllm_backend import VLLMEngine

        assert isinstance(session.draft, VLLMEngine)
        warm_out = session.draft.generate_trace(prompt, max_tokens=max(target_warm + 64, 160))
        text = warm_out["response_text"]
        blocks = [b for b in text.split("\n\n") if b.strip()]
        if len(blocks) >= 2:
            keep = max(1, min(len(blocks) - 1, 1 + (attempts % max(1, len(blocks) - 1))))
            prefix = prompt + "\n\n".join(blocks[:keep]).rstrip() + "\n\n"
            depth = keep
        else:
            prefix = prompt + (text[: max(80, len(text) // 2)].rstrip() + "\n\n")
            depth = 1

        # Greedy next step from draft
        greedy = generate_one_step_vllm(
            session.draft.llm,
            session.draft.llm.get_tokenizer(),
            prefix,
            max_tokens=step_max_tokens,
            temperature=0.0,
            n=1,
        )[0]
        if not greedy.text.strip():
            n_empty += 1
            continue

        # Warm shared prefix on target then score greedy
        score = session.verifier.score_greedy(
            question=prob["question"],
            prefix_text=prefix,
            greedy_step=greedy.text,
            tau_accept=tau_accept,
        )
        score_samples.append(score.score)
        if attempts <= 5 or attempts % 10 == 0:
            print(
                f"[v3.6 collect] attempt={attempts} score={score.score:.3f} "
                f"logA={score.logp_accept:.2f} logR={score.logp_reject:.2f} "
                f"accepted={score.score >= tau_accept}"
            )
        if score.score >= tau_accept:
            n_accept += 1
            continue

        n_reject += 1
        bucket = reject_bucket(score.score, tau_accept)
        # Skip if this bucket is already full, or this problem hit its cap.
        if bucket_counts.get(bucket, 0) >= bucket_targets.get(bucket, 0):
            continue
        if per_problem.get(prob["problem_id"], 0) >= per_problem_cap:
            continue

        tok = session.target_tok
        prefix_tokens = len(tok.encode(prefix))
        state_id = f"{prob['problem_id']}_d{depth}_t{prefix_tokens}_a{attempts}"
        if state_id in done_ids:
            continue
        row = {
            "state_id": state_id,
            "problem_id": prob["problem_id"],
            "question": prob["question"],
            "prefix_text": prefix,
            "prefix_tokens": prefix_tokens,
            "reasoning_depth": depth,
            "greedy_step": greedy.text,
            "greedy_step_tokens": greedy.num_tokens,
            "greedy_step_status": greedy.status,
            "greedy_verifier_score": score.score,
            "greedy_logp_accept": score.logp_accept,
            "greedy_logp_reject": score.logp_reject,
            "reject_bucket": bucket,
            "tau_accept": tau_accept,
            "split": None,  # assigned later by problem_id
        }
        _append_jsonl(out_path, row)
        done_ids.add(state_id)
        collected.append(row)
        per_problem[prob["problem_id"]] = per_problem.get(prob["problem_id"], 0) + 1
        bucket_counts[bucket] += 1
        print(
            f"[v3.6 collect] {len(done_ids)}/{n_states} rejected "
            f"score={score.score:.2f} bucket={bucket} depth={depth} tok={prefix_tokens} "
            f"buckets={bucket_counts}"
        )

    if score_samples:
        mean_s = sum(score_samples) / len(score_samples)
        print(
            f"[v3.6 collect] done attempts={attempts} accept={n_accept} reject={n_reject} "
            f"empty={n_empty} score_mean={mean_s:.3f} "
            f"score_min={min(score_samples):.3f} score_max={max(score_samples):.3f}"
        )
        print(
            f"[v3.6 collect] buckets={bucket_counts} "
            f"problems_covered={len(per_problem)} "
            f"depth_range=[{min((r['reasoning_depth'] for r in collected), default=0)},"
            f"{max((r['reasoning_depth'] for r in collected), default=0)}]"
        )
    else:
        print(f"[v3.6 collect] done attempts={attempts} no scores (empty={n_empty})")
    return collected


def assign_splits(
    states_path: Path,
    *,
    seed: int = 42,
    calib_frac: float = 0.25,
    dev_frac: float = 0.25,
) -> None:
    rows = _load_jsonl(states_path)
    problems = sorted({r["problem_id"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(problems)
    n = len(problems)
    n_cal = max(1, int(n * calib_frac))
    n_dev = max(1, int(n * dev_frac))
    cal = set(problems[:n_cal])
    dev = set(problems[n_cal : n_cal + n_dev])
    for r in rows:
        pid = r["problem_id"]
        r["split"] = "calibration" if pid in cal else ("development" if pid in dev else "test")
    with states_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[v3.6] splits: cal={len(cal)} dev={len(dev)} test={n - len(cal) - len(dev)} problems")


def main() -> None:
    p = argparse.ArgumentParser(description="V3.6 collect greedy-rejected states")
    p.add_argument("--draft-model", default="/root/autodl-tmp/specreason/models/DeepSeek-R1-Distill-Qwen-1.5B")
    p.add_argument("--target-model", default="/root/autodl-tmp/specreason/models/DeepSeek-R1-Distill-Qwen-32B-AWQ")
    p.add_argument("--problems", default="/root/autodl-tmp/reasonbranch/data/aime_train.jsonl")
    p.add_argument("--out", default="/root/autodl-tmp/reasonbranch/outputs/action_study_v36/rejected_states.jsonl")
    p.add_argument("--n-states", type=int, default=64)
    p.add_argument("--n-problems", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tau-accept", type=float, default=0.0, help="calibrated logP(Accept)-logP(Reject) threshold")
    p.add_argument("--tau-from", default="", help="read calibrated tau from a tau_sweep.json (recommended.tau)")
    p.add_argument("--per-problem-cap", type=int, default=4)
    p.add_argument("--bucket-near", type=int, default=0, help="override target count for near bucket (0=auto)")
    p.add_argument("--bucket-medium", type=int, default=0)
    p.add_argument("--bucket-hard", type=int, default=0)
    p.add_argument("--assign-splits-only", action="store_true")
    args = p.parse_args()

    out = Path(args.out)
    if args.assign_splits_only:
        assign_splits(out, seed=args.seed)
        return

    tau = args.tau_accept
    if args.tau_from:
        rep = json.loads(Path(args.tau_from).read_text())
        rec = rep.get("recommended")
        if rec and rec.get("tau") is not None:
            tau = float(rec["tau"])
            print(f"[v3.6 collect] using calibrated tau={tau:.3f} from {args.tau_from}")
        else:
            print(f"[v3.6 collect] WARN: {args.tau_from} has no recommended tau; using {tau}")

    bucket_targets = None
    if args.bucket_near or args.bucket_medium or args.bucket_hard:
        bucket_targets = {
            "near": args.bucket_near,
            "medium": args.bucket_medium,
            "hard": args.bucket_hard,
        }

    problems = load_problems(Path(args.problems), args.n_problems, args.seed)
    session = DualResidentSession(
        draft_model=args.draft_model,
        target_model=args.target_model,
        verify_tau=tau,
    )
    collect_rejected_states(
        session,
        problems,
        n_states=args.n_states,
        out_path=out,
        seed=args.seed,
        tau_accept=tau,
        per_problem_cap=args.per_problem_cap,
        bucket_targets=bucket_targets,
    )
    assign_splits(out, seed=args.seed)


if __name__ == "__main__":
    main()

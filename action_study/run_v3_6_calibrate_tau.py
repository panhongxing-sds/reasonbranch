"""V3.6 verifier threshold (tau_A) calibration (user review §3).

Builds an independent candidate set (NOT the mechanism states), labels each
candidate with the semantic oracle (DeepSeek V4 Pro), then sweeps the verifier
score threshold tau and reports precision / recall / false-acceptance-rate /
coverage. Goal: pick tau such that P(A=1 | v>=tau) is high (near-lossless).

Output:
  - candidates.jsonl : {question, prefix, prefix_hash, candidate, cand_hash,
                        verifier_score, logp_accept, logp_reject, oracle_label,
                        oracle_detail}
  - tau_sweep.json   : per-tau precision/recall/FAR/coverage + recommended tau
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from reasoning_branch_dataset.action_study.run_v3_6_pilot import (
    _load_jsonl,
    maybe_oracle_label,
)
from reasoning_branch_dataset.action_study.v36_counterfactual import DualResidentSession
from reasoning_branch_dataset.action_study.v36_step_gen import generate_one_step_vllm
from reasoning_branch_dataset.model_utils import build_prompt


def _sha1(t: str) -> str:
    return hashlib.sha1((t or "").encode("utf-8")).hexdigest()


def _load_problems(path: Path, n: int, seed: int) -> list[dict[str, Any]]:
    rows = _load_jsonl(path)
    cleaned = []
    for i, r in enumerate(rows):
        q = r.get("problem") or r.get("question") or ""
        if q.strip():
            cleaned.append(
                {"problem_id": str(r.get("id", r.get("problem_id", f"p{i:04d}"))), "question": q.strip()}
            )
    rng = random.Random(seed)
    return cleaned if len(cleaned) <= n else rng.sample(cleaned, n)


def collect_candidates(
    session: DualResidentSession,
    problems: list[dict[str, Any]],
    *,
    n_candidates: int,
    out_path: Path,
    oracle_cache: dict[str, Any],
    seed: int = 42,
    per_prefix: int = 4,
    step_max_tokens: int = 256,
    warm_tokens: tuple[int, ...] = (128, 384, 768, 1280),
) -> list[dict[str, Any]]:
    from reasoning_branch_dataset.action_study.vllm_backend import VLLMEngine

    assert isinstance(session.draft, VLLMEngine)
    rng = random.Random(seed)
    collected: list[dict[str, Any]] = list(_load_jsonl(out_path))
    seen = {(r["prefix_hash"], r["cand_hash"]) for r in collected}
    idx = 0
    attempts = 0
    max_attempts = n_candidates * 4 + 50

    while len(collected) < n_candidates and attempts < max_attempts:
        attempts += 1
        prob = problems[idx % len(problems)]
        idx += 1
        prompt = build_prompt(prob["question"])
        target_warm = warm_tokens[attempts % len(warm_tokens)]
        warm_out = session.draft.generate_trace(prompt, max_tokens=max(target_warm + 64, 160))
        text = warm_out["response_text"]
        blocks = [b for b in text.split("\n\n") if b.strip()]
        if len(blocks) >= 2:
            keep = max(1, min(len(blocks) - 1, 1 + (attempts % max(1, len(blocks) - 1))))
            prefix = prompt + "\n\n".join(blocks[:keep]).rstrip() + "\n\n"
        else:
            prefix = prompt + (text[: max(80, len(text) // 2)].rstrip() + "\n\n")

        pool = generate_one_step_vllm(
            session.draft.llm,
            session.draft.llm.get_tokenizer(),
            prefix,
            max_tokens=step_max_tokens,
            temperature=0.7,
            top_p=0.95,
            n=per_prefix,
            seed=1000 + attempts,
        )
        texts = [p.text or " " for p in pool]
        vres = session.verifier.score_batch(
            question=prob["question"], prefix_text=prefix, candidates=texts,
            tau_accept=session.verify_tau,
        )
        details: dict[str, Any] = {}
        labels = maybe_oracle_label(
            question=prob["question"], prefix_text=prefix, candidates=texts,
            skip_oracle=False, cache=oracle_cache, prefix_id=f"calib_a{attempts}",
            details_out=details,
        )
        bj = details.get("branch_judgments") or [None] * len(texts)
        for i, t in enumerate(texts):
            ph, ch = _sha1(prefix), _sha1(t)
            if (ph, ch) in seen:
                continue
            lab = labels[i]
            if lab is None:  # oracle unavailable → skip (can't calibrate on unknown)
                continue
            row = {
                "problem_id": prob["problem_id"],
                "question": prob["question"],
                "prefix_text": prefix,
                "prefix_hash": ph,
                "candidate": t,
                "cand_hash": ch,
                "verifier_score": vres.scores[i].score,
                "logp_accept": vres.scores[i].logp_accept,
                "logp_reject": vres.scores[i].logp_reject,
                "oracle_label": bool(lab),
                "oracle_detail": bj[i] if i < len(bj) else None,
            }
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            collected.append(row)
            seen.add((ph, ch))
        print(f"[calib] collected={len(collected)}/{n_candidates} attempts={attempts}")

    return collected


def sweep_tau(rows: list[dict[str, Any]], *, target_precision: float = 0.99) -> dict[str, Any]:
    scored = [(r["verifier_score"], bool(r["oracle_label"])) for r in rows]
    n = len(scored)
    n_pos = sum(1 for _, y in scored if y)
    n_neg = n - n_pos
    taus = sorted({round(s, 3) for s, _ in scored})
    grid = []
    for tau in taus:
        acc = [(s, y) for s, y in scored if s >= tau]
        tp = sum(1 for _, y in acc if y)
        fp = sum(1 for _, y in acc if not y)
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / n_pos if n_pos else float("nan")
        far = fp / n_neg if n_neg else float("nan")  # false acceptance among true-negatives
        coverage = (tp + fp) / n if n else 0.0
        grid.append(
            {
                "tau": tau,
                "precision": precision,
                "recall": recall,
                "false_acceptance_rate": far,
                "coverage": coverage,
                "n_accept": tp + fp,
                "tp": tp,
                "fp": fp,
            }
        )
    # Recommend smallest tau meeting target precision (max coverage under constraint).
    feasible = [g for g in grid if g["precision"] == g["precision"] and g["precision"] >= target_precision and g["n_accept"] > 0]
    recommended = min(feasible, key=lambda g: g["tau"]) if feasible else None
    return {
        "n": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "base_rate": (n_pos / n) if n else float("nan"),
        "target_precision": target_precision,
        "recommended": recommended,
        "grid": grid,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="V3.6 verifier tau calibration")
    p.add_argument("--draft-model", default="/root/autodl-tmp/specreason/models/DeepSeek-R1-Distill-Qwen-1.5B")
    p.add_argument("--target-model", default="/root/autodl-tmp/specreason/models/DeepSeek-R1-Distill-Qwen-32B-AWQ")
    p.add_argument("--problems", default="/root/autodl-tmp/reasonbranch/data/aime_train.jsonl")
    p.add_argument("--out-dir", default="/root/autodl-tmp/reasonbranch/outputs/action_study_v36_calib")
    p.add_argument("--n-candidates", type=int, default=200)
    p.add_argument("--n-problems", type=int, default=30)
    p.add_argument("--per-prefix", type=int, default=4)
    p.add_argument("--target-precision", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sweep-only", action="store_true", help="skip generation; sweep existing candidates.jsonl")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cand_path = out_dir / "candidates.jsonl"

    if not args.sweep_only:
        session = DualResidentSession(
            draft_model=args.draft_model,
            target_model=args.target_model,
            verify_tau=0.0,
        )
        problems = _load_problems(Path(args.problems), args.n_problems, args.seed)
        oracle_cache: dict[str, Any] = {}
        collect_candidates(
            session, problems,
            n_candidates=args.n_candidates, out_path=cand_path,
            oracle_cache=oracle_cache, seed=args.seed, per_prefix=args.per_prefix,
        )

    rows = _load_jsonl(cand_path)
    if not rows:
        raise SystemExit(f"No labeled candidates in {cand_path}")
    report = sweep_tau(rows, target_precision=args.target_precision)
    (out_dir / "tau_sweep.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))

    rec = report["recommended"]
    print(f"[calib] n={report['n']} pos={report['n_pos']} neg={report['n_neg']} base_rate={report['base_rate']:.3f}")
    if rec:
        print(
            f"[calib] recommended tau={rec['tau']:.3f} "
            f"precision={rec['precision']:.3f} recall={rec['recall']:.3f} "
            f"FAR={rec['false_acceptance_rate']:.3f} coverage={rec['coverage']:.3f}"
        )
    else:
        print(f"[calib] no tau reaches target precision={args.target_precision}; see grid")


if __name__ == "__main__":
    main()

"""V4.0 Phase-0 de-risk collector.

Question: does any near-zero-cost draft-intrinsic signal separate
oracle-acceptable from unacceptable draft-generated steps, where the 32B
Accept/Reject verifier has ~0 discriminative power (V3.6: AUC~0.5)?

For each (prefix, candidate) we record, on the SAME candidate:
  - the 32B verifier score  (baseline, known to fail)
  - draft self-confidence signals (teacher-forced, near-free)  [the hypothesis]
  - the offline oracle acceptability label (ground truth)

Data: GSM8K (easy -> higher acceptable base rate, more positives) + AIME
(hard, matches the mechanism regime). Existing V3.6 calibration candidates are
reused via --reuse-calib (their oracle labels are free; we only backfill signals).

Output: outputs/action_study_v40_derisk/candidates.jsonl (resumable).
Run v40_derisk_analyze.py afterwards for the AUC decision report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from reasoning_branch_dataset.action_study.run_v3_6_pilot import _load_jsonl, maybe_oracle_label
from reasoning_branch_dataset.action_study.v36_counterfactual import DualResidentSession
from reasoning_branch_dataset.action_study.v36_step_gen import generate_one_step_vllm
from reasoning_branch_dataset.action_study.v40_self_signals import DraftSelfSignalExtractor
from reasoning_branch_dataset.model_utils import build_prompt


def _sha1(t: str) -> str:
    return hashlib.sha1((t or "").encode("utf-8")).hexdigest()


class DraftOnlySession:
    """Minimal draft-only session (no 32B) for low-GPU-footprint de-risk.

    Used when the GPU cannot host the dual-resident 32B (e.g. shared with
    another job). The verifier baseline is then taken from the reused V3.6
    calib subset (which already carries verifier_score) plus the V3.6 finding.
    """

    def __init__(self, draft_model: str, *, gpu_util: float = 0.12, max_model_len: int = 4096) -> None:
        from reasoning_branch_dataset.action_study.vllm_backend import VLLMEngine

        self.draft = VLLMEngine(draft_model, gpu_memory_utilization=gpu_util, max_model_len=max_model_len)
        self.verifier = None

    def warmup(self, prefix: str, *, n: int = 5) -> None:
        for _ in range(n):
            generate_one_step_vllm(self.draft.llm, self.draft.llm.get_tokenizer(), prefix, max_tokens=16, n=1)


def _load_dataset(name: str, path: Path, n: int, seed: int) -> list[dict[str, Any]]:
    rows = _load_jsonl(path)
    cleaned: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        q = r.get("problem") or r.get("question") or ""
        if q.strip():
            pid = str(r.get("id", r.get("problem_id", i)))
            cleaned.append({"problem_id": f"{name}_{pid}", "question": q.strip(), "dataset": name})
    rng = random.Random(seed)
    if len(cleaned) > n:
        cleaned = rng.sample(cleaned, n)
    return cleaned


def _block_repetition(blocks: list[str]) -> float:
    """Fraction of near-duplicate blocks (degenerate-trace guard)."""
    if len(blocks) < 2:
        return 0.0
    norm = [b.strip()[:120] for b in blocks]
    return 1.0 - (len(set(norm)) / len(norm))


def build_prefixes(
    session: DualResidentSession,
    question: str,
    *,
    depths: tuple[int, ...],
    trace_max_tokens: int,
) -> list[dict[str, Any]]:
    """Greedy draft trace -> split into steps -> prefixes at given depths.

    Returns [{prefix_text, depth}]. Skips degenerate (repeated-block) traces.
    """
    prompt = build_prompt(question)
    trace = session.draft.generate_trace(prompt, max_tokens=trace_max_tokens)["response_text"]
    blocks = [b for b in trace.split("\n\n") if b.strip()]
    if len(blocks) < 2 or _block_repetition(blocks) > 0.4:
        return []
    out: list[dict[str, Any]] = []
    for d in depths:
        if d < 1 or d >= len(blocks):
            continue
        prefix = prompt + "\n\n".join(blocks[:d]).rstrip() + "\n\n"
        out.append({"prefix_text": prefix, "depth": d})
    return out


def _gen_candidates(session: DualResidentSession, prefix: str, *, step_max_tokens: int, seed: int) -> list[dict[str, Any]]:
    """1 greedy (temp 0) + 3 sampled (temp 0.7) candidates = 4 (oracle slot size)."""
    tok = session.draft.llm.get_tokenizer()
    greedy = generate_one_step_vllm(
        session.draft.llm, tok, prefix, max_tokens=step_max_tokens, temperature=0.0, n=1, seed=seed
    )
    sampled = generate_one_step_vllm(
        session.draft.llm, tok, prefix, max_tokens=step_max_tokens, temperature=0.7, top_p=0.95, n=3, seed=seed + 1
    )
    cands: list[dict[str, Any]] = []
    for r in greedy[:1]:
        cands.append({"text": r.text or " ", "gen_mode": "greedy", "status": r.status})
    for r in sampled[:3]:
        cands.append({"text": r.text or " ", "gen_mode": "sample", "status": r.status})
    return cands


def collect(
    session: DualResidentSession,
    extractor: DraftSelfSignalExtractor,
    problems: list[dict[str, Any]],
    *,
    out_path: Path,
    oracle_cache: dict[str, Any],
    depths: tuple[int, ...],
    target_candidates: int,
    step_max_tokens: int,
    trace_max_tokens: int,
    seed: int,
) -> None:
    collected = _load_jsonl(out_path)
    seen = {(r["prefix_hash"], r["cand_hash"]) for r in collected}
    n_start = len(collected)
    print(f"[derisk] resume: {n_start} existing rows in {out_path.name}")

    for pi, prob in enumerate(problems):
        if len(collected) >= target_candidates:
            break
        try:
            prefixes = build_prefixes(
                session, prob["question"], depths=depths, trace_max_tokens=trace_max_tokens
            )
        except Exception as exc:
            print(f"[derisk] prefix build failed for {prob['problem_id']}: {exc}")
            continue
        if not prefixes:
            print(f"[derisk] {prob['problem_id']}: degenerate/short trace, skip")
            continue

        for pf in prefixes:
            if len(collected) >= target_candidates:
                break
            prefix = pf["prefix_text"]
            ph = _sha1(prefix)
            cand_seed = (seed + pi * 97 + pf["depth"] * 13) % (2**31)
            cands = _gen_candidates(session, prefix, step_max_tokens=step_max_tokens, seed=cand_seed)
            texts = [c["text"] for c in cands]

            vres = None
            if getattr(session, "verifier", None) is not None:
                vres = session.verifier.score_batch(
                    question=prob["question"], prefix_text=prefix, candidates=texts, tau_accept=0.0
                )
            details: dict[str, Any] = {}
            labels = maybe_oracle_label(
                question=prob["question"], prefix_text=prefix, candidates=texts,
                skip_oracle=False, cache=oracle_cache, prefix_id=f"v40d_{prob['problem_id']}_{pf['depth']}",
                details_out=details,
            )
            if all(l is None for l in labels):
                print(f"[derisk] oracle unavailable ({details.get('reason')}); aborting")
                return
            bj = details.get("branch_judgments") or [None] * len(texts)
            sigs = extractor.extract_batch([{"prefix_text": prefix, "candidate": t} for t in texts])

            n_new = 0
            for i, c in enumerate(cands):
                if labels[i] is None:
                    continue
                ch = _sha1(c["text"])
                if (ph, ch) in seen:
                    continue
                row = {
                    "dataset": prob["dataset"],
                    "problem_id": prob["problem_id"],
                    "question": prob["question"],
                    "prefix_text": prefix,
                    "prefix_hash": ph,
                    "depth": pf["depth"],
                    "candidate": c["text"],
                    "cand_hash": ch,
                    "gen_mode": c["gen_mode"],
                    "step_status": c["status"],
                    "verifier_score": (vres.scores[i].score if vres is not None else None),
                    "logp_accept": (vres.scores[i].logp_accept if vres is not None else None),
                    "logp_reject": (vres.scores[i].logp_reject if vres is not None else None),
                    "oracle_label": bool(labels[i]),
                    "oracle_detail": bj[i] if i < len(bj) else None,
                    "self_signals": sigs[i].to_dict(),
                }
                with out_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                collected.append(row)
                seen.add((ph, ch))
                n_new += 1
            n_pos = sum(1 for r in collected if r["oracle_label"])
            print(
                f"[derisk] {prob['problem_id']} d{pf['depth']}: +{n_new} "
                f"total={len(collected)} pos={n_pos} ({n_pos/max(1,len(collected)):.1%})"
            )

    n_pos = sum(1 for r in collected if r["oracle_label"])
    print(f"[derisk] done: total={len(collected)} pos={n_pos} neg={len(collected)-n_pos}")


def reuse_calib(
    extractor: DraftSelfSignalExtractor,
    session: DualResidentSession,
    calib_path: Path,
    out_path: Path,
) -> None:
    """Backfill draft self-signals onto existing V3.6 calib candidates (free labels)."""
    rows = _load_jsonl(calib_path)
    if not rows:
        print(f"[derisk] no calib rows at {calib_path}")
        return
    existing = _load_jsonl(out_path)
    seen = {(r["prefix_hash"], r["cand_hash"]) for r in existing}
    sigs = extractor.extract_batch(
        [{"prefix_text": r["prefix_text"], "candidate": r["candidate"]} for r in rows]
    )
    n_new = 0
    for r, sig in zip(rows, sigs):
        ph, ch = r["prefix_hash"], r["cand_hash"]
        if (ph, ch) in seen:
            continue
        row = {
            "dataset": "aime_calib",
            "problem_id": f"aime_{r.get('problem_id')}",
            "question": r["question"],
            "prefix_text": r["prefix_text"],
            "prefix_hash": ph,
            "depth": -1,
            "candidate": r["candidate"],
            "cand_hash": ch,
            "gen_mode": "calib",
            "step_status": None,
            "verifier_score": r["verifier_score"],
            "logp_accept": r.get("logp_accept"),
            "logp_reject": r.get("logp_reject"),
            "oracle_label": bool(r["oracle_label"]),
            "oracle_detail": r.get("oracle_detail"),
            "self_signals": sig.to_dict(),
        }
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        seen.add((ph, ch))
        n_new += 1
    print(f"[derisk] reuse_calib: backfilled {n_new} rows from {calib_path.name}")


def main() -> None:
    p = argparse.ArgumentParser(description="V4.0 Phase-0 de-risk collector")
    p.add_argument("--draft-model", default="/root/autodl-tmp/specreason/models/DeepSeek-R1-Distill-Qwen-1.5B")
    p.add_argument("--target-model", default="/root/autodl-tmp/specreason/models/DeepSeek-R1-Distill-Qwen-32B-AWQ")
    p.add_argument("--out-dir", default="/root/autodl-tmp/reasonbranch/outputs/action_study_v40_derisk")
    p.add_argument("--gsm8k", default="/root/autodl-tmp/reasonbranch/data/gsm8k_test.jsonl")
    p.add_argument("--aime", default="/root/autodl-tmp/reasonbranch/data/aime_train.jsonl")
    p.add_argument("--calib", default="/root/autodl-tmp/reasonbranch/outputs/action_study_v36_calib/candidates.jsonl")
    p.add_argument("--n-gsm8k", type=int, default=40)
    p.add_argument("--n-aime", type=int, default=20)
    p.add_argument("--target-candidates", type=int, default=240)
    p.add_argument("--depths", default="1,2,3")
    p.add_argument("--step-max-tokens", type=int, default=256)
    p.add_argument("--trace-max-tokens", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reuse-calib", action="store_true", default=True)
    p.add_argument("--no-reuse-calib", dest="reuse_calib", action="store_false")
    p.add_argument("--skip-collect", action="store_true", help="only backfill calib, no new generation")
    p.add_argument("--draft-only", action="store_true",
                   help="load only the 1.5B draft (low GPU footprint); no 32B verifier baseline on new candidates")
    p.add_argument("--draft-gpu-util", type=float, default=0.12)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "candidates.jsonl"
    depths = tuple(int(x) for x in args.depths.split(",") if x.strip())

    if args.draft_only:
        session: Any = DraftOnlySession(args.draft_model, gpu_util=args.draft_gpu_util)
    else:
        session = DualResidentSession(
            draft_model=args.draft_model, target_model=args.target_model, verify_tau=0.0
        )
    session.warmup("Solve: 2+2=?", n=5)
    extractor = DraftSelfSignalExtractor(session.draft.llm, session.draft.llm.get_tokenizer())

    if args.reuse_calib:
        reuse_calib(extractor, session, Path(args.calib), out_path)

    if not args.skip_collect:
        problems: list[dict[str, Any]] = []
        problems += _load_dataset("gsm8k", Path(args.gsm8k), args.n_gsm8k, args.seed)
        problems += _load_dataset("aime", Path(args.aime), args.n_aime, args.seed)
        random.Random(args.seed).shuffle(problems)
        oracle_cache: dict[str, Any] = {}
        collect(
            session, extractor, problems,
            out_path=out_path, oracle_cache=oracle_cache, depths=depths,
            target_candidates=args.target_candidates, step_max_tokens=args.step_max_tokens,
            trace_max_tokens=args.trace_max_tokens, seed=args.seed,
        )

    print(f"[derisk] candidates at {out_path}")


if __name__ == "__main__":
    main()

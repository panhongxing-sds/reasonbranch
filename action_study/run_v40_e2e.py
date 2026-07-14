"""V4.0 end-to-end sequential reasoning with the draft-confidence gate.

Compares four step-level policies to a final \\boxed{} answer:

  target_only     every step by the 32B target        (accuracy ceiling, slow)
  draft_only      every step by the 1.5B draft greedy  (fast, low accuracy)
  selfconf (OURS) draft greedy step; accept if the conformal draft-confidence
                  gate fires (near-free), else handoff to the 32B target
  target_verify   ConfSpec-style: draft greedy step, 32B Accept/Reject verifier
                  decides accept vs handoff (known to mis-rank; costs a 32B pass)

Reports per-policy accuracy, wall-clock, #handoffs (target calls), accept rate,
and verification overhead. The headline: selfconf matches target accuracy at a
fraction of the cost with ~0 verification overhead, where target_verify pays a
32B verification pass yet gains little because its ranking is unreliable
(V3.6 finding).

The gate (fusion + tau) is fit on the Phase-0 de-risk candidates; the target
verifier threshold is calibrated the same way for a fair baseline.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reasoning_branch_dataset.action_study.run_v3_6_pilot import _load_jsonl
from reasoning_branch_dataset.action_study.v36_counterfactual import DualResidentSession, _cuda_sync
from reasoning_branch_dataset.action_study.v36_step_gen import generate_one_step_vllm
from reasoning_branch_dataset.action_study.v40_conformal_gate import (
    ConformalGate,
    FusionModel,
    calibrate_threshold,
    fit_fusion,
)
from reasoning_branch_dataset.action_study.v40_self_signals import DraftSelfSignalExtractor
from reasoning_branch_dataset.grading import grade_math_answer, has_boxed_answer
from reasoning_branch_dataset.model_utils import build_prompt


# ---- problems + gold ------------------------------------------------------

def _gsm8k_gold(answer: str) -> str:
    if "####" in answer:
        return answer.split("####")[-1].strip().replace(",", "")
    return answer.strip()


def load_problems(datasets: dict[str, str], n_each: dict[str, int], seed: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name, path in datasets.items():
        rows = _load_jsonl(Path(path))
        cleaned = []
        for i, r in enumerate(rows):
            q = r.get("problem") or r.get("question") or ""
            if not q.strip():
                continue
            if name == "gsm8k":
                gold = _gsm8k_gold(r.get("answer", ""))
            else:
                gold = str(r.get("answer") or r.get("gold_answer") or r.get("solution") or "").strip()
            if not gold:
                continue
            cleaned.append({
                "problem_id": f"{name}_{r.get('id', r.get('problem_id', i))}",
                "question": q.strip(), "gold": gold, "dataset": name,
            })
        rng = random.Random(seed)
        if len(cleaned) > n_each.get(name, 0):
            cleaned = rng.sample(cleaned, n_each.get(name, 0))
        out += cleaned
    return out


# ---- gate construction from de-risk data ---------------------------------

def _empirical_operating_tau(scores: list[float], labels: list[bool], *, target_precision: float,
                             min_accept: int = 5) -> float | None:
    """Max-coverage tau whose EMPIRICAL precision among accepted >= target.
    Deployable operating point (not the certified one)."""
    pairs = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    best_tau = None
    tp = fp = 0
    for s, y in pairs:
        if y:
            tp += 1
        else:
            fp += 1
        n_acc = tp + fp
        if n_acc >= min_accept and (tp / n_acc) >= target_precision:
            best_tau = s  # keep lowering while precision holds -> max coverage
    return best_tau


@dataclass
class GateBundle:
    fusion: FusionModel
    tau_ours: float
    tau_verifier: float
    tau_ours_certified: float | None
    epsilon: float
    delta: float
    info: dict[str, Any] = field(default_factory=dict)


def build_gate(derisk_rows: list[dict[str, Any]], *, mode: str, target_precision: float,
               epsilon: float, delta: float) -> GateBundle:
    fusion = fit_fusion(derisk_rows, mode=mode)
    scores = [fusion.score(r) for r in derisk_rows]
    labels = [bool(r["oracle_label"]) for r in derisk_rows]

    tau_emp = _empirical_operating_tau(scores, labels, target_precision=target_precision)
    cal = calibrate_threshold(scores, labels, epsilon=epsilon, delta=delta)
    tau_cert = cal.tau
    # Deployable tau: prefer certified; else empirical operating point; else median.
    tau_ours = tau_cert if tau_cert is not None else tau_emp
    if tau_ours is None:
        tau_ours = sorted(scores)[len(scores) // 2] if scores else 0.0

    # Target verifier baseline threshold, calibrated the same way (empirical).
    vrows = [r for r in derisk_rows if r.get("verifier_score") is not None]
    if vrows:
        vs = [float(r["verifier_score"]) for r in vrows]
        vy = [bool(r["oracle_label"]) for r in vrows]
        tau_v = _empirical_operating_tau(vs, vy, target_precision=target_precision)
        if tau_v is None:
            tau_v = 0.0  # V3.6 default
    else:
        tau_v = 0.0

    return GateBundle(
        fusion=fusion, tau_ours=float(tau_ours), tau_verifier=float(tau_v),
        tau_ours_certified=tau_cert, epsilon=epsilon, delta=delta,
        info={
            "mode": mode, "target_precision": target_precision,
            "tau_empirical": tau_emp, "cal_coverage": cal.coverage,
            "n_derisk": len(derisk_rows), "n_verifier_rows": len(vrows),
        },
    )


# ---- rollout --------------------------------------------------------------

@dataclass
class StepTiming:
    draft_sec: float = 0.0
    verify_sec: float = 0.0  # self-signal extraction OR 32B verifier
    target_sec: float = 0.0


def _draft_step(session: DualResidentSession, prefix: str, *, max_tokens: int):
    _cuda_sync()
    t0 = time.perf_counter()
    r = generate_one_step_vllm(
        session.draft.llm, session.draft.llm.get_tokenizer(), prefix,
        max_tokens=max_tokens, temperature=0.0, n=1
    )[0]
    _cuda_sync()
    return r, time.perf_counter() - t0


def _target_step(session: DualResidentSession, prefix: str, *, max_tokens: int):
    _cuda_sync()
    t0 = time.perf_counter()
    r = generate_one_step_vllm(
        session.target.llm, session.target_tok, prefix,
        max_tokens=max_tokens, temperature=0.0, n=1
    )[0]
    _cuda_sync()
    return r, time.perf_counter() - t0


def run_rollout(
    session: DualResidentSession,
    extractor: DraftSelfSignalExtractor,
    problem: dict[str, Any],
    *,
    policy: str,
    gate: GateBundle,
    max_steps: int,
    step_max_tokens: int,
    tau_override: float | None = None,
) -> dict[str, Any]:
    prompt = build_prompt(problem["question"])
    prefix = prompt + "\n\n"
    body = ""  # accumulated raw reasoning (faithful continuation)
    handoffs = accepts = 0
    tot = StepTiming()
    empties = 0
    finish = "MAX_STEP"

    for _ in range(max_steps):
        if policy == "target_only":
            res, dt = _target_step(session, prefix, max_tokens=step_max_tokens)
            tot.target_sec += dt
            handoffs += 1
        elif policy == "draft_only":
            res, dt = _draft_step(session, prefix, max_tokens=step_max_tokens)
            tot.draft_sec += dt
            accepts += 1
        elif policy == "selfconf":
            res, dt = _draft_step(session, prefix, max_tokens=step_max_tokens)
            tot.draft_sec += dt
            t0 = time.perf_counter()
            sig = extractor.extract_batch([{"prefix_text": prefix, "candidate": res.text}])[0]
            tot.verify_sec += time.perf_counter() - t0
            tau_use = gate.tau_ours if tau_override is None else tau_override
            if res.text.strip() and gate.fusion.score({"self_signals": sig.to_dict()}) >= tau_use:
                accepts += 1
            else:
                res, dt2 = _target_step(session, prefix, max_tokens=step_max_tokens)
                tot.target_sec += dt2
                handoffs += 1
        elif policy == "target_verify":
            res, dt = _draft_step(session, prefix, max_tokens=step_max_tokens)
            tot.draft_sec += dt
            t0 = time.perf_counter()
            vres = session.verifier.score_batch(
                question=problem["question"], prefix_text=prefix, candidates=[res.text or " "]
            )
            tot.verify_sec += time.perf_counter() - t0
            if res.text.strip() and vres.scores[0].score >= gate.tau_verifier:
                accepts += 1
            else:
                res, dt2 = _target_step(session, prefix, max_tokens=step_max_tokens)
                tot.target_sec += dt2
                handoffs += 1
        else:
            raise ValueError(f"unknown policy {policy}")

        # Accumulate the RAW step text so the reasoning chain flows to </think>+boxed.
        chunk = (res.raw_text or "").rstrip()
        if not chunk:
            empties += 1
            if empties >= 6:
                finish = "DEGENERATE"
                break
            continue
        body += chunk + "\n\n"
        prefix = prompt + "\n\n" + body
        if has_boxed_answer(chunk) or has_boxed_answer(body):
            finish = "FINAL_ANSWER"
            break

    grade = grade_math_answer(body, problem["gold"])
    wall = tot.draft_sec + tot.verify_sec + tot.target_sec
    return {
        "problem_id": problem["problem_id"],
        "dataset": problem["dataset"],
        "policy": policy,
        "is_correct": grade.get("is_correct"),
        "evaluation_status": grade.get("evaluation_status"),
        "final_answer": grade.get("final_answer"),
        "n_steps": accepts + handoffs,
        "accepts": accepts,
        "handoffs": handoffs,
        "finish": finish,
        "wall_sec": wall,
        "draft_sec": tot.draft_sec,
        "verify_sec": tot.verify_sec,
        "target_sec": tot.target_sec,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="V4.0 end-to-end sequential reasoning")
    p.add_argument("--draft-model", default="/root/autodl-tmp/specreason/models/DeepSeek-R1-Distill-Qwen-1.5B")
    p.add_argument("--target-model", default="/root/autodl-tmp/specreason/models/DeepSeek-R1-Distill-Qwen-32B-AWQ")
    p.add_argument("--derisk", default="/root/autodl-tmp/reasonbranch/outputs/action_study_v40_derisk/candidates.jsonl")
    p.add_argument("--out-dir", default="/root/autodl-tmp/reasonbranch/outputs/action_study_v40_e2e")
    p.add_argument("--gsm8k", default="/root/autodl-tmp/reasonbranch/data/gsm8k_test.jsonl")
    p.add_argument("--aime", default="/root/autodl-tmp/reasonbranch/data/aime_train.jsonl")
    p.add_argument("--n-gsm8k", type=int, default=30)
    p.add_argument("--n-aime", type=int, default=10)
    p.add_argument("--policies", default="target_only,draft_only,selfconf,target_verify")
    p.add_argument("--tau-sweep", default="", help="comma-separated fusion-score thresholds; runs selfconf once per tau (Pareto sweep)")
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--step-max-tokens", type=int, default=256)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gate-mode", default="logistic", choices=["logistic", "single"])
    p.add_argument("--target-precision", type=float, default=0.85)
    p.add_argument("--epsilon", type=float, default=0.15)
    p.add_argument("--delta", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    derisk_rows = _load_jsonl(Path(args.derisk))
    if not derisk_rows:
        raise SystemExit(f"No de-risk candidates at {args.derisk}; run run_v40_derisk first.")
    gate = build_gate(derisk_rows, mode=args.gate_mode, target_precision=args.target_precision,
                      epsilon=args.epsilon, delta=args.delta)
    print(f"[e2e] gate tau_ours={gate.tau_ours:.4f} (certified={gate.tau_ours_certified}) "
          f"tau_verifier={gate.tau_verifier:.4f} info={gate.info}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gate.json").write_text(json.dumps(gate.to_dict() if hasattr(gate, "to_dict") else {
        "tau_ours": gate.tau_ours, "tau_verifier": gate.tau_verifier,
        "tau_ours_certified": gate.tau_ours_certified, "info": gate.info,
        "fusion": gate.fusion.to_dict(),
    }, ensure_ascii=False, indent=2))

    session = DualResidentSession(draft_model=args.draft_model, target_model=args.target_model,
                                  max_model_len=args.max_model_len, verify_tau=0.0)
    session.warmup("Solve: 2+2=?", n=5)
    extractor = DraftSelfSignalExtractor(session.draft.llm, session.draft.llm.get_tokenizer())

    problems = load_problems({"gsm8k": args.gsm8k, "aime": args.aime},
                             {"gsm8k": args.n_gsm8k, "aime": args.n_aime}, args.seed)
    policies = [x.strip() for x in args.policies.split(",") if x.strip()]
    tau_sweep = [float(x) for x in args.tau_sweep.split(",") if x.strip()]
    # Build the (label, base_policy, tau_override) work list.
    jobs: list[tuple[str, str, float | None]] = []
    for policy in policies:
        if policy == "selfconf" and tau_sweep:
            for tau in tau_sweep:
                jobs.append((f"selfconf@{tau:g}", "selfconf", tau))
        else:
            jobs.append((policy, policy, None))

    res_path = out_dir / "e2e_results.jsonl"
    done = {(r["problem_id"], r["policy"]) for r in _load_jsonl(res_path)}

    for prob in problems:
        for label, base_policy, tau_override in jobs:
            if (prob["problem_id"], label) in done:
                continue
            row = run_rollout(session, extractor, prob, policy=base_policy, gate=gate,
                              max_steps=args.max_steps, step_max_tokens=args.step_max_tokens,
                              tau_override=tau_override)
            row["policy"] = label
            row["tau"] = tau_override if tau_override is not None else (
                gate.tau_verifier if base_policy == "target_verify" else gate.tau_ours)
            with res_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[e2e] {prob['problem_id']} {label}: correct={row['is_correct']} "
                  f"steps={row['n_steps']} handoff={row['handoffs']} wall={row['wall_sec']:.2f}s "
                  f"finish={row['finish']}")

    print(f"[e2e] results -> {res_path}")


if __name__ == "__main__":
    main()

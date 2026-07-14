"""V3.6 verifier scorer sanity checks (user review §4).

Confirms four things about the paired prompt_logprob scorer before it is allowed
to gate ACCEPT/REJECT:

  1. Labels ` Accept` / ` Reject` are single tokens for THIS tokenizer.
  2. The two paired requests share identical context (only the final label token
     differs), verified by token-id comparison.
  3. The logprob is read at the *label* position (last prompt token), not BOS/EOS
     /space/newline — verified by checking prompt_logprobs length == prompt len
     and that the scored token id equals the label id.
  4. A hand-built sanity set of obviously-correct vs obviously-wrong candidates
     is clearly separated by verifier score.

Prints a PASS/FAIL per gate and writes a JSON report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reasoning_branch_dataset.action_study.logit_step_verifier import LogitStepVerifier
from reasoning_branch_dataset.action_study.target_verifier import build_target_verifier


# Obvious cases: same trivial arithmetic prefix; correct vs wrong next steps.
SANITY_QUESTION = "Compute the value step by step."
SANITY_PREFIX = "We are asked to evaluate simple arithmetic facts.\n\n"

CORRECT_STEPS = [
    "Since 2 + 2 = 4, the sum is 4.",
    "We have 3 * 5 = 15, so the product is 15.",
    "Because 10 - 7 = 3, the difference is 3.",
    "Note that 12 / 4 = 3, hence the quotient is 3.",
    "As 6 + 6 = 12, the total equals 12.",
    "Given 9 * 9 = 81, the square of 9 is 81.",
    "Since 100 - 1 = 99, the result is 99.",
    "We compute 7 + 8 = 15, so the answer is 15.",
    "Because 4^2 = 16, the value is 16.",
    "Observe that 5 * 4 = 20, giving 20.",
    "Since 8 / 2 = 4, the quotient is 4.",
    "We have 11 + 1 = 12, so the sum is 12.",
    "As 2 * 2 * 2 = 8, the cube of 2 is 8.",
    "Because 15 - 5 = 10, the difference is 10.",
    "Given 6 * 7 = 42, the product is 42.",
    "Since 3 + 4 = 7, the total is 7.",
    "We find 20 / 5 = 4, so the answer is 4.",
    "Note 9 + 10 = 19, hence 19.",
    "Because 13 - 3 = 10, the result is 10.",
    "As 5^2 = 25, the square is 25.",
]

WRONG_STEPS = [
    "Since 2 + 2 = 5, the sum is 5.",
    "We have 3 * 5 = 20, so the product is 20.",
    "Because 10 - 7 = 5, the difference is 5.",
    "Note that 12 / 4 = 6, hence the quotient is 6.",
    "As 6 + 6 = 13, the total equals 13.",
    "Given 9 * 9 = 72, the square of 9 is 72.",
    "Since 100 - 1 = 98, the result is 98.",
    "We compute 7 + 8 = 14, so the answer is 14.",
    "Because 4^2 = 12, the value is 12.",
    "Observe that 5 * 4 = 25, giving 25.",
    "Since 8 / 2 = 5, the quotient is 5.",
    "We have 11 + 1 = 13, so the sum is 13.",
    "As 2 * 2 * 2 = 6, the cube of 2 is 6.",
    "Because 15 - 5 = 11, the difference is 11.",
    "Given 6 * 7 = 48, the product is 48.",
    "Since 3 + 4 = 8, the total is 8.",
    "We find 20 / 5 = 5, so the answer is 5.",
    "Note 9 + 10 = 20, hence 20.",
    "Because 13 - 3 = 11, the result is 11.",
    "As 5^2 = 30, the square is 30.",
]


def gate1_single_token(tokenizer) -> dict[str, Any]:
    a = tokenizer.encode(" Accept", add_special_tokens=False)
    r = tokenizer.encode(" Reject", add_special_tokens=False)
    ok = len(a) == 1 and len(r) == 1
    return {"gate": "single_token_labels", "pass": ok, "accept_ids": a, "reject_ids": r}


def gate23_context_and_position(verifier: LogitStepVerifier, tokenizer) -> dict[str, Any]:
    """Rebuild the two paired prompts and confirm they differ only in last token,
    and that scoring reads the label position."""
    q, prefix, cand = SANITY_QUESTION, SANITY_PREFIX, CORRECT_STEPS[0]
    full = verifier.build_prompt(question=q, prefix_text=prefix, candidate=cand)
    base = tokenizer.encode(full, add_special_tokens=False)
    aid = verifier.labels.accept_token_id
    rid = verifier.labels.reject_token_id
    seq_a = base + [aid]
    seq_r = base + [rid]

    context_ok = seq_a[:-1] == seq_r[:-1]
    last_diff_ok = seq_a[-1] == aid and seq_r[-1] == rid and aid != rid

    # Position check: run the scorer and confirm the read token is the label.
    from vllm import SamplingParams

    params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1, detokenize=False)
    outs = verifier.llm.generate(
        [{"prompt_token_ids": seq_a}, {"prompt_token_ids": seq_r}], params
    )
    pa = outs[0].prompt_logprobs
    pr = outs[1].prompt_logprobs
    pos_ok = (
        pa is not None and pr is not None
        and len(pa) == len(seq_a) and len(pr) == len(seq_r)
        and (aid in (pa[-1] or {})) and (rid in (pr[-1] or {}))
    )
    return {
        "gate": "context_identical_and_position",
        "pass": bool(context_ok and last_diff_ok and pos_ok),
        "context_identical": context_ok,
        "last_token_is_label": last_diff_ok,
        "logprob_at_label_position": pos_ok,
        "prompt_len": len(seq_a),
        "prompt_logprobs_len": (len(pa) if pa else None),
    }


def gate4_separation(verifier: LogitStepVerifier) -> dict[str, Any]:
    def score_all(steps: list[str]) -> list[float]:
        res = verifier.score_batch(
            question=SANITY_QUESTION, prefix_text=SANITY_PREFIX, candidates=steps
        )
        return [s.score for s in res.scores]

    correct = score_all(CORRECT_STEPS)
    wrong = score_all(WRONG_STEPS)
    c_mean = sum(correct) / len(correct)
    w_mean = sum(wrong) / len(wrong)
    c_min = min(correct)
    w_max = max(wrong)
    # Separation: correct clearly above wrong. Report margin and overlap.
    overlap = sum(1 for c in correct if c <= w_max) + sum(1 for w in wrong if w >= c_min)
    # AUC-style: fraction of (correct,wrong) pairs correctly ordered.
    pairs = [(c, w) for c in correct for w in wrong]
    correct_order = sum(1 for c, w in pairs if c > w)
    auc = correct_order / len(pairs)
    return {
        "gate": "obvious_separation",
        "pass": bool(auc >= 0.95 and c_mean > w_mean),
        "auc": auc,
        "correct_mean": c_mean,
        "wrong_mean": w_mean,
        "correct_min": c_min,
        "wrong_max": w_max,
        "margin_mean": c_mean - w_mean,
        "overlap_count": overlap,
        "correct_scores": [round(x, 3) for x in correct],
        "wrong_scores": [round(x, 3) for x in wrong],
    }


def main() -> None:
    p = argparse.ArgumentParser(description="V3.6 verifier scorer sanity checks")
    p.add_argument("--target-model", default="/root/autodl-tmp/specreason/models/DeepSeek-R1-Distill-Qwen-32B-AWQ")
    p.add_argument("--target-quantization", default="awq")
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--out", default="/root/autodl-tmp/reasonbranch/outputs/action_study_v36/verifier_sanity.json")
    args = p.parse_args()

    quant = args.target_quantization if "awq" in args.target_model.lower() else None
    target = build_target_verifier(
        args.target_model,
        engine="vllm",
        gpu_memory_utilization=0.70,
        max_model_len=args.max_model_len,
        quantization=quant,
        dual_resident=True,
        enable_prefix_caching=True,
    )
    tokenizer = target.tokenizer
    verifier = LogitStepVerifier(target.llm, tokenizer)

    g1 = gate1_single_token(tokenizer)
    print(f"[gate1] single_token_labels: pass={g1['pass']} A={g1['accept_ids']} R={g1['reject_ids']}")
    g23 = gate23_context_and_position(verifier, tokenizer)
    print(
        f"[gate2/3] context_identical={g23['context_identical']} "
        f"last_token_is_label={g23['last_token_is_label']} "
        f"logprob_at_label_position={g23['logprob_at_label_position']} pass={g23['pass']}"
    )
    g4 = gate4_separation(verifier)
    print(
        f"[gate4] separation: pass={g4['pass']} auc={g4['auc']:.3f} "
        f"correct_mean={g4['correct_mean']:.3f} wrong_mean={g4['wrong_mean']:.3f} "
        f"correct_min={g4['correct_min']:.3f} wrong_max={g4['wrong_max']:.3f}"
    )

    report = {"gate1": g1, "gate2_3": g23, "gate4": g4, "all_pass": all(g["pass"] for g in (g1, g23, g4))}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[v3.6 verifier-sanity] all_pass={report['all_pass']} -> {out}")


if __name__ == "__main__":
    main()

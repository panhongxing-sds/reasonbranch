"""Local 14B step verifier — zero-shot ACCEPT/REJECT (no API)."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from reasoning_branch_dataset.action_study.gpt_step_oracle import (
    BRANCH_KEYS,
    GREEDY_KEY,
    oracle_action_from_acceptability,
)
from reasoning_branch_dataset.action_study.run_utility_scoring import _load_jsonl
from reasoning_branch_dataset.action_study.target_verifier import build_target_verifier

VERIFIER_PROMPT = """You are a strict math reasoning step verifier.

Problem:
{question}

Current reasoning prefix:
{prefix_tail}

Candidate next step:
{candidate}

Is the candidate mathematically correct, consistent with the problem and prefix, substantive, and safe to append?

Answer with exactly one word: ACCEPT or REJECT."""


def _clip(text: str, n: int = 1200) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[: n - 3] + "..."


def parse_accept_reject(text: str) -> bool | None:
    t = (text or "").strip().upper()
    if re.search(r"\bACCEPT\b", t) and not re.search(r"\bREJECT\b", t):
        return True
    if re.search(r"\bREJECT\b", t):
        return False
    return None


class LocalStepVerifier:
    def __init__(self, model_path: str, *, max_model_len: int = 4096) -> None:
        self.verifier = build_target_verifier(
            model_path,
            engine="vllm",
            gpu_memory_utilization=0.90,
            max_model_len=max_model_len,
        )

    def judge_candidate(self, *, question: str, prefix_text: str, candidate: str) -> dict[str, Any]:
        from vllm import SamplingParams

        prompt = VERIFIER_PROMPT.format(
            question=_clip(question, 800),
            prefix_tail=_clip(prefix_text, 1000),
            candidate=_clip(candidate, 600),
        )
        ids = self.verifier.tokenize(prompt)
        params = SamplingParams(max_tokens=8, temperature=0.0, top_p=1.0)
        out = self.verifier.llm.generate([{"prompt_token_ids": ids}], params)[0]
        raw = self.verifier.tokenizer.decode(out.outputs[0].token_ids, skip_special_tokens=True)
        pred = parse_accept_reject(raw)
        return {"raw": raw.strip(), "acceptable": pred, "parsed": pred is not None}


def eval_verifier(
    dataset_path: Path,
    *,
    model_path: str,
    max_samples: int | None = 200,
    seed: int = 42,
) -> dict[str, Any]:
    import random

    rows = _load_jsonl(dataset_path)
    rng = random.Random(seed)
    if max_samples and len(rows) > max_samples:
        rows = rng.sample(rows, max_samples)

    lv = LocalStepVerifier(model_path)
    preds: list[dict[str, Any]] = []
    for r in rows:
        out = lv.judge_candidate(
            question=r["question"],
            prefix_text=r["prefix_text"],
            candidate=r["candidate_step"],
        )
        gold = bool(r["acceptable"])
        pred = out["acceptable"]
        preds.append(
            {
                "prefix_id": r["prefix_id"],
                "candidate_key": r["candidate_key"],
                "gold": gold,
                "pred": pred,
                "correct": pred == gold if pred is not None else None,
                "raw": out["raw"],
            }
        )

    parsed = [p for p in preds if p["pred"] is not None]
    correct = sum(1 for p in parsed if p["correct"])
    # action-level on prefixes (greedy + 4 branches)
    by_prefix: dict[str, list[dict]] = {}
    for p in preds:
        by_prefix.setdefault(p["prefix_id"], []).append(p)

    return {
        "n_candidates": len(preds),
        "parse_rate": len(parsed) / len(preds) if preds else 0,
        "candidate_accuracy": correct / len(parsed) if parsed else 0,
        "predictions": preds[:50],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="/mnt/afs/L202500372/reasoning_branch_dataset/outputs/verifier_dataset/candidate_labels.jsonl",
    )
    parser.add_argument(
        "--model",
        default="/mnt/afs/L202500372/specreason/models/DeepSeek-R1-Distill-Qwen-14B",
    )
    parser.add_argument("--out", default="/mnt/afs/L202500372/reasoning_branch_dataset/outputs/verifier_eval/summary.json")
    parser.add_argument("--max-samples", type=int, default=100)
    args = parser.parse_args()
    summary = eval_verifier(Path(args.dataset), model_path=args.model, max_samples=args.max_samples)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "predictions"}, indent=2))


if __name__ == "__main__":
    main()

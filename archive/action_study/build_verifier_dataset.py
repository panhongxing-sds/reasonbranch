"""Build candidate-level ACCEPT/REJECT dataset from V3.3 stable GPT labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from reasoning_branch_dataset.action_study.gpt_step_oracle import BRANCH_KEYS, CANDIDATE_KEYS, GREEDY_KEY
from reasoning_branch_dataset.action_study.run_utility_scoring import _load_jsonl


def build_verifier_rows(v3_dir: Path, v2_dir: Path) -> list[dict[str, Any]]:
    labels = _load_jsonl(v3_dir / "gpt_step_labels.jsonl")
    prefixes = {r["prefix_id"]: r for r in _load_jsonl(v2_dir / "prefixes.jsonl")}
    problems = {r["problem_id"]: r for r in _load_jsonl(v2_dir / "problems.jsonl")}

    rows: list[dict[str, Any]] = []
    for lab in labels:
        if not lab.get("oracle_stable"):
            continue
        p1 = lab.get("pass1") or {}
        judgments = p1.get("candidate_judgments") or {}
        label_map = p1.get("label_to_key") or {}
        # invert anon -> canonical key
        key_by_anon = {v: k for k, v in label_map.items()} if label_map else {}
        pfx = prefixes.get(lab["prefix_id"], {})
        prob = problems.get(lab["problem_id"], {})
        question = prob.get("question", "")
        prefix_text = pfx.get("prefix_text", lab.get("prefix_text", ""))

        steps = {
            GREEDY_KEY: lab.get("greedy_step", ""),
            **{k: (lab.get("branch_steps") or [""] * 4)[i] for i, k in enumerate(BRANCH_KEYS)},
        }
        for ck in CANDIDATE_KEYS:
            cand = steps.get(ck, "")
            # find judgment via anon labels in judgments dict keys
            j = None
            for anon, canon in key_by_anon.items():
                if canon == ck:
                    j = judgments.get(anon) or judgments.get(anon.upper())
                    break
            if j is None:
                j = judgments.get(ck) or {}
            acceptable = bool(j.get("acceptable")) if j else False
            rows.append(
                {
                    "problem_id": lab["problem_id"],
                    "prefix_id": lab["prefix_id"],
                    "candidate_key": ck,
                    "question": question,
                    "prefix_text": prefix_text,
                    "candidate_step": cand,
                    "acceptable": acceptable,
                    "label": "ACCEPT" if acceptable else "REJECT",
                    "oracle_action": lab.get("oracle_action"),
                    "quality": j.get("quality") if j else None,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3-dir", default="/mnt/afs/L202500372/reasoning_branch_dataset/outputs/action_study_pilot_v3")
    parser.add_argument("--v2-dir", default="/mnt/afs/L202500372/reasoning_branch_dataset/outputs/action_study_pilot_v2")
    parser.add_argument("--out-dir", default="/mnt/afs/L202500372/reasoning_branch_dataset/outputs/verifier_dataset")
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = build_verifier_rows(Path(args.v3_dir), Path(args.v2_dir))
    with (out / "candidate_labels.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "n_candidates": len(rows),
        "accept_rate": sum(1 for r in rows if r["acceptable"]) / len(rows) if rows else 0,
        "by_key": dict(Counter(r["candidate_key"] for r in rows)),
        "accept_by_key": {
            k: sum(1 for r in rows if r["candidate_key"] == k and r["acceptable"])
            for k in CANDIDATE_KEYS
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

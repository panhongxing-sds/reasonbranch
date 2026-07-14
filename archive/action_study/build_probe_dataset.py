"""Build two-stage probe datasets from V3.3 GPT labels + V2 prefix features."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from reasoning_branch_dataset.action_study.run_utility_scoring import _load_jsonl


LOGIT_FEATURES = (
    "entropy",
    "top1_prob",
    "top2_prob",
    "margin",
    "diversity_entropy",
    "reasoning_progress",
)


def _marker_features(prefix_text: str) -> dict[str, float]:
    tail = prefix_text[-800:].lower()
    return {
        "has_wait": float("wait" in tail),
        "has_but": float(" but " in tail or tail.startswith("but ")),
        "has_alternatively": float("alternatively" in tail),
        "prefix_chars": float(len(prefix_text)),
        "prefix_blocks": float(prefix_text.count("\n\n")),
    }


def build_probe_rows(v3_dir: Path, v2_dir: Path) -> tuple[list[dict], list[dict]]:
    labels = _load_jsonl(v3_dir / "gpt_step_labels.jsonl")
    prefixes = {r["prefix_id"]: r for r in _load_jsonl(v2_dir / "prefixes.jsonl")}

    stage1: list[dict[str, Any]] = []
    stage2: list[dict[str, Any]] = []
    for row in labels:
        if not row.get("oracle_eligible_for_probe"):
            continue
        action = row.get("oracle_action")
        if action not in ("CONTINUE", "BRANCH", "HANDOFF"):
            continue
        pfx = prefixes.get(row["prefix_id"], {})
        feats = {k: float(pfx.get(k, 0.0) or 0.0) for k in LOGIT_FEATURES}
        feats.update(_marker_features(pfx.get("prefix_text", row.get("prefix_text", ""))))
        base = {
            "problem_id": row["problem_id"],
            "prefix_id": row["prefix_id"],
            "oracle_action": action,
            **feats,
        }
        stage1.append({**base, "y_intervention": int(action != "CONTINUE")})
        if action in ("BRANCH", "HANDOFF"):
            stage2.append({**base, "y_branch": int(action == "BRANCH")})

    return stage1, stage2


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3-dir", default="/mnt/afs/L202500372/reasoning_branch_dataset/outputs/action_study_pilot_v3")
    parser.add_argument("--v2-dir", default="/mnt/afs/L202500372/reasoning_branch_dataset/outputs/action_study_pilot_v2")
    parser.add_argument("--out-dir", default="/mnt/afs/L202500372/reasoning_branch_dataset/outputs/probe_datasets")
    args = parser.parse_args()
    out = Path(args.out_dir)
    s1, s2 = build_probe_rows(Path(args.v3_dir), Path(args.v2_dir))
    write_jsonl(out / "stage1_continue_vs_intervention.jsonl", s1)
    write_jsonl(out / "stage2_branch_vs_handoff.jsonl", s2)
    summary = {
        "stage1_n": len(s1),
        "stage1_continue": sum(1 for r in s1 if r["y_intervention"] == 0),
        "stage1_intervention": sum(1 for r in s1 if r["y_intervention"] == 1),
        "stage2_n": len(s2),
        "stage2_branch": sum(1 for r in s2 if r["y_branch"] == 1),
        "stage2_handoff": sum(1 for r in s2 if r["y_branch"] == 0),
        "stage1_actions": dict(Counter(r["oracle_action"] for r in s1)),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

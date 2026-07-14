#!/usr/bin/env python3
"""Quick health check for action_study pilot output."""

from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "reasoning_branch_dataset/outputs/action_study_pilot_v2")


def main() -> int:
    issues: list[str] = []
    for name in ("problems", "traces", "prefixes", "actions", "outcome_results"):
        p = OUT / f"{name}.jsonl"
        if not p.exists() or p.stat().st_size == 0:
            issues.append(f"missing or empty: {name}.jsonl")
    ckpt = OUT / "checkpoints" / "done_problems.json"
    done = 0
    if ckpt.exists():
        done = len(json.loads(ckpt.read_text()))
    if done == 0 and not issues:
        print(f"OK: starting up (checkpoint={done})")
        return 0
    if issues and done == 0:
        print("FAIL:", "; ".join(issues))
        return 1
    traces = [json.loads(l) for l in (OUT / "traces.jsonl").read_text().splitlines() if l.strip()]
    wrong = sum(1 for t in traces if t.get("is_correct") == 0)
    print(f"OK: {done} problems done, {wrong}/{len(traces)} greedy wrong, prefixes={(OUT/'prefixes.jsonl').stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

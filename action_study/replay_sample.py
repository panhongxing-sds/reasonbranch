"""Stratified sampling for target acceptance replay."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _progress_bucket(progress: float | None) -> str:
    if progress is None:
        return "unknown"
    if progress < 0.33:
        return "early"
    if progress < 0.67:
        return "middle"
    return "late"


def _diversity_bucket(label: str | None) -> str:
    if label in {"MULTIPLE_GENUINE_STRATEGIES", "HIGH_DIVERSITY"}:
        return "high"
    return "low"


def _validity_bucket(status: str | None) -> str:
    if status in {"VALID", "INVALID", "UNCLEAR"}:
        return status
    return "UNCLEAR"


def stratified_sample_prefixes(
    prefixes: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    *,
    n: int = 300,
    seed: int = 42,
    admission_col: str = "admission_main",
    max_per_problem: int = 3,
    reachable_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Sample prefixes stratified by validity × diversity × progress × continue correctness."""
    continue_ok = {
        a["prefix_id"]: a.get("is_correct")
        for a in actions
        if a.get("action_type") == "continue" and a.get("evaluation_status") == "OK"
    }

    pool = [p for p in prefixes if p.get(admission_col) or p.get("include_in_main_experiment")]
    if not pool:
        pool = [p for p in prefixes if p.get("prefix_substantiveness") == "SUBSTANTIVE"]
    if reachable_ids is not None:
        pool = [p for p in pool if p["prefix_id"] in reachable_ids]

    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for p in pool:
        key = (
            _validity_bucket(p.get("prefix_validity")),
            _diversity_bucket(p.get("strategy_diversity") or p.get("diversity_label")),
            _progress_bucket(p.get("reasoning_progress")),
            "correct" if continue_ok.get(p["prefix_id"]) == 1 else "wrong",
        )
        buckets[key].append(p)

    rng = random.Random(seed)
    for items in buckets.values():
        rng.shuffle(items)

    # round-robin across buckets for balance
    bucket_keys = list(buckets.keys())
    rng.shuffle(bucket_keys)
    selected: list[dict] = []
    per_problem: dict[str, int] = defaultdict(int)
    idx = 0
    while len(selected) < n and bucket_keys:
        key = bucket_keys[idx % len(bucket_keys)]
        items = buckets[key]
        if not items:
            bucket_keys.remove(key)
            if not bucket_keys:
                break
            continue
        cand = items.pop(0)
        pid = cand["problem_id"]
        if per_problem[pid] >= max_per_problem:
            idx += 1
            continue
        selected.append(cand)
        per_problem[pid] += 1
        idx += 1

    return selected[:n]


def write_sample_manifest(data_dir: Path, samples: list[dict[str, Any]]) -> Path:
    path = data_dir / "target_replay_sample.jsonl"
    path.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in samples), encoding="utf-8")
    meta = {
        "n_samples": len(samples),
        "n_problems": len({s["problem_id"] for s in samples}),
        "prefix_ids": [s["prefix_id"] for s in samples],
    }
    (data_dir / "target_replay_sample_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return path

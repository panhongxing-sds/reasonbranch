#!/usr/bin/env python3
"""Find research-valuable action-study prefixes."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from reasoning_branch_dataset.action_study.api_validity import ValidityClient
from reasoning_branch_dataset.action_study.diversity import compute_diversity, state_bucket

OUT = Path("reasoning_branch_dataset/outputs/action_study_v1")


def load(name: str, *, rescored: bool = False) -> list[dict]:
    suffix = ".rescored" if rescored and name in {"traces", "actions", "action_results"} else ""
    path = OUT / f"{name}{suffix}.jsonl"
    if rescored and not path.exists():
        path = OUT / f"{name}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def reasoning_only(prefix_text: str) -> str:
    if "</think>" in prefix_text:
        return prefix_text.split("</think>", 1)[1].strip()
    if "Problem:" in prefix_text:
        tail = prefix_text.split("Problem:", 1)[1]
        return tail[tail.find("\n") + 1 :] if "\n" in tail else tail
    return prefix_text


def main() -> None:
    traces = {t["problem_id"]: t for t in load("traces", rescored=True)}
    prefixes = load("prefixes")
    results = load("action_results", rescored=True)
    next_by: dict[str, list] = defaultdict(list)
    for n in load("next_step_samples"):
        next_by[n["prefix_id"]].append(n)

    by_pfx: dict[str, dict] = defaultdict(dict)
    for r in results:
        by_pfx[r["prefix_id"]][r["action"]] = r

    client = ValidityClient.from_env(cache_path=OUT / "api_cache_fixed.jsonl")
    candidates = []

    for p in prefixes:
        pid = p["prefix_id"]
        acts = by_pfx.get(pid, {})
        if not acts:
            continue
        c = acts.get("continue", {})
        b = acts.get("branch", {})
        rb = acts.get("rollback", {})
        c_ok = c.get("is_correct") if c.get("evaluation_status") == "OK" else None
        b_ok = b.get("pass_at_k") if b.get("evaluation_status") == "OK" else None
        r_ok = rb.get("pass_at_k") if rb.get("evaluation_status") == "OK" else None

        prog = p["reasoning_progress"]
        prob = traces.get(p["problem_id"], {})
        rp = reasoning_only(p["prefix_text"])
        steps = [x["text"] for x in sorted(next_by[pid], key=lambda x: x["sample_id"])]

        val = client.label_prefix(
            prefix_id=pid,
            question=prob.get("question", ""),
            gold_answer=prob.get("gold_answer", ""),
            reasoning_prefix=rp,
        )
        cl = (
            client.cluster_next_steps(
                prefix_id=pid,
                question=prob.get("question", ""),
                reasoning_prefix=rp,
                next_steps=steps,
            )
            if steps
            else {"clusters": []}
        )
        div = compute_diversity(steps, api_clusters=cl.get("clusters") or None)
        bucket = state_bucket(val["prefix_validity"], div["diversity_label"])

        score = 0
        reasons: list[str] = []
        if prog < 0.85:
            score += 2
            reasons.append("non-terminal")
        else:
            reasons.append("late/terminal")
        if c_ok != b_ok or c_ok != r_ok or b_ok != r_ok:
            score += 5
            reasons.append(f"action_split C={c_ok} B={b_ok} R={r_ok}")
        if bucket not in {"Stable", "UNCLEAR", "API_ERROR"}:
            score += 4
            reasons.append(f"bucket={bucket}")
        if val["prefix_validity"] == "INVALID":
            score += 4
            reasons.append("INVALID")
        if div["diversity_label"] == "HIGH_DIVERSITY":
            score += 3
            reasons.append("HIGH_DIV")
        if c_ok == 0 and (b_ok == 1 or r_ok == 1):
            score += 6
            reasons.append("recoverable")
        if b_ok == 1 and r_ok == 0:
            score += 5
            reasons.append("branch>rollback")
        if r_ok == 1 and b_ok == 0:
            score += 5
            reasons.append("rollback>branch")

        candidates.append(
            {
                "prefix_id": pid,
                "problem_id": p["problem_id"],
                "progress": prog,
                "score": score,
                "reasons": reasons,
                "continue": c_ok,
                "branch": b_ok,
                "rollback": r_ok,
                "validity": val["prefix_validity"],
                "confidence": val.get("confidence", 0),
                "diversity": div["diversity_label"],
                "num_clusters": div["num_clusters"],
                "bucket": bucket,
                "trace_correct": prob.get("is_correct"),
            }
        )

    candidates.sort(key=lambda x: (-x["score"], x["progress"]))

    print("=== TOP RESEARCH CANDIDATES ===")
    for c in candidates[:15]:
        print(
            f"{c['score']:2d} {c['prefix_id']} prog={c['progress']:.3f} "
            f"C/B/R={c['continue']}/{c['branch']}/{c['rollback']} "
            f"V={c['validity']} D={c['diversity']} bucket={c['bucket']} | {', '.join(c['reasons'])}"
        )

    splits = [
        c
        for c in candidates
        if c["continue"] != c["branch"] or c["continue"] != c["rollback"] or c["branch"] != c["rollback"]
    ]
    print("\n=== ACTION SPLITS ===")
    for c in splits:
        print(
            f"{c['prefix_id']} C={c['continue']} B={c['branch']} R={c['rollback']} "
            f"{c['bucket']} prog={c['progress']:.3f}"
        )
    print(f"\nTotal: {len(candidates)}, splits: {len(splits)}")

    # save ranking
    rank_path = OUT / "research_candidates.json"
    rank_path.write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {rank_path}")


if __name__ == "__main__":
    main()

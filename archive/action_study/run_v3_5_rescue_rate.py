"""V3.5 Experiment B — Rescue@K under greedy reject.

Estimates:
  r_K = P(exists accepted branch among first K | greedy rejected)

Sources (in priority order):
1. Final-stack API labels on fixed prefixes (preferred)
2. Local 32B batch verifier labels (proxy)
3. Provisional V3.3 GPT oracle rates (prior only)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from reasoning_branch_dataset.action_study.cost_rescue import (
    V33_PROVISIONAL_RESCUE,
    RescueRates,
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def rescue_at_k_from_accept_masks(
    masks: list[list[bool]],
    *,
    k: int,
    n_perm: int = 50,
    seed: int = 0,
) -> float:
    """Average P(any of first K accepted) over random branch permutations."""
    if not masks:
        return 0.0
    rng = random.Random(seed)
    hits = 0
    trials = 0
    for mask in masks:
        m = list(mask)
        if not m:
            continue
        for _ in range(n_perm):
            rng.shuffle(m)
            trials += 1
            if any(m[:k]):
                hits += 1
    return hits / trials if trials else 0.0


def estimate_from_oracle_rows(rows: list[dict[str, Any]]) -> RescueRates:
    """Estimate r_K from rows with greedy_acceptable + branch_acceptable lists.

    Expected fields:
      - greedy_acceptable: bool
      - branch_acceptable: list[bool] length >= 1
    """
    rejected = [r for r in rows if r.get("greedy_acceptable") is False]
    masks = [list(r.get("branch_acceptable") or []) for r in rejected]
    masks = [m for m in masks if m]
    return RescueRates(
        r1=rescue_at_k_from_accept_masks(masks, k=1),
        r2=rescue_at_k_from_accept_masks(masks, k=2),
        r4=rescue_at_k_from_accept_masks(masks, k=4),
        n_greedy_reject=len(masks),
        source="oracle_rows",
    )


def estimate_from_v33_step_oracle(path: Path) -> RescueRates:
    """Parse V3.3 step-oracle jsonl if present; else return provisional constants."""
    rows = _load_jsonl(path)
    if not rows:
        return V33_PROVISIONAL_RESCUE

    parsed: list[dict[str, Any]] = []
    for r in rows:
        # Support both raw oracle dumps and probe-dataset style rows.
        g = r.get("greedy_acceptable")
        if g is None and "acceptability" in r:
            acc = r["acceptability"]
            g = bool(acc.get("greedy") or acc.get("g0"))
            branches = [
                bool(acc.get(f"b{i}", acc.get(f"branch_{i}", False)))
                for i in range(1, 5)
            ]
        else:
            branches = list(r.get("branch_acceptable") or [])
            if not branches:
                # alternate keys used in some exports
                for key in ("b1", "b2", "b3", "b4"):
                    if key in r:
                        branches.append(bool(r[key]))
        if g is None:
            continue
        if not isinstance(g, bool):
            g = bool(g)
        parsed.append({"greedy_acceptable": g, "branch_acceptable": branches})

    if not parsed:
        return V33_PROVISIONAL_RESCUE
    rates = estimate_from_oracle_rows(parsed)
    return RescueRates(
        r1=rates.r1,
        r2=rates.r2,
        r4=rates.r4,
        n_greedy_reject=rates.n_greedy_reject,
        source=f"v3.3_file:{path}",
    )


def run_local_verifier_rescue(
    *,
    states_path: Path,
    draft_costs_path: Path,
    target_model: str,
    out_path: Path,
    max_states: int = 200,
    seed: int = 42,
    target_quantization: str | None = "awq",
    gpu_util: float = 0.92,
    max_model_len: int = 4096,
) -> RescueRates:
    """Label greedy+branches with local 32B batch verifier; estimate r_K | G=0."""
    from reasoning_branch_dataset.action_study.batch_step_verifier import BatchStepVerifier
    from reasoning_branch_dataset.action_study.target_verifier import build_target_verifier

    states = {r["state_id"]: r for r in _load_jsonl(states_path)}
    drafts = {r["state_id"]: r for r in _load_jsonl(draft_costs_path)}
    ids = sorted(set(states) & set(drafts))
    rng = random.Random(seed)
    if len(ids) > max_states:
        ids = rng.sample(ids, max_states)

    if "awq" not in Path(target_model).name.lower():
        target_quantization = None

    target = build_target_verifier(
        target_model,
        engine="vllm",
        gpu_memory_utilization=gpu_util,
        max_model_len=max_model_len,
        quantization=target_quantization,
    )
    verifier = BatchStepVerifier(target.llm, target.tokenizer)

    labeled: list[dict[str, Any]] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for sid in ids:
            st = states[sid]
            br = (drafts[sid].get("branches") or {}).get("4") or {}
            texts = list(br.get("texts") or [])
            # Also need a greedy candidate — use first branch text slot if missing.
            greedy = st.get("example_step") or (texts[0] if texts else "")
            cands = [greedy] + texts[:4]
            while len(cands) < 5:
                cands.append(greedy)
            res = verifier.verify_batch(
                question=st["question"],
                prefix_text=st["prefix_text"],
                candidates=cands[:5],
            )
            row = {
                "state_id": sid,
                "greedy_acceptable": res.acceptable[0],
                "branch_acceptable": [bool(a) for a in res.acceptable[1:5]],
                "raw": res.raw_outputs,
                "latency_sec": res.latency_sec,
            }
            # Treat unparsed as reject (conservative for rescue rate).
            if row["greedy_acceptable"] is None:
                row["greedy_acceptable"] = False
            row["branch_acceptable"] = [
                bool(a) if a is not None else False for a in res.acceptable[1:5]
            ]
            labeled.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"[v3.5B] {sid}: G={row['greedy_acceptable']} "
                f"branches={row['branch_acceptable']}"
            )

    rates = estimate_from_oracle_rows(labeled)
    return RescueRates(
        r1=rates.r1,
        r2=rates.r2,
        r4=rates.r4,
        n_greedy_reject=rates.n_greedy_reject,
        source=f"local_32b_verifier:{target_model}",
    )


def main() -> None:
    p = argparse.ArgumentParser(description="V3.5 Experiment B: Rescue@K")
    p.add_argument(
        "--mode",
        choices=["provisional", "from-oracle", "local-verifier"],
        default="provisional",
    )
    p.add_argument(
        "--oracle-path",
        default="",
        help="V3.3 / API oracle jsonl for --mode from-oracle",
    )
    p.add_argument(
        "--states-path",
        default="/root/autodl-tmp/reasonbranch/outputs/action_study_v35_latency/prefix_states.jsonl",
    )
    p.add_argument(
        "--draft-costs-path",
        default="/root/autodl-tmp/reasonbranch/outputs/action_study_v35_latency/draft_costs.jsonl",
    )
    p.add_argument(
        "--target-model",
        default="/root/autodl-tmp/specreason/models/DeepSeek-R1-Distill-Qwen-32B-AWQ",
    )
    p.add_argument(
        "--out-dir",
        default="/root/autodl-tmp/reasonbranch/outputs/action_study_v35_rescue",
    )
    p.add_argument("--max-states", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "provisional":
        rates = V33_PROVISIONAL_RESCUE
    elif args.mode == "from-oracle":
        rates = estimate_from_v33_step_oracle(Path(args.oracle_path))
    else:
        rates = run_local_verifier_rescue(
            states_path=Path(args.states_path),
            draft_costs_path=Path(args.draft_costs_path),
            target_model=args.target_model,
            out_path=out_dir / "local_verifier_labels.jsonl",
            max_states=args.max_states,
            seed=args.seed,
        )

    out = rates.to_dict()
    (out_dir / "rescue_rates.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

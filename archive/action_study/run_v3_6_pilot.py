"""V3.6 Pilot — paired one-step counterfactual Cost–Rescue Gate.

For each greedy-rejected state:
  - time Direct Handoff (T_H)
  - time Branch@1/2/4 pipelines (T_B)
  - optional offline oracle labels for Safe Rescue
  - write trial rows for v36_analyze
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from reasoning_branch_dataset.action_study.v36_analyze import (
    analyze_trials,
    compute_rescue_flags,
    render_report,
)
from reasoning_branch_dataset.action_study.v36_counterfactual import (
    DualResidentSession,
    draft_branch_pool,
    gamma_margin,
    run_branch_pipeline,
    run_direct_handoff,
    summarize_reps,
)
from reasoning_branch_dataset.action_study.v36_step_gen import generate_one_step_vllm


def _sha1(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def maybe_oracle_label_step(
    *,
    question: str,
    prefix_text: str,
    step_text: str,
    skip_oracle: bool,
    cache: dict[str, Any],
    prefix_id: str = "v36",
) -> tuple[bool | None, dict[str, Any]]:
    """Oracle acceptability for a SINGLE step (used for the Handoff step A(h)).

    Reuses the pooled judge by placing the step in the greedy slot; returns
    (acceptable_or_None, detail_dict).
    """
    details: dict[str, Any] = {}
    if skip_oracle:
        return None, {"source": "unavailable", "reason": "skip_oracle"}
    try:
        from reasoning_branch_dataset.action_study.gpt_step_oracle import (
            BRANCH_KEYS,
            GREEDY_KEY,
            GPTStepOracleClient,
        )
    except Exception as exc:  # pragma: no cover
        return None, {"source": "unavailable", "reason": f"import_error: {exc}"}

    client = cache.get("oracle")
    if client is None:
        try:
            cache_path = Path(
                "/root/autodl-tmp/reasonbranch/outputs/action_study_v36/oracle_cache.jsonl"
            )
            client = GPTStepOracleClient.from_env(cache_path=cache_path)
            cache["oracle"] = client
        except Exception as exc:
            return None, {"source": "unavailable", "reason": f"client_init_error: {exc}"}
    if not client.enabled or not client.api_key:
        return None, {"source": "unavailable", "reason": "disabled_or_no_key"}

    step = step_text.strip() or "(empty)"
    ckey = ("handoff", question[:120], prefix_text[-200:], step[:400])
    if ckey in cache:
        cached = cache[ckey]
        return cached["label"], cached["details"]

    steps = {
        GREEDY_KEY: step,
        BRANCH_KEYS[0]: "(empty)",
        BRANCH_KEYS[1]: "(empty)",
        BRANCH_KEYS[2]: "(empty)",
        BRANCH_KEYS[3]: "(empty)",
    }
    try:
        out = client.judge_shuffled_pass(
            prefix_id=f"{prefix_id}_handoff",
            question=question,
            prefix_tail=prefix_text[-1500:],
            steps=steps,
            shuffle_seed=0,
            pass_id=1,
        )
    except Exception as exc:
        return None, {"source": "unavailable", "reason": f"api_exception: {exc}"}
    if out.get("api_error"):
        return None, {"source": "unavailable", "reason": f"api_error: {out['api_error']}"}

    label = bool(out.get("g_acceptable"))
    judgments = out.get("candidate_judgments") or {}
    details = {
        "source": "api",
        "prefix_status": out.get("prefix_status"),
        "judgment": judgments.get(GREEDY_KEY),
        "model": getattr(client, "model", None),
        "cache_key": _oracle_cache_key(client, f"{prefix_id}_handoff", steps, 0, 1),
    }
    cache[ckey] = {"label": label, "details": details}
    return label, details


def _oracle_cache_key(client: Any, prefix_id: str, steps: dict[str, str], seed: int, pass_id: int) -> str | None:
    """Best-effort reconstruction of the oracle cache key for audit trails."""
    try:
        from reasoning_branch_dataset.action_study.gpt_step_oracle import (
            PROMPT_VERSION,
            _content_hash,
        )

        return f"gptstep:{prefix_id}:{PROMPT_VERSION}:p{pass_id}:{_content_hash(steps, seed)}"
    except Exception:
        return None


def maybe_oracle_label(
    *,
    question: str,
    prefix_text: str,
    candidates: list[str],
    skip_oracle: bool,
    cache: dict[str, Any],
    prefix_id: str = "v36",
    details_out: dict[str, Any] | None = None,
) -> list[bool | None]:
    """Offline semantic labels via DeepSeek V4 Pro step oracle (OpenRouter).

    When ``details_out`` is provided, it is populated with reproducibility info:
    per-candidate judgments, prefix_status, api_error, and a source flag. This
    lets the trial row record *why* the oracle judged each branch.
    """
    def _fail(reason: str) -> list[None]:
        if details_out is not None:
            details_out.update({"source": "unavailable", "reason": reason})
        return [None] * len(candidates)

    if skip_oracle:
        return _fail("skip_oracle")
    try:
        from reasoning_branch_dataset.action_study.gpt_step_oracle import (
            BRANCH_KEYS,
            GREEDY_KEY,
            GPTStepOracleClient,
        )
    except Exception as exc:
        return _fail(f"import_error: {exc}")

    client = cache.get("oracle")
    if client is None:
        try:
            cache_path = Path(
                "/root/autodl-tmp/reasonbranch/outputs/action_study_v36/oracle_cache.jsonl"
            )
            client = GPTStepOracleClient.from_env(cache_path=cache_path)
            cache["oracle"] = client
        except Exception as exc:
            return _fail(f"client_init_error: {exc}")
    if not client.enabled or not client.api_key:
        return _fail("disabled_or_no_key")

    cands = list(candidates[:4])
    while len(cands) < 4:
        cands.append("(empty)")
    steps = {
        GREEDY_KEY: "(placeholder — not used for branch labeling)",
        BRANCH_KEYS[0]: cands[0],
        BRANCH_KEYS[1]: cands[1],
        BRANCH_KEYS[2]: cands[2],
        BRANCH_KEYS[3]: cands[3],
    }
    pool_key = ("pool", question[:120], prefix_text[-200:], tuple(cands))
    if pool_key in cache:
        cached = cache[pool_key]
        if details_out is not None:
            details_out.update(cached.get("details", {"source": "cache"}))
        return list(cached["labels"])[: len(candidates)]

    try:
        out = client.judge_shuffled_pass(
            prefix_id=prefix_id,
            question=question,
            prefix_tail=prefix_text[-1500:],
            steps=steps,
            shuffle_seed=0,
            pass_id=1,
        )
    except Exception as exc:
        print(f"[v3.6 oracle] error: {exc}")
        return _fail(f"api_exception: {exc}")

    if out.get("api_error"):
        print(f"[v3.6 oracle] api_error: {out['api_error']}")
        return _fail(f"api_error: {out['api_error']}")

    branch_flags = list(out.get("branch_acceptables") or [False] * 4)
    labels: list[bool | None] = [bool(x) for x in branch_flags[: len(candidates)]]
    while len(labels) < len(candidates):
        labels.append(False)

    # Reproducibility: per-candidate judgments keyed by BRANCH slot (b1..b4).
    judgments = out.get("candidate_judgments") or {}
    branch_judgments = [judgments.get(bk) for bk in BRANCH_KEYS[: len(candidates)]]
    details = {
        "source": "api",
        "prefix_status": out.get("prefix_status"),
        "n_acceptable_branches": out.get("n_acceptable_branches"),
        "branch_judgments": branch_judgments,
        "model": getattr(client, "model", None),
        "cache_key": _oracle_cache_key(client, prefix_id, steps, 0, 1),
    }
    if details_out is not None:
        details_out.update(details)
    cache[pool_key] = {"labels": labels, "details": details}
    return labels


def run_state_pilot(
    session: DualResidentSession,
    state: dict[str, Any],
    *,
    ks: tuple[int, ...] = (1, 2, 4),
    n_reps: int = 5,
    n_seeds: int = 3,
    max_tokens: int = 256,
    skip_oracle: bool = True,
    oracle_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Two-layer design:

    - Layer 1 (timing noise): for a FIXED branch pool, repeat verify+fallback+
      handoff timing `n_reps` times.
    - Layer 2 (content variance): draw `n_seeds` independent branch pools.

    All K share ONE pool per seed (nested K1⊂K2⊂K4). Verifier / oracle / timing
    all use the SAME pool. Full texts, hashes, logprobs, and cache keys are saved.
    """
    rng = random.Random(hash(state["state_id"]) & 0xFFFFFFFF)
    oracle_cache = oracle_cache if oracle_cache is not None else {}
    prefix_text = state["prefix_text"]
    question = state["question"]
    pool_size = max(ks)

    # --- Handoff timing (fixed content: greedy temp=0 → deterministic step) ---
    h_walls = []
    h_example = None
    for _ in range(n_reps):
        th = run_direct_handoff(session, prefix_text=prefix_text, max_tokens=max_tokens)
        h_walls.append(th.wall_sec)
        if h_example is None:
            h_example = th
    h_stats = summarize_reps(h_walls)
    handoff_step = h_example.step_text if h_example else ""

    # --- Handoff oracle label A(h) ---
    handoff_oracle_label, handoff_oracle_details = maybe_oracle_label_step(
        question=question,
        prefix_text=prefix_text,
        step_text=handoff_step,
        skip_oracle=skip_oracle,
        cache=oracle_cache,
        prefix_id=state["state_id"],
    )

    # --- Layer 2: n_seeds independent pools ---
    seed_rows = []
    for seed_i in range(n_seeds):
        pool_seed = (hash(state["state_id"]) & 0xFFFF) * 100 + seed_i
        # Draw ONE fixed pool for this seed; measure draft cost once.
        pool, pool_draft_sec = draft_branch_pool(
            session,
            prefix_text=prefix_text,
            pool_size=pool_size,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.95,
            seed=pool_seed,
        )
        texts = [p.text or " " for p in pool]
        statuses = [p.status for p in pool]
        tokens = [p.num_tokens for p in pool]
        text_hashes = [_sha1(t) for t in texts]

        # Verifier scores on the fixed pool (single call; reused for all reps/K)
        vres = session.verifier.score_batch(
            question=question,
            prefix_text=prefix_text,
            candidates=texts,
            tau_accept=session.verify_tau,
        )
        scores = [s.score for s in vres.scores]
        logp_accept = [s.logp_accept for s in vres.scores]
        logp_reject = [s.logp_reject for s in vres.scores]
        verifier_prompt_hashes = [
            _sha1(session.verifier.build_prompt(question=question, prefix_text=prefix_text, candidate=t))
            for t in texts
        ]

        # Oracle labels on the SAME fixed pool
        oracle_details: dict[str, Any] = {}
        oracle_labels = maybe_oracle_label(
            question=question,
            prefix_text=prefix_text,
            candidates=texts,
            skip_oracle=skip_oracle,
            cache=oracle_cache,
            prefix_id=f"{state['state_id']}_s{seed_i}",
            details_out=oracle_details,
        )
        print(
            f"[v3.6 oracle] {state['state_id']} seed{seed_i} labels={oracle_labels} "
            f"src={oracle_details.get('source')} prefix={oracle_details.get('prefix_status')} "
            f"handoff_label={handoff_oracle_label}"
        )
        label_bools = list(oracle_labels)

        # Layer 1: fixed-content timing per nested K, n_reps each.
        b_stats = {}
        b_example = {}
        used_fb = {}
        for k in ks:
            walls = []
            ex = None
            for _ in range(n_reps):
                tb = run_branch_pipeline(
                    session,
                    question=question,
                    prefix_text=prefix_text,
                    k=k,
                    max_tokens=max_tokens,
                    branch_texts=texts,       # nested slice of the SAME pool
                    branch_statuses=statuses,
                    draft_sec=pool_draft_sec,  # shared draft cost
                )
                walls.append(tb.wall_sec)
                if ex is None:
                    ex = tb
            b_stats[str(k)] = summarize_reps(walls)
            b_example[str(k)] = ex
            used_fb[str(k)] = bool(ex.used_fallback if ex else True)

        rescue = {}
        deltas = {}
        profitable = {}
        for k in ks:
            # Nested K: candidates 0..k-1 of the fixed pool (deterministic).
            labs = label_bools[:k]
            scs = scores[:k]
            flags = compute_rescue_flags(
                {
                    "branch_oracle_labels": labs,
                    "branch_verifier_scores": scs,
                    "tau_accept": session.verify_tau,
                },
                k,
            )
            known = [x for x in labs if x is not None]
            th_med = h_stats["median"] or 0.0
            tb_med = b_stats[str(k)]["median"] or 0.0
            delta = th_med - tb_med
            gamma = gamma_margin(th_med)
            exist_rate = float(flags["exist"]) if known else float("nan")
            safe_rate = float(flags["safe"]) if known else float("nan")
            # Local safety condition A(b*) >= A(h): selected branch acceptable AND
            # (if handoff known) not strictly worse than handoff.
            sel = flags.get("selected_index")
            sel_label = labs[sel] if (sel is not None and sel < len(labs)) else None
            safe_vs_handoff = None
            if sel_label is not None and handoff_oracle_label is not None:
                safe_vs_handoff = bool(sel_label) or (not handoff_oracle_label)
            rescue[str(k)] = {
                "exist": exist_rate,
                "accepted": (max(scs) >= session.verify_tau) if scs else False,
                "safe": safe_rate,
                "safe_vs_handoff": safe_vs_handoff,
                "oracle_known": bool(known),
                "selected_index": flags.get("selected_index"),
                "selected_oracle_label": sel_label,
            }
            deltas[str(k)] = delta
            profitable[str(k)] = bool(known and safe_rate >= 0.5 and delta > gamma)

        seed_rows.append(
            {
                "branch_seed": seed_i,
                "pool_seed": pool_seed,
                "branch_steps": texts,
                "branch_hashes": text_hashes,
                "branch_tokens": tokens,
                "branch_statuses": statuses,
                "branch_verifier_scores": scores,
                "branch_logp_accept": logp_accept,
                "branch_logp_reject": logp_reject,
                "verifier_prompt_hashes": verifier_prompt_hashes,
                "branch_oracle_labels": oracle_labels,
                "branch_oracle_details": oracle_details,
                "oracle_cache_key": oracle_details.get("cache_key"),
                "draft_sec": pool_draft_sec,
                "branch_pipeline_sec": {str(k): b_stats[str(k)]["median"] for k in ks},
                "branch_pipeline_stats": b_stats,
                "branch_used_fallback": used_fb,
                "rescue": rescue,
                "delta_sec": deltas,
                "profitable": profitable,
            }
        )

    # Aggregate across seeds (mean of medians)
    def _nanmean(vals: list[float]) -> float:
        clean = [v for v in vals if v == v]  # drop NaN
        return sum(clean) / len(clean) if clean else float("nan")

    agg_pipe = {}
    agg_rescue = {}
    agg_delta = {}
    agg_prof = {}
    for k in ks:
        key = str(k)
        agg_pipe[key] = sum(s["branch_pipeline_sec"][key] or 0 for s in seed_rows) / len(seed_rows)
        agg_delta[key] = sum(s["delta_sec"][key] for s in seed_rows) / len(seed_rows)
        agg_rescue[key] = {
            "exist": _nanmean([s["rescue"][key]["exist"] for s in seed_rows]),
            "accepted": sum(float(s["rescue"][key]["accepted"]) for s in seed_rows) / len(seed_rows),
            "safe": _nanmean([s["rescue"][key]["safe"] for s in seed_rows]),
        }
        agg_prof[key] = sum(float(s["profitable"][key]) for s in seed_rows) / len(seed_rows)

    s0 = seed_rows[0] if seed_rows else {}
    return {
        "state_id": state["state_id"],
        "problem_id": state["problem_id"],
        "split": state.get("split"),
        # --- reproducibility: full source texts + hashes ---
        "question": question,
        "prefix_text": prefix_text,
        "prefix_hash": _sha1(prefix_text),
        "greedy_step": state.get("greedy_step"),
        "greedy_step_hash": _sha1(state.get("greedy_step", "")),
        "prefix_tokens": state.get("prefix_tokens"),
        "reasoning_depth": state.get("reasoning_depth"),
        "greedy_verifier_score": state.get("greedy_verifier_score"),
        "tau_accept": session.verify_tau,
        # --- handoff ---
        "handoff_wall_sec": h_stats["median"],
        "handoff_stats": h_stats,
        "target_step": handoff_step,
        "target_step_hash": _sha1(handoff_step),
        "target_step_tokens": h_example.step_tokens if h_example else 0,
        "target_step_status": h_example.step_status if h_example else "",
        "handoff_oracle_label": handoff_oracle_label,
        "handoff_oracle_details": handoff_oracle_details,
        # --- aggregates ---
        "branch_pipeline_sec": agg_pipe,
        "branch_used_fallback": {
            str(k): any(s["branch_used_fallback"][str(k)] for s in seed_rows) for k in ks
        },
        "rescue": agg_rescue,
        "delta_sec": agg_delta,
        "profitable_rate": agg_prof,
        "seeds": seed_rows,
        # --- flat fields (first seed) for analyze + quick audit ---
        "branch_oracle_labels": s0.get("branch_oracle_labels", []),
        "branch_verifier_scores": s0.get("branch_verifier_scores", []),
        "branch_logp_accept": s0.get("branch_logp_accept", []),
        "branch_logp_reject": s0.get("branch_logp_reject", []),
        "branch_steps": s0.get("branch_steps", []),
        "branch_hashes": s0.get("branch_hashes", []),
        "branch_statuses": s0.get("branch_statuses", []),
        "branch_tokens": s0.get("branch_tokens", []),
        "verifier_prompt_hashes": s0.get("verifier_prompt_hashes", []),
        "oracle_cache_key": s0.get("oracle_cache_key"),
        "branch_oracle_details": s0.get("branch_oracle_details", {}),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="V3.6 one-step counterfactual pilot")
    p.add_argument("--states", default="/root/autodl-tmp/reasonbranch/outputs/action_study_v36/rejected_states.jsonl")
    p.add_argument("--out-dir", default="/root/autodl-tmp/reasonbranch/outputs/action_study_v36")
    p.add_argument("--draft-model", default="/root/autodl-tmp/specreason/models/DeepSeek-R1-Distill-Qwen-1.5B")
    p.add_argument("--target-model", default="/root/autodl-tmp/specreason/models/DeepSeek-R1-Distill-Qwen-32B-AWQ")
    p.add_argument("--max-states", type=int, default=64)
    p.add_argument("--n-reps", type=int, default=3, help="timing reps (pilot default 3; full=5)")
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--tau-accept", type=float, default=0.0)
    p.add_argument("--skip-oracle", action="store_true", default=True)
    p.add_argument("--with-oracle", action="store_true")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--split", default="", help="optional: calibration|development|test")
    args = p.parse_args()

    skip_oracle = not args.with_oracle
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trials_path = out_dir / "trials.jsonl"

    states = _load_jsonl(Path(args.states))
    if args.split:
        states = [s for s in states if s.get("split") == args.split]
    states = states[: args.max_states]
    if not states:
        raise SystemExit(f"No states in {args.states}. Run run_v3_6_collect_states first.")

    done = {r["state_id"] for r in _load_jsonl(trials_path)}
    session = DualResidentSession(
        draft_model=args.draft_model,
        target_model=args.target_model,
        verify_tau=args.tau_accept,
    )
    print(f"[v3.6] warmup {args.warmup} on first prefix")
    session.warmup(states[0]["prefix_text"], n=args.warmup)

    oracle_cache: dict[str, Any] = {}
    for st in states:
        if st["state_id"] in done:
            continue
        print(f"[v3.6] trial {st['state_id']}")
        row = run_state_pilot(
            session,
            st,
            n_reps=args.n_reps,
            n_seeds=args.n_seeds,
            skip_oracle=skip_oracle,
            oracle_cache=oracle_cache,
        )
        _append_jsonl(trials_path, row)
        d1 = row["delta_sec"].get("1")
        d4 = row["delta_sec"].get("4")
        print(f"  T_H={row['handoff_wall_sec']:.3f}s  Δ1={d1:.3f}s  Δ4={d4:.3f}s")

    rows = _load_jsonl(trials_path)
    summary = analyze_trials(rows)
    (out_dir / "v36_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = render_report(summary)
    (out_dir / "v36_report.md").write_text(md, encoding="utf-8")
    Path("/root/autodl-tmp/reasonbranch/outputs/pilot_v3_6_report.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()

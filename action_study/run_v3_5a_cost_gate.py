"""V3.5a Formal Cost Gate — per-state / per-bucket r_K^* for K∈{1,2,4}.

Does NOT estimate rescue rates (that is V3.5b). Goal:
  measure whether Branch@K can ever beat Handoff under current impl.

Outputs:
  - decomposed C_T, C_DK, C_VK
  - optional dual-resident pipeline C_pipeK = wall-clock(draft@K → verify@K)
  - r_K^*(s) and tables by target-step / prefix buckets
  - C_T(L) curve for forced lengths
  - scoped decision (dominated / never / needs rescue / heterogeneity)
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path
from typing import Any

from reasoning_branch_dataset.action_study.batch_step_verifier import BatchStepVerifier
from reasoning_branch_dataset.action_study.cost_rescue import (
    break_even,
    decide_from_bucket_stars,
    decide_policy,
    prefix_length_bucket,
    step_length_bucket,
    summarize_latencies,
)
from reasoning_branch_dataset.action_study.target_verifier import build_target_verifier
from reasoning_branch_dataset.action_study.vllm_backend import VLLMEngine
from reasoning_branch_dataset.model_utils import build_prompt


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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _cuda_sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _free_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def load_problems(path: Path, n: int, seed: int) -> list[dict[str, Any]]:
    rows = _load_jsonl(path)
    cleaned: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        q = r.get("problem") or r.get("question") or r.get("prompt") or ""
        if not q.strip():
            continue
        cleaned.append(
            {
                "problem_id": str(r.get("id", r.get("problem_id", f"p{i:04d}"))),
                "question": q.strip(),
            }
        )
    if not cleaned:
        raise FileNotFoundError(f"No problems in {path}")
    rng = random.Random(seed)
    return cleaned if len(cleaned) <= n else rng.sample(cleaned, n)


def sample_prefix_states(
    draft: VLLMEngine,
    problems: list[dict[str, Any]],
    *,
    n_states: int,
    seed: int,
    warm_tokens: tuple[int, ...] = (128, 384, 768, 1280, 1800),
    step_max_tokens: int = 256,
) -> list[dict[str, Any]]:
    tok = draft.llm.get_tokenizer()
    states: list[dict[str, Any]] = []
    idx = 0
    while len(states) < n_states and problems:
        prob = problems[idx % len(problems)]
        idx += 1
        prompt = build_prompt(prob["question"])
        target_warm = warm_tokens[len(states) % len(warm_tokens)]
        gen = draft.generate_trace(prompt, max_tokens=max(target_warm + 64, 160))
        text = gen["response_text"]
        blocks = text.split("\n\n")
        if len(blocks) >= 2:
            keep = max(1, min(len(blocks) - 1, 1 + (len(states) % max(1, len(blocks) - 1))))
            prefix_tail = "\n\n".join(blocks[:keep]).rstrip() + "\n\n"
        else:
            cut = max(80, int(len(text) * target_warm / max(gen["num_tokens"], 1)))
            prefix_tail = text[:cut].rstrip() + "\n\n"
        prefix = prompt + prefix_tail
        step_rows = draft.generate_next_steps(
            prefix, k=1, max_tokens=step_max_tokens, temperature=0.0, top_p=1.0
        )
        step = (step_rows[0]["text"] if step_rows else "").strip()
        if not step:
            step = "Let me reconsider the previous calculation carefully."
        prefix_tokens = len(tok.encode(prefix))
        step_tokens = len(tok.encode(step))
        states.append(
            {
                "state_id": f"{prob['problem_id']}_s{len(states):03d}",
                "problem_id": prob["problem_id"],
                "question": prob["question"],
                "prefix_text": prefix,
                "example_step": step,
                "prefix_tokens": prefix_tokens,
                "draft_step_tokens": step_tokens,
                "prefix_bucket": prefix_length_bucket(prefix_tokens),
                "draft_step_bucket": step_length_bucket(step_tokens),
                "warm_target_tokens": target_warm,
            }
        )
    return states


def measure_draft_costs(
    draft: VLLMEngine,
    state: dict[str, Any],
    *,
    ks: tuple[int, ...] = (1, 2, 4),
    step_max_tokens: int = 256,
    reps: int = 3,
    warmup: int = 1,
) -> dict[str, Any]:
    prefix = state["prefix_text"]
    out: dict[str, Any] = {"branches": {}}
    for _ in range(warmup):
        _ = draft.generate_next_steps(prefix, k=1, max_tokens=16, temperature=0.0, top_p=1.0)

    for k in ks:
        latencies: list[float] = []
        texts: list[str] = []
        ntoks: list[int] = []
        for r in range(reps):
            _cuda_sync()
            t0 = time.perf_counter()
            rows = draft.generate_next_steps(
                prefix,
                k=k,
                max_tokens=step_max_tokens,
                temperature=0.7 if k > 1 else 0.0,
                top_p=0.95 if k > 1 else 1.0,
            )
            _cuda_sync()
            latencies.append(time.perf_counter() - t0)
            if r == 0:
                texts = [x["text"].strip() for x in rows]
                ntoks = [int(x.get("num_tokens", 0)) for x in rows]
                while len(texts) < k:
                    texts.append(state["example_step"])
        out["branches"][str(k)] = {
            "latency_sec": summarize_latencies(latencies)["median"],
            "latency_stats": summarize_latencies(latencies),
            "texts": texts[:k],
            "num_tokens": ntoks[:k],
        }
    return out


def measure_target_costs(
    verifier: BatchStepVerifier,
    target_llm: Any,
    target_tok: Any,
    state: dict[str, Any],
    branch_payloads: dict[str, Any],
    *,
    ks: tuple[int, ...] = (1, 2, 4),
    step_max_tokens: int = 256,
    forced_lengths: tuple[int, ...] = (32, 64, 96, 128, 192, 256),
    reps: int = 3,
    warmup: int = 1,
) -> dict[str, Any]:
    from vllm import SamplingParams

    prefix = state["prefix_text"]
    question = state["question"]
    prefix_ids = target_tok.encode(prefix, add_special_tokens=False)

    for _ in range(warmup):
        _ = target_llm.generate(
            [{"prompt_token_ids": prefix_ids}],
            SamplingParams(max_tokens=8, temperature=0.0, top_p=1.0, detokenize=True),
        )

    # Natural C_T: stop at paragraph boundary.
    nat_lats: list[float] = []
    nat_toks = 0
    for r in range(reps):
        params = SamplingParams(
            max_tokens=step_max_tokens,
            temperature=0.0,
            top_p=1.0,
            stop=["\n\n"],
            detokenize=True,
        )
        _cuda_sync()
        t0 = time.perf_counter()
        gen = target_llm.generate([{"prompt_token_ids": prefix_ids}], params)[0]
        _cuda_sync()
        nat_lats.append(time.perf_counter() - t0)
        if r == 0:
            nat_toks = len(list(gen.outputs[0].token_ids))

    # Forced-length C_T(L) curve (decode budget L; may EOS early).
    ct_by_l: dict[str, Any] = {}
    for L in forced_lengths:
        lats: list[float] = []
        got = 0
        for r in range(max(1, reps - 1)):
            params = SamplingParams(
                max_tokens=L,
                min_tokens=min(L, 8),
                temperature=0.0,
                top_p=1.0,
                detokenize=True,
            )
            _cuda_sync()
            t0 = time.perf_counter()
            gen = target_llm.generate([{"prompt_token_ids": prefix_ids}], params)[0]
            _cuda_sync()
            lats.append(time.perf_counter() - t0)
            if r == 0:
                got = len(list(gen.outputs[0].token_ids))
        ct_by_l[str(L)] = {
            "latency_sec": summarize_latencies(lats)["median"],
            "latency_stats": summarize_latencies(lats),
            "actual_tokens": got,
            "requested_tokens": L,
        }

    verify: dict[str, Any] = {}
    for k in ks:
        texts = branch_payloads.get(str(k), {}).get("texts") or [state["example_step"]] * k
        texts = (texts + [state["example_step"]] * k)[:k]
        lats: list[float] = []
        meta = None
        for r in range(reps):
            res = verifier.verify_batch(
                question=question, prefix_text=prefix, candidates=texts
            )
            lats.append(res.latency_sec)
            if r == 0:
                meta = {
                    "parsed_rate": res.parsed_rate,
                    "shared_stem_tokens": res.shared_stem_tokens,
                    "prompt_tokens": res.prompt_tokens,
                    "output_tokens": res.output_tokens,
                    "n_accepted": sum(1 for a in res.acceptable if a is True),
                    "raw": res.raw_outputs,
                }
        verify[str(k)] = {
            "latency_sec": summarize_latencies(lats)["median"],
            "latency_stats": summarize_latencies(lats),
            **(meta or {}),
        }

    return {
        "c_t": summarize_latencies(nat_lats)["median"],
        "c_t_stats": summarize_latencies(nat_lats),
        "target_step_tokens": nat_toks,
        "target_step_bucket": step_length_bucket(nat_toks),
        "c_t_by_length": ct_by_l,
        "verify": verify,
    }


def measure_pipeline_e2e(
    draft: VLLMEngine,
    verifier: BatchStepVerifier,
    state: dict[str, Any],
    *,
    ks: tuple[int, ...] = (1, 2, 4),
    step_max_tokens: int = 256,
    reps: int = 2,
) -> dict[str, Any]:
    """True dual-resident pipeline: draft@K then verify@K in one wall-clock span."""
    out: dict[str, Any] = {}
    prefix = state["prefix_text"]
    question = state["question"]
    for k in ks:
        lats: list[float] = []
        for _ in range(reps):
            _cuda_sync()
            t0 = time.perf_counter()
            rows = draft.generate_next_steps(
                prefix,
                k=k,
                max_tokens=step_max_tokens,
                temperature=0.7 if k > 1 else 0.0,
                top_p=0.95 if k > 1 else 1.0,
            )
            texts = [r["text"].strip() or state["example_step"] for r in rows]
            while len(texts) < k:
                texts.append(state["example_step"])
            _ = verifier.verify_batch(
                question=question, prefix_text=prefix, candidates=texts[:k]
            )
            _cuda_sync()
            lats.append(time.perf_counter() - t0)
        out[str(k)] = {
            "latency_sec": summarize_latencies(lats)["median"],
            "latency_stats": summarize_latencies(lats),
        }
    return out


def _agg_by(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(str(r.get(key, "unk")), []).append(r)
    out = []
    for bucket, grp in sorted(groups.items()):
        def col(name: str) -> list[float]:
            return [float(x[name]) for x in grp if x.get(name) is not None]

        c_t = summarize_latencies(col("c_t"))["median"]
        row: dict[str, Any] = {"bucket": bucket, "n": len(grp), "c_t": c_t}
        for k in (1, 2, 4):
            cd = summarize_latencies(col(f"c_d{k}"))["median"]
            cv = summarize_latencies(col(f"c_v{k}"))["median"]
            cp = summarize_latencies(col(f"c_pipe{k}"))["median"]
            row[f"c_d{k}"] = cd
            row[f"c_v{k}"] = cv
            row[f"c_pipe{k}"] = cp
            row[f"r{k}_star"] = (
                break_even(cd, cv, c_t) if (cd is not None and cv is not None and c_t) else None
            )
            row[f"r{k}_star_pipe"] = (
                (cp / c_t) if (cp is not None and c_t and c_t > 0) else None
            )
        out.append(row)
    return out


def render_report(summary: dict[str, Any]) -> str:
    o = summary.get("overall") or {}
    lines = [
        "# V3.5a — Formal Cost Gate",
        "",
        "> Scope: **current impl / current length distribution / current hardware**.",
        "> Does **not** claim Branch is forever useless.",
        "",
        f"- draft: `{summary.get('draft_model')}`",
        f"- target: `{summary.get('target_model')}`",
        f"- states: **{summary.get('n_states', 0)}**",
        f"- dual_resident_pipeline: `{summary.get('dual_resident')}`",
        f"- prefix_caching: `{summary.get('prefix_caching')}`",
        f"- verifier max_tokens: `{summary.get('verify_max_tokens')}`",
        "",
        "## Overall (median)",
        "",
        "| Metric | Value |",
        "|--------|------:|",
        f"| $C_T$ | {_s(o.get('c_t'))} |",
        f"| $C_{{D1}} / C_{{V1}}$ | {_s(o.get('c_d1'))} / {_s(o.get('c_v1'))} |",
        f"| $C_{{D2}} / C_{{V2}}$ | {_s(o.get('c_d2'))} / {_s(o.get('c_v2'))} |",
        f"| $C_{{D4}} / C_{{V4}}$ | {_s(o.get('c_d4'))} / {_s(o.get('c_v4'))} |",
        f"| $r_1^*$ | {_pct(o.get('r1_star'))} |",
        f"| $r_2^*$ | {_pct(o.get('r2_star'))} |",
        f"| $r_4^*$ | {_pct(o.get('r4_star'))} |",
        f"| $r_4^*$ (pipeline) | {_pct(o.get('r4_star_pipe'))} |",
        "",
        "## By natural target step length",
        "",
        "| Bucket | N | $C_T$ | $r_1^*$ | $r_2^*$ | $r_4^*$ |",
        "|--------|--:|------:|-------:|-------:|-------:|",
    ]
    for row in summary.get("by_target_step_bucket") or []:
        lines.append(
            f"| {row['bucket']} | {row['n']} | {_s(row.get('c_t'))} | "
            f"{_pct(row.get('r1_star'))} | {_pct(row.get('r2_star'))} | {_pct(row.get('r4_star'))} |"
        )
    lines += [
        "",
        "## By prefix length",
        "",
        "| Bucket | N | $C_T$ | $r_1^*$ | $r_2^*$ | $r_4^*$ |",
        "|--------|--:|------:|-------:|-------:|-------:|",
    ]
    for row in summary.get("by_prefix_bucket") or []:
        lines.append(
            f"| {row['bucket']} | {row['n']} | {_s(row.get('c_t'))} | "
            f"{_pct(row.get('r1_star'))} | {_pct(row.get('r2_star'))} | {_pct(row.get('r4_star'))} |"
        )

    lines += ["", "## $C_T(L)$ forced-length curve (median over states)", ""]
    ct_curve = summary.get("c_t_length_curve") or {}
    if ct_curve:
        lines += [
            "| L (requested) | median $C_T$ |",
            "|-------------:|-------------:|",
        ]
        for L in sorted(ct_curve.keys(), key=lambda x: int(x)):
            lines.append(f"| {L} | {_s(ct_curve[L])} |")

    lines += ["", "## Scoped decisions (Cost Gate A only)", ""]
    for d in summary.get("decisions") or []:
        lines.append(f"- **K={d.get('k')}**: `{d.get('decision')}` — {d.get('rationale')}")

    het = summary.get("heterogeneity") or {}
    if het:
        lines += ["", "### Heterogeneity check", ""]
        for k, h in sorted(het.items(), key=lambda x: int(x[0])):
            lines.append(
                f"- K={k}: `{h.get('decision')}` — {h.get('rationale')} "
                f"(spread={_pct(h.get('spread'))})"
            )

    lines += [
        "",
        "## Interpretation rules",
        "",
        "- If $r_K^*\\ge 100\\%$: Branch@K is **strictly dominated** even at perfect rescue.",
        "- Router is useful **only** when action value is state-heterogeneous "
        "(some buckets above, some below break-even after Rescue Gate).",
        "- Do **not** train Branch router until V3.5b $r_K^{select}$ is measured.",
        "",
    ]
    return "\n".join(lines)


def _s(x: Any) -> str:
    return "—" if x is None else f"{float(x):.3f}s"


def _pct(x: Any) -> str:
    return "—" if x is None else f"{100 * float(x):.1f}%"


def run_v35a(
    *,
    draft_model: str,
    target_model: str,
    problems_path: Path,
    out_dir: Path,
    n_states: int = 48,
    n_problems: int = 30,
    seed: int = 42,
    step_max_tokens: int = 256,
    reps: int = 3,
    dual_resident: bool = True,
    target_quantization: str | None = "awq",
    draft_gpu_util: float = 0.18,
    target_gpu_util: float = 0.70,
    max_model_len: int = 4096,
    verify_max_tokens: int = 2,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    states_path = out_dir / "prefix_states.jsonl"
    draft_path = out_dir / "draft_costs.jsonl"
    samples_path = out_dir / "cost_samples.jsonl"
    summary_path = out_dir / "cost_gate_summary.json"
    report_path = out_dir / "cost_gate_report.md"

    if target_quantization and "awq" not in Path(target_model).name.lower():
        target_quantization = None

    problems = load_problems(problems_path, n_problems, seed)
    ks = (1, 2, 4)

    if dual_resident:
        print(f"[v3.5a] dual-resident load target={target_model}")
        target = build_target_verifier(
            target_model,
            engine="vllm",
            gpu_memory_utilization=target_gpu_util,
            max_model_len=max_model_len,
            quantization=target_quantization,
            dual_resident=True,
            enable_prefix_caching=True,
        )
        print(f"[v3.5a] dual-resident load draft={draft_model}")
        draft = VLLMEngine(
            draft_model,
            gpu_memory_utilization=draft_gpu_util,
            max_model_len=max_model_len,
        )
        verifier = BatchStepVerifier(target.llm, target.tokenizer, max_tokens=verify_max_tokens)

        if states_path.exists():
            states = _load_jsonl(states_path)
        else:
            states = sample_prefix_states(
                draft, problems, n_states=n_states, seed=seed, step_max_tokens=step_max_tokens
            )
            _write_jsonl(states_path, states)

        done = {r["state_id"] for r in _load_jsonl(samples_path)}
        for st in states:
            sid = st["state_id"]
            if sid in done:
                continue
            print(f"[v3.5a] state {sid}")
            d = measure_draft_costs(
                draft, st, ks=ks, step_max_tokens=step_max_tokens, reps=reps
            )
            _append_jsonl(draft_path, {"state_id": sid, **d})
            t = measure_target_costs(
                verifier,
                target.llm,
                target.tokenizer,
                st,
                d.get("branches", {}),
                ks=ks,
                step_max_tokens=step_max_tokens,
                reps=reps,
            )
            pipe = measure_pipeline_e2e(
                draft, verifier, st, ks=ks, step_max_tokens=step_max_tokens, reps=max(1, reps - 1)
            )
            row = {
                "state_id": sid,
                "problem_id": st["problem_id"],
                "prefix_tokens": st["prefix_tokens"],
                "prefix_bucket": st["prefix_bucket"],
                "c_t": t["c_t"],
                "c_t_stats": t["c_t_stats"],
                "target_step_tokens": t["target_step_tokens"],
                "target_step_bucket": t["target_step_bucket"],
                "c_t_by_length": t["c_t_by_length"],
                "c_d1": d["branches"]["1"]["latency_sec"],
                "c_d2": d["branches"]["2"]["latency_sec"],
                "c_d4": d["branches"]["4"]["latency_sec"],
                "c_v1": t["verify"]["1"]["latency_sec"],
                "c_v2": t["verify"]["2"]["latency_sec"],
                "c_v4": t["verify"]["4"]["latency_sec"],
                "c_pipe1": pipe["1"]["latency_sec"],
                "c_pipe2": pipe["2"]["latency_sec"],
                "c_pipe4": pipe["4"]["latency_sec"],
                "verify_meta": t["verify"],
                "draft_meta": {k: {kk: vv for kk, vv in v.items() if kk != "texts"} for k, v in d["branches"].items()},
            }
            for k in ks:
                cd, cv, ct = row[f"c_d{k}"], row[f"c_v{k}"], row["c_t"]
                row[f"r{k}_star"] = break_even(cd, cv, ct) if ct else None
                cp = row[f"c_pipe{k}"]
                row[f"r{k}_star_pipe"] = (cp / ct) if (cp is not None and ct) else None
            _append_jsonl(samples_path, row)
            print(
                f"  C_T={row['c_t']:.3f}s step_tok={row['target_step_tokens']} "
                f"r4*={row['r4_star']:.2f} pipe_r4*={row['r4_star_pipe']:.2f}"
            )
    else:
        # Sequential swap path (same as smoke): draft then target.
        print(f"[v3.5a] sequential draft={draft_model}")
        draft = VLLMEngine(draft_model, gpu_memory_utilization=0.90, max_model_len=max_model_len)
        if states_path.exists():
            states = _load_jsonl(states_path)
        else:
            states = sample_prefix_states(
                draft, problems, n_states=n_states, seed=seed, step_max_tokens=step_max_tokens
            )
            _write_jsonl(states_path, states)
        done_draft = {r["state_id"] for r in _load_jsonl(draft_path)}
        for st in states:
            if st["state_id"] in done_draft:
                continue
            d = measure_draft_costs(draft, st, ks=ks, step_max_tokens=step_max_tokens, reps=reps)
            _append_jsonl(draft_path, {"state_id": st["state_id"], **d})
        del draft
        _free_cuda()

        print(f"[v3.5a] sequential target={target_model}")
        target = build_target_verifier(
            target_model,
            engine="vllm",
            gpu_memory_utilization=0.92,
            max_model_len=max_model_len,
            quantization=target_quantization,
            enable_prefix_caching=True,
        )
        verifier = BatchStepVerifier(target.llm, target.tokenizer, max_tokens=verify_max_tokens)
        drafts = {r["state_id"]: r for r in _load_jsonl(draft_path)}
        done = {r["state_id"] for r in _load_jsonl(samples_path)}
        for st in states:
            sid = st["state_id"]
            if sid in done:
                continue
            d = drafts[sid]
            t = measure_target_costs(
                verifier,
                target.llm,
                target.tokenizer,
                st,
                d.get("branches", {}),
                ks=ks,
                step_max_tokens=step_max_tokens,
                reps=reps,
            )
            row = {
                "state_id": sid,
                "problem_id": st["problem_id"],
                "prefix_tokens": st["prefix_tokens"],
                "prefix_bucket": st["prefix_bucket"],
                "c_t": t["c_t"],
                "target_step_tokens": t["target_step_tokens"],
                "target_step_bucket": t["target_step_bucket"],
                "c_t_by_length": t["c_t_by_length"],
                "c_d1": d["branches"]["1"]["latency_sec"],
                "c_d2": d["branches"]["2"]["latency_sec"],
                "c_d4": d["branches"]["4"]["latency_sec"],
                "c_v1": t["verify"]["1"]["latency_sec"],
                "c_v2": t["verify"]["2"]["latency_sec"],
                "c_v4": t["verify"]["4"]["latency_sec"],
                "c_pipe1": None,
                "c_pipe2": None,
                "c_pipe4": None,
                "verify_meta": t["verify"],
            }
            for k in ks:
                row[f"r{k}_star"] = break_even(row[f"c_d{k}"], row[f"c_v{k}"], row["c_t"])
                row[f"r{k}_star_pipe"] = None
            _append_jsonl(samples_path, row)

    rows = _load_jsonl(samples_path)
    by_step = _agg_by(rows, "target_step_bucket")
    by_pref = _agg_by(rows, "prefix_bucket")
    overall_list = _agg_by([{**r, "all": "all"} for r in rows], "all")
    overall = overall_list[0] if overall_list else {}

    # C_T(L) curve across states
    ct_curve: dict[str, float] = {}
    length_map: dict[str, list[float]] = {}
    for r in rows:
        for L, meta in (r.get("c_t_by_length") or {}).items():
            if meta and meta.get("latency_sec") is not None:
                length_map.setdefault(str(L), []).append(float(meta["latency_sec"]))
    for L, xs in length_map.items():
        ct_curve[L] = summarize_latencies(xs)["median"]  # type: ignore[assignment]

    decisions = []
    for k in ks:
        decisions.append(
            decide_policy(r_k=None, r_k_star=overall.get(f"r{k}_star"), k=k).to_dict()
        )

    heterogeneity = {}
    for k in ks:
        bucket_stars = {b["bucket"]: b.get(f"r{k}_star") for b in by_step}
        heterogeneity[str(k)] = decide_from_bucket_stars(bucket_stars, k=k)

    summary = {
        "n_states": len(rows),
        "draft_model": draft_model,
        "target_model": target_model,
        "dual_resident": dual_resident,
        "prefix_caching": True,
        "verify_max_tokens": verify_max_tokens,
        "step_max_tokens": step_max_tokens,
        "reps": reps,
        "overall": overall,
        "by_target_step_bucket": by_step,
        "by_prefix_bucket": by_pref,
        "c_t_length_curve": ct_curve,
        "decisions": decisions,
        "heterogeneity": heterogeneity,
        "scope_note": (
            "never/dominated applies to current impl + length + hardware only; "
            "not a claim that Branch is forever useless"
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(render_report(summary), encoding="utf-8")
    # Convenience mirrors
    mirror = Path("/root/autodl-tmp/reasonbranch/outputs/pilot_v3_5a_cost_gate_report.md")
    mirror.write_text(render_report(summary), encoding="utf-8")
    print(f"[v3.5a] wrote {report_path}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="V3.5a formal Cost Gate")
    p.add_argument("--draft-model", default="/root/autodl-tmp/specreason/models/DeepSeek-R1-Distill-Qwen-1.5B")
    p.add_argument("--target-model", default="/root/autodl-tmp/specreason/models/DeepSeek-R1-Distill-Qwen-32B-AWQ")
    p.add_argument("--target-quantization", default="awq")
    p.add_argument("--problems", default="/root/autodl-tmp/reasonbranch/data/aime_train.jsonl")
    p.add_argument("--out-dir", default="/root/autodl-tmp/reasonbranch/outputs/action_study_v35a_cost_gate")
    p.add_argument("--n-states", type=int, default=48)
    p.add_argument("--n-problems", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--step-max-tokens", type=int, default=256)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--verify-max-tokens", type=int, default=2)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--draft-gpu-util", type=float, default=0.18)
    p.add_argument("--target-gpu-util", type=float, default=0.70)
    p.add_argument("--no-dual-resident", action="store_true")
    args = p.parse_args()

    run_v35a(
        draft_model=args.draft_model,
        target_model=args.target_model,
        problems_path=Path(args.problems),
        out_dir=Path(args.out_dir),
        n_states=args.n_states,
        n_problems=args.n_problems,
        seed=args.seed,
        step_max_tokens=args.step_max_tokens,
        reps=args.reps,
        dual_resident=not args.no_dual_resident,
        target_quantization=args.target_quantization,
        draft_gpu_util=args.draft_gpu_util,
        target_gpu_util=args.target_gpu_util,
        max_model_len=args.max_model_len,
        verify_max_tokens=args.verify_max_tokens,
    )


if __name__ == "__main__":
    main()

"""Main data collection pipeline with API-guided prefix selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from reasoning_branch_dataset.api_teacher import TeacherClient, TeacherConfig
from reasoning_branch_dataset.config import DatasetConfig
from reasoning_branch_dataset.datasets import load_problems
from reasoning_branch_dataset.grading import extract_math_answer, math_equal
from reasoning_branch_dataset.hidden_export import extract_prefix_hidden, flush_hidden_store, save_hidden_batch
from reasoning_branch_dataset.io_utils import save_table
from reasoning_branch_dataset.model_utils import (
    build_prompt,
    generate_continuation,
    generate_with_trace,
    js_divergence,
    kl_divergence,
    load_model_and_tokenizer,
    score_correctness,
)
from reasoning_branch_dataset.prefix_select import ScoredPrefix, extract_candidate_pool, select_prefixes_for_rollout
from reasoning_branch_dataset.spec_verify import run_speculative_round


def _stable_id(*parts: str) -> str:
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:12]


def _load_checkpoint(path: Path) -> set[str]:
    if path.exists():
        return set(json.loads(path.read_text()))
    return set()


def _save_checkpoint(path: Path, done: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(done)))


def _append_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


@torch.no_grad()
def _prefix_token_features(
    draft_model,
    target_model,
    tokenizer,
    prefix_text: str,
    topk_save: int,
) -> dict[str, Any]:
    inputs = tokenizer(prefix_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(draft_model.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(draft_model.device)

    draft_logits = draft_model(input_ids=input_ids, attention_mask=attention_mask).logits
    target_logits = target_model(input_ids=input_ids, attention_mask=attention_mask).logits

    pos = input_ids.shape[1] - 1
    d_score = draft_logits[0, pos]
    t_score = target_logits[0, pos]
    d_lp = torch.log_softmax(d_score.float(), dim=-1)
    t_lp = torch.log_softmax(t_score.float(), dim=-1)
    d_prob = d_lp.exp()

    topv, topi = torch.topk(d_lp, k=min(topk_save, d_lp.numel()))
    topk_ids = topi.tolist()
    topk_probs = topv.exp().tolist()
    entropy = float(-(d_prob * d_lp).sum().item())
    top1 = float(topk_probs[0]) if topk_probs else 0.0
    top2 = float(topk_probs[1]) if len(topk_probs) > 1 else 0.0
    margin = top1 - top2

    t_topv, t_topi = torch.topk(t_lp, k=min(topk_save, t_lp.numel()))
    proposed = int(input_ids[0, -1].item()) if pos > 0 else int(topi[0])
    accept_ratio = float(torch.exp(t_lp[proposed]) / torch.exp(d_lp[proposed]).clamp(min=1e-12))

    return {
        "entropy": entropy,
        "margin": margin,
        "top1_prob": top1,
        "top2_prob": top2,
        "topk_token_ids": topk_ids,
        "topk_probs": topk_probs,
        "target_topk_token_ids": t_topi.tolist(),
        "target_topk_probs": t_topv.exp().tolist(),
        "draft_target_kl": kl_divergence(d_score, t_score),
        "draft_target_js": js_divergence(d_score, t_score),
        "proposed_token_accept_ratio": accept_ratio,
    }


def _process_prefix_rollout(
    sp: ScoredPrefix,
    *,
    prob: dict,
    prompt: str,
    draft_model,
    target_model,
    tokenizer,
    cfg: DatasetConfig,
    rng: random.Random,
    teacher: TeacherClient,
    full_reasoning: str,
) -> tuple[list[dict], list[dict], list[dict], list[dict], dict[str, torch.Tensor], list[dict], list[dict]]:
    cand = sp.candidate
    prefix_id = sp.prefix_id
    prefix_full = prompt + cand.prefix_text

    tf = _prefix_token_features(draft_model, target_model, tokenizer, prefix_full, cfg.topk_save)
    token_features_row = {"prefix_id": prefix_id, "problem_id": prob["problem_id"], **tf}

    hidden_store: dict[str, torch.Tensor] = {}
    for source, model in (("draft", draft_model), ("target", target_model)):
        pooled = extract_prefix_hidden(
            model, tokenizer, prefix_full, cand.token_index, cand.step_index, cfg.hidden_layers
        )
        save_hidden_batch(hidden_store, prefix_id, source, pooled)

    token_branch_rows: list[dict] = []
    top_ids = tf["topk_token_ids"][: cfg.token_branch_k]
    top_probs = tf["topk_probs"][: cfg.token_branch_k]
    for tid, probv in zip(top_ids, top_probs):
        cont = generate_continuation(
            draft_model,
            tokenizer,
            prefix_full,
            max_new_tokens=cfg.next_step_max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            do_sample=False,
            forced_first_token_id=int(tid),
            stop_at_paragraph=True,
        )
        token_branch_rows.append(
            {
                "prefix_id": prefix_id,
                "branch_type": "token_branch",
                "first_token_id": int(tid),
                "first_token_text": tokenizer.decode([int(tid)]),
                "first_token_prob": float(probv),
                "continuation_text": cont["continuation_text"],
                "continuation_token_ids": json.dumps(cont["continuation_token_ids"]),
                "stop_reason": cont["stop_reason"],
            }
        )

    step_branch_rows: list[dict] = []
    do_full = rng.random() < cfg.full_answer_branch_frac
    branch_payload: list[dict[str, str]] = []
    for k in range(cfg.step_branch_k):
        next_step = generate_continuation(
            draft_model,
            tokenizer,
            prefix_full,
            max_new_tokens=cfg.next_step_max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            do_sample=True,
            stop_at_paragraph=True,
        )
        branch_text = next_step["continuation_text"]
        branch_id = f"{prefix_id}_ns{k}"
        step_branch_rows.append(
            {
                "prefix_id": prefix_id,
                "branch_id": branch_id,
                "branch_mode": "next_step",
                "branch_text": branch_text,
                "branch_tokens": json.dumps(next_step["continuation_token_ids"]),
                "branch_length": len(next_step["continuation_token_ids"]),
                "contains_wait": int("Wait" in branch_text),
                "contains_but": int("But" in branch_text),
                "final_answer": extract_math_answer(prefix_full + branch_text),
                "is_correct": int(math_equal(prefix_full + branch_text, prob["gold_answer"])),
            }
        )
        branch_payload.append({"branch_id": branch_id, "branch_text": branch_text})

        if do_full:
            full = generate_continuation(
                draft_model,
                tokenizer,
                prefix_full,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                do_sample=True,
                stop_at_paragraph=False,
            )
            full_text = full["continuation_text"]
            branch_id = f"{prefix_id}_fa{k}"
            step_branch_rows.append(
                {
                    "prefix_id": prefix_id,
                    "branch_id": branch_id,
                    "branch_mode": "full_answer",
                    "branch_text": full_text,
                    "branch_tokens": json.dumps(full["continuation_token_ids"]),
                    "branch_length": len(full["continuation_token_ids"]),
                    "contains_wait": int("Wait" in full_text),
                    "contains_but": int("But" in full_text),
                    "final_answer": extract_math_answer(prefix_full + full_text),
                    "is_correct": int(math_equal(prefix_full + full_text, prob["gold_answer"])),
                }
            )

    api_branch_rows: list[dict] = []
    if cfg.use_api_teacher and branch_payload:
        try:
            api_branch_rows = teacher.rank_branches(
                prefix_id=prefix_id,
                question=prob["question"],
                gold_answer=prob["gold_answer"],
                prefix_tail=cand.prefix_text,
                branches=branch_payload,
            )
        except Exception as exc:
            api_branch_rows = [{"prefix_id": prefix_id, "api_error": str(exc)}]

    api_rows: list[dict] = []
    if cfg.use_trace_aware_api:
        try:
            api_rows.append(
                teacher.annotate_trace_aware(
                    prefix_id=prefix_id,
                    question=prob["question"],
                    gold_answer=prob["gold_answer"],
                    full_reasoning=full_reasoning,
                    prefix_tail=cand.prefix_text,
                )
            )
        except Exception as exc:
            api_rows.append({"prefix_id": prefix_id, "annotation_mode": "trace_aware", "api_error": str(exc)})

    verification_rows: list[dict] = []
    spec_logs = run_speculative_round(
        draft_model,
        target_model,
        tokenizer,
        prefix_full,
        gamma=cfg.spec_gamma,
        max_new_tokens=min(cfg.next_step_max_tokens, 64),
    )
    for log in spec_logs:
        verification_rows.append({"problem_id": prob["problem_id"], "prefix_id": prefix_id, **log})

    return (
        [token_features_row],
        token_branch_rows,
        step_branch_rows,
        verification_rows,
        hidden_store,
        api_rows,
        api_branch_rows,
    )


def collect_dataset(cfg: DatasetConfig, *, run_analysis: bool = False) -> Path:
    cfg.ensure_dirs()
    out = cfg.output_dir
    ckpt_path = out / "checkpoints" / "done_problems.json"
    done = _load_checkpoint(ckpt_path) if cfg.resume else set()

    teacher_cfg = TeacherConfig.from_env(cache_path=out / "api_cache.jsonl")
    teacher_cfg.base_url = cfg.teacher_base_url
    teacher_cfg.model = cfg.teacher_model
    teacher_cfg.enabled = cfg.use_api_teacher and teacher_cfg.enabled
    teacher = TeacherClient(teacher_cfg)
    if not teacher_cfg.api_key:
        print("WARN: TEACHER_API_KEY not set — using heuristic fallback for API scores")

    print("Loading models...")
    draft_model, tokenizer = load_model_and_tokenizer(cfg.draft_model, cfg.device, cfg.dtype)
    target_model, _ = load_model_and_tokenizer(cfg.target_model, cfg.device, cfg.dtype)

    problems: list[dict] = []
    for ds in cfg.datasets:
        limit = cfg.max_problems_per_dataset.get(ds)
        problems.extend(load_problems(ds, limit=limit))

    rng = random.Random(42)
    hidden_store: dict[str, torch.Tensor] = {}
    if (out / "hidden.safetensors").exists() and cfg.resume:
        from safetensors.torch import load_file

        hidden_store = dict(load_file(str(out / "hidden.safetensors")))

    for prob in tqdm(problems, desc="problems"):
        if prob["problem_id"] in done:
            continue

        prompt = build_prompt(prob["question"])
        trace = generate_with_trace(
            draft_model,
            tokenizer,
            prompt,
            max_new_tokens=cfg.max_new_tokens,
            device=cfg.device,
            topk_save=cfg.topk_save,
        )
        final_answer, is_correct = score_correctness(trace["response_text"], prob["gold_answer"])

        _append_rows(
            out / "traces.jsonl",
            [
                {
                    "problem_id": prob["problem_id"],
                    "dataset": prob["dataset"],
                    "question": prob["question"],
                    "gold_answer": prob["gold_answer"],
                    "model_name": cfg.draft_model,
                    "decoding_config": json.dumps({"greedy": True, "max_new_tokens": cfg.max_new_tokens}),
                    "full_reasoning": trace["response_text"],
                    "final_answer": final_answer,
                    "is_correct": int(is_correct),
                    "token_ids": json.dumps(trace["token_ids"]),
                }
            ],
        )

        seed = abs(hash(prob["problem_id"])) % 10_000
        candidates = extract_candidate_pool(
            trace["response_text"], trace["token_texts"], trace["token_trace"], seed=seed
        )

        scored: list[ScoredPrefix] = []
        api_annotation_rows: list[dict] = []
        for i, cand in enumerate(candidates):
            prefix_id = f"{prob['problem_id']}_c{i:02d}_{_stable_id(cand.prefix_type, str(cand.token_index))}"
            prefix_full = prompt + cand.prefix_text
            tf = _prefix_token_features(draft_model, target_model, tokenizer, prefix_full, cfg.topk_save)

            api_row: dict[str, Any] = {}
            if cfg.use_api_teacher:
                try:
                    api_row = teacher.annotate_prefix_only(
                        prefix_id=prefix_id,
                        question=prob["question"],
                        prefix_tail=cand.prefix_text,
                        prefix_type=cand.prefix_type,
                        progress=cand.reasoning_progress,
                    )
                except Exception as exc:
                    api_row = {
                        "prefix_id": prefix_id,
                        "annotation_mode": "prefix_only",
                        "api_error": str(exc),
                        "branch_worthiness": 0.0,
                        "rollback_risk": 0.0,
                        "decision_point_score": 0.0,
                    }
                api_annotation_rows.append(api_row)

            scored.append(
                ScoredPrefix(
                    candidate=cand,
                    prefix_id=prefix_id,
                    entropy=tf["entropy"],
                    margin=tf["margin"],
                    branch_worthiness=float(api_row.get("branch_worthiness", tf["entropy"])),
                    rollback_risk=float(api_row.get("rollback_risk", tf["draft_target_kl"])),
                    decision_point_score=float(api_row.get("decision_point_score", 0.0)),
                )
            )

        selected = select_prefixes_for_rollout(
            scored,
            top_branch=cfg.api_top_branch,
            top_rollback=cfg.api_top_rollback,
            max_wait_but=cfg.max_wait_but_selected,
            max_paragraph=cfg.max_paragraph_selected,
            n_random=cfg.n_random_control,
            n_low_control=cfg.n_low_score_control,
            rng=random.Random(seed),
        )
        selected_ids = {sp.prefix_id for sp in selected}

        # Save all candidate prefixes (metadata only for non-selected)
        all_prefix_rows: list[dict] = []
        for sp in scored:
            cand = sp.candidate
            all_prefix_rows.append(
                {
                    "prefix_id": sp.prefix_id,
                    "problem_id": prob["problem_id"],
                    "prefix_type": cand.prefix_type,
                    "token_index": cand.token_index,
                    "step_index": cand.step_index,
                    "prefix_text": prompt + cand.prefix_text,
                    "local_window_before": cand.local_window_before,
                    "local_window_after": cand.local_window_after,
                    "reasoning_progress": cand.reasoning_progress,
                    "selected_for_rollout": int(sp.prefix_id in selected_ids),
                    "selection_reason": sp.selection_reason if sp.selected else "",
                    "api_branch_worthiness": sp.branch_worthiness,
                    "api_rollback_risk": sp.rollback_risk,
                    "api_decision_point_score": sp.decision_point_score,
                    "entropy_at_cut": sp.entropy,
                }
            )
        _append_rows(out / "prefixes.jsonl", all_prefix_rows)
        _append_rows(out / "api_annotations.jsonl", api_annotation_rows)

        for sp in selected:
            (
                tf_rows,
                tb_rows,
                sb_rows,
                ver_rows,
                hs,
                trace_api,
                branch_api,
            ) = _process_prefix_rollout(
                sp,
                prob=prob,
                prompt=prompt,
                draft_model=draft_model,
                target_model=target_model,
                tokenizer=tokenizer,
                cfg=cfg,
                rng=rng,
                teacher=teacher,
                full_reasoning=trace["response_text"],
            )
            _append_rows(out / "token_features.jsonl", tf_rows)
            _append_rows(out / "token_branches.jsonl", tb_rows)
            _append_rows(out / "step_branches.jsonl", sb_rows)
            _append_rows(out / "verification_logs.jsonl", ver_rows)
            _append_rows(out / "api_annotations.jsonl", trace_api + branch_api)
            hidden_store.update(hs)

        flush_hidden_store(out / "hidden.safetensors", hidden_store)
        done.add(prob["problem_id"])
        _save_checkpoint(ckpt_path, done)

    # Final parquet export from jsonl
    _export_parquet_from_jsonl(out)
    print(f"Dataset written to {out}")
    return out


def _export_parquet_from_jsonl(out: Path) -> None:
    from reasoning_branch_dataset.io_utils import save_table
    import json

    for name in (
        "traces",
        "prefixes",
        "token_features",
        "token_branches",
        "step_branches",
        "verification_logs",
        "api_annotations",
    ):
        jsonl = out / f"{name}.jsonl"
        if jsonl.exists() and jsonl.stat().st_size > 0:
            rows = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
            save_table(rows, out / f"{name}.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Reasoning Branch/Rollback dataset")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--math500-limit", type=int, default=50)
    parser.add_argument("--aime-limit", type=int, default=30)
    parser.add_argument("--no-analysis", action="store_true")
    parser.add_argument("--no-api", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    cfg = DatasetConfig()
    if args.output_dir:
        cfg.output_dir = Path(args.output_dir)
    cfg.max_problems_per_dataset = {"math500": args.math500_limit, "aime": args.aime_limit}
    cfg.use_api_teacher = not args.no_api
    cfg.resume = not args.no_resume
    if args.math500_limit == 0:
        cfg.datasets = [d for d in cfg.datasets if d != "math500"]
    if args.aime_limit == 0:
        cfg.datasets = [d for d in cfg.datasets if d != "aime"]
    collect_dataset(cfg, run_analysis=not args.no_analysis)


if __name__ == "__main__":
    main()

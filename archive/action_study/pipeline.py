"""Phase-1 uncertainty study pipeline (small model only).

Research question:
  Can reasoning prefixes be classified into Stable / Future-diverse / Current-unreliable,
  and can hidden/logits predict these states?

Operations:
  - Continue-full: greedy continuation to final answer
  - Branch: sample K next steps, cluster, continue each to answer

Rollback is NOT a primary action. Optional counterfactual_regeneration diagnostic only.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm

from reasoning_branch_dataset.action_study.actions import (
    action_rows_from_results,
    run_branch,
    run_continue,
    run_rollback,
)
from reasoning_branch_dataset.action_study.admission import prefix_admission_pass, trace_is_complete
from reasoning_branch_dataset.action_study.api_validity import ValidityClient
from reasoning_branch_dataset.action_study.config import ActionStudyConfig
from reasoning_branch_dataset.action_study.diversity import (
    behavior_state,
    compute_diversity,
    future_system_action,
    state_bucket,
)
from reasoning_branch_dataset.action_study.prefix_substantiveness import prefix_substantiveness
from reasoning_branch_dataset.action_study.hidden import DraftHiddenExporter
from reasoning_branch_dataset.action_study.prefix_extract import extract_study_prefixes
from reasoning_branch_dataset.action_study.vllm_backend import build_engine
from reasoning_branch_dataset.datasets import load_problems
from reasoning_branch_dataset.grading import classify_generation_outcome, has_boxed_answer
from reasoning_branch_dataset.io_utils import save_table
from reasoning_branch_dataset.model_utils import build_prompt


def _load_done(path: Path) -> set[str]:
    return set(json.loads(path.read_text())) if path.exists() else set()


def _save_done(path: Path, done: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(done)))


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _needs_review(val: dict, div: dict, cont: dict, branch: dict) -> str | None:
    status = val.get("prefix_validity", "UNCLEAR")
    b_pass = branch.get("pass_at_k", 0)
    c_ok = cont.get("is_correct", 0)
    if status == "INVALID" and b_pass and not c_ok:
        return "current_unreliable_but_branch_oracle_wins"
    if status == "VALID" and div.get("diversity_label") == "HIGH_DIVERSITY" and b_pass and not c_ok:
        return "future_diverse_branch_rescues"
    if status == "VALID" and div.get("diversity_label") == "LOW_DIVERSITY" and b_pass and not c_ok:
        return "stable_branch_rescues"
    if val.get("confidence", 1.0) < 0.5:
        return "low_confidence_validity"
    return None


def _outcome_row(
    *,
    prob: dict,
    pfx_id: str,
    state: str,
    result: dict,
) -> dict:
    action = result["action"]
    oracle = result.get("pass_at_k", result.get("is_correct"))
    draft_tokens = result.get("num_tokens", result.get("total_tokens", 0))
    row = {
        "problem_id": prob["problem_id"],
        "prefix_id": pfx_id,
        "state_bucket": state,
        "behavior_state": result.get("behavior_state", state),
        "future_system_action": future_system_action(result.get("behavior_state", state)),
        "operation": action,
        "oracle_recoverable": oracle,
        "draft_generated_tokens": draft_tokens,
        "evaluation_status": result.get("evaluation_status", "OK"),
        "pass_at_k": oracle,
        "is_correct": result.get("is_correct", oracle),
        "num_tokens": draft_tokens,
        "debug_latency_sec": result.get("latency_sec"),
    }
    if action == "branch":
        row.update(
            {
                "branch_correct_count": result.get("branch_correct_count"),
                "branch_accuracy_at_k": result.get("branch_accuracy_at_k"),
                "branch_evaluated_count": result.get("branch_evaluated_count"),
                "branch_evaluation_errors": result.get("branch_evaluation_errors"),
                "recovery_profile": result.get("recovery_profile"),
            }
        )
    return row


def _counterfactual_row(
    *,
    prob: dict,
    pfx_id: str,
    state: str,
    result: dict,
    current_continue: dict,
) -> dict:
    cf_oracle = result.get("pass_at_k")
    cur_oracle = current_continue.get("is_correct")
    delta = None
    if cf_oracle is not None and cur_oracle is not None:
        delta = int(cf_oracle) - int(cur_oracle)
    return {
        "problem_id": prob["problem_id"],
        "prefix_id": pfx_id,
        "state_bucket": state,
        "regeneration_start": "previous_checkpoint",
        "oracle_recoverable": cf_oracle,
        "continue_oracle_at_current": cur_oracle,
        "delta_contamination": delta,
        "discarded_prefix_tokens": result.get("discarded_prefix_tokens", result.get("wasted_tokens", 0)),
        "draft_generated_tokens": result.get("total_tokens", 0),
        "debug_latency_sec": result.get("latency_sec"),
    }


def collect(cfg: ActionStudyConfig) -> Path:
    cfg.ensure_dirs()
    out = cfg.output_dir
    ckpt = out / "checkpoints" / "done_problems.json"
    done = _load_done(ckpt) if cfg.resume else set()

    api_client = ValidityClient.from_env(cache_path=out / "api_cache.jsonl")
    engine = build_engine(cfg)
    hidden_exporter = None
    if cfg.engine == "hf" and (cfg.save_draft_hidden or cfg.save_prefix_logits):
        hidden_exporter = DraftHiddenExporter(
            cfg.model_path,
            cfg.hidden_layers,
            topk_logits=cfg.topk_logits,
        )

    problems = []
    for ds in cfg.datasets:
        problems.extend(
            load_problems(
                ds,
                limit=cfg.max_problems.get(ds),
                offset=cfg.dataset_offsets.get(ds, 0),
            )
        )

    problem_rows = []
    for prob in tqdm(problems, desc="uncertainty_study"):
        if prob["problem_id"] in done:
            continue

        visual_meta = {
            "requires_visual_input": prob.get("requires_visual_input", False),
            "input_complete": prob.get("input_complete", True),
            "exclusion_reason": prob.get("exclusion_reason"),
        }
        problem_row = {
            "problem_id": prob["problem_id"],
            "dataset": prob["dataset"],
            "question": prob["question"],
            "gold_answer": prob["gold_answer"],
            **visual_meta,
        }
        problem_rows.append(problem_row)

        if not prob.get("input_complete", True):
            _append_jsonl(
                out / "excluded_problems.jsonl",
                [
                    {
                        **problem_row,
                        "excluded": True,
                    }
                ],
            )
            _append_jsonl(out / "problems.jsonl", [problem_row])
            done.add(prob["problem_id"])
            _save_done(ckpt, done)
            continue

        prompt = build_prompt(prob["question"])
        trace = engine.generate_trace(prompt, max_tokens=cfg.max_new_tokens)
        score = classify_generation_outcome(
            trace["response_text"],
            prob["gold_answer"],
            finish_reason=trace.get("finish_reason"),
            require_marker=False,
        )
        pred_answer = score["predicted_answer"]
        is_correct = score["is_correct"]
        trace_complete = trace_is_complete(
            {
                "evaluation_status": score.get("evaluation_status"),
                "finish_reason": trace.get("finish_reason"),
                "has_final_answer": score.get("evaluation_status") == "OK",
            }
        )

        trace_row = {
            "problem_id": prob["problem_id"],
            "dataset": prob["dataset"],
            "question": prob["question"],
            "gold_answer": prob["gold_answer"],
            "full_reasoning": trace["response_text"],
            "predicted_answer": pred_answer,
            "is_correct": is_correct,
            "evaluation_status": score.get("evaluation_status", "OK"),
            "evaluation_error": score.get("evaluation_error"),
            "finish_reason": trace.get("finish_reason"),
            "has_boxed_answer": score.get("has_boxed_answer", has_boxed_answer(trace["response_text"])),
            "has_final_answer": score.get("evaluation_status") == "OK",
            "is_truncated": trace.get("finish_reason") == "length"
            or score.get("evaluation_status") == "TRUNCATED",
            "generated_tokens": trace["num_tokens"],
            "token_ids": json.dumps(trace.get("token_ids", [])),
            "token_count": trace["num_tokens"],
            "generation_latency": trace["latency_sec"],
            "needs_retry": not trace_complete,
        }
        _append_jsonl(out / "traces.jsonl", [trace_row])

        if not trace_complete:
            _append_jsonl(out / "problems.jsonl", [problem_row])
            done.add(prob["problem_id"])
            _save_done(ckpt, done)
            continue

        prefixes = extract_study_prefixes(
            trace["response_text"],
            prob["problem_id"],
            max_markers=cfg.max_marker_prefixes,
            max_paragraphs=cfg.max_paragraph_prefixes,
            trace_complete=True,
        )

        prefix_rows = []
        next_step_rows = []
        validity_rows = []
        cluster_rows = []
        outcome_rows = []
        action_detail_rows = []
        counterfactual_rows = []
        review_rows = []

        for pfx in prefixes:
            full_prefix = prompt + pfx.reasoning_prefix

            next_steps = engine.generate_next_steps(
                full_prefix,
                k=cfg.branch_k,
                max_tokens=cfg.next_step_max_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
            )
            step_texts = [s["text"] for s in next_steps]
            for i, ns in enumerate(next_steps):
                next_step_rows.append(
                    {
                        "prefix_id": pfx.prefix_id,
                        "sample_id": i,
                        "text": ns["text"],
                        "num_tokens": ns["num_tokens"],
                        "latency_sec": ns["latency_sec"],
                    }
                )

            api_clusters: list[int] | None = None
            api_num_clusters: int | None = None
            multiple_genuine: bool | None = None
            if cfg.use_api_clustering and api_client.enabled:
                cluster_resp = api_client.cluster_next_steps(
                    prefix_id=pfx.prefix_id,
                    question=prob["question"],
                    reasoning_prefix=pfx.reasoning_prefix,
                    next_steps=step_texts,
                )
                cluster_rows.append(cluster_resp)
                if cluster_resp.get("clusters"):
                    api_clusters = cluster_resp["clusters"]
                    api_num_clusters = cluster_resp.get("num_semantic_clusters")
                    multiple_genuine = cluster_resp.get("multiple_genuine_strategies")

            div = compute_diversity(
                step_texts,
                api_clusters=api_clusters,
                api_num_clusters=api_num_clusters,
                multiple_genuine=multiple_genuine,
            )

            if cfg.use_api_validity:
                val = api_client.label_prefix(
                    prefix_id=pfx.prefix_id,
                    question=prob["question"],
                    gold_answer=prob["gold_answer"],
                    reasoning_prefix=pfx.reasoning_prefix,
                )
            else:
                val = {
                    "prefix_id": pfx.prefix_id,
                    "prefix_validity": "UNCLEAR",
                    "prefix_status": "UNCLEAR",
                    "prefix_substantiveness": "SUBSTANTIVE",
                    "confidence": 0.0,
                }
            validity_rows.append(val)

            subst = prefix_substantiveness(
                full_prefix,
                api_label=val.get("prefix_substantiveness"),
            )

            state = state_bucket(val["prefix_validity"], div["diversity_label"])

            logits_feat: dict = {}
            if hidden_exporter is not None:
                logits_feat = hidden_exporter.export(pfx.prefix_id, full_prefix, pfx.step_index)

            cont = run_continue(
                engine,
                prompt,
                pfx,
                prob["gold_answer"],
                max_tokens=cfg.continuation_max_tokens,
                retry_max_tokens=cfg.continuation_retry_tokens,
            )
            branch = run_branch(
                engine,
                prompt,
                pfx,
                prob["gold_answer"],
                k=cfg.branch_k,
                max_tokens=cfg.continuation_max_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                retry_max_tokens=cfg.continuation_retry_tokens,
            )

            admitted, admission_reason = prefix_admission_pass(
                problem_row=problem_row,
                trace_row=trace_row,
                continue_result=cont,
                branch_result=branch,
            )

            b_state = behavior_state(
                prefix_validity=val["prefix_validity"],
                prefix_substantiveness=subst,
                strategy_diversity=div["strategy_diversity"],
                recovery_profile=branch.get("recovery_profile") or "UNKNOWN",
                continue_correct=cont.get("is_correct"),
                branch_pass_at_k=branch.get("pass_at_k"),
            )
            cont["behavior_state"] = b_state
            branch["behavior_state"] = b_state

            prefix_rows.append(
                {
                    "problem_id": prob["problem_id"],
                    "prefix_id": pfx.prefix_id,
                    "prefix_type": pfx.prefix_type,
                    "prefix_text": full_prefix,
                    "previous_checkpoint": pfx.previous_checkpoint,
                    "reasoning_progress": pfx.reasoning_progress,
                    "prefix_validity": val["prefix_validity"],
                    "prefix_status": val.get("prefix_status", val["prefix_validity"]),
                    "prefix_substantiveness": subst,
                    "include_in_main_experiment": subst == "SUBSTANTIVE" and admitted,
                    "admission_pass": admitted,
                    "admission_reason": admission_reason,
                    "error_type": val.get("error_type", "none"),
                    "validity_confidence": val.get("confidence", 0.0),
                    "strategy_diversity": div["strategy_diversity"],
                    "diversity_label": div["diversity_label"],
                    "diversity_entropy": div["diversity_entropy"],
                    "num_clusters": div["num_clusters"],
                    "multiple_genuine_strategies": div.get("multiple_genuine_strategies", False),
                    "cluster_source": div.get("cluster_source", "heuristic_conservative"),
                    "state_bucket": state,
                    "behavior_state": b_state,
                    "recovery_profile": branch.get("recovery_profile"),
                    "future_system_action": future_system_action(b_state),
                    "entropy": logits_feat.get("entropy", 0.0),
                    "top1_prob": logits_feat.get("top1_prob", 0.0),
                    "top2_prob": logits_feat.get("top2_prob", 0.0),
                    "margin": logits_feat.get("margin", 0.0),
                    "topk_token_ids": logits_feat.get("topk_token_ids", "[]"),
                    "topk_probs": logits_feat.get("topk_probs", "[]"),
                }
            )

            outcome_rows.append(_outcome_row(prob=prob, pfx_id=pfx.prefix_id, state=state, result=cont))
            outcome_rows.append(_outcome_row(prob=prob, pfx_id=pfx.prefix_id, state=state, result=branch))

            action_detail_rows.extend(
                action_rows_from_results(
                    [cont, branch],
                    problem_id=prob["problem_id"],
                    prefix_id=pfx.prefix_id,
                    state_bucket=b_state,
                    continue_checkpoint=pfx.reasoning_prefix,
                    rollback_checkpoint="",
                    temperature=cfg.temperature,
                )
            )

            if cfg.run_counterfactual_regeneration and pfx.previous_checkpoint:
                cf = run_rollback(
                    engine,
                    prompt,
                    pfx,
                    prob["gold_answer"],
                    k=cfg.branch_k,
                    max_tokens=cfg.continuation_max_tokens,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    retry_max_tokens=cfg.continuation_retry_tokens,
                )
                if cf is not None:
                    counterfactual_rows.append(
                        _counterfactual_row(
                            prob=prob,
                            pfx_id=pfx.prefix_id,
                            state=state,
                            result=cf,
                            current_continue=cont,
                        )
                    )

            review_reason = _needs_review(val, div, cont, branch)
            if review_reason and cfg.use_api_review and api_client.enabled:
                review_rows.append(
                    api_client.review_case(
                        prefix_id=pfx.prefix_id,
                        question=prob["question"],
                        gold_answer=prob["gold_answer"],
                        prefix_status=val["prefix_validity"],
                        diversity_label=div["diversity_label"],
                        branch_pass=branch.get("pass_at_k", 0),
                        rollback_pass=0,
                        continue_correct=cont.get("is_correct", 0),
                        reason=review_reason,
                    )
                )

        _append_jsonl(out / "problems.jsonl", [problem_rows[-1]])
        _append_jsonl(out / "prefixes.jsonl", prefix_rows)
        _append_jsonl(out / "next_step_samples.jsonl", next_step_rows)
        _append_jsonl(out / "validity_labels.jsonl", validity_rows)
        _append_jsonl(out / "cluster_labels.jsonl", cluster_rows)
        _append_jsonl(out / "outcome_results.jsonl", outcome_rows)
        # Legacy alias for downstream scripts
        _append_jsonl(out / "action_results.jsonl", outcome_rows)
        _append_jsonl(out / "actions.jsonl", action_detail_rows)
        if counterfactual_rows:
            _append_jsonl(out / "counterfactual_regeneration.jsonl", counterfactual_rows)
        _append_jsonl(out / "api_reviews.jsonl", review_rows)

        if hidden_exporter is not None:
            hidden_exporter.flush(out / "hidden.safetensors")

        done.add(prob["problem_id"])
        _save_done(ckpt, done)

    _export_tables(out)
    if hidden_exporter is not None:
        hidden_exporter.unload()
    print(f"Uncertainty study data written to {out}")
    return out


def _export_tables(out: Path) -> None:
    for name in (
        "excluded_problems",
        "problems",
        "traces",
        "prefixes",
        "next_step_samples",
        "validity_labels",
        "cluster_labels",
        "outcome_results",
        "action_results",
        "actions",
        "counterfactual_regeneration",
        "api_reviews",
    ):
        jsonl = out / f"{name}.jsonl"
        if jsonl.exists() and jsonl.stat().st_size > 0:
            rows = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
            save_table(rows, out / f"{name}.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-1 uncertainty study (small model)")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--math500-limit", type=int, default=200)
    parser.add_argument("--math500-offset", type=int, default=0)
    parser.add_argument("--aime-limit", type=int, default=0)
    parser.add_argument("--deepscaler-limit", type=int, default=0)
    parser.add_argument("--deepscaler-offset", type=int, default=0)
    parser.add_argument("--gsm8k-limit", type=int, default=100)
    parser.add_argument("--engine", choices=["vllm", "hf"], default="vllm")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--no-api", action="store_true")
    parser.add_argument("--no-hidden", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--counterfactual-regeneration",
        action="store_true",
        help="Optional diagnostic: regenerate from previous checkpoint",
    )
    args = parser.parse_args()

    cfg = ActionStudyConfig()
    if args.output_dir:
        cfg.output_dir = Path(args.output_dir)
    cfg.max_problems = {
        "math500": args.math500_limit,
        "aime": args.aime_limit,
        "deepscaler": args.deepscaler_limit,
        "gsm8k": args.gsm8k_limit,
    }
    cfg.dataset_offsets = {
        "math500": args.math500_offset,
        "deepscaler": args.deepscaler_offset,
    }
    if os.environ.get("ACTION_STUDY_MAX_MARKERS"):
        cfg.max_marker_prefixes = int(os.environ["ACTION_STUDY_MAX_MARKERS"])
    if os.environ.get("ACTION_STUDY_MAX_PARAGRAPHS"):
        cfg.max_paragraph_prefixes = int(os.environ["ACTION_STUDY_MAX_PARAGRAPHS"])
    if args.math500_limit == 0:
        cfg.datasets = [d for d in cfg.datasets if d != "math500"]
    if args.aime_limit == 0:
        cfg.datasets = [d for d in cfg.datasets if d != "aime"]
    if args.deepscaler_limit == 0:
        cfg.datasets = [d for d in cfg.datasets if d != "deepscaler"]
    if args.gsm8k_limit == 0:
        cfg.datasets = [d for d in cfg.datasets if d != "gsm8k"]
    cfg.engine = args.engine
    cfg.use_api_validity = not args.no_api
    cfg.use_api_clustering = not args.no_api
    cfg.use_api_review = not args.no_api
    cfg.save_draft_hidden = not args.no_hidden
    cfg.save_prefix_logits = not args.no_hidden
    cfg.resume = not args.no_resume
    cfg.run_counterfactual_regeneration = args.counterfactual_regeneration
    if args.max_new_tokens is not None:
        cfg.max_new_tokens = args.max_new_tokens
    if os.environ.get("CONTINUATION_MAX_TOKENS"):
        cfg.continuation_max_tokens = int(os.environ["CONTINUATION_MAX_TOKENS"])
    if os.environ.get("CONTINUATION_RETRY_TOKENS"):
        cfg.continuation_retry_tokens = int(os.environ["CONTINUATION_RETRY_TOKENS"])
    collect(cfg)


if __name__ == "__main__":
    main()

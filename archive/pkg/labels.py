"""Post-process branch utility, rollback, and uncertainty labels."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import pandas as pd

from reasoning_branch_dataset.grading import extract_math_answer, math_equal


def answer_entropy(answers: list[str]) -> tuple[float, int]:
    if not answers:
        return 0.0, 0
    canon = [extract_math_answer(a) for a in answers]
    counts = Counter(canon)
    total = sum(counts.values())
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log(p + 1e-12)
    return ent, len(counts)


def compute_branch_labels(
    prefixes_df: pd.DataFrame,
    step_branches_df: pd.DataFrame,
    traces_df: pd.DataFrame,
    *,
    branch_k: int = 4,
) -> pd.DataFrame:
    trace_map = traces_df.set_index("problem_id").to_dict("index")
    rows: list[dict[str, Any]] = []

    for prefix_id, grp in step_branches_df.groupby("prefix_id"):
        meta = prefixes_df[prefixes_df["prefix_id"] == prefix_id].iloc[0]
        problem_id = meta["problem_id"]
        trace = trace_map[problem_id]
        gold = trace["gold_answer"]

        single_correct = bool(trace["is_correct"])
        branch_correct_flags = []
        branch_answers = []
        best_branch_id = None
        for _, br in grp.iterrows():
            full_text = meta["prefix_text"] + br["branch_text"]
            ans = br.get("final_answer") or extract_math_answer(full_text)
            ok = math_equal(ans, gold)
            branch_correct_flags.append(ok)
            branch_answers.append(ans)
            if ok and best_branch_id is None:
                best_branch_id = br["branch_id"]

        branch_oracle_correct = int(any(branch_correct_flags))
        branch_gain = branch_oracle_correct - int(single_correct)
        single_acc = float(single_correct)
        oracle_acc = float(branch_oracle_correct)
        cost = len(grp) / max(branch_k, 1)
        branch_utility = (oracle_acc / max(single_acc, 1e-6)) - 0.1 * cost
        branch_label = int(branch_utility > 0)

        ans_ent, n_clusters = answer_entropy(branch_answers)
        rows.append(
            {
                "prefix_id": prefix_id,
                "problem_id": problem_id,
                "single_correct": int(single_correct),
                "branch_oracle_correct": branch_oracle_correct,
                "branch_gain": branch_gain,
                "branch_utility": branch_utility,
                "branch_label": branch_label,
                "best_branch_id": best_branch_id,
                "answer_entropy": ans_ent,
                "num_answer_clusters": n_clusters,
            }
        )

    return pd.DataFrame(rows)


def compute_rollback_labels(
    verification_df: pd.DataFrame,
    *,
    accept_threshold: float = 0.25,
    reject_pos_threshold: int = 2,
    kl_threshold: float = 0.5,
) -> pd.DataFrame:
    rows = []
    for _, row in verification_df.iterrows():
        gamma = max(int(row.get("gamma", 1)), 1)
        tau = int(row.get("accepted_length", 0))
        accept_ratio = tau / gamma
        first_reject = row.get("first_reject_position")
        if pd.isna(first_reject):
            first_reject = None
        kl = float(row.get("target_draft_KL", 0.0) or 0.0)
        rows.append(
            {
                "problem_id": row["problem_id"],
                "round_id": row["round_id"],
                "prefix_id": row.get("prefix_id"),
                "rollback_accept": int(accept_ratio < accept_threshold),
                "rollback_reject": int(
                    first_reject is not None and int(first_reject) <= reject_pos_threshold
                ),
                "rollback_kl": int(kl > kl_threshold),
            }
        )
    return pd.DataFrame(rows)


def merge_labels(
    branch_labels_df: pd.DataFrame,
    rollback_labels_df: pd.DataFrame,
    token_features_df: pd.DataFrame,
) -> pd.DataFrame:
    out = branch_labels_df.copy()
    if not rollback_labels_df.empty and "prefix_id" in rollback_labels_df.columns:
        rb = (
            rollback_labels_df.dropna(subset=["prefix_id"])
            .groupby("prefix_id", as_index=False)
            .agg(
                rollback_accept=("rollback_accept", "max"),
                rollback_reject=("rollback_reject", "max"),
                rollback_kl=("rollback_kl", "max"),
            )
        )
        out = out.merge(rb, on="prefix_id", how="left")

    if not token_features_df.empty:
        tf = token_features_df[
            [
                "prefix_id",
                "entropy",
                "margin",
                "top1_prob",
                "draft_target_kl",
                "draft_target_js",
            ]
        ].copy()
        tf = tf.rename(
            columns={
                "entropy": "token_entropy",
                "margin": "token_margin",
            }
        )
        out = out.merge(tf, on="prefix_id", how="left")
    return out

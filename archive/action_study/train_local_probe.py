"""Train two-stage local probes with problem-level GroupKFold (no API)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from reasoning_branch_dataset.action_study.build_probe_dataset import LOGIT_FEATURES
from reasoning_branch_dataset.action_study.run_utility_scoring import _load_jsonl


FEATURE_KEYS = list(LOGIT_FEATURES) + [
    "has_wait",
    "has_but",
    "has_alternatively",
    "prefix_chars",
    "prefix_blocks",
]


def _standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    return (X - mu) / sigma, mu, sigma


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))


def _fit_logreg(X: np.ndarray, y: np.ndarray, *, lr: float = 0.1, steps: int = 800) -> np.ndarray:
    n, d = X.shape
    w = np.zeros(d + 1, dtype=np.float64)
    Xb = np.concatenate([np.ones((n, 1)), X], axis=1)
    pos = max(int(y.sum()), 1)
    neg = max(int((1 - y).sum()), 1)
    weight_pos = n / (2 * pos)
    weight_neg = n / (2 * neg)
    sample_w = np.where(y == 1, weight_pos, weight_neg)
    for _ in range(steps):
        p = _sigmoid(Xb @ w)
        grad = (Xb.T @ (sample_w * (p - y))) / n
        w -= lr * grad
    return w


def _predict_proba(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    Xb = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    return _sigmoid(Xb @ w)


def _auroc(y: np.ndarray, prob: np.ndarray) -> float:
    order = np.argsort(prob)
    y_sorted = y[order]
    n_pos = y_sorted.sum()
    n_neg = len(y_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = np.arange(1, len(y_sorted) + 1)
    sum_ranks_pos = ranks[y_sorted == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _pr_auc(y: np.ndarray, prob: np.ndarray) -> float:
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-prob)
    y_sorted = y[order]
    tp = 0
    fp = 0
    precisions: list[float] = []
    for yi in y_sorted:
        if yi == 1:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
    return float(np.mean(precisions))


def _group_kfold(groups: np.ndarray, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    uniq = np.unique(groups)
    rng = np.random.default_rng(0)
    rng.shuffle(uniq)
    folds = np.array_split(uniq, n_splits)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_splits):
        te_groups = set(folds[i].tolist())
        te_mask = np.array([g in te_groups for g in groups])
        tr_mask = ~te_mask
        splits.append((np.where(tr_mask)[0], np.where(te_mask)[0]))
    return splits
def _xy(rows: list[dict], label_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.array([[float(r.get(k, 0.0)) for k in FEATURE_KEYS] for r in rows], dtype=np.float64)
    y = np.array([int(r[label_key]) for r in rows], dtype=np.int64)
    groups = np.array([r["problem_id"] for r in rows])
    return X, y, groups


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    out = [[0, 0], [0, 0]]
    for yt, yp in zip(y_true, y_pred):
        out[int(yt)][int(yp)] += 1
    return out


def _eval_fold(y_true: np.ndarray, y_prob: np.ndarray, *, positive: int = 1) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    out: dict[str, float] = {}
    out["auroc"] = _auroc(y_true, y_prob)
    out["pr_auc"] = _pr_auc(y_true, y_prob)
    pos = y_true == positive
    out["recall_pos"] = float((y_pred[pos] == positive).mean()) if pos.any() else float("nan")
    neg = y_true != positive
    out["false_pos_rate"] = float((y_pred[neg] == positive).mean()) if neg.any() else float("nan")
    return out


def train_stage(
    rows: list[dict],
    *,
    label_key: str,
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    X, y, groups = _xy(rows, label_key)
    unique_groups = np.unique(groups)
    n_splits = min(n_splits, len(unique_groups))
    fold_metrics: list[dict[str, float]] = []
    oof_prob = np.zeros(len(rows), dtype=np.float64)

    for tr_idx, te_idx in _group_kfold(groups, n_splits):
        X_tr, mu, sigma = _standardize(X[tr_idx])
        X_te = (X[te_idx] - mu) / sigma
        w = _fit_logreg(X_tr, y[tr_idx].astype(np.float64))
        prob = _predict_proba(X_te, w)
        oof_prob[te_idx] = prob
        fold_metrics.append(_eval_fold(y[te_idx], prob))

    y_pred = (oof_prob >= 0.5).astype(int)
    return {
        "n_samples": len(rows),
        "label_key": label_key,
        "class_balance": {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "fold_metrics_mean": {
            k: float(np.nanmean([m[k] for m in fold_metrics])) for k in fold_metrics[0]
        },
        "oof_confusion_matrix": _confusion(y, y_pred),
        "feature_keys": FEATURE_KEYS,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/mnt/afs/L202500372/reasoning_branch_dataset/outputs/probe_datasets")
    parser.add_argument("--out-dir", default="/mnt/afs/L202500372/reasoning_branch_dataset/outputs/probe_models")
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()
    data = Path(args.data_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    s1 = _load_jsonl(data / "stage1_continue_vs_intervention.jsonl")
    s2 = _load_jsonl(data / "stage2_branch_vs_handoff.jsonl")
    r1 = train_stage(s1, label_key="y_intervention", n_splits=args.n_splits)
    r2 = train_stage(s2, label_key="y_branch", n_splits=args.n_splits)
    summary = {"stage1": r1, "stage2": r2}
    (out / "probe_cv_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Local Probe CV Summary",
        "",
        "## Stage 1: Continue vs Intervention",
        f"- n={r1['n_samples']} | balance={r1['class_balance']}",
        f"- OOF AUROC={r1['fold_metrics_mean']['auroc']:.3f} PR-AUC={r1['fold_metrics_mean']['pr_auc']:.3f}",
        f"- intervention recall={r1['fold_metrics_mean']['recall_pos']:.3f} false-intervention rate={r1['fold_metrics_mean']['false_pos_rate']:.3f}",
        "",
        "## Stage 2: Branch vs Handoff",
        f"- n={r2['n_samples']} | balance={r2['class_balance']}",
        f"- OOF AUROC={r2['fold_metrics_mean']['auroc']:.3f} PR-AUC={r2['fold_metrics_mean']['pr_auc']:.3f}",
        f"- Branch recall={r2['fold_metrics_mean']['recall_pos']:.3f}",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Probe and exploratory analysis for the dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except ImportError:
    sns = None
from safetensors.torch import load_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def _load_hidden_matrix(
    hidden_path: Path,
    prefixes_df: pd.DataFrame,
    *,
    source: str = "draft",
    layer: int = 32,
    pool: str = "last",
) -> tuple[np.ndarray, list[str]]:
    tensors = load_file(str(hidden_path))
    prefix_ids = prefixes_df["prefix_id"].tolist()
    vecs = []
    kept = []
    for pid in prefix_ids:
        key = f"{pid}/{source}/layer{layer}/{pool}"
        if key not in tensors:
            continue
        vecs.append(tensors[key].float().numpy())
        kept.append(pid)
    if not vecs:
        return np.zeros((0, 1)), []
    return np.stack(vecs), kept


def _safe_auc(y: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, scores))


def _safe_ap(y: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, scores))


def run_probe(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    random_state: int = 0,
) -> dict[str, float]:
    if features.shape[0] < 8 or len(np.unique(labels)) < 2:
        return {"auroc": float("nan"), "pr_auc": float("nan")}
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=random_state, stratify=labels
    )
    clf = LogisticRegression(max_iter=1000)
    clf.fit(x_train, y_train)
    prob = clf.predict_proba(x_test)[:, 1]
    return {"auroc": _safe_auc(y_test, prob), "pr_auc": _safe_ap(y_test, prob)}


def analysis_prefix_type_branch_rate(labels_df: pd.DataFrame, prefixes_df: pd.DataFrame) -> pd.DataFrame:
    merged = labels_df.merge(prefixes_df[["prefix_id", "prefix_type"]], on="prefix_id", how="left")
    out = (
        merged.groupby("prefix_type", as_index=False)
        .agg(
            n=("branch_label", "count"),
            branch_rate=("branch_label", "mean"),
            mean_utility=("branch_utility", "mean"),
        )
        .sort_values("branch_rate", ascending=False)
    )
    return out


def analysis_api_enrichment(labels_df: pd.DataFrame, prefixes_df: pd.DataFrame) -> dict[str, float]:
    merged = labels_df.merge(
        prefixes_df[["prefix_id", "selection_reason", "api_branch_worthiness"]],
        on="prefix_id",
        how="left",
    )
    if merged.empty or "branch_label" not in merged.columns:
        return {"api_enrichment": float("nan"), "random_branch_rate": float("nan"), "api_top_branch_rate": float("nan")}

    random = merged[merged["selection_reason"] == "random_control"]
    api_top = merged[merged["selection_reason"] == "api_top_branch"]
    rand_rate = float(random["branch_label"].mean()) if len(random) else float("nan")
    api_rate = float(api_top["branch_label"].mean()) if len(api_top) else float("nan")
    enrichment = api_rate / max(rand_rate, 1e-6) if rand_rate == rand_rate else float("nan")
    return {
        "api_enrichment": enrichment,
        "random_branch_rate": rand_rate,
        "api_top_branch_rate": api_rate,
    }


def analysis_entropy_correlation(labels_df: pd.DataFrame) -> dict[str, float]:
    if "token_entropy" not in labels_df.columns:
        return {"corr_entropy_utility": float("nan"), "auc_entropy": float("nan")}
    sub = labels_df.dropna(subset=["token_entropy", "branch_utility"])
    if sub.empty:
        return {"corr_entropy_utility": float("nan"), "auc_entropy": float("nan")}
    corr = float(sub["token_entropy"].corr(sub["branch_utility"]))
    y = sub["branch_label"].astype(int).values
    scores = sub["token_entropy"].values
    return {
        "corr_entropy_utility": corr,
        "auc_entropy": _safe_auc(y, scores),
    }


def analysis_hidden_branch_probe(
    labels_df: pd.DataFrame,
    prefixes_df: pd.DataFrame,
    hidden_path: Path,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    label_map = labels_df.set_index("prefix_id")["branch_label"].to_dict()

    for source in ("draft", "target"):
        for layer in (16, 28, 32):
            x, kept = _load_hidden_matrix(hidden_path, prefixes_df, source=source, layer=layer, pool="last")
            if x.shape[0] == 0:
                continue
            y = np.array([label_map[pid] for pid in kept], dtype=int)
            metrics = run_probe(x, y)
            results[f"{source}_layer{layer}_last"] = metrics
    return results


def analysis_rollback_probe(
    labels_df: pd.DataFrame,
    prefixes_df: pd.DataFrame,
    hidden_path: Path,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    if "rollback_accept" not in labels_df.columns:
        return results
    sub = labels_df.dropna(subset=["rollback_accept"])
    if sub.empty:
        return results
    label_map = sub.set_index("prefix_id")["rollback_accept"].to_dict()
    pfx = prefixes_df[prefixes_df["prefix_id"].isin(sub["prefix_id"])]

    x, kept = _load_hidden_matrix(hidden_path, pfx, source="draft", layer=32, pool="last")
    if x.shape[0] == 0:
        return results
    y = np.array([label_map[pid] for pid in kept], dtype=int)
    results["hidden_draft_layer32"] = run_probe(x, y)

    if "token_entropy" in sub.columns:
        ent = sub.set_index("prefix_id")["token_entropy"].to_dict()
        scores = np.array([ent.get(pid, np.nan) for pid in kept])
        mask = ~np.isnan(scores)
        if mask.sum() > 0 and len(np.unique(y[mask])) > 1:
            results["entropy_baseline"] = {
                "auroc": _safe_auc(y[mask], scores[mask]),
                "pr_auc": _safe_ap(y[mask], scores[mask]),
            }
    return results


def save_analysis_outputs(
    output_dir: Path,
    prefix_rates: pd.DataFrame,
    entropy_stats: dict[str, float],
    branch_probe: dict[str, Any],
    rollback_probe: dict[str, Any],
    api_enrichment: dict[str, float] | None = None,
) -> Path:
    report_path = output_dir / "analysis_report.md"
    lines = ["# Reasoning Branch/Rollback Analysis Report", ""]
    lines.append("## 1. Branch rate by prefix type")
    lines.append(prefix_rates.to_markdown(index=False))
    lines.append("")
    lines.append("## 2. Entropy vs branch utility")
    for k, v in entropy_stats.items():
        lines.append(f"- {k}: {v:.4f}" if v == v else f"- {k}: nan")
    lines.append("")
    lines.append("## 3. Hidden → branch utility probe")
    for k, v in branch_probe.items():
        lines.append(f"- {k}: AUROC={v.get('auroc', float('nan')):.4f}, PR-AUC={v.get('pr_auc', float('nan')):.4f}")
    lines.append("")
    lines.append("## 4. Hidden → rollback probe")
    for k, v in rollback_probe.items():
        if isinstance(v, dict):
            lines.append(f"- {k}: AUROC={v.get('auroc', float('nan')):.4f}, PR-AUC={v.get('pr_auc', float('nan')):.4f}")
    lines.append("")
    if api_enrichment:
        lines.append("## 5. API prefix enrichment")
        for k, v in api_enrichment.items():
            lines.append(f"- {k}: {v:.4f}" if v == v else f"- {k}: nan")
    report_path.write_text("\n".join(lines))

    if not prefix_rates.empty:
        plt.figure(figsize=(8, 4))
        if sns is not None:
            sns.barplot(data=prefix_rates, x="prefix_type", y="branch_rate")
        else:
            plt.bar(prefix_rates["prefix_type"], prefix_rates["branch_rate"])
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / "figures" / "branch_rate_by_prefix_type.png", dpi=150)
        plt.close()

    summary = {
        "prefix_rates": prefix_rates,
        "entropy_stats": entropy_stats,
        "branch_probe": branch_probe,
        "rollback_probe": rollback_probe,
    }
    return report_path

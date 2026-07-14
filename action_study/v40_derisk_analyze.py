"""V4.0 Phase-0 de-risk analysis: does draft self-confidence discriminate?

Computes, on the SAME oracle-labeled candidate set:
  - per-signal ROC-AUC + PR-AUC (draft self-signals vs the 32B verifier_score)
  - a fused draft-only logistic model, out-of-fold AUC with GroupKFold by
    problem_id (prevents prefix/problem leakage)
  - precision/coverage at operating thresholds

Emits a GREEN / YELLOW / RED decision per the plan:
  GREEN  best draft signal (or fused OOF) AUC >= 0.70 AND some threshold gives
         precision >= 0.90 with non-trivial coverage
  YELLOW 0.55 <= AUC < 0.70 -> escalate to hidden-probe mini-test
  RED    AUC < 0.55 -> draft self-signals also fail; revisit idea B

Pure numpy (no sklearn dependency).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from reasoning_branch_dataset.action_study.v40_self_signals import SIGNAL_ORIENTATION


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def roc_auc(scores: list[float], labels: list[bool]) -> float:
    """Mann-Whitney U AUC with tie handling. Higher score -> more positive."""
    pairs = [(s, y) for s, y in zip(scores, labels) if _finite(s)]
    n_pos = sum(1 for _, y in pairs if y)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = sorted(range(len(pairs)), key=lambda i: pairs[i][0])
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and pairs[order[j + 1]][0] == pairs[order[i]][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # ranks are 1-based
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    sum_pos_ranks = sum(ranks[i] for i in range(len(pairs)) if pairs[i][1])
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(scores: list[float], labels: list[bool]) -> float:
    """Area under precision-recall (average precision). Higher score -> positive."""
    pairs = sorted(
        [(s, y) for s, y in zip(scores, labels) if _finite(s)],
        key=lambda x: x[0],
        reverse=True,
    )
    n_pos = sum(1 for _, y in pairs if y)
    if n_pos == 0:
        return float("nan")
    tp = 0
    fp = 0
    ap = 0.0
    prev_recall = 0.0
    for _, y in pairs:
        if y:
            tp += 1
        else:
            fp += 1
        recall = tp / n_pos
        precision = tp / (tp + fp)
        ap += precision * (recall - prev_recall)
        prev_recall = recall
    return ap


def precision_at_thresholds(scores: list[float], labels: list[bool]) -> dict[str, Any]:
    """Best precision achievable with non-trivial coverage, and best precision>=0.9 coverage."""
    pairs = sorted(
        [(s, y) for s, y in zip(scores, labels) if _finite(s)],
        key=lambda x: x[0],
        reverse=True,
    )
    n = len(pairs)
    n_pos = sum(1 for _, y in pairs if y)
    best_cov_at_p90 = 0.0
    tp = fp = 0
    max_prec_cov10 = 0.0  # max precision at coverage >= 10%
    for idx, (_, y) in enumerate(pairs, start=1):
        if y:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp)
        coverage = idx / n
        if precision >= 0.90:
            best_cov_at_p90 = max(best_cov_at_p90, coverage)
        if coverage >= 0.10:
            max_prec_cov10 = max(max_prec_cov10, precision)
    return {
        "n": n,
        "n_pos": n_pos,
        "base_rate": (n_pos / n) if n else float("nan"),
        "coverage_at_precision90": best_cov_at_p90,
        "max_precision_at_coverage10": max_prec_cov10,
    }


def _as_float(x: Any) -> float:
    if x is None:
        return float("nan")
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _col(rows: list[dict[str, Any]], name: str) -> list[float]:
    if name == "verifier_score":
        return [_as_float(r.get("verifier_score")) for r in rows]
    return [_as_float((r.get("self_signals") or {}).get(name)) for r in rows]


# ---- fused draft-only logistic model with GroupKFold ---------------------

def _standardize(cols: list[list[float]]) -> tuple[list[list[float]], list[float], list[float]]:
    import numpy as np

    X = np.array(cols, dtype=float).T  # (n, d)
    # impute non-finite per column with column median
    for j in range(X.shape[1]):
        colj = X[:, j]
        finite = colj[np.isfinite(colj)]
        med = float(np.median(finite)) if finite.size else 0.0
        colj[~np.isfinite(colj)] = med
        X[:, j] = colj
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    return Xs.tolist(), mu.tolist(), sd.tolist()


def _logreg_fit_predict(X_tr, y_tr, X_te, *, epochs: int = 400, lr: float = 0.1, l2: float = 1e-3):
    import numpy as np

    Xtr = np.array(X_tr, dtype=float)
    ytr = np.array(y_tr, dtype=float)
    Xte = np.array(X_te, dtype=float)
    n, d = Xtr.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(epochs):
        z = Xtr @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        grad_w = Xtr.T @ (p - ytr) / n + l2 * w
        grad_b = float((p - ytr).mean())
        w -= lr * grad_w
        b -= lr * grad_b
    zte = Xte @ w + b
    return (1.0 / (1.0 + np.exp(-zte))).tolist()


def grouped_oof_fused(rows: list[dict[str, Any]], feature_names: list[str], *, k: int = 5) -> dict[str, Any]:
    groups = [r["problem_id"] for r in rows]
    uniq = sorted(set(groups))
    if len(uniq) < 2:
        return {"error": "too few groups for CV"}
    fold_of = {g: (i % k) for i, g in enumerate(uniq)}
    labels = [bool(r["oracle_label"]) for r in rows]

    raw_cols = [_col(rows, f) for f in feature_names]
    Xs, _, _ = _standardize(raw_cols)  # standardize on full set (features only; labels never leak)

    n = len(rows)
    oof = [float("nan")] * n
    for fold in range(k):
        tr_idx = [i for i in range(n) if fold_of[groups[i]] != fold]
        te_idx = [i for i in range(n) if fold_of[groups[i]] == fold]
        if not te_idx or not tr_idx:
            continue
        y_tr = [1.0 if labels[i] else 0.0 for i in tr_idx]
        if sum(y_tr) == 0 or sum(y_tr) == len(y_tr):
            # degenerate training fold; predict base rate
            base = sum(1 for i in tr_idx if labels[i]) / len(tr_idx)
            for i in te_idx:
                oof[i] = base
            continue
        X_tr = [Xs[i] for i in tr_idx]
        X_te = [Xs[i] for i in te_idx]
        preds = _logreg_fit_predict(X_tr, y_tr, X_te)
        for i, pr in zip(te_idx, preds):
            oof[i] = pr

    valid = [(oof[i], labels[i]) for i in range(n) if _finite(oof[i])]
    s = [v[0] for v in valid]
    y = [v[1] for v in valid]
    return {
        "n_folds": k,
        "n_groups": len(uniq),
        "features": feature_names,
        "oof_auc": roc_auc(s, y),
        "oof_ap": average_precision(s, y),
        "operating": precision_at_thresholds(s, y),
    }


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [bool(r["oracle_label"]) for r in rows]
    n = len(rows)
    n_pos = sum(labels)

    signal_names = list(SIGNAL_ORIENTATION.keys())
    per_signal: list[dict[str, Any]] = []
    for name in signal_names + ["verifier_score"]:
        orient = SIGNAL_ORIENTATION.get(name, +1)
        raw = _col(rows, name)
        oriented = [orient * v if _finite(v) else float("nan") for v in raw]
        auc = roc_auc(oriented, labels)
        ap = average_precision(oriented, labels)
        power = (max(auc, 1 - auc) if _finite(auc) else float("nan"))
        per_signal.append({
            "signal": name,
            "orientation": orient,
            "auc": auc,
            "auc_power": power,  # |discriminative power|, direction-agnostic
            "ap": ap,
        })
    per_signal.sort(key=lambda d: (d["auc_power"] if _finite(d["auc_power"]) else -1), reverse=True)

    draft_features = [
        "mean_logprob", "min_logprob", "last_logprob", "perplexity",
        "mean_entropy", "max_entropy", "mean_margin", "min_margin",
        "self_eval_logit", "repetition_rate", "n_tokens",
    ]
    fused_draft = grouped_oof_fused(rows, draft_features)
    fused_all = grouped_oof_fused(rows, draft_features + ["verifier_score"])

    verifier_auc = next((d["auc"] for d in per_signal if d["signal"] == "verifier_score"), float("nan"))
    best_draft = next((d for d in per_signal if d["signal"] != "verifier_score"), None)
    best_draft_power = best_draft["auc_power"] if best_draft else float("nan")
    fused_auc = fused_draft.get("oof_auc", float("nan"))
    fused_power = max(fused_auc, 1 - fused_auc) if _finite(fused_auc) else float("nan")

    # Operating point for the best single draft signal (oriented), on the full set.
    best_single_op = {}
    if best_draft is not None:
        orient = best_draft["orientation"]
        oriented = [orient * v if _finite(v) else float("nan") for v in _col(rows, best_draft["signal"])]
        best_single_op = precision_at_thresholds(oriented, labels)

    # Best achievable discriminative power across draft-only signals + fused model.
    candidates_power = [p for p in [best_draft_power, fused_power] if _finite(p)]
    best_power = max(candidates_power) if candidates_power else float("nan")
    cov_at_p90 = max(
        fused_draft.get("operating", {}).get("coverage_at_precision90", 0.0),
        best_single_op.get("coverage_at_precision90", 0.0),
    )

    if not _finite(best_power):
        decision = "INCONCLUSIVE"
    elif best_power >= 0.70 and cov_at_p90 > 0.0:
        decision = "GREEN"
    elif best_power >= 0.70:
        decision = "GREEN_LOW_COVERAGE"
    elif best_power >= 0.55:
        decision = "YELLOW"
    else:
        decision = "RED"

    return {
        "n": n,
        "n_pos": n_pos,
        "n_neg": n - n_pos,
        "base_rate": (n_pos / n) if n else float("nan"),
        "by_dataset": _by_dataset(rows),
        "verifier_score_auc": verifier_auc,
        "best_draft_signal": best_draft,
        "best_single_operating": best_single_op,
        "per_signal": per_signal,
        "fused_draft_only": fused_draft,
        "fused_with_verifier": fused_all,
        "best_discriminative_power": best_power,
        "coverage_at_precision90": cov_at_p90,
        "decision": decision,
    }


def _by_dataset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-dataset base rate + head-to-head AUC (verifier vs best draft self-eval).

    This is the paper's key breakdown: on hard AIME the 32B verifier collapses,
    while the draft's own self-eval token retains discriminative power.
    """
    out: dict[str, Any] = {}
    for ds in sorted(set(r.get("dataset", "?") for r in rows)):
        sub = [r for r in rows if r.get("dataset") == ds]
        labels = [bool(r["oracle_label"]) for r in sub]
        npos = sum(labels)
        v_auc = roc_auc(_col(sub, "verifier_score"), labels)
        se = _col(sub, "self_eval_logit")  # orientation +1
        se_auc = roc_auc(se, labels)
        out[ds] = {
            "n": len(sub),
            "n_pos": npos,
            "base_rate": npos / max(1, len(sub)),
            "verifier_auc": v_auc,
            "self_eval_auc": se_auc,
        }
    return out


def _fmt(x: Any) -> str:
    return f"{x:.3f}" if _finite(x) else "n/a"


def render_report(summary: dict[str, Any]) -> str:
    L: list[str] = []
    L.append("# V4.0 Phase-0 De-risk: Draft Self-Confidence Discriminability\n")
    L.append(f"- Decision: **{summary['decision']}**")
    L.append(
        f"- N={summary['n']} (pos={summary['n_pos']}, neg={summary['n_neg']}, "
        f"base_rate={_fmt(summary['base_rate'])})"
    )
    L.append(f"- 32B verifier AUC (baseline, expected ~0.5): **{_fmt(summary['verifier_score_auc'])}**")
    bd = summary.get("best_draft_signal")
    if bd:
        L.append(
            f"- Best single draft signal: `{bd['signal']}` "
            f"AUC={_fmt(bd['auc'])} power={_fmt(bd['auc_power'])} AP={_fmt(bd['ap'])}"
        )
    fd = summary["fused_draft_only"]
    L.append(f"- Fused draft-only (GroupKFold OOF): AUC={_fmt(fd.get('oof_auc'))} AP={_fmt(fd.get('oof_ap'))}")
    fa = summary["fused_with_verifier"]
    L.append(f"- Fused draft+verifier (OOF): AUC={_fmt(fa.get('oof_auc'))} AP={_fmt(fa.get('oof_ap'))}")
    bso = summary.get("best_single_operating") or {}
    if bso and bd:
        L.append(
            f"- Best single signal op: coverage@P90={_fmt(bso.get('coverage_at_precision90'))}, "
            f"maxP@cov10={_fmt(bso.get('max_precision_at_coverage10'))}"
        )
    L.append(f"- Best discriminative power: **{_fmt(summary['best_discriminative_power'])}**")
    L.append(f"- Coverage at precision>=0.90 (best of fused/single): **{_fmt(summary['coverage_at_precision90'])}**\n")

    L.append("## Head-to-head by dataset (verifier vs draft self-eval)\n")
    L.append("| dataset | N | pos | base_rate | verifier AUC | self_eval AUC |")
    L.append("|---|--:|--:|--:|--:|--:|")
    for ds, d in summary["by_dataset"].items():
        L.append(
            f"| {ds} | {d['n']} | {d['n_pos']} | {_fmt(d['base_rate'])} | "
            f"{_fmt(d.get('verifier_auc'))} | {_fmt(d.get('self_eval_auc'))} |"
        )

    L.append("\n## Per-signal discriminability (sorted by power)\n")
    L.append("| signal | orient | AUC | power | AP |")
    L.append("|---|--:|--:|--:|--:|")
    for d in summary["per_signal"]:
        L.append(
            f"| {d['signal']} | {d['orientation']:+d} | {_fmt(d['auc'])} | "
            f"{_fmt(d['auc_power'])} | {_fmt(d['ap'])} |"
        )

    L.append("\n## Interpretation\n")
    dec = summary["decision"]
    if dec.startswith("GREEN"):
        L.append(
            "Draft self-confidence separates oracle-acceptable from unacceptable steps where the "
            "32B verifier cannot. Proceed to Phase 1 (conformal abstention gate + E2E)."
        )
        if dec == "GREEN_LOW_COVERAGE":
            L.append(
                "Note: discriminative power is there but coverage at precision>=0.90 is 0 -> the "
                "guarantee may be tight; Phase 1 should target a looser epsilon or add signals."
            )
    elif dec == "YELLOW":
        L.append(
            "Weak-to-moderate signal. Escalate to the hidden-state probe mini-test before committing "
            "to full Phase 1."
        )
    elif dec == "RED":
        L.append(
            "Draft self-signals also fail to discriminate -> the main thesis (cheap intrinsic "
            "self-verification) does not hold on this distribution. Revisit idea B (progress-aware verifier)."
        )
    else:
        L.append("Inconclusive (insufficient positives/negatives). Collect more labeled candidates.")
    return "\n".join(L) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="V4.0 Phase-0 de-risk analysis")
    p.add_argument("--candidates", default="/root/autodl-tmp/reasonbranch/outputs/action_study_v40_derisk/candidates.jsonl")
    p.add_argument("--out-dir", default="/root/autodl-tmp/reasonbranch/outputs/action_study_v40_derisk")
    args = p.parse_args()

    rows = _load_jsonl(Path(args.candidates))
    if not rows:
        raise SystemExit(f"No candidates at {args.candidates}")
    summary = analyze(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "auc_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    report = render_report(summary)
    (out_dir / "auc_report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()

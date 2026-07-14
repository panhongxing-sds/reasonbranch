"""V3.6 analysis: Exist/Safe rescue, Δ latency, profitability, three-way decision."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _median(xs: list[float]) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    n = len(ys)
    return ys[n // 2] if n % 2 else 0.5 * (ys[n // 2 - 1] + ys[n // 2])


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _p90(xs: list[float]) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    return ys[min(len(ys) - 1, int(math.ceil(0.9 * len(ys)) - 1))]


def cluster_bootstrap_mean(
    pairs: list[tuple[str, float]],
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, float | None]:
    """Problem-clustered bootstrap for E[x]. pairs=(problem_id, value)."""
    if not pairs:
        return {"mean": None, "ci_low": None, "ci_high": None}
    by: dict[str, list[float]] = defaultdict(list)
    for pid, v in pairs:
        by[pid].append(v)
    keys = list(by.keys())
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        sample_keys = [keys[rng.randrange(len(keys))] for _ in keys]
        vals = []
        for k in sample_keys:
            vals.extend(by[k])
        boots.append(sum(vals) / len(vals))
    boots.sort()
    return {
        "mean": sum(v for _, v in pairs) / len(pairs),
        "ci_low": boots[int(0.025 * n_boot)],
        "ci_high": boots[int(0.975 * n_boot)],
    }


def compute_rescue_flags(row: dict[str, Any], k: int) -> dict[str, Any]:
    """From a trial row with oracle labels + verifier scores.

    None labels mean unknown (do not treat as False).
    """
    labels = list(row.get("branch_oracle_labels") or [])
    scores = list(row.get("branch_verifier_scores") or [])
    tau = float(row.get("tau_accept", 0.0))
    while len(labels) < k:
        labels.append(None)
    labels = labels[:k]
    scores = (scores + [-1e9] * k)[:k]
    known = [x for x in labels if x is not None]
    exist = any(bool(x) for x in known) if known else False
    if not scores:
        return {
            "exist": exist,
            "accepted": False,
            "safe": False,
            "selected_index": None,
            "selector_gap_unit": float(exist),
            "oracle_known": bool(known),
        }
    k_star = max(range(k), key=lambda i: scores[i])
    accepted = scores[k_star] >= tau
    sel = labels[k_star]
    safe = bool(accepted and sel is True)
    return {
        "exist": exist,
        "accepted": accepted,
        "safe": safe,
        "selected_index": k_star,
        "selector_gap_unit": float(exist) - float(safe),
        "oracle_known": bool(known),
    }


def analyze_trials(rows: list[dict[str, Any]], *, ks: tuple[int, ...] = (1, 2, 4)) -> dict[str, Any]:
    """Aggregate V3.6 pilot/full trial jsonl."""
    def _is_num(x: Any) -> bool:
        return isinstance(x, (int, float)) and x == x  # not None, not NaN

    by_k: dict[str, dict[str, Any]] = {}
    for k in ks:
        key = str(k)
        deltas = []
        delta_pairs = []
        exist_r = []
        accept_r = []
        safe_r = []
        profitable = []
        t_h = []
        t_b = []
        t_b_success = []
        t_b_fail = []
        # 4-quadrant (branch_selected_acceptable × handoff_acceptable)
        quad = {"bok_hok": 0, "bbad_hok": 0, "bok_hbad": 0, "bbad_hbad": 0, "known": 0}
        for r in rows:
            th = r.get("handoff_wall_sec")
            tb = (r.get("branch_pipeline_sec") or {}).get(key)
            if th is None or tb is None:
                continue
            d = float(th) - float(tb)
            deltas.append(d)
            delta_pairs.append((r["problem_id"], d))
            t_h.append(float(th))
            t_b.append(float(tb))
            flags = (r.get("rescue") or {}).get(key) or {}
            if not flags and r.get("branch_oracle_labels") is not None:
                tmp = {
                    "branch_oracle_labels": (r.get("branch_oracle_labels") or [])[:k],
                    "branch_verifier_scores": (r.get("branch_verifier_scores") or [])[:k],
                    "tau_accept": r.get("tau_accept", 0.0),
                }
                flags = compute_rescue_flags(tmp, k)
            # Only aggregate rescue over oracle-known trials (NaN-safe).
            if _is_num(flags.get("exist")):
                exist_r.append(float(flags["exist"]))
            if _is_num(flags.get("safe")):
                safe_r.append(float(flags["safe"]))
            accept_r.append(float(bool(flags.get("accepted"))))
            gamma = max(0.05, 0.05 * float(th))
            prof = _is_num(flags.get("safe")) and float(flags["safe"]) >= 0.5 and d > gamma
            profitable.append(float(prof))
            # 4-quadrant using selected branch label vs handoff label
            sel_lab = flags.get("selected_oracle_label")
            h_lab = r.get("handoff_oracle_label")
            if sel_lab is not None and h_lab is not None:
                quad["known"] += 1
                bok, hok = bool(sel_lab), bool(h_lab)
                quad["bok_hok"] += int(bok and hok)
                quad["bbad_hok"] += int((not bok) and hok)
                quad["bok_hbad"] += int(bok and (not hok))
                quad["bbad_hbad"] += int((not bok) and (not hok))
            if r.get("branch_used_fallback", {}).get(key):
                t_b_fail.append(float(tb))
            else:
                t_b_success.append(float(tb))

        boot = cluster_bootstrap_mean(delta_pairs)
        by_k[key] = {
            "n": len(deltas),
            "handoff_ms": {
                "median": _ms(_median(t_h)),
                "mean": _ms(_mean(t_h)),
                "p90": _ms(_p90(t_h)),
            },
            "branch_pipeline_ms": {
                "median": _ms(_median(t_b)),
                "mean": _ms(_mean(t_b)),
                "p90": _ms(_p90(t_b)),
                "success_median": _ms(_median(t_b_success)),
                "fail_median": _ms(_median(t_b_fail)),
            },
            "delta_ms": {
                "median": _ms(_median(deltas)),
                "mean": _ms(_mean(deltas)),
                "p_positive": (sum(1 for d in deltas if d > 0) / len(deltas)) if deltas else None,
            },
            "delta_boot": {
                "mean_ms": _ms(boot["mean"]),
                "ci_low_ms": _ms(boot["ci_low"]),
                "ci_high_ms": _ms(boot["ci_high"]),
            },
            "rescue": {
                "exist": _mean(exist_r),
                "accepted": _mean(accept_r),
                "safe": _mean(safe_r),
                "selector_gap": (_mean(exist_r) or 0) - (_mean(safe_r) or 0) if exist_r else None,
                "n_oracle_known": len(safe_r),
            },
            "quadrant": quad,
            "profitable_rate": _mean(profitable),
        }

    decision = decide_v36(by_k)
    return {"by_k": by_k, "decision": decision, "n_rows": len(rows)}


def decide_v36(by_k: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Three-way: fixed_handoff / fixed_branch / need_router."""
    candidates = []
    for k, m in by_k.items():
        dmean = (m.get("delta_boot") or {}).get("mean_ms")
        ci_lo = (m.get("delta_boot") or {}).get("ci_low_ms")
        safe = (m.get("rescue") or {}).get("safe")
        ppos = (m.get("delta_ms") or {}).get("p_positive")
        prof = m.get("profitable_rate")
        if dmean is None:
            continue
        candidates.append(
            {
                "k": int(k),
                "delta_mean_ms": dmean,
                "ci_low_ms": ci_lo,
                "safe": safe,
                "p_positive": ppos,
                "profitable_rate": prof,
            }
        )

    if not candidates:
        return {"decision": "insufficient_data", "rationale": "no trials"}

    # Fixed Branch if some K has positive CI and decent safe+profitable mass
    good = [
        c
        for c in candidates
        if (c["ci_low_ms"] is not None and c["ci_low_ms"] > 0)
        and (c["safe"] or 0) >= 0.15
        and (c["p_positive"] or 0) >= 0.6
    ]
    if good:
        best = max(good, key=lambda c: c["delta_mean_ms"])
        return {
            "decision": "fixed_branch",
            "k_star": best["k"],
            "rationale": (
                f"K={best['k']} has positive Δ CI and safe rescue; "
                "use Reject→Fixed Branch (no router)"
            ),
            "details": best,
        }

    # Heterogeneity proxy: profitable_rate mid-range with mixed signs
    mixed = [
        c
        for c in candidates
        if 0.15 <= (c["profitable_rate"] or 0) <= 0.70 and (c["p_positive"] or 0) not in (0, 1)
    ]
    if mixed and any((c["delta_mean_ms"] or 0) > 0 for c in candidates):
        return {
            "decision": "need_router",
            "rationale": (
                "some states profitable, others not — train Y_profitable router before V3.7"
            ),
            "details": mixed,
        }

    return {
        "decision": "fixed_handoff",
        "rationale": (
            "no K shows reliable positive safe Δ — Reject→Handoff; do not train Branch router"
        ),
        "details": candidates,
    }


def _ms(x: float | None) -> float | None:
    return None if x is None else 1000.0 * float(x)


def render_report(summary: dict[str, Any]) -> str:
    d = summary.get("decision") or {}
    lines = [
        "# V3.6 — One-Step Counterfactual Cost–Rescue Report",
        "",
        f"- trials: **{summary.get('n_rows', 0)}**",
        f"- decision: `{d.get('decision')}`",
        f"- rationale: {d.get('rationale')}",
        "",
        "## Profitability / Δ",
        "",
        "| K | median Δ | mean Δ (boot CI) | P(Δ>0) | P(profitable) | Safe Rescue |",
        "|--:|---------:|-----------------:|-------:|--------------:|------------:|",
    ]
    for k, m in sorted((summary.get("by_k") or {}).items(), key=lambda x: int(x[0])):
        dm = m.get("delta_ms") or {}
        db = m.get("delta_boot") or {}
        rs = m.get("rescue") or {}
        lines.append(
            "| {k} | {med} | {mean} [{lo},{hi}] | {pp} | {pr} | {sf} |".format(
                k=k,
                med=_fmt(dm.get("median")),
                mean=_fmt(db.get("mean_ms")),
                lo=_fmt(db.get("ci_low_ms")),
                hi=_fmt(db.get("ci_high_ms")),
                pp=_pct(dm.get("p_positive")),
                pr=_pct(m.get("profitable_rate")),
                sf=_pct(rs.get("safe")),
            )
        )
    lines += [
        "",
        "## Rescue decomposition",
        "",
        "| K | Exist | Accepted | Safe | Selector Gap |",
        "|--:|------:|---------:|-----:|-------------:|",
    ]
    for k, m in sorted((summary.get("by_k") or {}).items(), key=lambda x: int(x[0])):
        rs = m.get("rescue") or {}
        lines.append(
            f"| {k} | {_pct(rs.get('exist'))} | {_pct(rs.get('accepted'))} | "
            f"{_pct(rs.get('safe'))} | {_pct(rs.get('selector_gap'))} |"
        )
    lines += [
        "",
        "## Latency",
        "",
        "| K | Handoff med | Branch pipe med | Success med | Fail med |",
        "|--:|------------:|----------------:|------------:|---------:|",
    ]
    for k, m in sorted((summary.get("by_k") or {}).items(), key=lambda x: int(x[0])):
        h = m.get("handoff_ms") or {}
        b = m.get("branch_pipeline_ms") or {}
        lines.append(
            f"| {k} | {_fmt(h.get('median'))} | {_fmt(b.get('median'))} | "
            f"{_fmt(b.get('success_median'))} | {_fmt(b.get('fail_median'))} |"
        )
    lines += [
        "",
        "## Branch vs Handoff quadrant (oracle-known)",
        "",
        "| K | known | B✓H✓ (race) | B✗H✓ (handoff) | B✓H✗ (branch better) | B✗H✗ (both fail) |",
        "|--:|------:|-----------:|---------------:|---------------------:|-----------------:|",
    ]
    for k, m in sorted((summary.get("by_k") or {}).items(), key=lambda x: int(x[0])):
        q = m.get("quadrant") or {}
        lines.append(
            f"| {k} | {q.get('known', 0)} | {q.get('bok_hok', 0)} | {q.get('bbad_hok', 0)} | "
            f"{q.get('bok_hbad', 0)} | {q.get('bbad_hbad', 0)} |"
        )
    lines += [
        "",
        "## Next",
        "",
        "- `fixed_handoff` → SpecReason first version; skip Branch router",
        "- `fixed_branch` → sequential V3.7 with Fixed Branch@K*",
        "- `need_router` → train $Y^{profitable}$ then V3.7",
        "",
    ]
    return "\n".join(lines)


def _fmt(x: Any) -> str:
    if x is None or (isinstance(x, float) and x != x):  # None or NaN
        return "—"
    return f"{float(x):.0f}ms"


def _pct(x: Any) -> str:
    if x is None or (isinstance(x, float) and x != x):  # None or NaN
        return "—"
    return f"{100 * float(x):.1f}%"


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--trials", default="/root/autodl-tmp/reasonbranch/outputs/action_study_v36/trials.jsonl")
    p.add_argument("--out-dir", default="/root/autodl-tmp/reasonbranch/outputs/action_study_v36")
    args = p.parse_args()
    rows = _load_jsonl(Path(args.trials))
    summary = analyze_trials(rows)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "v36_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = render_report(summary)
    (out / "v36_report.md").write_text(md, encoding="utf-8")
    Path("/root/autodl-tmp/reasonbranch/outputs/pilot_v3_6_report.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()

"""V3.5 — Combine Experiment A+B into a cost–rescue decision report.

Compares empirical (or provisional) r_K against measured r_K^* and recommends:
  always_branch | never_branch | train_predictor
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reasoning_branch_dataset.action_study.cost_rescue import (
    V33_PROVISIONAL_RESCUE,
    decide_policy,
    expected_branch_cost,
    speedup_vs_handoff,
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_decision_report(
    *,
    latency_summary: dict[str, Any],
    rescue: dict[str, Any],
) -> dict[str, Any]:
    overall = latency_summary.get("overall") or {}
    c_t = overall.get("c_t") or 0.0
    decisions = []
    for k, r_key, star_key, cd_key, cv_key in (
        (1, "r1", "r1_star", "c_d1", "c_v1"),
        (2, "r2", "r2_star", "c_d2", "c_v2"),
        (4, "r4", "r4_star", "c_d4", "c_v4"),
    ):
        r = rescue.get(r_key)
        star = overall.get(star_key)
        d = decide_policy(r_k=r, r_k_star=star, k=k)
        cd = overall.get(cd_key)
        cv = overall.get(cv_key)
        extra: dict[str, Any] = d.to_dict()
        if r is not None and cd is not None and cv is not None and c_t:
            extra["expected_branch_cost_sec"] = expected_branch_cost(cd, cv, c_t, r)
            extra["expected_handoff_cost_sec"] = c_t
            extra["speedup_vs_handoff"] = speedup_vs_handoff(cd, cv, c_t, r)
        decisions.append(extra)

    primary = next((d for d in decisions if d["k"] == 4), decisions[-1] if decisions else {})
    return {
        "latency_overall": overall,
        "rescue": rescue,
        "decisions": decisions,
        "primary_k4": primary,
        "recommendation": primary.get("decision"),
        "rationale": primary.get("rationale"),
    }


def render_markdown(result: dict[str, Any], *, latency_path: str, rescue_path: str) -> str:
    o = result.get("latency_overall") or {}
    rescue = result.get("rescue") or {}
    primary = result.get("primary_k4") or {}
    lines = [
        "# V3.5 — Cost–Rescue Gate Decision",
        "",
        "> SpecExit-style: measure signals/costs first; set operating point by threshold,",
        "> do **not** train a Branch/Handoff action classifier until break-even requires it.",
        "",
        f"- latency summary: `{latency_path}`",
        f"- rescue rates: `{rescue_path}` (source: `{rescue.get('source')}`)",
        "",
        "## Primary recommendation (K=4)",
        "",
        f"**Decision: `{result.get('recommendation')}`**",
        "",
        f"{result.get('rationale')}",
        "",
        "## Cost–rescue table",
        "",
        "| K | $r_K$ | $r_K^*$ | margin | E[$C_{branch}$] | $C_T$ | speedup | decision |",
        "|--:|------:|--------:|-------:|----------------:|------:|--------:|----------|",
    ]
    for d in result.get("decisions") or []:
        lines.append(
            "| {k} | {rk} | {star} | {margin} | {cb} | {ct} | {sp} | `{dec}` |".format(
                k=d["k"],
                rk=_pct(d.get("r_k")),
                star=_pct(d.get("r_k_star")),
                margin=_pct(d.get("margin")),
                cb=_s(d.get("expected_branch_cost_sec")),
                ct=_s(d.get("expected_handoff_cost_sec")),
                sp=_x(d.get("speedup_vs_handoff")),
                dec=d.get("decision"),
            )
        )
    lines += [
        "",
        "## Measured costs (overall)",
        "",
        "| Metric | Value |",
        "|--------|------:|",
        f"| $C_T$ | {_s(o.get('c_t'))} |",
        f"| $C_{{D4}}$ | {_s(o.get('c_d4'))} |",
        f"| $C_{{V4}}$ | {_s(o.get('c_v4'))} |",
        f"| $(C_{{D4}}+C_{{V4}})/C_T$ | {_pct(o.get('r4_star'))} |",
        "",
        "## What to do next",
        "",
    ]
    dec = result.get("recommendation")
    if dec == "always_branch":
        lines += [
            "1. Skip Branch/Handoff router training.",
            "2. Enter clean mechanism comparison: **SpecReason vs Fixed Branch@4**.",
            "3. Focus remaining work on acceptance threshold $\\tau_{accept}$ (accuracy vs accept rate).",
            "",
        ]
    elif dec == "never_branch":
        lines += [
            "1. Do not pay Branch@4 by default.",
            "2. Re-measure with K=1/2, or improve batch verify / shared-prefix KV.",
            "3. Only then consider a selective Branch predictor.",
            "",
        ]
    else:
        lines += [
            "1. Near break-even → train a simple binary label:",
            "   $Y_{rescue}=\\mathbb{1}[\\exists k: B_k\\ \\mathrm{acceptable}]$.",
            "2. Gate with $\\tau_{branch}=r_K^*$.",
            "3. Do **not** construct sequential utility / three-way cost-aware labels yet.",
            "",
        ]
    if str(rescue.get("source", "")).startswith("v3.3"):
        lines += [
            "> Note: rescue rates are **provisional** (V3.3 / different draft stack).",
            "> Re-run Experiment B on final 1.5B+32B before locking the decision.",
            "",
        ]
    return "\n".join(lines)


def _pct(x: Any) -> str:
    if x is None:
        return "—"
    return f"{100 * float(x):.1f}%"


def _s(x: Any) -> str:
    if x is None:
        return "—"
    return f"{float(x):.3f}s"


def _x(x: Any) -> str:
    if x is None:
        return "—"
    return f"{float(x):.2f}×"


def main() -> None:
    p = argparse.ArgumentParser(description="V3.5 cost–rescue decision report")
    p.add_argument(
        "--latency-summary",
        default="/root/autodl-tmp/reasonbranch/outputs/action_study_v35_latency/latency_summary.json",
    )
    p.add_argument(
        "--rescue-rates",
        default="/root/autodl-tmp/reasonbranch/outputs/action_study_v35_rescue/rescue_rates.json",
    )
    p.add_argument(
        "--out-dir",
        default="/root/autodl-tmp/reasonbranch/outputs/action_study_v35_cost_rescue",
    )
    p.add_argument(
        "--use-provisional-rescue",
        action="store_true",
        help="If rescue_rates.json missing, use V3.3 provisional r_K",
    )
    args = p.parse_args()

    latency = _load_json(Path(args.latency_summary))
    rescue_path = Path(args.rescue_rates)
    rescue = _load_json(rescue_path)
    if not rescue and args.use_provisional_rescue:
        rescue = V33_PROVISIONAL_RESCUE.to_dict()

    if not latency:
        raise SystemExit(f"Missing latency summary: {args.latency_summary}")
    if not rescue:
        raise SystemExit(
            f"Missing rescue rates: {args.rescue_rates} "
            "(pass --use-provisional-rescue or run Experiment B)"
        )

    result = build_decision_report(latency_summary=latency, rescue=rescue)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "decision.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md = render_markdown(
        result,
        latency_path=str(args.latency_summary),
        rescue_path=str(rescue_path if rescue_path.exists() else "provisional:v3.3"),
    )
    (out_dir / "cost_rescue_decision.md").write_text(md, encoding="utf-8")
    # Also mirror under outputs/ for INDEX convenience.
    mirror = Path("/root/autodl-tmp/reasonbranch/outputs/pilot_v3_5_cost_rescue_report.md")
    mirror.write_text(md, encoding="utf-8")
    print(md)
    print(f"\n[v3.5] wrote {out_dir / 'decision.json'}")
    print(f"[v3.5] wrote {out_dir / 'cost_rescue_decision.md'}")


if __name__ == "__main__":
    main()

"""Run SD audits in order B → A → C and write combined report."""

from __future__ import annotations

import json
from pathlib import Path

from reasoning_branch_dataset.action_study.sd_audit.audit_a_budget import run_audit_a
from reasoning_branch_dataset.action_study.sd_audit.audit_b_residual import run_audit_b
from reasoning_branch_dataset.action_study.sd_audit.audit_c_profile import run_audit_c


def render_report(results: dict) -> str:
    b, a, c = results["B"], results["A"], results["C"]
    lines = [
        "# SD Audits B → A → C — 执行报告\n",
        f"执行顺序: 2(B) → 1(A) → 3(C)\n",
        "## Audit B — 请求内 residual 稳定性（方向②）\n",
        f"- 决策: **{b['decision']}**",
        f"- 记录数: {b['n_records']} ({b['n_prompts']} prompts)",
        f"- pooled test delta_top1 (EMA): **{b['pooled']['test'].get('delta_top1_ema', 0):+.3f}**",
        f"- pooled test delta_KL: **{b['pooled']['test'].get('delta_kl_ema', 0):+.3f}**",
        f"- 改善 prompt 比例: {b['n_prompts_improved_top1']}/{b['n_prompts']}\n",
        "## Audit A — 两阶段验证回本（方向①）\n",
        f"- 决策: **{a['decision']}**",
        f"- {a.get('note', '')}\n",
        "| gamma | baseline(s) | twostage(s) | savings | r_hat | pass |",
        "|--:|--:|--:|--:|--:|:--:|",
    ]
    for row in a.get("by_gamma", []):
        lines.append(
            f"| {row['gamma']} | {row['mean_baseline_sec']:.4f} | {row['mean_twostage_sec']:.4f} | "
            f"{row['mean_savings_frac']*100:.1f}% | {row['mean_r_hat']:.1f} | {row['passed']} |"
        )
    lines += [
        "\n## Audit C — LM-head profile（方向③）\n",
        f"- 决策: **{c['decision']}**",
        f"- target ρ_head (mean): **{c.get('target_mean_rho_head', 0)*100:.1f}%**",
        f"- draft ρ_head (mean): **{c.get('draft_mean_rho_head', 0)*100:.1f}%**",
        f"- target verify total: {c.get('target_mean_total_ms', 0):.1f} ms, lm_head: {c.get('target_mean_lm_head_ms', 0):.1f} ms\n",
        "## 综合建议\n",
    ]
    recs = []
    if b["decision"] == "PASS":
        recs.append("方向② **继续**：请求内 EMA residual 在 held-out 段有稳定 top-1/KL 改善。")
    else:
        recs.append("方向② **暂停**：residual 时间稳定性不足或未通过 kill gate。")
    if a["decision"] == "PASS":
        recs.append("方向① **继续**：两阶段验证有 ≥5% wall-clock 节省。")
    else:
        recs.append("方向① **封存**：额外 INT4 扫描 + 短 verify 未回本（符合 memory-bound 预期）。")
    if c.get("decision") == "CONTINUE":
        recs.append("方向③ **可探索**：target LM-head 占比 ≥10%。")
    else:
        recs.append("方向③ **封存**：target LM-head 占比 <10%，跳 head 收益不足。")
    lines.extend(recs)
    return "\n".join(lines) + "\n"


def main() -> None:
    out_dir = Path("/root/autodl-tmp/reasonbranch/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    draft = "/root/autodl-tmp/specreason/models/DeepSeek-R1-Distill-Qwen-1.5B"
    target = "/root/autodl-tmp/specreason/models/DeepSeek-R1-Distill-Qwen-32B"
    data = "/root/autodl-tmp/reasonbranch/data/gsm8k_test.jsonl"

    print("[213] Audit B ...", flush=True)
    res_b = run_audit_b(
        draft_path=draft, target_path=target, data_path=data,
        n_prompts=6, gamma=8, max_cycles=16,
    )
    (out_dir / "sd_audit_b.json").write_text(json.dumps(res_b, indent=2, default=str))
    print(f"[213] Audit B done: {res_b['decision']}", flush=True)

    print("[213] Audit A ...", flush=True)
    res_a = run_audit_a(
        draft_path=draft, approx_target_path=target, data_path=data, n_prompts=4,
    )
    (out_dir / "sd_audit_a.json").write_text(json.dumps(res_a, indent=2, default=str))
    print(f"[213] Audit A done: {res_a['decision']}", flush=True)

    print("[213] Audit C ...", flush=True)
    res_c = run_audit_c(
        draft_path=draft, target_path=target, data_path=data, n_prompts=3,
    )
    (out_dir / "sd_audit_c.json").write_text(json.dumps(res_c, indent=2, default=str))
    print(f"[213] Audit C done: {res_c['decision']}", flush=True)

    combined = {"B": res_b, "A": res_a, "C": res_c}
    (out_dir / "sd_audit_213.json").write_text(json.dumps(combined, indent=2, ensure_ascii=False))
    report = render_report(combined)
    (out_dir / "sd_audit_213_report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()

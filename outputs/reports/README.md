# 研究报告总索引（唯一入口）

**日期**: 2026-07-14  
**怎么用**: 只打开本目录。下面每份报告都是完整自洽的方法文档；旧散落文件保留作原始数据，不再作为阅读入口。

---

## 总览（先看这个）

| 优先级 | 方法 | 判决 | 报告 |
|--:|---|---|---|
| ★★★ | **★ EAGLE-3 Drafter 转向(当前主结果 2.21×)** | **系统加速已验证** | [`eagle3_drafter_pivot.md`](eagle3_drafter_pivot.md) |
| ★★ | **Method 候选:置信自适应深度** | 绿灯,未证实超越基线 | [`method_confidence_gated_adaptive_depth.md`](method_confidence_gated_adaptive_depth.md) |
| ★ | **下一步方向 + 怎么思考** | — | [`99_next_directions.md`](99_next_directions.md) |
| — | **总决策矩阵** | — | [`00_overview.md`](00_overview.md) |
| 历史 | V3.5 Cost–Rescue（Branch vs Handoff） | 工程先验：Branch 不便宜 | [`v35_cost_rescue.md`](v35_cost_rescue.md) |
| 历史 | V3.6 One-Step Verification Gap | Target-as-verifier **死** | [`v36_verification_gap.md`](v36_verification_gap.md) |
| 历史 | V4.0 Draft-Confidence Selective Speculative Reasoning | 判别力 **GREEN**；加速温和 | [`v40_draft_confidence.md`](v40_draft_confidence.md) |
| SD① | Quantized Budget Predictor | Audit **FAIL**（−100% savings） | [`sd1_quantized_budget.md`](sd1_quantized_budget.md) |
| SD② | Request-Local Residual Calibration | Audit **FAIL**（残差不稳定） | [`sd2_request_local_calib.md`](sd2_request_local_calib.md) |
| SD③ | Certified Candidate-Only Projection | Audit **KILL**（ρ_head=1.9%） | [`sd3_candidate_projection.md`](sd3_candidate_projection.md) |
| SD④ | Layerwise Verification Trajectory | Kill-gate **FAIL**（M4≈M3） | [`sd4_layerwise_trajectory.md`](sd4_layerwise_trajectory.md) |
| 路线B | Layer-Adaptive Early-Reject（compute-bound） | **KILL**(精度墙,可实现 0.4%) | [`routeB_compute_bound_precision_wall.md`](routeB_compute_bound_precision_wall.md) |

**一句话现状**: 验证侧四条切口 + 层间轨迹 + compute-bound 剪枝全部被证死(经济墙 + 精度墙);转 drafter 侧后拿到唯一确定的系统加速 **EAGLE-3 head 2.21×**(欠训,headroom 大),method novelty 待"训强 head"确认。

---

## 原始数据 / 旧报告（不必先读）

| 内容 | 路径 |
|---|---|
| V4 原始结题稿 | `../v40_report.md` |
| V4 derisk AUC | `../action_study_v40_derisk/` |
| V4 E2E | `../action_study_v40_e2e{,_tp70}/` |
| SD 三方向文献核实 | `../spec_decoding_directions_verification.md` |
| SD Audit A/B/C 原始 | `../sd_audit_213_report.md` + `../sd_audit_{a,b,c}.json` |
| LVD 原始 | `../lvd_report.md` + `../lvd_cycles.jsonl` |
| V3.6 smoke 复盘 | `../v36_smoke_review.md` |
| 更早 V2–V3.4 索引 | [`../INDEX.md`](../INDEX.md)（已改为指向本目录） |

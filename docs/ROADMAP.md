# Roadmap

Priority (SpecExit-style, one-step counterfactual first):

```
V3.6 One-Step Cost–Rescue Gate → (only if Branch wins) V3.7 sequential
```

## P0 — V3.6 One-Step Counterfactual Gate（当前）

在 **greedy 已被拒** 的固定 prefix 上，配对测：

- $T_H$: Direct Handoff（32B 生成一个完整 next step）
- $T_B^{(K)}$: Branch@K 完整 pipeline（含失败 fallback）
- $R^{\mathrm{safe}}$ vs $R^{\mathrm{exist}}$（selector gap）
- $\Delta=T_H-T_B$, $Y^{\mathrm{profitable}}$

然后在三种结局中选一：**Fixed Handoff / Fixed Branch@K / Router**。

- [x] 框架文档 [`v3_6_one_step_cost_rescue.md`](v3_6_one_step_cost_rescue.md)
- [x] Logit Accept/Reject verifier（单 token ` Accept`/` Reject`）
- [x] Step boundary 生成（`<STEP_END>` / paragraph + status）
- [x] Counterfactual timing engine（dual-resident pipeline）
- [x] Rejected-state collector + Pilot runner + analyze
- [ ] Pilot：64 rejected states × Handoff × Branch@{1,2,4}
- [ ] Calibration：$\tau_A$ via DeepSeek V4 Pro / API oracle（precision≥99%）
- [ ] 锁定三种结论之一；通过后才进 V3.7

```bash
source /root/autodl-tmp/activate_reasonbranch.sh
bash scripts/run_v3_6_pilot.sh
```

## Deprioritized / frozen

### V3.5 / V3.5a microbenchmark

- Smoke 有用：证伪了 $C_{D4}+C_{V4}\ll C_T$ 的乐观先验
- **不再**用强制 `step_max_tokens=96` 的分解成本下终局 never-branch 结论
- V3.5a 长跑已停止；正式机制结论以 **V3.6 配对 pipeline** 为准

### Earlier work

- V3.3 GPT step oracle 标签资产保留
- V3.4 sequential：污染 pilot，等 V3.6 通过后再做干净 V3.7
- Local probe / verifier 蒸馏：不阻塞 V3.6

## Do not do yet

- 训 Branch/Handoff router（除非 V3.6 显示状态异质性）
- Sequential cascade / 完整 trajectory（V3.7）
- 用旧 4B V3.3 rescue rate 当最终 1.5B 数字
- 预测 32B step 长度当作主路由信号

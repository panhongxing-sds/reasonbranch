# 00 — 方法探索总决策矩阵

日期: 2026-07-14 · 模型主线: DeepSeek-R1-Distill-Qwen 1.5B (draft) / 32B (target)

> ## ★ 当前主结果(唯一已验证、可保存的系统加速)
> 验证侧全灭后转 **drafter 侧**:为 DeepSeek-R1-Distill-Qwen-32B 训出**首个 EAGLE-3 推理 head**,
> vLLM 同引擎 **2.21× 无损加速**(46.7 vs 21.1 tok/s),且 head **严重欠训**、headroom 大。
> 详见 [`eagle3_drafter_pivot.md`](eagle3_drafter_pivot.md)。方法候选(置信自适应深度)见
> [`method_confidence_gated_adaptive_depth.md`](method_confidence_gated_adaptive_depth.md);
> 下一步见 [`99_next_directions.md`](99_next_directions.md)。

## 1. 我们探索了什么

三幕:

1. **Reasoning-step 系统（V3.5 → V4.0）**  
   决策单位是语义推理步，不是 token。问题是 Continue / Branch / Handoff / Draft-self-gate。

2. **Verification-centric Speculative Decoding（SD①–④ + 路线B）**  
   决策单位是 token 级 speculative decoding。问题是 verification 里的浪费与未用信息。**全灭**。

3. **【当前】Drafter-side 加速（EAGLE-3 转向)**  
   放弃验证侧,转而抬高单步接受率 α。产出唯一确定的系统加速 2.21×。

## 2. 判决总表（含第二轮严格重做）

| ID | 方法 | 核心假设 | 结果（重做后） | 关键诊断 |
|---|---|---|---|---|
| V3.5 | Cost–Rescue Branch | Branch@K 比 Handoff 便宜且能救援 | **FAIL** | ≈1.06×Handoff |
| V3.6 | Target-as-step-verifier | 32B 能判别 1.5B step | **FAIL** | 真实分布 AUC≈0.5 |
| **V4.0** | Draft-confidence gate | Draft 自信号替代跨模型验证 | **GREEN** | self_eval AUC 0.84 |
| SD① | Quantized budget / 自预算探针 | 廉价估计接受边界缩短 verify | **FAIL（v1 无效测试→v2 经 T1 重定向仍证伪）** | v1 用同一 32B 双跑=同义反复；T1 浅层拒绝探针精度=基率 0.21 |
| SD② | Request-local residual | 请求内前段 residual 改善后段 | **FAIL（强证伪）** | 诚实 Δ=−0.001；**oracle 上界也仅 +1.5%**→信号不存在 |
| SD③ | Candidate-only LM-head | 跳过 target 全词表投影 | **KILL（物理封死）** | ρ_head 恒 ~2%、**batch 无关**；head 大的 drafter 又用不上 |
| SD④ | Layerwise trajectory | 层间轨迹含最终 logits 外信息 | **部分翻案** | v1 因 RMSNorm/logit-lens bug 误杀；见 §3 的 T1/T2 |
| 路线B | Layer-adaptive early-reject（compute-bound） | T1 剪验证 tail 省算力 | **KILL（精度墙）** | oracle 上界 47.5%,无损可实现仅 0.4%;浅层能排序(AUC 0.86)不能无损判定首-reject |

> 重做产物：`outputs/vsignal/{tokens.jsonl, sd4_redo.json, sd2_redo.json, sd3_reframe.json, b_layeradaptive.json}`；代码 `sd_audit/{run_vsignal_collect,run_sd4_redo_analyze,run_sd2_redo,run_sd3_reframe,run_b_layeradaptive_derisk,run_adaptive_draftlen_derisk}.py`。路线B 详见 [`routeB_compute_bound_precision_wall.md`](routeB_compute_bound_precision_wall.md)。

## 3. 仍然成立的发现（论文诊断/机制节）

1. **Verification gap（V3.6）**：32B 在手工对/错上 AUC 高，在真实 1.5B 候选上崩溃（plausibility ≠ progress）。
2. **Draft self-eval 有判别力（V4.0）**：`self_eval_logit` AUC 0.84 ≫ verifier 0.67；E2E 上 selfconf 追平 target 准确率。
3. **【新】T1 — 决策解析非对称性**：11520 token 上，**拒绝在浅层就锁定（中位第 0 层），接受要到顶层（中位 0.98 深度），p≈0**。target 浅层能快速否掉 draft token，但确认接受必须跑满深度。修正了 SD④ v1 的 logit-lens bug 后才显现。
4. **【新】T2 — hard-negative 解析更晚**：draft 自信却被拒的 token（draft-confidence gate 会误接受的那批）在 target 里**解析更晚、路径更纠结**（path_speed p=5e-11）——为 verification gap 提供**表征级机制解释**（但效应小，AUC≈0.57，未超过 final margin）。

## 3.5 统一根因（为何所有 verification-side idea 都失败）

> **单请求 SD 验证是内存带宽受限的**：一次 32B forward 成本由权重加载主导，验证 8 个 token ≈ 验证 1 个。于是"验证注定被丢弃的 token"几乎免费，一切"少验证 / 更聪明地验证 / 早停 / 自适应 γ"都没有 headroom，最优平凡地是"γ 拉满"；换大 batch 变计算受限后，最优又退化成"γ=1 不投机"。两个极端都不给耍聪明的空间。

证据链：SD①（缩短精确验证=−100%）、SD③（ρ_head 恒 2%、batch 无关）、SD④/T1 early-exit（零损失⇒零节省）、自适应 γ（内存受限 0.987、计算受限 0.999，均打不过最优固定 γ）。附带一个**真但无用**的预测信号：draft entropy → 接受边界 AUC 0.60（`t1_adaptive_gamma.json`）。

**推论**：要在验证侧拿真加速，必须离开"单请求 memory-bound"设定（异构 INT4 硬件 / 真大 batch / 起草成为瓶颈的场景）。

## 4. 已严格证伪、不要再堆的方向

- Branch / 多候选救援；Target-as-step-verifier 作主方法。
- **SD①②③、early-exit、用轨迹预测下一轮 γ**：均有决定性诊断（非草率），当前 1.5B/32B 设定下无系统加速空间。
- 仅在换 regime 才可能复活：SD② 换"格式/实体强重复"workload；SD③ 换"大词表 + 中小验证器"。

## 5. 综合：现在手里能讲的最强故事

**没有一个 idea 产出了系统加速**；但严格迭代后，手里有一个**统计无懈可击的科学发现 + 一条自洽的机制叙事**：

> **《为什么小 drafter 的投机验证很难：verification gap 的表征级解释》**（分析/机制论文，非加速论文）
> - 现象：verification gap（V3.6）——大模型对小模型退化候选失去判别力。
> - 内因一：draft 自评比跨模型验证更有判别力（V4.0），但有精度天花板。
> - 内因二（新）：拒绝早/接受晚的层间非对称（T1）；draft 自信的误判恰是 target 深层才翻转的 hard-negative（T2）。
> - 边界（负结果加固"这是本质而非工程差距"）：SD①②③ + early-exit 的系统加速尝试均有决定性证伪。

**若要系统加速论文**：验证侧已穷尽,已据此转 drafter 侧(见 §6)。

## 6. 第三幕:Drafter-side 加速(当前进行中)

统一根因(§3.5)推出:**唯一能提速的杠杆是 drafter**——让每次昂贵的 target 前向多吃被接受的
token(抬高 α)。于是转 EAGLE-3。

| 里程碑 | 结果 |
|---|---|
| 阶段A:vLLM EAGLE-3 栈验证(官方 8B 推理 head) | **3.24×**(87→282 tok/s) |
| 阶段B:32B 自蒸馏数学 CoT 语料 | 1600 条 |
| 阶段C:SpecForge 训 Qwen-32B EAGLE-3 head | epoch6 收敛(train accept ~0.92);磁盘满中断但 head 完整 |
| Benchmark:自训 head vs AR(vLLM 同引擎) | **2.21×**(46.7 vs 21.1 tok/s, accept 2.36) |

**科学观察**:EAGLE head 是"短跑者"(第一步 0.63、深层崩),完整 1.5B 是"马拉松者"(20% 块吃满
γ=8)。这决定了下游"置信自适应深度"method 的可行性。

**method 候选 de-risk**:draft 置信度预测接受 AUC 0.86、校准干净(prob>0.95→accept 0.97)、
双峰长 run 存在——**绿灯**,但在欠训 head 上前提暂不成立。详见
[`method_confidence_gated_adaptive_depth.md`](method_confidence_gated_adaptive_depth.md)。

**判决**:2.21× 是唯一确定可保存的系统加速,但光训 head = 复现;method novelty 待"训强 head +
验证长 run"确认。下一步见 [`99_next_directions.md`](99_next_directions.md)。

# V3.6 Smoke 运行完整复盘（供审查）

> 目的：把 V3.6「一步反事实 Cost–Rescue Gate」两次 smoke 的**全部原始结果**、我做的**所有代码改动**、以及我发现的**遗留问题**摊开，供你判断我有没有跑错。

---

## 0. 一句话结论

- 管线**已跑通**（采集 → 配对计时 → oracle 打标 → 分析 → 报告）。
- 两次 smoke 的机制方向一致：**`fixed_handoff`**（被拒时直接交给 32B，Branch 不划算）。
- **但样本极小（2 / 4 个 state），且 oracle 全判 False**，只能当"跑通验证"，**不能当结论**。下面第 4 节列了我认为需要你定夺的疑点。

---

## 1. 这次我改了哪些代码（按问题→修复）

| # | 问题 | 现象 | 修复 |
|---|------|------|------|
| 1 | verifier 打分塌成 0 | 第一次 smoke 采到 **0 个** rejected state，`logP(Accept)-logP(Reject)` 几乎恒 ≥0 | verifier 从"读生成 token 的 top-k logprobs"改为**成对 `prompt_logprobs`**：把 ` Accept` / ` Reject` 各自拼到 prompt 末尾，分别取该位置的 logprob，保证两个标签都拿得到分数 |
| 2 | `logprobs=64` 超限 | vLLM 报 `VLLMValidationError: max allowed 20` | 改用 `prompt_logprobs=1`，不再依赖生成 top-k |
| 3 | 采集计数重复 | `len(collected)+len(done_ids)` 双重计数，采到 **2 个就提前停** | 循环条件改为只看 `len(done_ids)` |
| 4 | 采集无可观测性 | 全程只有"0 states"，无法判断是 accept 太多还是 reject 没触发 | 加了每次尝试的 `score/logA/logR/accepted` 打印 + 结尾 `accept/reject/empty/score_mean/min/max` 汇总 |
| 5 | oracle 之前是桩函数 | pilot 里 oracle 标签恒为 `None` | 接上真实 `GPTStepOracleClient.judge_shuffled_pass`（DeepSeek-V4-Pro / OpenRouter） |
| 6 | `None` 被当 False | 未知标签被 `bool(None)=False` 污染 Exist/Safe | pilot + analyze 都改为**保留 `None`=未知**，只在有已知标签时计入 Exist/Safe，并记录 `oracle_known_frac` |
| 7 | `pkill -f run_v3_6` 自杀 | 重启脚本时把脚本自己 kill 了 | 去掉 pkill，靠 GPU 空闲直接重启 |

---

## 2. smoke2（干净的一次，N=4，reps=2，seeds=1，with-oracle）

### 2.1 采集到的 4 个 rejected state

| state_id | depth | prefix_tok | greedy 打分 | logA | logR |
|---|--:|--:|--:|--:|--:|
| 67_d7_t293_a6 | 7 | 293 | −0.750 | −7.01 | −6.26 |
| 77_d1_t524_a10 | 1 | 524 | −0.750 | −7.02 | −6.27 |
| 63_d15_t486_a14 | 15 | 486 | −0.995 | −7.89 | −6.89 |
| 67_d19_t641_a18 | 19 | 641 | −0.750 | −7.01 | −6.26 |

采集统计：`attempts=18, accept=14, reject=4, empty=0, score_mean=1.80, min=−0.995, max=4.346`
→ reject 率约 4/18，被拒的分差都很小（−0.75 ~ −1.0），说明这些"被拒"其实是**边缘拒绝**，不是硬错误。

### 2.2 每个 state 的完整 trial 数据

**67_d7_t293_a6** (calibration)
- handoff = **0.538s**；branch_pipe = {K1: 0.956, K2: 1.097, K4: 1.411}
- Δ = {K1: −0.419, K2: −0.559, K4: −0.874}（全负）
- verifier_scores = [−0.82, −0.74, −0.75, −0.83]（4 条分支都被拒）
- oracle = [False, False, False, False]；used_fallback = 全 True

**77_d1_t524_a10** (test)
- handoff = **1.749s**；branch_pipe = {K1: 3.958, K2: 1.111, K4: 1.550}
- Δ = {K1: −2.209, K2: **+0.639**, K4: **+0.199**}（K2/K4 时间上更快）
- verifier_scores = [1.64, 1.19, 1.23, −0.44]（有 3 条被 verifier 接受）
- oracle = [False, False, False, False]（但 oracle 认为全不合格）；used_fallback = {K1: True, K2: False, K4: False}

**63_d15_t486_a14** (development)
- handoff = **0.320s**；branch_pipe = {K1: 0.609, K2: 0.797, K4: 1.191}
- Δ = {K1: −0.289, K2: −0.478, K4: −0.871}（全负）
- verifier_scores = [−0.70, −1.01, −0.81, −0.70]
- oracle = [False, False, False, False]

**67_d19_t641_a18** (calibration)
- handoff = **0.565s**；branch_pipe = {K1: 0.959, K2: 1.148, K4: 1.423}
- Δ = {K1: −0.394, K2: −0.584, K4: −0.859}（全负）
- verifier_scores = [−0.82, −0.74, −0.75, −0.83]
- oracle = [False, False, False, False]

### 2.3 聚合报告

| K | Δ 中位 | mean Δ (boot CI) | P(Δ>0) | P(profitable) | Safe Rescue |
|--:|--:|--:|--:|--:|--:|
| 1 | −407ms | −828ms [−2209, −348] | 0% | 0% | 0% |
| 2 | −518ms | −245ms [−572, +639] | 25% | 0% | 0% |
| 4 | −865ms | −601ms [−869, +199] | 25% | 0% | 0% |

Rescue 分解：Exist / Safe **全 0%**；Accepted（verifier 层面）25%。
延迟：Handoff 中位 **551ms** < 任何 Branch@K 管线（最快 958ms）。
决策：**`fixed_handoff`** — 不训 Branch router。

---

## 3. smoke1（第一次，N=2，仅供对照，已被 smoke2 取代）

采到 2 个 state；机制方向相同（Safe=0%，Handoff 更快）。但那次 verifier 用的是旧打分（scores 极端到 ±13），且计数 bug 导致只采 2 个，**不作数**。

---

## 4. 我发现的问题 / 需要你定夺的疑点

1. **oracle 全 False 的真实原因（重要）**
   查 oracle 缓存原文：候选被判 `correct=true, safe_to_append=true`，但 `substantive_progress=false, acceptable=false`，理由是"**只是把题目重述一遍，没有实质推理进展**"。
   → 说明不是 oracle bug，而是**1.5B 在这些被拒位置生成的分支候选质量确实差**（AIME 难题上小模型重复/复述）。这会系统性压低 Safe Rescue。

2. **`branch_steps` 没落盘（可复现性缺陷）**
   trial 行里**没有保存 oracle 实际打分的候选文本**，事后无法逐条核对 oracle 判罚是否合理。建议加上再跑正式版。

3. **"被拒"太边缘**
   4 个 state 的 greedy 分差只有 −0.75 ~ −1.0（tau=0），属于勉强被拒。真正"硬错误"的状态可能表现不同，当前采集没区分。

4. **verifier 接受 ≠ oracle 接受**
   77 这个 state：verifier 给 3 条分支正分（Accepted=True），但 oracle 判全 False → **selector 会选到 verifier 觉得好、实际不合格的分支**。样本大了这块（selector gap）值得看。

5. **样本量 & seeds**
   N=4、seeds=1，boot CI 极宽（跨 0）。任何"划算/不划算"都还不显著。

---

## 5. 相关文件

- 报告：`reasonbranch/outputs/action_study_v36_smoke2/v36_report.md`
- 汇总：`.../v36_summary.json`
- 原始 trial：`.../trials.jsonl`、状态：`.../rejected_states.jsonl`
- oracle 缓存：`reasonbranch/outputs/action_study_v36/oracle_cache.jsonl`（3 条）
- 运行日志：`reasonbranch/logs/v3_6_smoke2.log`

---

## 6. 复用/碰撞排查结论（新增，落盘 branch_steps 后交叉验证）

对 smoke3（已落盘 `branch_steps` + `branch_oracle_details`）逐条哈希核对，结论：

### 6.1 「跨 state 候选完全相同」= 可能 A（模型重复），**不是缓存 bug**
- `67_d7` 与 `67_d19`：4 条候选逐字节相同（hash `6ea046c981`），文本是
  `"To solve this problem, you need to find the values of x and y..."`——**本题是 a+b+c=300，根本没有 x/y**，是 1.5B 对退化 prefix 的默认口水话。
- 两个不同深度的 prefix 都塌成同一句 → 撞车。但 `63_d15` 的 4 条各不相同，证明**没有跨 state 复用/缓存碰撞**。
- 根因：`generate_one_step_vllm(n=4)` 在引擎 `seed=0` + 1.5B 高置信重复下，temp=0.7 仍全塌成同一序列。

### 6.2 真实 bug（必须修）
1. **`prefix_text` 未落盘** —— trial 里没存，prefix_hash 全是空串 sha1（`da39a3ee5e`），无法审计。
2. **打分候选 ≠ 计时候选**：`run_state_pilot` 顶部生成一次 `texts` 给 verifier+oracle；而 `run_branch_pipeline(branch_texts=None)` 内部**又各自独立重新生成**候选做计时。两组候选不是同一批。
3. **K 非嵌套**：每个 K、每个 rep 都在 `run_branch_pipeline` 内重采一套新候选 → K1/K2/K4 之间、rep 之间内容都不同，违背「同一 pool 做 K1⊂K2⊂K4」，且把「内容随机」与「计时噪声」混在一起。
4. **K=1 强制 temp=0**（`temperature if k>1 else 0.0`）→ K=1 与 K≥2 的候选生成分布不一致，K 对比不公平。

### 6.3 对结论的影响
- `fixed_handoff` **不成立为机制结论**：延迟对比建立在「非嵌套、每次重采、打分与计时脱节」的候选上。
- 但 `state 77` 的核心观察仍有效（Handoff 慢时 Branch 时间可回本、瓶颈是 verifier false-acceptance），因为它不依赖 K 嵌套。

### 6.4 下一步重构要点（对齐用户 11/12 节）
- 生成**一个** branch pool（固定内容、带 seed），K1⊂K2⊂K4 从同一 pool 切片；
- verifier / oracle / 计时 **全部用同一 pool**；
- 落盘：problem / prefix / prefix_hash / greedy / 4×branch(+hash) / handoff step / verifier_prompt_hash / oracle_cache_key / logA / logR；
- 两层重复：固定内容计时 ≥5 次 + 每 state 3 个 pool；
- 先做 verifier 人工 sanity set + τ 校准，再采正式 rejected states（分桶：near/medium/hard）。

---

## 7. 全部修复实施记录（对齐用户 §2–§12）

### 7.1 候选生成重构（§2 bug2/3/4、§8）— `v36_counterfactual.py` / `v36_step_gen.py` / `run_v3_6_pilot.py`
- 新增 `draft_branch_pool()`：每个 seed 只抽**一个固定 pool**（`pool_size=max(K)`），测一次 draft 成本。
- `run_branch_pipeline` 改为**接收固定 pool**（`branch_texts` 必传），按 `k` 切片 → K1⊂K2⊂K4 **严格嵌套**；draft 成本通过 `draft_sec` 参数公平计入 wall。
- `generate_one_step_vllm` 增加 `seed` 参数 → pool 采样可复现，且不再被引擎全局 seed 强制确定化。
- `run_state_pilot` 彻底重写为**两层重复**（§7）：Layer1 固定内容计时 `n_reps` 次；Layer2 抽 `n_seeds` 个独立 pool。verifier / oracle / 计时**全部用同一 pool**。
- 去掉了「K=1 强制 temp=0」的不公平分支。

### 7.2 全量原文落盘（§2、§5、§11）— `run_v3_6_pilot.py`
每条 trial 现在保存：`question` / `prefix_text` / `prefix_hash` / `greedy_step(+hash)` / `branch_steps` / `branch_hashes` / `branch_tokens` / `branch_statuses` / `branch_verifier_scores` / `branch_logp_accept` / `branch_logp_reject` / `verifier_prompt_hashes` / `branch_oracle_labels` / `branch_oracle_details`（含逐条 `acceptable/substantive_progress/correct/brief_reason`）/ `oracle_cache_key` / 每 seed 全部保留在 `seeds[]`。

### 7.3 Handoff step 也做 oracle 标注（§9）
- 新增 `maybe_oracle_label_step()`：对 32B 的 replacement step 求 `A(h)`。
- trial 保存 `handoff_oracle_label` / `handoff_oracle_details`。
- analyze 产出 **4 象限表**（B✓H✓ race / B✗H✓ handoff / B✓H✗ branch-better / B✗H✗ both-fail），实现 §9 的 `A(b*) ≥ A(h)` 局部安全判据（`rescue[k].safe_vs_handoff`）。

### 7.4 verifier scorer sanity（§4）— `run_v3_6_verifier_sanity.py`（新）
四道门全过（报告 `outputs/action_study_v36/verifier_sanity.json`）：
- Gate1 单 token：`Accept`=[20829] / `Reject`=[87293] 均 1 token ✅
- Gate2 上下文一致：两 prompt 仅末位 label 不同 ✅
- Gate3 位置正确：logprob 读在 label 位、`prompt_logprobs` 长度对齐 ✅
- Gate4 明显对错分离：**AUC=0.973**，correct_mean=2.13 vs wrong_mean=−0.38 ✅
- ⚠️ 关键发现：`wrong_max=1.69 > correct_min=1.19` → **τ=0 附近有明显错误候选被打正分**，直接证明 §3「τ=0 不足以做无损 ACCEPT」，必须校准。

### 7.5 τ 校准（§3）— `run_v3_6_calibrate_tau.py`（新）
- 独立候选集（非机制 state）→ oracle 标注 → 扫 τ，报 **precision / recall / FAR / coverage**；
- 推荐 τ = 满足 `P(A=1|v≥τ) ≥ target` 的最小 τ（默认 0.99，pilot 用 0.95）。
- 输出 `action_study_v36_calib/{candidates.jsonl, tau_sweep.json}`。

### 7.6 采集分桶（§6）— `run_v3_6_collect_states.py`
- `reject_bucket()` 按 `v−τ` 分 near `[−1,0)` / medium `[−3,−1)` / hard `<−3`；
- 三桶目标配额 + `per_problem_cap`（默认 4）保证覆盖 depth / prefix_len / problem；
- 支持 `--tau-from tau_sweep.json` 直接读校准 τ；落盘 `reject_bucket`。

### 7.7 analyze 适配（新 schema）— `v36_analyze.py`
- rescue 聚合 **NaN-safe**（oracle 未知不计入，不再污染均值）；
- `compute_rescue_flags` 保留 `None`=未知；
- 报告新增 4 象限表；`profitable` 用 `safe≥0.5 ∧ Δ>γ`。
- 单测新增 3 例（None 未知 / 全未知 / NaN-safe+象限），`14 passed`。

### 7.8 τ 校准结果（关键发现 · 48 候选）

用重构后的干净流程（固定 pool + 逐条 oracle 标注）在**独立候选集**上校准，得到一个**改变方向的结论**：

- 样本：n=48，oracle 认可 **pos=3**，否决 neg=45 → **base_rate=6.2%**。
- verifier score 分布（正例 vs 负例）：
  - 正例（oracle✓）：min=−0.14, **median=0.81**, max=2.44
  - 负例（oracle✗）：min=−1.62, **median=1.82**, max=4.30
  - → **负例中位分(1.82) > 正例中位分(0.81)**，方向都反了。
- τ 扫描：整个 τ 范围内 **最高 precision 仅 7.1%**（≈ base rate），**没有任何 τ 达到 90%**。

**结论：32B verifier 在 1.5B 真实生成候选的分布上，对「oracle 是否可接受」几乎零判别力。**
- 与 Gate4 对比：人工构造的显然对错能分（AUC 0.97），但真实退化候选分不开。
- 这彻底、量化地证实用户 §3 判断，而且比预期更严重：**V3.6 的瓶颈不是延迟，而是 verifier reliability**。
- 直接解释了 state 77 的 false acceptance：verifier 给退化候选打高分是系统性的。

⚠️ 样本仅 48、正例仅 3，绝对数字需扩样，但方向极其明确。

### 7.9 方向性影响（需用户拍板）
- 「rejected state」的定义依赖 verifier，而 verifier 现已被证无判别力 → **继续按旧 verifier 采集 states / 跑 Sanity 意义不大**。
- `fixed_handoff` 结论**已撤回**；真正待解决的是先把 verifier 做到可用（换 prompt/标签/加校验维度/或换判别方式），否则 Branch 的 selector 必然失败。
- 待用户决定：(a) 先攻 verifier reliability；(b) 仍按现状跑 10-state Sanity 看流程；(c) 扩大校准样本量确认 6% base_rate。

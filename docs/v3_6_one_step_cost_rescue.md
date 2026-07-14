# V3.6 — One-Step Counterfactual Cost–Rescue Gate

> **定义**：在相同的、greedy 已被 32B 拒绝的固定 prefix 上，分别真实执行 **Direct Handoff** 与 **Branch@K next-step pipeline**，测产生下一个完整 reasoning step 的 wall-clock，并用离线语义 oracle 判断 Branch 是否在不降质量下真正省时。  
> **不是** sequential rollout；**不**跑到最终答案。

## 与 V3.5 / V3.5a 的关系

| | V3.5 smoke / V3.5a | **V3.6** |
|--|-------------------|----------|
| 计时对象 | 分解 $C_D+C_V$ 或强制长度 | **完整 pipeline** $T_H$ vs $T_B^{(K)}$ |
| 状态 | 任意 prefix | **仅 greedy 被拒后** |
| 质量 | 无 / 延后 | **$R^{\mathrm{safe}}$**（verifier 选中 ∧ oracle 可接受）|
| 决策量 | $r_K^*$ | $\Delta=T_H-T_B$ 与 $Y^{\mathrm{profitable}}$ |

V3.5 smoke 仍有工程价值（证伪了 $C_{D4}+C_{V4}\ll C_T$ 的乐观先验），但 **不能**用它直接下 never-branch 终局结论——设置偏向短 step、且未做 one-step 反事实配对。

## 核心公式

$$
\Delta_{i,K}=T_{H,i}-T_{B,i}^{(K)}
$$

$$
Y_{i,K}^{\mathrm{profitable}}
=\mathbb{1}\big[R_{i,K}^{\mathrm{safe}}=1\ \land\ \Delta_{i,K}>\gamma\big]
$$

$$
\gamma=\max(50\mathrm{ms},\ 0.05\,T_{H,i})
$$

三种 Rescue：

| 符号 | 含义 |
|------|------|
| $R^{\mathrm{exist}}$ | $\exists k: A_{i,k}=1$（上限） |
| $R^{\mathrm{accepted}}$ | $v_{i,k^*}\ge\tau_A$ |
| $R^{\mathrm{safe}}$ | accepted ∧ $A_{i,k^*}=1$（**主指标**） |

## 配置

| 角色 | 模型 |
|------|------|
| Draft | DeepSeek-R1-Distill-Qwen-1.5B |
| Target / online verifier | DeepSeek-R1-Distill-Qwen-32B-AWQ（双常驻） |
| Offline oracle | DeepSeek V4 Pro / 兼容 OpenAI API（不计入 latency） |
| $K$ | $\{1,2,4\}$ |

要求：双模型常驻、prefix caching、warm shared-prefix 主结果、verifier **不写解释**（logit 打分）。

> Tokenizer 注意：`ACCEPT` 单 token、`REJECT` 多 token。在线打分使用单 token 对 **` Accept` / ` Reject`**（prompt 以 `Judgment:` 结尾），语义仍映射为 ACCEPT/REJECT。

## 计时起点（公平）

Sunk（两边相同，不计入）：

- 1.5B greedy generation
- 32B greedy verification

从「已知 greedy 被拒、需在 Branch vs Handoff 间选择」开始计 $T_H$ / $T_B$。

主结果：**warm shared-prefix**；cold-cache 仅作消融。

## 决策（三种结局）

| 情况 | 条件 | 动作 |
|------|------|------|
| A Fixed Handoff | 所有 $K$ 几乎无正收益 | Reject→Handoff；**不训 router** |
| B Fixed Branch@$K^*$ | 某 $K^*$ 整体 $\Delta>0$ 且 safe | Reject→Fixed Branch；**不训 router** |
| C Router | 状态异质且可预测 | 训 $Y^{\mathrm{profitable}}$ 后再进 V3.7 |

## 规模

1. **Pilot**：64 rejected states × Handoff + Branch@{1,2,4} × 3 seeds × 5 timing reps  
2. **Full**：仅当 Pilot 显示某 $K$ 有潜力时扩到 256–500

## 明确不做

- 不预测 32B step 长度；不跑完整 trajectory / cascade  
- 不直接训 Continue/Branch/Handoff 三分类  
- 不用 API latency；不用 oracle-best 代替 verifier selection  
- 不用旧 4B V3.3 $r_4$ 当最终 1.5B rescue  

## 代码入口

| 模块 | 作用 |
|------|------|
| `logit_step_verifier.py` | 单 token ` Accept`/` Reject` logit 分数 |
| `v36_step_gen.py` | `<STEP_END>` / paragraph boundary + status |
| `v36_counterfactual.py` | 配对 $T_H$ / $T_B^{(K)}$ pipeline |
| `run_v3_6_collect_states.py` | 采集 greedy-rejected states |
| `run_v3_6_pilot.py` | Pilot 主跑 |
| `v36_analyze.py` | 主结果表 + 三种结论 |
| `scripts/run_v3_6_pilot.sh` | 启动脚本 |

```bash
source /root/autodl-tmp/activate_reasonbranch.sh
# 1) 采集 rejected states（本地 verifier，无 API）
python -m reasoning_branch_dataset.action_study.run_v3_6_collect_states --n-states 64
# 2) Pilot timing（可 --skip-oracle 先只看延迟）
bash scripts/run_v3_6_pilot.sh
```

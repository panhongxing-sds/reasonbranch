# V3.5 — Cost–Rescue Gate（先测 Branch 是否划算）

> **定位**：SpecExit 式 gate experiment。  
> **先测成本门槛，再决定要不要 Branch / 要不要训 router。**  
> Smoke 已证伪最乐观先验；正式结论必须按 **K × bucket** 给出，并限定实现/长度/硬件。

## Smoke 已证明什么（需限定 scope）

在当前 smoke（N=4, `step_max_tokens=96`, 1.5B+32B-AWQ）下：

$$
C_{D4}+C_{V4}\approx 1.098\text{s} > C_T\approx 1.033\text{s}
\Rightarrow r_4^*\approx 106\%>100\%
$$

因此：

$$
\boxed{\text{当前实现 / 当前长度 / 当前硬件下，Branch@4 被 Handoff 严格支配}}
$$

即使 $r_4=100\%$，Branch@4 仍更慢。完美 router 也救不了——瓶颈是 **Branch@4 本身太贵**，不是“不知道何时 Branch”。

**不能**扩大成“Branch 永远没用”。尚未否定：Branch@1/2、更长 step、prefix-KV、轻量 verifier、特定 bucket。

## Router 何时才有价值

| 情况 | 动作 |
|------|------|
| 所有 bucket $r_K^*\ge 100\%$ 或 $r_K\ll r_K^*$ | 固定 Handoff / SpecReason，**不训 router** |
| 所有 bucket $r_K\gg r_K^*$ | 固定 Branch@K，**不训 router** |
| **状态异质**：有的 bucket 过线、有的不过 | 才训 $P(\mathrm{rescue}\mid s,K)$ vs $r_K^*(s)$ |

$$
\boxed{\text{只有动作收益具有状态异质性时，router 才真正有价值。}}
$$

## V3.5a Formal Cost Gate（当前）

```bash
source /root/autodl-tmp/activate_reasonbranch.sh
N_STATES=48 bash scripts/run_v3_5a_cost_gate.sh
```

产出：按 target-step / prefix bucket 的 $r_1^*,r_2^*,r_4^*$，外加 pipeline E2E 与 $C_T(L)$ 曲线。

## V3.5b Rescue Gate（Cost 完成后再做）

比较 $r_K^{\mathrm{select}}$ vs $r_K^*$；不要只用 V3.3 的 $r_4^{\mathrm{exist}}=44.6\%$。

## 明确不做

- 现在不进 V3.4 sequential rollout
- 现在不训 Branch/Handoff classifier
- 不用污染 V3.4 数字下机制结论

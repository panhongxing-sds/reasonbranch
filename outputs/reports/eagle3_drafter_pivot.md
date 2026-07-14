# EAGLE-3 Drafter 转向 —— 当前唯一确定的系统加速主结果

日期: 2026-07-14 · 模型: DeepSeek-R1-Distill-Qwen-32B (target) · MATH-500 · 单请求 batch=1

> **一句话**：在验证侧四条切口(SD①–④)全部被证死后,转到 **drafter 侧**。我们为
> DeepSeek-R1-Distill-Qwen-32B 训出**据我们所知首个 EAGLE-3 推理草稿 head**,vLLM 同引擎
> 下 **2.21× 无损加速**(46.7 vs 21.1 tok/s)。这是目前唯一**已验证、可作为主结果保存**的系统加速。
> 而且这个 head 是**严重欠训**的(1600 条语料 / 2048 截断 / 7 有效 epoch),headroom 很大。

---

## 0. 为什么转到 drafter 侧(动机)

前期结论(见 `00_overview.md` §3.5 统一根因):单请求 SD 验证是**内存带宽受限**,验证"注定被拒的 token"几乎免费 → 一切验证侧"耍聪明"都没有经济 headroom。因此**唯一能提速的杠杆是 drafter**:让每次昂贵的 target 前向吃下更多被接受的 token(= 提高接受长度 α)。EAGLE-3 是当前 SOTA 的 drafter 方案(训一个复用 target 多层隐状态的小 head,大幅抬高 α)。

文献侧同时确认:token-level SD 有 ~1.4× 的算法天花板(全对概率随长度指数衰减),领域前沿是抬高单步 α(EAGLE 系)与 step-level 投机。EAGLE-3 属于前者,且有成熟训练/推理栈。

---

## 1. 基线(必须先钉死,否则加速比无意义)

| 配置 | 引擎 | tok/s | TPOT | 接受长度 | speedup |
|---|---|---|---|---|---|
| target-only AR 32B | vLLM | 21.14 | 47.3 ms | 1.0 | 1.0× |
| target-only AR 32B | UMbreLLa | 18.2 | 55.0 ms | 1.0 | 1.0× |
| 动态树 SD (1.5B→32B) | UMbreLLa | 40.6 | 24.6 ms | 4.84 | 2.23× |

> UMbreLLa = 一个 tree speculative decoding 引擎(reasonv4 目录),用完整 1.5B 作 draft。
> 它的 2.23× 是我们要超越的**强参考**。

---

## 2. 三阶段实施

### 阶段 A —— 验证 vLLM EAGLE-3 栈(用官方 8B 推理 head)
目的:先证明"训 EAGLE head 能加速推理模型"这条路本身成立,再自己训。

| 配置 | tok/s | TPOT | speedup |
|---|---|---|---|
| DeepSeek-R1-Distill-8B target-only | 87.0 | 11.49 ms | 1.0× |
| + 官方 `yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B` | 282.4 | 3.54 ms | **3.24×** |

结论:EAGLE-3 在推理模型上确有 3× 级加速,路线成立。栈坑:Blackwell(sm120)上需
`VLLM_USE_FLASHINFER_SAMPLER=0` 关掉 flashinfer sampler 才能跑。

### 阶段 B —— 生成 32B 自蒸馏数学 CoT 语料
- 脚本 `reasonv4/examples/gen_cot_corpus.py`,vLLM 批量生成。
- 源:GSM8K + AIME + MATH-500,共 **1600 条**;32B 自己生成 CoT(temperature 0.6, top-p 0.95, max 3072 tok),ShareGPT 格式。
- 产物 `reasonv4/data/cot_corpus.jsonl`。

### 阶段 C —— SpecForge 训 Qwen-32B EAGLE-3 head
- 框架:SpecForge(SGLang 团队,支持 Qwen,活跃维护)。HF 后端(避免 sglang 依赖)。
- draft 结构:`LlamaForCausalLMEagle3`,单 decoder 层,hidden 5120(对齐 32B),
  `draft_vocab_size=32000`(频率裁剪 + d2t/t2d 映射)。
- 训练:1600 条 / batch 1 / lr 1e-4 / max_len **2048**(4096 会 OOM) / `expandable_segments`。
- 收敛:**epoch 6 起训练接受率就稳在 ~0.92**(epoch 7-9 无提升,早收敛)。
- 事故:**磁盘写满**,epoch 9 存 checkpoint 时 IO 崩溃;但 **epoch_6 head 完整**(1.5G safetensors),不影响结果。

坑与修复:
1. SpecForge `deepseek-r1-distill` 模板 `end_of_turn_token=None` → `re.escape(None)` 崩
   (`decoding to str: NoneType`)。**修**:改成真实 eos `<｜end▁of▁sentence｜>`
   (`specforge/data/template.py`)。
2. `max_len=4096` OOM(32B 全量前向取多层 hidden)。**修**:降 2048 + expandable_segments。
3. `train_eagle3.py --help` 需要 sglang。**修**:`specforge/args.py` 把 sglang import 包 try/except。

---

## 3. 主结果:自训 head 的真实加速

vLLM 同引擎,batch=1,MATH-500 n=15,greedy,max_tokens 512:

| 配置 | tok/s | TPOT | 接受长度 | speedup |
|---|---|---|---|---|
| target-only AR 32B | 21.14 | 47.3 ms | 1.0 | 1.0× |
| **自训 EAGLE-3 head (k=5)** | **46.69** | 21.4 ms | 2.36 | **2.21×** |
| 自训 EAGLE-3 head (k=8) | 45.70 | 21.9 ms | 2.51 | 2.16× |
| (参考) UMbreLLa 树 | 40.6 | 24.6 ms | 4.84 | 2.23× |

**要点**:
1. **2.21× 无损加速**,绝对 tok/s(46.7)已**超过** UMbreLLa 树(40.6),尽管接受长度(2.36)
   远低于树(4.84)——因为 EAGLE head 草稿成本极低(一个小层 vs 每 token 跑一次 1.5B)。
2. **加深固定深度反而更慢**(k=5→8:46.7→45.7)。逐位置接受率
   `0.63/0.33/0.21/0.14/0.09/0.04/0.04/0.03`,第 6+ 位近乎白挖。
3. head **严重欠训**(官方 EAGLE-3 通常用几万条 + 不截断,接受长度 4–5)。所以 2.21× 是**下界**。

---

## 4. 关键科学观察:EAGLE head 是"短跑者",完整 draft 是"马拉松者"

对比两种 drafter 的接受结构:

| drafter | 第 1 位接受率 | 深层行为 | 长 run(吃满 cap) |
|---|---|---|---|
| 完整 1.5B(tokens.jsonl, γ=8) | — | 衰减慢 | **20.2% 块吃满 8** |
| 自训 EAGLE head(k=8) | 0.63 | 衰减极快,第 8 位 0.027 | 几乎没有 |

EAGLE head 复用 target 特征,**第一步很准但自回归喂自己的预测后迅速失真** → 冲刺型。
完整 1.5B 有全模型的长程一致性 → 耐力型。**这个"短跑 vs 马拉松"的差异直接决定了下游
自适应深度方法的可行性(见 `method_confidence_gated_adaptive_depth.md`)。**

---

## 5. 产物与复现

- head: `SpecForge/outputs/r1-qwen-32b-eagle3-math/epoch_6_step_10000/`(config.json + model.safetensors)
- 语料: `reasonv4/data/cot_corpus.jsonl`(1600 条)
- bench 脚本: `reasonv4/examples/vllm_eagle_bench.py`
- bench 日志: `reasonv4/examples/bench_{eagle3_32b,eagle3_k8,aronly_32b}.log`
- 复现命令:
  ```bash
  VLLM_USE_FLASHINFER_SAMPLER=0 python3 vllm_eagle_bench.py \
    --base .../DeepSeek-R1-Distill-Qwen-32B \
    --eagle .../epoch_6_step_10000 --n 15 --max-tokens 512 --spec-tokens 5
  ```

## 6. 判决

| 维度 | 状态 |
|---|---|
| 系统加速 | ✅ **2.21× 已验证**(唯一确定可保存的主结果) |
| 作为论文主贡献 | ❌ 光训 head = 复现/工程,不够 novelty |
| headroom | 高(欠训;更多数据/不截断/树结构预期 3×+) |
| 下一步 | 在**强 head** 上验证自适应深度方法前提(长 run),或换 method 杠杆 |

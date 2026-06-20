# 第 11 章 · Speculative Decoding：Draft 预测 + Target 验证

> **本章导读**：第 7–10 章从量化、调度、Kernel、多卡等角度优化推理。但 **Decode 阶段**仍是交互式体验的核心瓶颈——每生成 1 个 token 就要跑一次大模型 forward。本章讲解 **Speculative Decoding（投机解码）**：用**小型 Draft 模型**快速「猜」若干 token，再用**大型 Target 模型**一次 forward **并行验证**，在**不损失输出质量**（与 Target 单独采样分布一致）的前提下，显著降低 **TPOT**（每 token 耗时）。**全部可运行代码见一个文件**：[`basic/chapter_11_speculative_decoding.py`](../basic/chapter_11_speculative_decoding.py)

---

## 11.1 Decode 为何是延迟瓶颈

回顾第 8 章两阶段推理：

| 阶段 | 用户感知 | 瓶颈 |
|------|----------|------|
| **Prefill** | 等第一个字（TTFT） | 计算量 ∝ L，可用 FlashAttention / Chunked Prefill |
| **Decode** | 每个后续字（流式输出） | **每 token 1 次大模型 forward**，Memory-bound |

```
标准 Decode（无投机）:
  Step 1: Target forward → "春"     ~30 ms
  Step 2: Target forward → "风"     ~30 ms
  Step 3: Target forward → "拂"     ~30 ms
  ...
  生成 100 token → 约 100 × 30 ms = 3 秒
```

**TPOT**（Time Per Output Token）主要由 **Target 模型单次 forward 延迟** 决定。KV Cache、算子融合、TP 能优化常数项，但**无法突破「每 token 至少 1 次大模型推理」**的下界——除非换算法。

**Speculative Decoding 的核心**：用便宜的小模型多猜几步，大模型**少跑几次**，每次 forward 却**产出多个 token**。

---

## 11.2 基本思想：猜 + 验

```
┌─────────────────────────────────────────────────────────────┐
│                    Speculative Decoding                      │
├─────────────────────────────────────────────────────────────┤
│  Draft 模型（小，快）  →  连续猜 γ 个 token：t+1, t+2, …, t+γ │
│                              ↓                               │
│  Target 模型（大，准）  →  1 次 forward 并行验证这 γ 个猜测    │
│                              ↓                               │
│  接受一致的 token；首个不一致处按规则重采样；丢弃后续猜测          │
└─────────────────────────────────────────────────────────────┘
```

类比：**实习生（Draft）先起草一段，专家（Target）一次审阅**——对的段落直接通过，第一个错处改完再继续，不必专家逐字写。

| 角色 | 典型规模 | 职责 |
|------|----------|------|
| **Draft** | 7B 以下 / 专用小模型 / 额外 head | 快速 autoregressive 猜 γ 个 token |
| **Target** | 70B 主模型 | 一次 forward 验证，保证分布正确 |

---

## 11.3 算法流程（Lossless Speculative Decoding）

以下描述 **Leviathan et al. / Chen et al.** 经典方案，保证输出与 **Target 单独自回归采样** 在统计上**完全一致**（不是近似）。

### 11.3.1 一轮迭代

设当前已生成 `x_1, …, x_t`，Draft 猜测长度 `γ`（如 4）：

```
① Draft 自回归生成 γ 个候选:
   x̃_{t+1}, x̃_{t+2}, …, x̃_{t+γ}     （γ 次小模型 forward，很快）

② Target 一次 forward（关键！）:
   输入 [x_1…x_t, x̃_{t+1}, …, x̃_{t+γ}]
   得到各位置 logits → 概率 p_T(·|context)

③ 逐个验证（Rejection Sampling）:
   for i = 1 to γ:
       若 Draft 的猜测 x̃_{t+i} 通过接受检验 → 接受，继续
       否则 → 从修正分布采样 1 个 token，停止验证，丢弃后面 Draft 猜测

④ 更新 KV Cache，进入下一轮
```

### 11.3.2 接受检验（为何「无损」）

对候选 token `x`，比较 Target 概率 `p_T(x)` 与 Draft 概率 `q_D(x)`：

```
以 min(1, p_T(x) / q_D(x)) 的概率接受 Draft 的 x
否则从修正分布 resample（与 Metropolis-Hastings 思想相关）
```

这保证**最终每个位置的边际分布 = Target 单独采样**，不会因为 Draft 猜错而「变糊」或「跑偏」。

### 11.3.3 示意图

```
Draft 猜:   [春] [风] [拂] [面]     ← 4 次小模型 forward
              ✓    ✓    ✗
Target 验:  一次 forward 看 4 个位置
接受:       春   风   (重采样→"轻")
产出:       3 个 token，仅 1 次 Target forward  （原本要 3 次）
```

---

## 11.4 加速比：接受率 α 与 γ

定义 **接受率 α**：Draft 猜的 token 被 Target 接受的概率（粗算可视为每步独立）。

每轮 Target **只做 1 次 forward**，期望接受 token 数：

```
E[接受数] ≈ 1 + α + α² + … + α^γ = (1 - α^{γ+1}) / (1 - α)
```

| α | γ=4 时期望接受 token 数 | 相对标准 Decode |
|---|------------------------|----------------|
| 0.5 | ~1.94 | ~2× |
| 0.7 | ~2.76 | ~2.8× |
| 0.9 | ~4.05 | ~4× |

**总加速**还取决于 Draft 成本：

```
加速 ≈ E[接受数] / (1 + γ × T_draft / T_target)
```

- `T_draft / T_target` 越小（Draft 远快于 Target）→ 越划算
- α 越高（Draft 与 Target 越对齐）→ 越划算
- γ 过大：Draft 算得久、后面猜测更易全拒 → 需调参

**经验**：α 在 0.6–0.8、γ=4–8 时，Decode **2–3× 加速**常见；与 vLLM 等实测一致。

---

## 11.5 Draft 模型从哪来

| 方案 | Draft 来源 | 优点 | 缺点 |
|------|-----------|------|------|
| **独立小 LM** | LLaMA-68M draft + LLaMA-70B target | 实现简单 | 需额外加载小模型；α 依赖二者对齐 |
| **同系列蒸馏** | 7B draft + 70B target（同词表） | 接受率较高 | 仍占显存 |
| **Medusa** | Target 上加多个解码头 | 无独立小模型 | 需微调 |
| **EAGLE** | 轻量特征预测网络 | α 高、速度快 | 需训练 |
| **Prompt lookup / n-gram** | 从输入或缓存 copy | 零成本 draft | 仅重复文本场景有效 |

生产常见：**同 family 小模型**（如 TinyLlama draft + LLaMA-70B target），或 **EAGLE / Medusa** 等专用结构。

---

## 11.6 与 KV Cache 的配合

Speculative Decoding 对 KV Cache 有特殊要求：

```
验证阶段 Target forward 输入:
  [已确认 tokens | Draft 猜的 γ tokens]
   ↑ KV 已有      ↑ 需要临时算，验证后可能丢弃部分
```

- **接受的 token**：KV 保留并追加
- **拒绝后**：从第一个拒绝位置起，**丢弃** Draft 带来的错误 KV，用重采样 token 重建

vLLM 等在 PagedAttention block 上实现 **KV rollback / 部分提交**，与 Continuous Batching 兼容是工程难点。

---

## 11.7 与其他优化手段的关系

```
                    Decode 延迟优化谱
    ┌───────────────────────────────────────────────────┐
    │  算法层（本章）                                     │
    │  Speculative Decoding → 减少 Target forward 次数   │
    ├───────────────────────────────────────────────────┤
    │  算子层（第 9 章）  FlashAttention · 算子融合       │
    │  → 每次 forward 更快                               │
    ├───────────────────────────────────────────────────┤
    │  系统层（第 8 章）  KV Cache · Continuous Batching  │
    ├───────────────────────────────────────────────────┤
    │  模型层（第 7 章）  量化                             │
    ├───────────────────────────────────────────────────┤
    │  分布式（第 10 章） TP · 多卡                       │
    └───────────────────────────────────────────────────┘
```

| 手段 | 优化什么 | 与 Speculative 关系 |
|------|----------|---------------------|
| KV Cache | 不重算历史 | 必需，Speculative 依赖 |
| FlashAttention | 单次 forward 更快 | 叠加；验证阶段一次算 γ+1 位置 |
| 量化 | 权重更小 | Draft+Target 都可量化 |
| TP | 大模型放多卡 | Target 验证仍可用 TP |
| **Speculative** | **少跑 Target forward** | **正交，可全部叠加** |

**注意**：Speculative 主要加速 **Decode**；**Prefill / TTFT** 几乎无帮助（Prefill 没有「猜后续 token」的空间）。

---

## 11.8 局限与适用场景

| 局限 | 说明 |
|------|------|
| **额外显存** | 需同时驻留 Draft + Target（或 Medusa 头） |
| **接受率依赖任务** | 创意写作 α 低；代码补全 / 重复模式 α 高 |
| **Draft–Target 对齐** | 词表、chat template 必须一致 |
| **实现复杂** | KV rollback、batch 内不同 γ 步长 |
| **不损质量 ≠ 更快** | α 太低时反而更慢（白跑 Draft） |

| 场景 | 是否推荐 |
|------|----------|
| 交互聊天（70B Target） | ✓ 常用 |
| 代码补全（重复多） | ✓✓ α  often 很高 |
| 高创意 / 高温度采样 | △ α 可能偏低 |
| 仅 Prefill 瓶颈 | ✗ 无效 |
| 极小 Target 已够快 | △ 收益有限 |

---

## 11.9 生产实践

| 框架 | 支持 | 配置要点 |
|------|------|----------|
| **vLLM** | `--speculative-model` | 指定 draft 模型路径、`num_speculative_tokens` |
| **TensorRT-LLM** | Medusa / Eagle 插件 | 需对应 checkpoint |
| **llama.cpp** | `--draft` | 本地 draft + target gguf |
| **SGLang** | EAGLE / 标准 speculative | 高吞吐 serving |

**vLLM 示意**：

```bash
python -m vllm.entrypoints.api_server \
    --model meta-llama/Llama-2-70b-hf \
    --speculative-model JackFram/llama-160m-arch \
    --num_speculative_tokens 5 \
    --tensor-parallel-size 4
```

---

## 11.10 本章小结

| 概念 | 一句话 |
|------|--------|
| **Speculative Decoding** | 小模型猜 γ token，大模型 1 次 forward 验证 |
| **无损** | Rejection sampling 保证与 Target 单独采样同分布 |
| **接受率 α** | 越高加速越大；取决于 Draft 与 Target 对齐程度 |
| **γ** | 每轮猜测长度；过大则拒收多、Draft 成本高 |
| **主要收益** | 降低 **TPOT** / Decode 延迟，不优化 TTFT |
| **叠加** | 与 KV Cache、FlashAttention、量化、TP 全部兼容 |

---

## 11.11 思考题与参考答案

### 思考题 1

Speculative Decoding 为何保证「无损」，而不是「Draft 猜错就凑合用」？

**参考答案**：

若猜错直接采用 Draft 输出，分布会偏向小模型，质量下降。经典 Speculative Decoding 用 **rejection sampling**：以 `min(1, p_T/q_D)` 接受；拒绝时从 **修正分布**重采样，可证明最终每个 token 的边际分布等于 Target 自回归采样。Draft 只提议候选，**最终决定权在 Target 概率**。

### 思考题 2

接受率 α=0.5，γ=3，忽略 Draft 耗时，期望每轮 Target forward 产出多少 token？

**参考答案**：

```
E ≈ (1 - 0.5⁴) / (1 - 0.5) = (1 - 0.0625) / 0.5 = 1.875
```

约 **1.9 个 token / 次 Target forward**，相对标准 Decode（1 token/次）约 **1.9×**。若 Draft 耗时不可忽略，实际加速略低。

### 思考题 3

为何 Speculative Decoding 对 TTFT 帮助不大？

**参考答案**：

TTFT 由 **Prefill**（处理用户 prompt）决定。Speculative 发生在 **Decode**：先有第一个 token，再猜后续。Prefill 阶段没有「已生成序列 + 猜未来 token」的结构；且第一个 token 必须先由 Target（或至少一次完整 forward）产生，Draft 无法跳过 Prefill。

---

## 相关资源

- 第 8 章：[`document/chapter_08_inference_pipeline.md`](chapter_08_inference_pipeline.md) — Prefill / Decode、TPOT
- 第 9 章：[`document/chapter_09_flash_attention_operator_fusion.md`](chapter_09_flash_attention_operator_fusion.md) — 验证阶段 forward 加速
- 第 10 章：[`document/chapter_10_distributed_inference.md`](chapter_10_distributed_inference.md) — Target 模型多卡
- **本章全部代码（一个文件）**：[`basic/chapter_11_speculative_decoding.py`](../basic/chapter_11_speculative_decoding.py) — 运行 `python3 basic/chapter_11_speculative_decoding.py`
- 论文：[Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)、[Medusa](https://arxiv.org/abs/2401.10774)、[EAGLE](https://arxiv.org/abs/2401.15077)

---

*下一章预告：第 12 章将讨论 **推理监控与 SLA**——TTFT、TPOT、P99 延迟、吞吐与成本如何在生产环境中度量、告警与容量规划。*

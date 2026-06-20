# 第 8 章 · 推理流水线：批处理、KV Cache 与连续批处理

> **本章导读**：第 7 章通过量化把「单个模型」变轻、变快。但线上服务真正的瓶颈，往往不在模型权重，而在**请求如何调度、GPU 是否被喂饱、自回归生成是否重复计算**。本章从推理流水线出发，讲解批处理策略、KV Cache 与 Continuous Batching——这三项是 modern LLM serving 系统的核心武器。**全部可运行代码见一个文件**：[`basic/chapter_08_inference_pipeline.py`](../basic/chapter_08_inference_pipeline.py)

---

## 8.1 从单机推理到推理流水线

一个完整的 LLM 推理服务，不是 `model(input)` 这么简单，而是一条流水线：

```
用户请求 → 请求队列 → 调度器 → Tokenizer → GPU 推理 → Detokenizer → 流式返回
                ↑                              ↑
           批处理策略                      KV Cache 管理
```

| 阶段 | 职责 | 优化目标 |
|------|------|----------|
| **请求队列** | 缓冲突发流量 | 削峰、公平调度 |
| **调度器** | 决定哪些请求一起上 GPU | 最大化吞吐、控制延迟 |
| **Prefill** | 处理用户输入的全部 token | 计算密集型（Compute-bound） |
| **Decode** | 逐 token 生成输出 | 内存带宽密集型（Memory-bound） |
| **KV Cache** | 缓存历史 Key/Value | 避免重复计算 |

**关键认知**：LLM 推理分两个阶段，瓶颈不同，优化手段也不同。

---

## 8.2 批处理策略：三种范式

GPU 擅长并行矩阵运算。**Batch Size 越大，单位 token 的算力成本通常越低**——但 batch 不是越大越好，还要考虑延迟、padding 浪费和内存上限。

### 8.2.1 无批处理（Batch Size = 1）

```
请求 A ──→ GPU ──→ 返回
请求 B ──→ 等待 ──→ GPU ──→ 返回
请求 C ──→ 等待 ──→ 等待 ──→ GPU ──→ 返回
```

- **延迟**：最低（每个请求立即处理）
- **吞吐**：最差（GPU 大量时间空闲）
- **适用**：极低延迟场景、调试、单用户交互

### 8.2.2 静态批处理（Static Batching）

等凑够 N 个请求，或等超时，打包成一个 batch 一次送入 GPU：

```
请求 A (len=10)  ─┐
请求 B (len=50)  ─┼─→ Batch ──→ GPU 一次前向 ──→ 分别返回
请求 C (len=30)  ─┘
```

**优点**：

- 实现简单（类似 `Problem_2_openAI_optimize_pipeline.py` 中的 `optimize_inference`：固定 batch 逐批处理）
- GPU 利用率显著提升

**缺点——Padding 浪费**：

为对齐序列长度，短请求必须 pad 到 batch 内最长序列：

```
请求 A: [tok₁, tok₂, PAD, PAD, ..., PAD]   ← 实际 10，pad 到 50
请求 B: [tok₁, tok₂, ..., tok₅₀]              ← 实际 50
```

Attention 计算量近似 O(L²)，padding 带来**无效计算**。在 LLM **Decode 阶段**（每个请求每次只生成 1 个 token），静态 batch 的问题更严重：

- 各请求生成长度不同 → 必须等**最慢的请求**完成，整 batch 才能释放
- 早完成的请求占着 GPU slot，**GPU 空转**

### 8.2.3 连续批处理（Continuous Batching）

也叫 **Iteration-level Batching** 或 **Dynamic Batching**（Orca、vLLM 等系统的核心思想）。

**核心思想**：不在「请求级别」组 batch，而在**每个 decode step（每一轮 token 生成）** 重新组 batch。

```
Step 1: Batch = {A, B, C}     → 各生成 1 个 token
Step 2: A 完成，D 加入       → Batch = {B, C, D}
Step 3: Batch = {B, C, D, E}  → 继续生成
Step 4: B 完成               → Batch = {C, D, E}
...
```

| 对比项 | 静态批处理 | 连续批处理 |
|--------|-----------|-----------|
| 组 batch 时机 | 请求到达时 | 每个 decode step |
| 请求完成后 | 等整 batch 结束 | 立即退出，slot 给新请求 |
| GPU 利用率 | 中等 | 高 |
| 实现复杂度 | 低 | 高 |
| 代表系统 | 早期 Triton 部署 | **vLLM**, TGI, TensorRT-LLM |

> **一句话**：静态 batch 像「等满一班车再开」；Continuous Batching 像「地铁随到随走，每站都可以上下人」。

---

## 8.3 KV Cache：自回归推理的加速器

### 8.3.1 为什么需要 KV Cache

Transformer 生成是**自回归**的：每步只预测 1 个新 token，下一步要把**全部历史**再喂进模型。

**没有 KV Cache 的朴素做法**（生成第 t 个 token）：

```
输入: [tok₁, tok₂, ..., tok_{t-1}]   ← 长度 t-1
      ↓ 完整 forward（所有层、所有 token）
输出: tok_t
```

每生成 1 个 token，都要对**所有历史 token** 重新计算 Q、K、V——大量重复计算。

**有 KV Cache**：

```
Step 1: 输入 [tok₁, tok₂, tok₃]  → 计算 K,V 并缓存
Step 2: 输入 [tok₄]              → 只算 tok₄ 的 Q,K,V
                                  → K,V 与缓存拼接
                                  → Attention 用「新 Q × 全部 K」
Step 3: 输入 [tok₅]              → 同理，只增量计算
```

**只算新 token 的 Q/K/V，历史 K/V 从缓存读取**——Decode 阶段从 O(L²) 重复计算，降为每步 O(L)。

### 8.3.2 KV Cache 存什么

对每一层、每一个 attention head，缓存：

- **K**：Key 矩阵，形状 `[seq_len, d_k]`
- **V**：Value 矩阵，形状 `[seq_len, d_v]`

生成新 token 时，新 K/V **追加（append）** 到 cache 末尾。

### 8.3.3 内存估算

单层、单请求的 KV Cache 大小：

```
KV_size = 2 × L × d_model × bytes_per_elem
```

- 因子 2：K 和 V 各一份
- L：当前序列长度（输入 + 已生成长度）
- FP16 下 `bytes_per_elem = 2`

**完整模型**（N 层、H 个头，通常 `d_model = H × d_k`）：

```
Total_KV = 2 × N × L × d_model × bytes
```

**示例**：LLaMA-7B（N=32, d_model=4096），FP16，序列长度 L=2048：

```
2 × 32 × 2048 × 4096 × 2 ≈ 1 GB
```

**仅 KV Cache 就占 1 GB**——这就是为什么「内存占用 = 模型权重 + KV Cache + 激活值」，也是高并发 serving 的主要显存压力来源。

### 8.3.4 KV Cache 与 Batch 的交互

Batch 中有 B 个并发请求，每个序列长度不同，KV Cache 也是**每请求独立**的一份：

```
Total_KV_batch = Σ KV_size(L_i)   (i = 1 到 B)
```

Continuous Batching 的难点之一：**动态增删请求时，如何高效管理、分配、回收 KV Cache 内存块**——vLLM 的 **PagedAttention** 正是为此设计（类似 OS 虚拟内存的分页机制）。

---

## 8.4 Prefill vs Decode：两阶段调度

| 阶段 | 输入 | 计算特征 | 优化重点 |
|------|------|----------|----------|
| **Prefill** | 用户 prompt 全部 token | 大矩阵乘，Compute-bound | 大 batch、FlashAttention |
| **Decode** | 每次 1 个新 token | 小矩阵乘 + 读 KV Cache，Memory-bound | KV Cache、Continuous Batching |

```
用户: "请写一首关于春天的诗"
         │
         ▼ Prefill（一次处理整段 prompt）
    [请, 写, 一, 首, ...]  →  建立 KV Cache
         │
         ▼ Decode（逐 token 生成）
    春  → 风  → 拂  → 面  → ...   每步只算 1 token，读全部 KV
```

**Chunked Prefill**（进阶）：超长 prompt 分块 prefill，避免一次占满 GPU，与 decode 请求交错执行——进一步平衡延迟与吞吐。

---

## 8.5 三种技术如何协同

```
                    ┌─────────────────────────────────┐
                    │         推理服务（如 vLLM）        │
                    └─────────────────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                         ▼
   ┌─────────────┐          ┌─────────────┐          ┌─────────────┐
   │  批处理策略   │          │  KV Cache   │          │  量化(INT8)  │
   │  喂饱 GPU    │          │  避免重复算  │          │  减小权重    │
   └─────────────┘          └─────────────┘          └─────────────┘
          │                         │                         │
          ▼                         ▼                         ▼
   提升 Throughput            降低 Decode 延迟           降低 Weight 内存
   减少 GPU 空转              降低 Compute 量            更多并发 slot
```

| 技术 | 主要优化对象 | 主要收益 |
|------|-------------|----------|
| **静态批处理** | GPU 并行度 | 吞吐 ↑，延迟 ↑ |
| **Continuous Batching** | GPU 空转 + padding | 吞吐 ↑↑，延迟可控 |
| **KV Cache** | 重复 Attention 计算 | Decode 延迟 ↓↓ |
| **PagedAttention** | KV Cache 内存碎片 | 并发数 ↑ |
| **量化（第 7 章）** | 模型权重 | 显存 ↓，权重加载更快 |

---

## 8.6 延迟 vs 吞吐：如何选型

没有「最优 batch size」，只有「最优权衡」：

```
         高吞吐
           ↑
           │     ● Continuous Batching + 大并发
           │
           │           ● 静态批处理 (batch=32)
           │
           │  ● 静态批处理 (batch=8)
           │
           │ ● Batch=1
           └────────────────────────→ 低延迟
```

| 场景 | 推荐策略 | 原因 |
|------|----------|------|
| 聊天机器人（交互式） | Continuous Batching + 适中并发上限 | 低 P99 延迟 + 合理吞吐 |
| 离线批量摘要 | 静态大 batch | 吞吐优先，延迟不敏感 |
| 代码补全（流式） | Batch=1 或小 batch + KV Cache | 首 token 延迟（TTFT）最重要 |
| 高并发 API | Continuous Batching + PagedAttention | 最大化 GPU 利用率 |

**关键指标**：

| 指标 | 含义 |
|------|------|
| **TTFT**（Time To First Token） | 从请求到第一个 token 的时间，受 Prefill 影响 |
| **TPOT**（Time Per Output Token） | 每个输出 token 的平均耗时，受 Decode + KV Cache 影响 |
| **Throughput** | tokens/sec 或 requests/sec |
| **P99 Latency** | 99% 请求的端到端延迟 |

---

## 8.7 生产级实现一览

| 框架 | 核心技术 | 特点 |
|------|----------|------|
| **vLLM** | PagedAttention + Continuous Batching | 开源 serving 事实标准，高吞吐 |
| **TGI**（HuggingFace） | Continuous Batching + Flash Attention | HF 生态集成好 |
| **TensorRT-LLM** | 算子融合 + In-flight Batching | NVIDIA 硬件深度优化 |
| **llama.cpp** | KV Cache + 量化 | CPU/边缘设备友好 |

**vLLM 伪代码逻辑**（帮助理解 Continuous Batching）：

```python
# 每个 decode step 执行一次
while running_requests:
    batch = scheduler.get_next_batch()   # 动态选取「本轮可运行」的请求
    # 可能：A 生成第 50 token，B 生成第 3 token，C 刚完成 prefill
    outputs = model.forward(
        input_tokens=[req.next_token for req in batch],
        kv_caches=[req.kv_cache for req in batch],  # 各请求独立 cache
    )
    for req, token in zip(batch, outputs):
        req.append_token(token)
        if req.is_done():
            scheduler.release(req)       # 立即释放 slot，不等其他请求
            running_requests.remove(req)
    scheduler.add_new_requests()         # 新请求填入空出的 slot
```

---

## 8.8 与第 7 章、Problem 2/3 的关系

| 章节/文件 | 优化层次 | 核心手段 |
|-----------|----------|----------|
| **第 7 章 / Problem 3** | 模型层 | INT8 量化、FP16、压缩比 |
| **Problem 2** | 数据流水线层 | 固定 batch 分批推理（传统 ML） |
| **第 8 章（本章）** | 系统调度层 | KV Cache、Continuous Batching |
| **`basic/chapter_08_inference_pipeline.py`** | 本章全部代码 | 静态/连续批处理、KV Cache、PagedAttention、调度器 |

Problem 2 的批处理：

```python
for i in range(0, len(X_test), batch_size):
    batch = X_test[i:i+batch_size]
    batch_pred = model.predict(batch)
```

这是**数据维度**的静态 batch——输入长度固定、一次 forward 出全部结果。

LLM serving 的 Continuous Batching 是**时间维度**的动态 batch——每轮 forward 只生成 1 token，batch 成员每轮都可能变化。**二者思想相通，但复杂度差一个数量级。**

---

## 8.9 本章小结

| 概念 | 一句话 |
|------|--------|
| **静态批处理** | 多个请求打包一次 forward，简单但有 padding 和等待浪费 |
| **Continuous Batching** | 每个 decode step 动态组 batch，完成即退出，GPU 利用率高 |
| **KV Cache** | 缓存历史 K/V，Decode 每步只算新 token，避免 O(L²) 重复 |
| **Prefill / Decode** | 两阶段瓶颈不同，需分别优化 |
| **PagedAttention** | KV Cache 分页管理，减少碎片、提高并发 |
| **延迟 vs 吞吐** | 没有万能 batch size，按业务场景权衡 TTFT 和 Throughput |

---

## 8.10 思考题与参考答案

### 思考题 1

一个 batch 有 4 个请求，生成长度分别为 10、50、100、20 tokens。静态批处理要跑多少 decode step？Continuous Batching 大约跑多少？

**参考答案**：

- **静态批处理**：必须等最长请求 → **100 decode steps**（其余 3 个请求完成后空等）
- **Continuous Batching**：约 **100 steps**，但每 step 活跃请求数递减（100→80→50→10→0），GPU 利用率随时间变化，无空等

### 思考题 2

KV Cache 占 1 GB，模型权重 INT8 量化后 2 GB，GPU 共 24 GB。理论上最多能同时服务多少个「平均序列长 2048」的请求？

**参考答案**（粗算）：

- 可用显存 ≈ 24 - 2 = 22 GB（忽略激活开销）
- 每请求 KV ≈ 1 GB → 理论并发约 **20 个**（实际更少，需预留激活和碎片）

---

## 相关资源

- 第 7 章：[`document/chapter_07_model_quantization.md`](chapter_07_model_quantization.md) — 模型量化与推理优化
- **本章全部代码（一个文件）**：[`basic/chapter_08_inference_pipeline.py`](../basic/chapter_08_inference_pipeline.py) — 运行 `python3 basic/chapter_08_inference_pipeline.py`
- 第 9 章：[`document/chapter_09_flash_attention_operator_fusion.md`](chapter_09_flash_attention_operator_fusion.md) — FlashAttention 与算子融合
- 第 10 章：[`document/chapter_10_distributed_inference.md`](chapter_10_distributed_inference.md) — 分布式推理与模型并行
- 第 11 章：[`document/chapter_11_speculative_decoding.md`](chapter_11_speculative_decoding.md) — Speculative Decoding
- 第 12 章：[`document/chapter_12_inference_monitoring_sla.md`](chapter_12_inference_monitoring_sla.md) — 推理监控与 SLA
- 代码实现：`openAI/Problem_3_openAI_optimize_inference_model.py`
- Problem 2 批处理基础：`openAI/Problem_2_openAI_optimize_pipeline.py`
- OpenAI 面试题梳理：`openAI/openAI_questions.md`

---

*下一章预告：第 9 章将讨论 **FlashAttention** 与 **算子融合**——从 Kernel 层面进一步降低 Attention 的内存与计算开销，与 KV Cache 形成「系统 + 算子」双层优化。详见 [`chapter_09_flash_attention_operator_fusion.md`](chapter_09_flash_attention_operator_fusion.md)。*

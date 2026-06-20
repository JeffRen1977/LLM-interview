# 第 10 章 · 分布式推理与模型并行

> **本章导读**：第 7–9 章在**单卡**内把模型压小、把请求调度好、把 Attention 算快。但当模型超过单卡显存（如 70B 参数），或高并发下 KV Cache 撑爆显存时，必须**跨多 GPU 扩展**。本章讲解 **Tensor Parallelism（TP）** 与 **Pipeline Parallelism（PP）**——两种最核心的模型并行策略，以及它们如何与量化、KV Cache、FlashAttention 协同。**全部可运行代码见一个文件**：[`basic/chapter_10_distributed_inference.py`](../basic/chapter_10_distributed_inference.py)

---

## 10.1 什么时候单卡不够

单卡显存大致被三部分瓜分：

```
GPU 显存 ≈ 模型权重 + KV Cache + 激活值 / 临时 buffer
```

| 压力来源 | 典型场景 | 第 7–9 章单卡手段 | 仍不够时 |
|----------|----------|------------------|----------|
| **权重太大** | LLaMA-70B FP16 ≈ 140 GB | INT8 量化 → ~70 GB | 多卡切分权重（TP / PP） |
| **KV Cache 太大** | 高并发 × 长序列 | PagedAttention、Continuous Batching | 多卡分摊 KV 或减并发 |
| **激活 / buffer** | 超长 Prefill、大 batch | FlashAttention、Chunked Prefill | TP 分摊激活 |

**示例**：LLaMA-7B FP16 权重 ~14 GB，24 GB 卡上还剩 ~10 GB 给 KV Cache。若每请求 KV ≈ 1 GB（L=2048），理论并发 ~10。换成 **70B 模型**，单卡连权重都放不下——必须模型并行。

**关键认知**：量化解决「权重能不能塞进去」；模型并行解决「塞不进去时怎么切开」；二者可叠加。

---

## 10.2 三种并行范式一览

```
                    ┌─────────────────────────────────────────┐
                    │           一个 LLM Forward Pass          │
                    └─────────────────────────────────────────┘
                                        │
          ┌─────────────────────────────┼─────────────────────────────┐
          ▼                             ▼                             ▼
   ┌─────────────┐              ┌─────────────┐              ┌─────────────┐
   │ Data Parallel│              │Tensor Parallel│              │Pipeline Parallel│
   │  (DP)        │              │  (TP)         │              │  (PP)           │
   └─────────────┘              └─────────────┘              └─────────────┘
   复制完整模型                   切分每一层的张量                 切分不同的层
   各卡处理不同请求/样本           各卡持有部分权重                 各卡持有部分层
   推理 serving 常见               大模型推理主力                   超大模型补充
```

| 范式 | 切什么 | 每卡有什么 | 主要通信 | 推理典型用途 |
|------|--------|-----------|----------|-------------|
| **Data Parallel (DP)** | 数据 / 请求 | 完整模型副本 | 梯度 AllReduce（训练） | 多副本负载均衡 |
| **Tensor Parallel (TP)** | 层内权重矩阵 | 部分权重 | AllReduce / AllGather | 单模型放多卡 |
| **Pipeline Parallel (PP)** | 层间（按 depth） | 连续若干层 | P2P 发送激活 | 极深 / 极大模型 |

> **推理 vs 训练**：训练三者都用；**大模型推理**最常见的是 **TP**（延迟低），PP 在超大模型上与 TP 组合，DP 用于多副本扩吞吐。

---

## 10.3 Tensor Parallelism（TP）：切分每一层

TP 的核心思想：**把单个矩阵乘法的权重按列或按行切到多张 GPU**，每张卡只算一部分，再通过通信合并结果。Megatron-LM 是事实标准。

### 10.3.1 线性层怎么切

线性层 `Y = X @ W`（X: [batch, in], W: [in, out]）：

**列并行（Column Parallel）**——按输出维度切 W：

```
GPU 0: W₀ [:, 0:out/2]   →  Y₀ = X @ W₀
GPU 1: W₁ [:, out/2:out] →  Y₁ = X @ W₁

最终 Y = concat(Y₀, Y₁)    ← AllGather 或只需各自持有半段
```

**行并行（Row Parallel）**——按输入维度切 W：

```
GPU 0: W₀ [0:in/2, :]   →  Y₀ = X₀ @ W₀
GPU 1: W₁ [in/2:in, :]  →  Y₁ = X₁ @ W₁

最终 Y = Y₀ + Y₁         ← AllReduce（求和）
```

一个 Transformer 层通常交替使用列并行 + 行并行，使 **AllReduce 次数最少**。

### 10.3.2 Attention 层的 TP

Multi-Head Attention 天然可按 **head 维度** 切分：

```
32 heads, TP=4  →  每卡 8 heads

GPU 0: head 0–7   的 Q,K,V 投影 + Attention + 输出
GPU 1: head 8–15
...
```

每卡独立算自己那部分 head 的 Attention（含各自 KV Cache），最后 **AllGather** 或输出投影用行并行合并。

### 10.3.3 TP 与 KV Cache

TP 切 head 时，**KV Cache 也按 head 分摊到各卡**：

```
单卡 KV (TP=1):  2 × N_layers × L × d_model
TP=4 每卡:       2 × N_layers × L × (d_model / 4)   ← 每卡只存 1/4 heads 的 K,V
```

这是 TP 除了切权重之外的重要收益：**KV Cache 显存也近似按 TP 度线性下降**。

### 10.3.4 通信代价

| 操作 | 通信类型 | 何时发生 |
|------|----------|----------|
| 列并行 Linear 输出拼接 | AllGather | 需要完整输出给下一层时 |
| 行并行 Linear 输出合并 | AllReduce | 每个 TP rank 算部分和 |
| Attention 多头合并 | AllGather / AllReduce | 依实现而定 |

**NVLink**（同机多卡）带宽 ~600 GB/s 级，TP 通常限制在同一节点内（TP=2/4/8）。跨节点 TP 通信延迟高，生产环境较少。

---

## 10.4 Pipeline Parallelism（PP）：切分不同的层

PP 把 **N 层 Transformer 按 depth 切成若干 stage**，每张 GPU 负责连续的一段层。

```
输入 ──→ [GPU 0: Layer 0–7] ──→ [GPU 1: Layer 8–15] ──→ ... ──→ [GPU 3: Layer 24–31] ──→ 输出
              Stage 0                  Stage 1                              Stage 3
```

### 10.4.1 微批次与流水线气泡

PP 要喂饱流水线，通常把 batch 拆成 **micro-batch** 流水线推进：

```
时间 →
GPU 0: [mb1][mb2][mb3][mb4]...
GPU 1:      [mb1][mb2][mb3][mb4]...
GPU 2:           [mb1][mb2][mb3]...
GPU 3:                [mb1][mb2]...
         ↑ 启动/结束阶段 GPU 有空泡（bubble）
```

**气泡（pipeline bubble）**：流水线刚启动和即将结束时，部分 GPU 空闲。micro-batch 越多，气泡占比越小，但 **latency 增加**。

### 10.4.2 PP 在推理中的特点

| 优点 | 缺点 |
|------|------|
| 可切极深模型，每卡只存 1/PP 的层权重 | 单请求 latency 高（需穿过所有 stage） |
| 层间通信量相对小（传激活，不传全量权重） | Decode batch=1 时 GPU 利用率低 |
| 与 TP 正交，可组合 | 实现与调度复杂 |

**推理直觉**：TP 像「一道菜多人同时切配」；PP 像「流水线厨房，每人负责一道工序」。低延迟交互式 serving 更偏 **TP**；超大模型不得不用 **TP + PP**。

---

## 10.5 TP + PP 组合部署

常见 8 卡部署 LLaMA-70B：

```
8 GPU = TP=2 × PP=4   （或 TP=4 × PP=2，依硬件拓扑）

┌──────── TP group 0 ────────┐  ┌──────── TP group 1 ────────┐
│ GPU0  GPU1  (Stage 0, L0–7) │  │ GPU2  GPU3  (Stage 1, L8–15)│
└─────────────────────────────┘  └─────────────────────────────┘
         │ P2P 激活                      │ P2P
┌──────── TP group 2 ────────┐  ┌──────── TP group 3 ────────┐
│ GPU4  GPU5  (Stage 2)      │  │ GPU6  GPU7  (Stage 3)      │
└─────────────────────────────┘  └─────────────────────────────┘
```

| 配置 | 每卡权重大致占比 | 适用 |
|------|-----------------|------|
| TP=1, PP=1 | 100% | 7B 单卡 |
| TP=2, PP=1 | ~50% 权重 + ~50% KV | 13B–34B 双卡 |
| TP=4, PP=2 | ~12.5% 层 × 并行 | 70B 八卡 |
| TP=8, PP=1 | ~12.5% | 70B 八卡低延迟（需 NVLink 全互联） |

---

## 10.6 显存预算：要不要上多卡

用第 8 章 KV 公式 + 第 7 章量化，粗算单请求显存：

```
Weight_mem = num_params × bytes_per_param
KV_mem     = 2 × N_layers × L × d_model × bytes
Total      ≈ Weight_mem + KV_mem × num_concurrent_requests
```

**LLaMA-70B，FP16，L=4096，1 请求**：

```
Weight: 70B × 2 bytes ≈ 140 GB
KV:     2 × 80 × 4096 × 8192 × 2 ≈ 10 GB+
Total:  > 150 GB  →  至少需要 2×80GB 或 8×24GB + TP/PP
```

**LLaMA-70B INT8 + TP=4**：

```
Weight 每卡: ~35 GB / 4 ≈ 8.75 GB
KV 每卡:     ~10 GB / 4 ≈ 2.5 GB
→ 24 GB 卡可跑，但并发有限
```

---

## 10.7 与第 7–9 章的完整优化栈

```
┌─────────────────────────────────────────────────────────────┐
│  第 10 章 · 分布式推理（本章）                                  │
│  TP / PP / 多副本 DP  →  跨卡扩展显存与算力                     │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│  第 8 章 · 系统调度                                           │
│  KV Cache · Continuous Batching · PagedAttention             │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│  第 9 章 · 算子 / Kernel                                      │
│  FlashAttention · 算子融合                                    │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│  第 7 章 · 模型压缩                                           │
│  INT8/FP16 量化                                               │
└─────────────────────────────────────────────────────────────┘
```

| 问题 | 优先手段 |
|------|----------|
| 权重放不下 | 量化 → TP → PP |
| KV Cache 放不下 | 减并发 / 缩短 L → TP（分摊 KV）→ 多卡 DP 副本 |
| Prefill 太慢 | FlashAttention → 更大 TP（更多算力） |
| 吞吐不够 | Continuous Batching → 多 GPU 数据并行副本 |

---

## 10.8 生产实践一览

| 框架 | 并行支持 | 特点 |
|------|----------|------|
| **vLLM** | TP（Ray 或多进程） | `--tensor-parallel-size 4`，与 PagedAttention 集成 |
| **TensorRT-LLM** | TP + PP | NVIDIA 官方，单机多卡性能优 |
| **Megatron-LM** | TP + PP + EP | 训练出身，推理 benchmark 常用 |
| **llama.cpp** | 层 offload / 多 GPU split | 消费级多卡、CPU+GPU 混合 |
| **DeepSpeed-Inference** | TP + kernel 注入 | 与 HuggingFace 模型集成 |

**vLLM 启动 TP 推理（示意）**：

```bash
python -m vllm.entrypoints.api_server \
    --model meta-llama/Llama-2-70b-hf \
    --tensor-parallel-size 4 \
    --dtype float16
```

**关键配置项**：

| 参数 | 含义 |
|------|------|
| `tensor_parallel_size` | TP 度，通常 = 同节点 GPU 数 |
| `pipeline_parallel_size` | PP 度，超大模型启用 |
| `max_num_seqs` | 最大并发序列（影响 KV 总量） |
| `gpu_memory_utilization` | 单卡显存占用上限比例 |

---

## 10.9 Data Parallel 与多副本 Serving

与 TP/PP 不同，**Data Parallel 推理**是在每张卡上放**完整模型副本**，不同卡处理**不同请求**：

```
         Load Balancer
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
  GPU 0     GPU 1     GPU 2
  完整 7B    完整 7B    完整 7B
  请求 A,B   请求 C,D   请求 E,F
```

| | TP / PP | Data Parallel 副本 |
|--|---------|-------------------|
| 目的 | 单模型放多卡 | 扩吞吐 |
| 每卡模型 | 一部分 | 完整副本 |
| 单请求延迟 | TP 低，PP 较高 | 与单卡相同 |
| 适用 | 70B+ 放不进单卡 | 7B 高 QPS API |

生产常见组合：**TP 把大模型切开 + 多套 TP group 做 DP 副本** 同时扩模型规模与吞吐。

---

## 10.10 本章小结

| 概念 | 一句话 |
|------|--------|
| **Tensor Parallelism** | 切层内权重/head，AllReduce 合并，低延迟，推理首选 |
| **Pipeline Parallelism** | 切层间 depth，流水线推进，适合超大模型，latency 较高 |
| **TP + KV Cache** | head 切分后 KV 按卡分摊，显存近似 ÷TP |
| **Data Parallel 副本** | 每卡完整模型，不同请求，扩吞吐 |
| **何时上多卡** | 权重量化后仍放不下，或 KV×并发超显存 |
| **通信** | TP 依赖 NVLink；跨节点 TP 慎用 |

---

## 10.11 思考题与参考答案

### 思考题 1

LLaMA-7B FP16 约 14 GB 权重，24 GB 卡 Prefill 够用。为何 70B 必须 TP，而 7B 高并发却可能用 DP 副本而不是 TP？

**参考答案**：

- **70B**：权重 ~140 GB FP16，**单卡物理放不下**，必须用 TP（或 PP）切权重。
- **7B**：单卡能放下完整模型，瓶颈往往是 **吞吐** 而非单卡显存。此时复制多份完整模型（DP 副本）让各卡独立处理不同请求，实现简单、无 TP 通信开销。只有单卡 KV Cache × 并发仍不够时，才考虑 TP 分摊 KV 或加卡。

### 思考题 2

TP=4 时，Attention 的 KV Cache 每卡占单卡的多少？通信发生在哪？

**参考答案**：

- 按 head 切分：每卡约 **1/4** 的 K/V（8 heads / 32 total）。
- 通信：各卡算完本地 head 的 Attention 后，输出投影或下一层输入处可能需要 **AllGather**（拼 head 输出）或 **AllReduce**（行并行 Linear），具体依 Megatron 实现。Decode 每步都有 TP 通信，因此 TP 度不宜过大（通常 ≤8）。

### 思考题 3

Pipeline Parallel 为何在 Decode（batch=1）时 GPU 利用率低？

**参考答案**：

PP 靠 **micro-batch 流水线** 填气泡。Decode 每步只处理 1 token，batch 极小，流水线各 stage **无法同时处理多个 micro-batch**，大量时间花在等上下游传递激活上，bubble 占比高。Prefill 批量大时 PP 相对高效；交互式 Decode 更依赖 TP 而非 PP。

---

## 相关资源

- 第 7 章：[`document/chapter_07_model_quantization.md`](chapter_07_model_quantization.md) — 量化减小权重
- 第 8 章：[`document/chapter_08_inference_pipeline.md`](chapter_08_inference_pipeline.md) — KV Cache 与批处理
- 第 9 章：[`document/chapter_09_flash_attention_operator_fusion.md`](chapter_09_flash_attention_operator_fusion.md) — FlashAttention 与算子融合
- **本章全部代码（一个文件）**：[`basic/chapter_10_distributed_inference.py`](../basic/chapter_10_distributed_inference.py) — 运行 `python3 basic/chapter_10_distributed_inference.py`
- Megatron-LM 论文：[Efficient Large-Scale Language Model Training](https://arxiv.org/abs/2104.04473)
- OpenAI 面试题梳理：`openAI/openAI_questions.md`（模型并行化章节）

---

*下一章预告：第 11 章将讨论 **Speculative Decoding 与推理加速**——用小型 Draft 模型预测、大型 Target 模型验证，在不损失质量的前提下进一步降低 Decode 延迟。详见 [`chapter_11_speculative_decoding.md`](chapter_11_speculative_decoding.md)。*

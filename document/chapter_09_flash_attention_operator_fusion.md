# 第 9 章 · FlashAttention 与算子融合：Kernel 层 Attention 优化

> **本章导读**：第 8 章从**系统调度层**解决了「请求怎么组 batch、历史 K/V 怎么缓存」的问题——KV Cache 让 Decode 不再重复计算，Continuous Batching 让 GPU 不再空转。但 Attention 本身仍是推理中最贵、最吃内存带宽的操作。本章下沉到 **GPU Kernel 层**，讲解 **FlashAttention** 与 **算子融合**——从算法与硬件 IO 视角进一步压缩 Attention 的内存与计算开销，与 KV Cache 形成「**系统 + 算子**」双层优化。**全部可运行代码见一个文件**：[`basic/chapter_09_flash_attention_operator_fusion.py`](../basic/chapter_09_flash_attention_operator_fusion.py)

---

## 9.1 从系统优化到算子优化：双层架构

LLM 推理优化可以分成两个正交的层次：

```
┌─────────────────────────────────────────────────────────────┐
│                    系统调度层（第 8 章）                       │
│   Continuous Batching · PagedAttention · Prefill/Decode 调度  │
│   → 决定「算什么、何时算、算哪些请求」                          │
└────────────────────────────┬────────────────────────────────┘
                             │ 每次 forward 仍要执行 Attention
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    算子 / Kernel 层（本章）                    │
│   FlashAttention · 算子融合 · 定制 CUDA Kernel                │
│   → 决定「单次 Attention 怎么算、读写多少 HBM」                 │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    模型压缩层（第 7 章）                       │
│   INT8/FP16 量化 · 剪枝 · 蒸馏                                │
│   → 决定「模型权重有多小、精度格式」                            │
└─────────────────────────────────────────────────────────────┘
```

| 层次 | 代表技术 | 主要优化对象 | 典型收益 |
|------|----------|-------------|----------|
| **系统层** | KV Cache、Continuous Batching | 重复计算、GPU 空转 | Decode 延迟 ↓↓、吞吐 ↑↑ |
| **算子层** | FlashAttention、算子融合 | HBM 读写、Kernel 启动 | Prefill 加速、显存 ↓、带宽利用 ↑ |
| **模型层** | INT8 量化 | 权重体积 | 显存 ↓、加载更快 |

**关键认知**：KV Cache 解决的是「**不算第二次**」；FlashAttention 解决的是「**算这一次时少搬数据**」。二者互补，不可互相替代。

---

## 9.2 标准 Attention 的内存瓶颈

回顾标准 Scaled Dot-Product Attention。**本章全部可运行代码见一个文件**：[`basic/chapter_09_flash_attention_operator_fusion.py`](../basic/chapter_09_flash_attention_operator_fusion.py)

```python
# 标准实现（概念示意）
scores = Q @ K.T / sqrt(d_k)     # [L, L]
weights = softmax(scores)        # [L, L]  ← 必须写回 HBM
output = weights @ V             # [L, d_v]
```

对序列长度 L、head 维度 d，**计算量**是 O(L²·d)——这是不可避免的（每个 query 要看所有 key）。

但 **内存访问** 才是实际瓶颈：

| 中间张量 | 形状 | FP16 大小（L=4096） |
|----------|------|---------------------|
| `scores` / `weights` | L × L | 4096² × 2 ≈ **32 MB** / head |
| Q, K, V | L × d | 相对较小 |

LLaMA-7B 有 32 层 × 32 heads，Prefill 阶段若 L=4096：

```
Attention 矩阵总 HBM 读写 ≈ 32 × 32 × 32 MB ≈ 32 GB（仅中间矩阵）
```

**GPU 算力增长远快于 HBM 带宽增长**——Attention 是典型的 **Memory-bound** 操作：计算单元在等数据，而非在等 FLOPs。

### 9.2.1 内存层级：SRAM vs HBM

```
┌──────────────────────────────────────────┐
│  GPU 芯片                                 │
│  ┌────────────┐   带宽 ~19 TB/s           │
│  │ SRAM       │   容量 ~20 MB (SM 共享)    │
│  │ (On-chip)  │   ← FlashAttention 在这里算 │
│  └─────┬──────┘                            │
│        │ 瓶颈！                             │
│  ┌─────▼──────┐   带宽 ~2 TB/s             │
│  │ HBM (显存)  │   容量 24–80 GB           │
│  │            │   ← 标准 Attention 反复读写 │
│  └────────────┘                            │
└──────────────────────────────────────────┘
```

标准实现把 L×L 的 Attention 矩阵**完整写入 HBM**，再读回来做 `@ V`——大量带宽浪费在**中间结果搬运**上，而非有效计算。

---

## 9.3 FlashAttention：IO-aware 分块 Attention

FlashAttention（Dao et al., 2022）的核心思想：**Never materialize the full attention matrix in HBM.**

### 9.3.1 分块（Tiling）

将 Q、K、V 切成能放进 SRAM 的小块（block），在外层循环中逐块计算：

```
Q 矩阵:  [Q₁ | Q₂ | ... | Q_Br]     ← Br 行一块
K 矩阵:  [K₁ | K₂ | ... | K_Bc]     ← Bc 列一块
V 矩阵:  [V₁ | V₂ | ... | V_Bc]
```

对每个 (Qᵢ, Kⱼ, Vⱼ) 块，在 **SRAM 内** 完成：

```
S_ij = Q_i @ K_j^T / sqrt(d)     # 小块，留在 SRAM
P_ij = softmax(S_ij)             # 不写入 HBM
O_ij = P_ij @ V_j                # 累加到输出
```

### 9.3.2 Online Softmax：跨块合并

Softmax 需要全局 max 和 sum，分块后不能直接做。FlashAttention 使用 **Online Softmax**（流式 Softmax）：

```
维护全局: m (running max), l (running sum)

每处理一个新块 S_new:
  m_new = max(m_old, max(S_new))
  l_new = exp(m_old - m_new) * l_old + sum(exp(S_new - m_new))
  O_new = (l_old / l_new) * exp(m_old - m_new) * O_old
        + (1 / l_new) * exp(S_new - m_new) * P_new @ V_new
```

这样可以在**不看到全部 scores 的情况下**，逐块更新最终 Attention 输出——数学上与标准 Softmax **完全等价**。

### 9.3.3 IO 复杂度对比

| 实现 | HBM 读写（Attention 矩阵） | 额外 HBM 占用 |
|------|---------------------------|---------------|
| **标准 Attention** | O(L²) 读写 L×L 矩阵 | L² 中间张量 |
| **FlashAttention** | O(L²·d²/M) ≈ 接近理论下界 | **O(L·d)** 仅输出 |

其中 M 是 SRAM 容量。FlashAttention 把 HBM 访问从「与 L² 成正比」压到「与计算量匹配的最优 IO 复杂度」。

### 9.3.4 算法流程（伪代码）

```python
# FlashAttention 外层逻辑（简化）
def flash_attention(Q, K, V, block_size):
    N, d = Q.shape
    O = zeros(N, d)
    m = full(-inf, N)    # running max per row
    l = zeros(N)         # running sum per row

    for j in range(0, N, block_size):       # 遍历 K,V 块
        K_j, V_j = K[j:j+block_size], V[j:j+block_size]
        for i in range(0, N, block_size):   # 遍历 Q 块
            Q_i = Q[i:i+block_size]
            S_ij = Q_i @ K_j.T / sqrt(d)    # 在 SRAM 中
            m_new, l_new, P_ij = online_softmax_update(S_ij, m, l)
            O[i:i+block_size] = rescale_and_accumulate(O, P_ij, V_j, ...)
            m, l = m_new, l_new
    return O
```

**实际 CUDA 实现**还包含：warp 级并行、共享内存布局、causal mask 的特殊处理等——这些细节由 `flash-attn` 库封装，用户无需手写。

---

## 9.4 FlashAttention-2 与后续演进

### 9.4.1 FlashAttention-2 改进点

| 维度 | FlashAttention-1 | FlashAttention-2 |
|------|-----------------|------------------|
| 并行策略 | 按 batch×head 并行 | 更细粒度：序列维度也并行 |
| 非 matmul 开销 | 较高 | 减少 rescaling、warp 同步 |
| 典型加速 | 基线 | 在 A100 上约 **~2×** |
| 支持 | 训练 + 推理 | 训练 + 推理，backward 同样优化 |

### 9.4.2 FlashAttention-3 与硬件协同

FlashAttention-3 面向 Hopper 架构（H100），利用 **Tensor Memory Accelerator (TMA)** 和 **FP8** 进一步压榨带宽与算力——属于「算法 + 硬件 co-design」的典范。

### 9.4.3 与 Serving 系统的集成

| 变体 | 用途 |
|------|------|
| **FlashAttention** | 标准 dense attention，Prefill 主力 |
| **FlashAttention + varlen** | 变长序列 batch，配合 Continuous Batching |
| **PagedAttention**（vLLM） | KV Cache 分页 + 定制 Attention Kernel |
| **FlashDecoding** | Decode 阶段 Q=1 时的 KV Cache 读取优化 |

> **PagedAttention** 是 vLLM 在系统层对 KV Cache 的管理方案，其底层 Kernel 与 FlashAttention 思想同源——都是减少无效 HBM 访问。

---

## 9.5 算子融合（Operator Fusion）

### 9.5.1 为什么需要融合

一次 Transformer 层的前向包含大量**细粒度操作**：

```
x → LayerNorm → Linear(QKV) → RoPE → Attention → Linear(Out) → Dropout → +残差 → ...
```

PyTorch 默认每个操作 = **一次 CUDA Kernel 启动**：

| 问题 | 影响 |
|------|------|
| **Kernel 启动开销** | 小 batch / Decode 时，启动时间可占显著比例 |
| **中间结果写 HBM** | 每个 op 输出都落盘，下个 op 再读回来 |
| **内存带宽浪费** | LayerNorm 输出 → 立刻被 Linear 消费，本可留在 SRAM |

**算子融合**：把多个连续、数据依赖的操作**合并为一个 Kernel**，中间结果留在寄存器 / 共享内存，不落 HBM。

### 9.5.2 典型融合模式

```
融合前（3 个 Kernel）:          融合后（1 个 Kernel）:
  x ──→ LayerNorm ──→ HBM       x ──→ [LayerNorm + Linear + BiasAdd] ──→ HBM
           │                              ↑ 中间结果在 SRAM
           ▼
        Linear ──→ HBM
           │
           ▼
        BiasAdd ──→ HBM
```

LLM 推理中常见的融合组合：

| 融合模式 | 包含操作 | 收益 |
|----------|----------|------|
| **QKV 融合** | 一次 Linear 同时投影 Q、K、V | 减少 3 次 launch → 1 次 |
| **Attention 融合** | Softmax + Dropout + @V | 减少中间 weights 落盘 |
| **FFN 融合** | Gate × Up + Activation（SwiGLU） | 减少激活张量读写 |
| **Norm + Linear** | RMSNorm/LayerNorm + 紧随的 Linear | Decode 小 batch 收益明显 |
| **RoPE + Attention** | 旋转位置编码嵌入 Attention Kernel | 减少 Q/K 写回 |

### 9.5.3 融合工具与框架

| 工具 | 方式 | 特点 |
|------|------|------|
| **`flash-attn`** | 手写 CUDA | Attention 专用，性能最优 |
| **Apex / Transformer Engine** | NVIDIA 融合 Kernel | LayerNorm、Linear、Softmax 等 |
| **`torch.compile`** | 编译期自动融合 | 通用，PyTorch 2.x 推荐 |
| **TensorRT-LLM** | 图级融合 + 定制 Kernel | 部署级，NVIDIA 硬件深度优化 |
| **Triton** | 用户写 DSL，编译为 GPU 代码 | 灵活，FlashAttention 早期原型用 Triton |

```python
# torch.compile 示例：自动融合相邻操作
model = torch.compile(model, mode="reduce-overhead")
# 编译器会识别可融合的子图，生成 fewer kernels
```

**注意**：融合不是免费的——Kernel 越大，寄存器压力越高，可能降低 occupancy（同时活跃的 warp 数）。生产环境需要 profiling 验证。

---

## 9.6 与 KV Cache 的协同：系统 + 算子双层优化

第 8 章 KV Cache 与 FlashAttention 在不同阶段、不同维度上发挥作用：

```
                    Prefill 阶段                    Decode 阶段
                 （处理整段 prompt）              （逐 token 生成）
              ┌─────────────────────┐          ┌─────────────────────┐
  系统层      │ 建立 KV Cache        │          │ 增量 append K,V      │
  (第 8 章)   │ 一次处理 L 个 token  │          │ 每步只输入 1 token   │
              └──────────┬──────────┘          └──────────┬──────────┘
                         │                                │
                         ▼                                ▼
  算子层      │ FlashAttention:          │          │ FlashDecoding /       │
  (本章)      │ Q,K,V 全长 L，           │          │ 融合 Kernel:          │
              │ 分块避免 L² 矩阵         │          │ Q=1，读全长 KV Cache  │
              │ Memory-bound → 加速      │          │ Memory-bound → 优化读 │
              └─────────────────────────┘          └─────────────────────┘
```

### 9.6.1 Prefill：FlashAttention 的主场

- 输入：L 个 token 的 Q、K、V 同时存在
- 瓶颈：L×L Attention 矩阵的 HBM 读写
- **FlashAttention** 直接消除 L² 中间矩阵，Prefill 加速 2–4× 很常见
- 与 **Chunked Prefill**（第 8 章）配合：长 prompt 分块，每块用 FlashAttention

### 9.6.2 Decode：KV Cache + 融合 Kernel

- 输入：Q 仅 1 个 token（新 token），K/V 从 Cache 读取全长 L
- 瓶颈：从 HBM **读取** 全部历史 K/V（Memory-bound）
- KV Cache（系统层）已避免**重算** K/V
- 算子层优化方向：
  - **FlashDecoding**：并行读取多个 head 的 KV，减少 latency
  - **QKV 投影 + RoPE 融合**：减少小矩阵操作的 launch 开销
  - **PagedAttention Kernel**：非连续 KV 块的高效 gather + attention

### 9.6.3 双层优化的量化感受

以 LLaMA-7B、L=2048、FP16 为例（粗算）：

| 优化 | 影响 | 效果 |
|------|------|------|
| 无优化 | 每 decode step 重算全部 K/V + 标准 Attention | 基线 |
| **+ KV Cache** | 每 step 只算 1 token 的 Q/K/V | Compute ↓ ~L 倍 |
| **+ FlashAttention** | Prefill 不写 L² 矩阵；Decode 优化 KV 读取 | HBM 带宽 ↓ 2–4× |
| **+ 算子融合** | 减少 Kernel launch + 中间张量 | Decode latency ↓ 10–30% |
| **+ 量化（第 7 章）** | 权重 INT8/FP16 | 显存 ↓，更多并发 |

**三者叠加**：量化腾出显存 → 更多并发 slot → Continuous Batching 喂饱 GPU → FlashAttention 让每次 Attention 更快 → 端到端吞吐显著提升。

---

## 9.7 生产实践一览

| 框架 | Kernel 层优化 | 系统层优化 | 说明 |
|------|--------------|-----------|------|
| **vLLM** | PagedAttention Kernel | Continuous Batching + Paged KV | 开源 serving 标杆 |
| **TGI** | FlashAttention-2 | Continuous Batching | HF 生态 |
| **TensorRT-LLM** | 全面算子融合 + 定制 Kernel | In-flight Batching | NVIDIA 部署首选 |
| **PyTorch 2.x + flash-attn** | `scaled_dot_product_attention` 自动选 backend | 需自行搭建 serving | 开发友好 |

**启用 FlashAttention 的常见方式**：

```python
# 方式 1：PyTorch 2.0+ 内置（自动选 flash 或 mem-efficient backend）
import torch.nn.functional as F
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

# 方式 2：flash-attn 库（更细粒度控制，支持 varlen / paged）
from flash_attn import flash_attn_func
out = flash_attn_func(q, k, v, causal=True)

# 方式 3：HuggingFace model（内部自动调用）
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    attn_implementation="flash_attention_2",  # 或 "sdpa"
)
```

**Profiling 建议**：用 `torch.profiler` 或 Nsight Systems 对比优化前后：

```
关注指标:
  - cudaLaunchKernel 次数        → 融合是否生效
  - HBM throughput (GB/s)        → FlashAttention 是否减少读写
  - Attention op 耗时占比         → 是否仍是瓶颈
  - SM occupancy                 → 融合 Kernel 是否寄存器溢出
```

---

## 9.8 与第 7/8 章、代码实现的关系

| 章节/文件 | 优化层次 | 核心手段 |
|-----------|----------|----------|
| **第 7 章 / Problem 3** | 模型层 | INT8/FP16 量化 |
| **第 8 章** | 系统调度层 | KV Cache、Continuous Batching |
| **第 9 章（本章）** | 算子 / Kernel 层 | FlashAttention、算子融合 |
| **`basic/chapter_09_flash_attention_operator_fusion.py`** | **本章全部代码（一个文件）** | 标准 Attention、FlashAttention、Online Softmax、KV Cache、算子融合、PyTorch SDPA |

标准 Attention 实现（本章的「优化前」对照）：

```python
# basic/transformer_implementation.py — 标准 Scaled Dot-Product Attention
attention_scores = torch.matmul(q, k.transpose(-2, -1))
attention_scores = attention_scores / math.sqrt(d_k)
attention_weights = F.softmax(attention_scores, dim=-1)  # L×L 写入 HBM
output = torch.matmul(attention_weights, v)
```

FlashAttention 在**数学上产生相同结果**，但**从不 materialize `attention_weights` 到 HBM**——这是面试中最常被问到的区别。

---

## 9.9 本章小结

| 概念 | 一句话 |
|------|--------|
| **HBM 墙** | Attention 的瓶颈在内存带宽，不在 FLOPs |
| **FlashAttention** | 分块 + Online Softmax，避免 L×L 矩阵落盘，IO 最优 |
| **FlashAttention-2** | 更好的并行与 warp 调度，约 2× 于 v1 |
| **算子融合** | 多 op 合并为一个 Kernel，减少 launch 和中间 HBM 读写 |
| **系统 + 算子双层** | KV Cache 减少算什么；FlashAttention 优化怎么算 |
| **Prefill vs Decode** | Prefill 用 FlashAttention 加速；Decode 优化 KV 读取 + 小 op 融合 |

---

## 9.10 思考题与参考答案

### 思考题 1

KV Cache 已经让 Decode 每步只算 1 个 token 的 Q/K/V，为什么还需要 FlashAttention？

**参考答案**：

KV Cache 解决的是**重复计算**——避免每步对全部历史 token 重新做 Q/K/V 投影和 Attention。但 Decode 每步仍要做一次 Attention：`Q_new @ K_cache^T`，其中 K_cache 长度随生成增长。这一步是 **Memory-bound**（读取全部 KV Cache），FlashDecoding 等 Kernel 优化可以并行读取、减少 latency。此外 **Prefill 阶段**没有「不算第二次」的空间——FlashAttention 对 Prefill 的 L×L 矩阵优化是 KV Cache 无法替代的。

### 思考题 2

标准 Attention 与 FlashAttention 的输出是否完全一致？为什么？

**参考答案**：

**数学上完全等价**（在相同浮点精度下）。FlashAttention 使用 Online Softmax 分块累加，是对标准 Softmax 的**精确重组**，不是近似。实际运行中可能因浮点累加顺序不同产生 **~1e-6 量级** 的数值差异，但不影响模型质量。这与「低秩近似 Attention」（如 Linformer）有本质区别——后者是算法近似，FlashAttention 是 **IO 优化**。

### 思考题 3

Decode 阶段 batch=1 时，算子融合还能带来收益吗？

**参考答案**：

**能，但收益小于 Prefill**。Decode 的瓶颈是读 KV Cache（Memory-bound），融合无法减少 KV 数据量。但 Decode 每步涉及大量小操作（RMSNorm、QKV Linear、RoPE、Attention、FFN），每个独立 Kernel 的 **launch 开销**（~5–20 μs）在 batch=1 时可占显著比例。融合 QKV 投影、Norm+Linear 等可削减 launch 次数，典型带来 **10–30%** 的 Decode latency 改善。batch 越大，Attention 本身占比越高，FlashAttention 收益越大。

---

## 相关资源

- 第 7 章：[`document/chapter_07_model_quantization.md`](chapter_07_model_quantization.md) — 模型量化与推理优化
- 第 8 章：[`document/chapter_08_inference_pipeline.md`](chapter_08_inference_pipeline.md) — KV Cache 与 Continuous Batching
- **本章全部代码（一个文件）**：[`basic/chapter_09_flash_attention_operator_fusion.py`](../basic/chapter_09_flash_attention_operator_fusion.py) — 运行 `python3 basic/chapter_09_flash_attention_operator_fusion.py`
- 标准 Attention 扩展阅读：`basic/transformer_implementation.py`（含完整 Multi-Head Attention）
- FlashAttention 论文：[FlashAttention (NeurIPS 2022)](https://arxiv.org/abs/2205.14135)、[FlashAttention-2 (2023)](https://arxiv.org/abs/2307.08691)
- 代码库：[Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
- OpenAI 面试题梳理：`openAI/openAI_questions.md`（算子融合与 FlashAttention 章节）

---

*下一章预告：第 10 章将讨论 **分布式推理与模型并行**——当单卡放不下模型或 KV Cache 时，如何通过 Tensor Parallelism、Pipeline Parallelism 将推理扩展到多 GPU。详见 [`chapter_10_distributed_inference.md`](chapter_10_distributed_inference.md)。*

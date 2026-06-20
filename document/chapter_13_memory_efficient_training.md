# 第 13 章 · 大模型训练基础设施：内存、并行与吞吐

> **本章导读**：第 7–12 章聚焦**推理**；本章补全 **AI Infra 面试里训练侧的高频考点**——显存构成、混合精度、梯度检查点、**分布式并行（DDP / TP / PP / ZeRO / FSDP）**、**FSDP / DeepSpeed 配置与 Megatron 1F1B**、**训练吞吐（tokens/s、MFU）**、数据流水线、Flash Attention、LoRA/QLoRA，以及 OOM 排查与 Checkpoint 容错。代码主线：`Problem_4_openAI_memory_efficient_training.py`（单卡 AMP + 检查点实验）；概念与估算 demo：**[`basic/chapter_13_memory_efficient_training.py`](../basic/chapter_13_memory_efficient_training.py)**。

---

## 13.1 训练 vs 推理：内存问题有什么不同

| 维度 | 训练 | 推理（第 7 章起） |
|------|------|-------------------|
| **核心目标** | 收敛、泛化 | 低延迟、高吞吐 |
| **额外内存** | 梯度、优化器状态、激活值（反向用） | 主要是 KV Cache（生成场景） |
| **精度策略** | AMP、Loss Scaling 防梯度下溢 | INT8/FP16 量化压缩权重 |
| **典型瓶颈** | 激活值随层数线性增长 | 请求调度、Attention IO |

**关键认知**：推理优化是「让模型变轻、调度变聪明」；训练优化是「在 backward 必须保存的信息与显存上限之间做权衡」。

---

## 13.2 训练时 GPU 显存都花在哪

一个训练 step 的峰值显存，大致来自四块：

```
GPU 显存 ≈ 模型参数 + 梯度 + 优化器状态 + 激活值（Activations）
```

| 组件 | 说明 | 典型规模（Adam, FP32） |
|------|------|------------------------|
| **模型参数** | 权重 W | P |
| **梯度** | ∂L/∂W | P |
| **优化器状态** | Adam 的一阶矩 m、二阶矩 v | 2P |
| **激活值** | 各层 forward 中间结果，backward 需要 | 与层数、batch、序列长度成正比 |

对 **Adam + FP32** 训练，仅「参数 + 梯度 + 优化器」就约 **4× 参数量** 的存储（每参数 4 字节 × 4 份 ≈ 16 字节/参数）。

**激活值**往往是深层 Transformer 的隐形大户：forward 时每一层都会产生中间 tensor；backward 默认需要这些 tensor 计算梯度。层数从 16 增到 64，激活内存近似线性增长——这正是梯度检查点要解决的问题。

### 13.2.1 先测量，再优化

与第 7 章推理优化相同的原则：**没有基线，就没有优化。**

```python
torch.cuda.reset_peak_memory_stats()
# ... 一次完整的 forward + backward + optimizer.step ...
peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
```

`Problem_4` 中的 `compare_memory_usage()` 正是按此思路，依次跑四种配置并对比峰值显存。

---

## 13.3 演示架构：为什么需要「内存密集型」模型

`Problem_4_openAI_memory_efficient_training.py` 用两个类模拟大模型训练场景：

```
Input
  ↓
input_projection
  ↓
[ MemoryIntensiveBlock × num_layers ]   ← 默认 32 层
  ↓
output_projection
  ↓
Output
```

### 13.3.1 MemoryIntensiveBlock

每个 block 模拟 Transformer FFN 的「扩维 → 激活 → 缩维」：

```
Input (hidden_dim)
  → Linear(hidden_dim → 4×hidden_dim)   ← 最大中间张量
  → ReLU
  → Linear(4×hidden_dim → hidden_dim)
  → LayerNorm
  → Output
```

**设计意图**：`4× hidden_dim` 的扩维层在 forward 时产生大量激活值，使不同优化手段的差异可被 `max_memory_allocated()` 清晰观测。

### 13.3.2 LargeModel 与梯度检查点开关

```python
class LargeModel(nn.Module):
    def __init__(self, num_layers=32, hidden_dim=1024, use_checkpointing=False):
        self.use_checkpointing = use_checkpointing
        self.layers = nn.ModuleList([
            MemoryIntensiveBlock(hidden_dim) for _ in range(num_layers)
        ])
        # input_projection, output_projection ...

    def forward(self, x):
        x = self.input_projection(x)
        for i, layer in enumerate(self.layers):
            if self.use_checkpointing and self.training:
                x = checkpoint(layer, x)   # 训练时用检查点
            else:
                x = layer(x)
        x = self.output_projection(x)
        return x
```

要点：

- `checkpoint(layer, x)` 只在 **`self.training == True`** 时生效；eval / 推理走普通 forward。
- 默认配置：`num_layers=32`, `hidden_dim=1024`，配合 `batch=16, seq_len=1024` 的合成输入，足以在多数消费级 GPU 上观察 OOM 与优化效果。

---

## 13.4 混合精度训练（Mixed Precision / AMP）

### 13.4.1 原理

在 forward 的大部分算子中使用 **FP16（或 BF16）**，减少激活值与部分计算的内存与带宽；权重更新仍可在 FP32 主副本上进行，保证数值稳定。

| 数据类型 | 每元素字节 | 相对 FP32 |
|----------|-----------|-----------|
| FP32 | 4 | 1× |
| FP16 / BF16 | 2 | ~0.5× |

在支持 **Tensor Core** 的 NVIDIA GPU（Volta 及以后）上，FP16 矩阵乘还有额外吞吐优势——训练可能**既省内存又更快**。

### 13.4.2 为什么需要 Loss Scaling

FP16 可表示范围远小于 FP32。反向传播时，小梯度可能 **underflow 变为 0**，导致部分参数不更新。做法是：

1. `scaler.scale(loss).backward()` — 放大 loss，梯度同比例放大
2. `scaler.step(optimizer)` — 更新前 unscale，若出现 inf/nan 则跳过本次 step
3. `scaler.update()` — 动态调整 scale 因子

### 13.4.3 代码实现（Problem_4）

```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler() if use_amp else None

with autocast(enabled=use_amp):
    output = model(input_data)
    loss = output.mean()

if use_amp:
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
else:
    loss.backward()
    optimizer.step()
```

**现代 PyTorch 提示**：新代码可优先使用 `torch.amp.autocast('cuda')` 与 `torch.amp.GradScaler('cuda')`；`Problem_4` 使用的是经典 `torch.cuda.amp` API，面试与旧项目里仍很常见。

### 13.4.4 预期效果

| 指标 | 典型变化 |
|------|----------|
| 激活值内存 | 约 **30–50%** 下降 |
| 权重（若全 FP16 存储） | 约 **50%** 下降 |
| 精度影响 | 多数 LLM 任务可忽略；不稳定时可试 BF16 |
| 硬件要求 | Volta+ GPU；CPU 训练收益有限 |

---

## 13.5 梯度检查点（Gradient Checkpointing）

### 13.5.1 核心思想：用计算换内存

标准 backward 需要 forward 保存的**全部激活值**。梯度检查点只保留**部分检查点**上的激活；其余在 backward 时**从最近检查点重新 forward 一遍**再算梯度。

```
标准训练:
  Forward:  存激活 [L1, L2, L3, ..., L32]  →  内存 O(L)
  Backward: 直接用缓存的激活

检查点训练:
  Forward:  只存少量检查点（或每层入口）
  Backward: 需要 L17 的激活？→ 从 L16 检查点重新算 L17 forward
```

### 13.5.2 权衡

| 优点 | 缺点 |
|------|------|
| 激活内存可降 **50–80%**（与分段策略有关） | 训练时间通常增加 **~20–30%** |
| 使「单卡训更深模型」成为可能 | 实现需注意 `checkpoint` 与 `dropout`/随机性的兼容 |
| 与 AMP 正交，可叠加 | 推理时不启用（`model.eval()` 下无 backward） |

### 13.5.3 PyTorch 用法

```python
from torch.utils.checkpoint import checkpoint

# 在 forward 内，对「整层」或「子模块」包裹
x = checkpoint(layer, x)

# 若层内有 random op（Dropout），可传 use_reentrant=False（PyTorch 2.x 推荐）
x = checkpoint(layer, x, use_reentrant=False)
```

Hugging Face Transformers 中常见：`model.gradient_checkpointing_enable()`，本质相同。

### 13.5.4 内存量级直觉

对 `L` 层、隐藏维 `d`、batch `B`、序列 `T` 的模型，粗略有：

- **无检查点**：激活峰值 ∝ **O(L × B × T × d)**
- **每层检查点**：峰值可接近 **O(B × T × d)**（常数级层缓存 + 重算开销）

`Problem_4` 注释中的「无检查点 ~ num_layers × hidden_dim²；有检查点 ~ hidden_dim² 量级」表达的是同一思想：**深度带来的线性激活堆积被压平**。

---

## 13.6 四种场景对比实验

`compare_memory_usage()` 在 CUDA GPU 上依次运行：

```
┌─────────────────────────────────────────────────────────────┐
│  场景 1 · 标准 FP32（无 AMP，无检查点）                        │
└────────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  场景 2 · 仅混合精度（AMP=True，无检查点）                   │
└────────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  场景 3 · 仅梯度检查点（AMP=False，checkpoint=True）         │
└────────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  场景 4 · AMP + 梯度检查点（组合）                            │
└────────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  汇总表格：峰值 GB、相对基线的节省百分比                        │
└─────────────────────────────────────────────────────────────┘
```

每轮实验后执行 `del model, optimizer` 与 `torch.cuda.empty_cache()`，避免上一轮显存污染下一轮测量。

### 13.6.1 结果解读模板

| 优化技术 | 内存使用 | 相对基线节省 | 计算开销 |
|----------|----------|--------------|----------|
| Standard FP32 | 基准 / 可能 OOM | 0% | 1× |
| Mixed Precision | ↓ | ~30–50% | ≤1×（常更快） |
| Gradient Checkpointing | ↓↓ | ~50–80% | ~1.2–1.3× |
| Combined | ↓↓↓ | ~70–90% | ~1.2–1.3× |

**注意**：具体数字依赖 GPU 型号、驱动、PyTorch 版本与输入形状；`Problem_4` 给出的是**实验框架 + 量级预期**，务必在目标环境上实测。

### 13.6.2 性能基准：`benchmark_performance()`

除峰值显存外，脚本还用 10 次迭代的平均 step 时间对比 speedup：

```python
configs = [
    ("Standard FP32", False, False),
    ("Mixed Precision", True, False),
    ("Gradient Checkpointing", False, True),
    ("Combined", True, True),
]
```

典型结论：

- **AMP**：往往 **≥1×**（有时明显加速）
- **检查点**：通常 **<1×**（重算带来额外 forward）
- **组合**：内存最优，速度介于两者之间

---

## 13.7 分布式训练：DDP、TP、PP 与 3D 并行

AI Infra 面试几乎必问：**模型放不进一张卡 / 训太慢时，怎么扩？** 先分清三种并行切的是什么。

```
                    ┌─────────────────────────────────────────┐
                    │           一个 Training Step             │
                    └─────────────────────────────────────────┘
                                        │
          ┌─────────────────────────────┼─────────────────────────────┐
          ▼                             ▼                             ▼
   ┌─────────────┐              ┌─────────────┐              ┌─────────────┐
   │ Data Parallel│              │Tensor Parallel│              │Pipeline Parallel│
   │  (DDP)       │              │  (TP)         │              │  (PP)           │
   └─────────────┘              └─────────────┘              └─────────────┘
   每卡完整模型副本               每层权重切多卡                   不同层放不同卡
   各卡不同 micro-batch           需 AllReduce / AllGather          需 P2P 传激活
   梯度 AllReduce 同步            Megatron 标准                    1F1B 调度减气泡
```

| 范式 | 切什么 | 解决什么问题 | 主要通信 | 典型框架 |
|------|--------|--------------|----------|----------|
| **DDP** | 数据（batch 维） | 加速训练、等效大 batch | **AllReduce** 梯度 | `torch.nn.parallel.DistributedDataParallel` |
| **TP** | 层内矩阵 / head | 单卡放不下单层权重 | AllReduce / AllGather | Megatron-LM |
| **PP** | 层间（depth） | 极深 / 极大模型 | **P2P** 发送激活 | Megatron PP、DeepSpeed PP |

### 13.7.1 DDP 训练一步在做什么

```
GPU 0: forward(micro_batch_0) → backward → grad_0 ─┐
GPU 1: forward(micro_batch_1) → backward → grad_1 ─┼─→ AllReduce(grad) → 各卡相同 grad → optimizer.step()
GPU 2: forward(micro_batch_2) → backward → grad_2 ─┤
GPU 3: forward(micro_batch_3) → backward → grad_3 ─┘
```

- **Global Batch** = `micro_batch_per_gpu × num_gpus × grad_accum_steps`
- **问题**：每卡仍存**完整**参数 + 梯度 + 优化器 → 7B 模型 DDP 4 卡，**每卡**仍 ~100 GB+ 状态（FP32 Adam），不能训 70B

### 13.7.2 TP / PP 训练侧要点（与第 10 章推理视角互补）

- **TP**：切权重也切激活，单卡峰值显存 ↓；每层有通信，**TP 度不宜过大**（常 2–8）
- **PP**：每卡只存部分层，但 **pipeline bubble** 导致 GPU 空转；用 **1F1B**（One Forward One Backward）调度缓解
- **3D 并行**：`DP × TP × PP = world_size`——例如 64 卡 = DP8 × TP4 × PP2，工业界训 100B+ 的常见组合

> **面试一句话**：DDP 扩吞吐但不减单卡显存；TP/PP 减单卡显存；ZeRO/FSDP 在 DDP 基础上再分片状态（见 13.8）。

---

## 13.8 ZeRO / FSDP：分片优化器状态与参数

传统 DDP 的内存冗余是面试高频陷阱：

| 组件 | DDP（每卡） | ZeRO-3 / FSDP FULL_SHARD（每卡约） |
|------|-------------|-------------------------------------|
| 参数 | P | P / N |
| 梯度 | P | P / N |
| 优化器 (Adam) | 2P | 2P / N |
| **合计** | **~4P** | **~4P / N** |

N = GPU 数量。ZeRO 分三阶段渐进：

| 阶段 | 分片内容 | 相对 DDP 优化器内存 |
|------|----------|---------------------|
| **ZeRO-1** | 优化器状态 | ~1/N |
| **ZeRO-2** | + 梯度 | 梯度也 1/N |
| **ZeRO-3** | + 参数 | 参数也 1/N，forward 前 **AllGather** 临时凑齐 |

**FSDP**（PyTorch 原生）≈ ZeRO-3 思想，常用策略：

| FSDP 策略 | 行为 |
|-----------|------|
| `FULL_SHARD` | 参数/梯度/优化器全分片（最省显存） |
| `SHARD_GRAD_OP` | 类似 ZeRO-2 |
| `NO_SHARD` | 接近 DDP |

**通信**：ZeRO-3 / FSDP 在 forward/backward 中穿插 AllGather / ReduceScatter；可与 **overlap**（`forward_prefetch`）隐藏部分延迟。

---

## 13.9 单卡常用手段：梯度累积与技术选型

### 13.9.1 梯度累积（Gradient Accumulation）

显存放不下大 batch 时，多个 micro-batch 累积梯度再 `optimizer.step()`：

```python
accum_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accum_steps
    loss.backward()
    if (i + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

- **Global batch** 变大，**峰值激活内存 ≈ 单个 micro-batch**
- 与 DDP 叠加：`global_batch = micro_batch × accum × num_gpus`

### 13.9.2 技术选型速查（面试表格题）

| 技术 | 主要省什么 | 内存效果 | 计算/通信 | 难度 | 何时用 |
|------|-----------|----------|-----------|------|--------|
| **AMP (BF16/FP16)** | 激活、部分权重 | ~30–50% | 常更快 | 低 | **默认开** |
| **Gradient Checkpointing** | 激活 | ~50–80% | +20–30% 时间 | 中 | 深模型 OOM |
| **Gradient Accumulation** | 不省显存，扩 effective batch | 峰值不变 | 步数变多 | 低 | batch 受限 |
| **ZeRO-1/2/3 / FSDP** | 优化器/梯度/参数 | ~4×~N× | AllGather 通信 | 高 | 7B+ 多卡 |
| **TP / PP** | 单层/单段放不下 | 按切分比 | TP/PP 通信 | 高 | 超大模型 |
| **LoRA / QLoRA** | 只训低秩适配器 | 梯度/优化器大减 | 略增前向 | 中 | 微调 |
| **8-bit Optimizer** | 优化器状态 | ~2× on opt | 略慢 | 低 | Adam 状态太大 |
| **CPU / NVMe Offload** | 优化器或参数 | 显著 | PCIe/磁盘 IO | 中 | 显存极紧 |

---

## 13.10 训练吞吐：tokens/s 与 MFU

Infra 面试不只问「能不能训」，还问「训得快不快、卡有没有喂饱」。

### 13.10.1 核心指标

| 指标 | 定义 | 用途 |
|------|------|------|
| **step time** | 一次 optimizer step  wall-clock 时间 | 对比优化手段、定位变慢原因 |
| **tokens/s** | `global_batch × seq_len / step_time` | 横向对比集群效率 |
| **samples/s** | `global_batch / step_time` | 非 LM 任务常用 |
| **MFU**（Model FLOPs Utilization） | 实际 FLOPs / 理论峰值 FLOPs | 衡量「算力利用率」 |

### 13.10.2 MFU 粗算（面试手算）

```
MFU = (实际吞吐 FLOPs/s) / (GPU 峰值 FLOPs/s × GPU 数)
```

对 Transformer 训练，经验公式（PaLM 论文量级）：

```
训练 FLOPs/token ≈ 6 × 参数量   （forward + backward）
tokens/s = global_batch × seq_len / step_time
实际 FLOPs/s ≈ 6 × P × tokens/s
```

**示例**：7B 模型，global batch=512，seq=2048，step=50s，8×A100 BF16 峰值 ~312 TFLOPS/卡：

```
tokens/s = 512 × 2048 / 50 ≈ 21,000
FLOPs/s ≈ 6 × 7×10⁹ × 21,000 ≈ 8.8×10¹⁴
集群峰值 ≈ 8 × 312×10¹² ≈ 2.5×10¹⁵
MFU ≈ 35%  （工业界 30–55% 常见）
```

MFU **>100%** 通常表示 step_time 估太小、未乘 GPU 数，或 batch 统计有误；面试手算时注意单位。

### 13.10.3 瓶颈判断速查

| 现象 | 可能原因 | 排查 |
|------|----------|------|
| GPU util 低、step 慢 | DataLoader 跟不上 | 增 `num_workers`、预 tokenize、WebDataset |
| 多卡扩展性差 | 通信占主导 | 检查 AllReduce 体积、是否 ZeRO-3 过度、网络 NCCL |
| 开 checkpoint 后 MFU 降 | 重算激活 | 预期内；可减小 checkpoint 粒度 |
| loss 正常但 tokens/s 低 | batch/seq 太小 | 增大 micro-batch 或 accum |

---

## 13.11 数据流水线：别让 GPU 等数据

训练集群贵，**GPU 空转等 batch** 是隐性大成本。

```
磁盘 / 对象存储 → 解压 / tokenize → DataLoader → CPU→GPU H2D → forward
                      ↑ 常见瓶颈              ↑ pin_memory, prefetch
```

| 手段 | 作用 |
|------|------|
| **`num_workers > 0`** | 多进程预取 batch |
| **`pin_memory=True`** | 加速 H2D 拷贝 |
| **`prefetch_factor`** | 每个 worker 预取批次数 |
| **离线 tokenize** | 训练时只读 id，避免 CPU 做 BPE |
| **WebDataset / MDS / Arrow mmap** | 大语料顺序读、少 random seek |
| **Gradient Accumulation** | 计算 bound 时隐藏部分 IO |

**面试题**：「8×A100 训练 loss 正常，但 GPU util 只有 40%？」→ 优先查 **DataLoader、tokenization、存储 IO**，而非先上 ZeRO。

---

## 13.12 Transformer 激活内存与 Flash Attention

### 13.12.1 标准 Attention 的激活（为何长序列 OOM）

Prefill / 训练 forward 中，标准 Attention 需物化 **S = QK^T**：

```
S 的大小 ≈ batch × n_heads × seq_len × seq_len × bytes
```

seq 从 2K → 8K，**激活内存 ≈ 16×**（平方关系）——这是长上下文训练 OOM 的首要原因。

FFN 激活（每层）粗算：`~ batch × seq × hidden × expansion × bytes × 常数`。

**总激活**（无检查点）≈ **O(layers × (T² + T×hidden))**。

### 13.12.2 Flash Attention 在训练中的价值

第 9 章从 Kernel 层讲了 FlashAttention；**训练侧**同样关键：

- 不物化完整 **N×N** attention matrix → 激活从 **O(T²)** 降到 **O(T)**
- 与 Gradient Checkpointing **正交**，长序列训练常 **FlashAttn + Checkpoint + AMP** 三件套
- PyTorch 2.x：`torch.nn.functional.scaled_dot_product_attention` 可自动选 flash/mem-efficient backend

---

## 13.13 参数高效微调：LoRA / QLoRA

全量微调 7B+ 时，**优化器状态**仍巨大。LoRA 只训低秩增量：

```
W' = W + B @ A     （W 冻结，A:[d,r], B:[r,d], r << d）
```

| 方式 | 可训参数 | 典型场景 |
|------|----------|----------|
| **Full Finetune** | 100% | 数据多、任务与预训练差异大 |
| **LoRA** | 0.1–1% | 指令微调、领域适配 |
| **QLoRA** | LoRA + 4bit 量化基座 | 单卡 24G 训 7B/13B |

**内存收益**：梯度与优化器只跟 **可训参数** 成正比；7B 全量 Adam ~100GB+ 状态 → LoRA 常 **<10GB** 量级。

---

## 13.14 Checkpoint、Offload 与数值稳定

### 13.14.1 训练 Checkpoint（≠ 梯度检查点）

| 概念 | 含义 |
|------|------|
| **Gradient Checkpointing** | 省激活内存，backward 重算 |
| **Training Checkpoint** | 定期存 `model + optimizer + scheduler + step`，故障恢复 |

生产建议：

- 存 **分布式一致** 的 checkpoint（FSDP `FULL_STATE_DICT` / sharded dict）
- **异步写入** 避免每 N step 全局阻塞
- 记录 **global step、RNG、data cursor**，保证 resume 可复现

### 13.14.2 CPU / NVMe Offload

DeepSpeed **ZeRO-Offload**：优化器状态放 CPU DRAM，forward/backward 时再搬回 GPU——用 **PCIe 带宽** 换 **显存**，适合单节点多卡显存紧、CPU 内存充裕。

### 13.14.3 其他数值与稳定手段

| 手段 | 作用 |
|------|------|
| **Gradient Clipping** | 防 loss spike、大梯度；`clip_grad_norm_(max_norm=1.0)` |
| **Warmup + Cosine LR** | 大 batch / 大模型训练稳定 |
| **BF16 vs FP16** | A100/H100 优先 **BF16**（动态范围大，常免 Loss Scaling） |
| **8-bit Adam** | 优化器状态减半，bitsandbytes |

---

## 13.15 工程实践：FSDP、DeepSpeed 与 Megatron 1F1B

前几节讲了「是什么」；AI Infra 面试还常考「**怎么配、怎么选**」。本节给出三类主流框架的**最小配置模板**与 **Pipeline Parallel 1F1B** 调度直觉。

### 13.15.1 PyTorch FSDP 最小训练骨架

FSDP ≈ ZeRO-3 的 PyTorch 原生实现。核心：用 `FullyShardedDataParallel` 包裹模型，forward 前 **AllGather** 参数，backward 后 **ReduceScatter** 梯度。

```python
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial

def build_fsdp_model(model, world_size_ok=True):
    """Minimal FSDP setup for LLM-style training."""
    mp = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    # 按 TransformerBlock 自动分片；也可 size_based_auto_wrap_policy
    wrap = partial(transformer_auto_wrap_policy, transformer_layer_cls={YourBlock})

    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3
        mixed_precision=mp,
        auto_wrap_policy=wrap,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,  # 便于 optimizer 按 name 分组 LR
    )
    return model

# 训练 loop 与 DDP 类似；注意 save/load 用 FSDP state_dict API
# from torch.distributed.fsdp import FullStateDictConfig, StateDictType
```

| FSDP 参数 | 面试要点 |
|-----------|----------|
| `FULL_SHARD` | 最省显存，通信最多 |
| `SHARD_GRAD_OP` | 类似 ZeRO-2，吞吐常更高 |
| `MixedPrecision(bfloat16)` | A100/H100 训练默认 |
| `auto_wrap_policy` | 按层分片，避免整模型一个 FSDP unit |
| `use_orig_params=True` | 多参数组 / LoRA 微调时需要 |
| `activation_checkpointing` | 可与 `apply_activation_checkpointing` 叠加 |

**Checkpoint 保存**（面试常忘）：

```python
from torch.distributed.fsdp import StateDictType, FullStateDictConfig

with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                          FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
    state = model.state_dict()  # 仅 rank 0 有完整 dict，可 torch.save
```

---

### 13.15.2 DeepSpeed ZeRO 配置要点

DeepSpeed 通过 **`ds_config.json`** 驱动，与 Hugging Face / PyTorch Lightning 集成广。

```json
{
  "train_batch_size": 512,
  "gradient_accumulation_steps": 4,
  "gradient_clipping": 1.0,
  "bf16": { "enabled": true },
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e8,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "none"
    }
  },
  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": false
  }
}
```

| 配置项 | 作用 | 面试一句话 |
|--------|------|------------|
| `zero_optimization.stage` | 1/2/3 对应 ZeRO 阶段 | stage 3 = 参数也分片 |
| `overlap_comm` | 通信与计算重叠 | 提 MFU，大模型常开 |
| `offload_optimizer` | 优化器放 CPU | 用 PCIe 换显存 |
| `offload_param` | 参数放 CPU/NVMe | 极省显存，更慢 |
| `train_batch_size` | **全局** batch | = micro_batch × accum × world_size |
| `activation_checkpointing` | DeepSpeed 版检查点 | 与 HF `gradient_checkpointing_enable` 二选一 |

**启动**：

```python
import deepspeed

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config="ds_config.json",
)
# 训练：model_engine.backward(loss); model_engine.step()
```

**FSDP vs DeepSpeed 怎么选？**

| 维度 | FSDP | DeepSpeed |
|------|------|-----------|
| 生态 | PyTorch 原生、TorchTitan | HF Trainer、Megatron 集成深 |
| 上手 | 中等 | 配置项多但文档全 |
| Offload | 有限 | **ZeRO-Offload / Infinity** 成熟 |
| 超大模型 | 需配合 TP/PP | **3D 并行**开箱 |
| 面试倾向 | Meta/Torch 栈 | Microsoft/HF 栈、超大预训练 |

---

### 13.15.3 Megatron Pipeline Parallel：GPipe vs 1F1B

PP 把模型按 **层** 切到不同 GPU。朴素 **GPipe** 先灌满 forward 再 backward，中间有大量 **bubble（空泡）**：

```
GPipe (4 stages, 4 microbatches):

时间 →
GPU0 [F1][F2][F3][F4][    ][B4][B3][B2][B1]
GPU1 [    ][F1][F2][F3][F4][    ][B4][B3][B2]
GPU2 [        ][F1][F2][F3][F4][    ][B4][B3]
GPU3 [            ][F1][F2][F3][F4][    ][B4]

F=forward microbatch, B=backward
[    ] = bubble（GPU 空转）
```

**1F1B（One Forward One Backward）**：每 stage 完成一个 microbatch forward 后尽快启动 backward，使 pipeline 始终有活干：

```
1F1B: forward 与 backward 交错，bubble 显著减小

GPU0 [F1][F2][B1][F3][B2][F4][B3][B4]
GPU1 [  ][F1][F2][B1][F3][B2][F4][B3][B4]
...
```

**Bubble 比例（面试手算）**：

```
GPipe bubble fraction ≈ (P - 1) / (P + M - 1)
1F1B bubble fraction ≈ (P - 1) / (M + 2(P - 1))

P = pipeline stages 数
M = microbatch 数量
```

**示例**：P=4，M=8

| 调度 | Bubble 比例 | 有效算力 |
|------|-------------|----------|
| GPipe | (4-1)/(4+8-1) = **27%** | ~73% |
| 1F1B | (4-1)/(8+2×3) = **21%** | ~79% |

M 越大 bubble 越小，但 **激活内存 ∝ M**——PP 要在吞吐与显存间折中。

Megatron-LM 训练栈典型组合：

```
3D Parallel = Data Parallel × Tensor Parallel × Pipeline Parallel
例：1024 GPU = DP128 × TP4 × PP2
```

| 并行 | Megatron 配置项 | 作用 |
|------|-----------------|------|
| TP | `--tensor-model-parallel-size` | 切 attention/FFN |
| PP | `--pipeline-model-parallel-size` | 切 layer depth |
| DP | `--data-parallel-size`（或由 world/tp/pp 推出） | 不同数据 |
| 序列并行 | `--sequence-parallel` | 进一步切 activation |

---

### 13.15.4 组合选型决策树（面试系统设计题）

```
                    模型能放进单卡 + Adam 状态？
                              │
                    ┌─────────┴─────────┐
                   Yes                   No
                    │                     │
              AMP + 可选 checkpoint    多卡
                    │                     │
                    │           ┌─────────┴─────────┐
                    │      单层/单段仍太大？      仅状态太大？
                    │           │                     │
                    │          TP                  FSDP/ZeRO-2/3
                    │           │                     │
                    │      仍太大 → PP (1F1B)    仍 OOM → +checkpoint
                    │           │                或 Offload / LoRA
                    └───────────┴─────────────────────┘
```

| 场景 | 推荐栈 |
|------|--------|
| 7B 全量微调，8×A100 80G | FSDP FULL_SHARD + BF16 + FlashAttn |
| 70B 预训练，数百卡 | Megatron **TP+PP+DP** 或 DeepSpeed 3D |
| 7B 指令微调，1×24G | QLoRA + AMP |
| 13B 全量，32×A100 | FSDP + TP2 或 DeepSpeed ZeRO-3 + TP |
| 长上下文 32K | FlashAttn + 序列并行 + checkpoint |

---

## 13.16 OOM 排查清单（面试场景题）

**题目**：「7B 模型 4×A100 80G OOM，怎么排查？」

```
1. 量化显存构成
   - 参数+梯度+Adam ≈ 4×7B×2B (BF16) 仍 ~56GB+ 仅状态
   - 激活 ∝ batch × seq² × layers

2. 逐项减压（由易到难）
   □ 开 BF16/AMP
   □ 减 micro_batch 或 seq_len
   □ gradient_checkpointing_enable()
   □ gradient_accumulation 保 global batch
   □ FSDP / ZeRO-2/3
   □ Flash Attention
   □ LoRA 代替 full finetune
   □ TP（单层仍太大时）

3. 测量验证
   torch.cuda.max_memory_allocated()
   torch.profiler 看 peak 算子
```

| 工具 | 用途 |
|------|------|
| `max_memory_allocated()` | 峰值显存 |
| `memory_summary()` | 分配明细 |
| `torch.profiler` | 算子级 memory / time |
| NCCL debug | 分布式 hang / 通信 |

---

## 13.17 与第 7–12 章的关系

```
                    ┌─────────────────────────────────────────┐
                    │              完整 LLM 生命周期            │
                    └─────────────────────────────────────────┘
                                        │
          ┌─────────────────────────────┴─────────────────────────────┐
          ▼                                                           ▼
   ┌─────────────────┐                                    ┌─────────────────┐
   │  训练（本章）     │                                    │  推理（第 7–12 章）│
   │  AMP · 检查点    │  ──── 权重 checkpoint ────▶       │  量化 · KV Cache │
   │  ZeRO · FSDP     │                                    │  批处理 · 监控   │
   └─────────────────┘                                    └─────────────────┘
```

| 章节 | 阶段 | 核心问题 |
|------|------|----------|
| **第 13 章（本章）** | 训练 | 显存不够时如何仍能把模型训完 |
| **第 7 章** | 推理 | 权重 INT8/FP16，降低部署内存 |
| **第 8 章** | 推理 | KV Cache、Continuous Batching |
| **第 10 章** | 推理 | 单卡放不下权重时的 TP/PP |

训练省下的显存，让你能用更大 batch 或更深模型；推理章节省下的显存，让你能服务更多并发——**同一套「内存三角：容量 / 速度 / 精度」**，阶段不同，手段不同。

---

## 13.18 生产训练 Checklist

### 推荐默认配置

1. **开启 AMP**（BF16 在 A100/H100 上往往比 FP16 更稳）
2. **层数 ≥ 24 或 OOM 时启用 gradient checkpointing**
3. **多卡 7B+ 优先 FSDP / ZeRO-2/3**，超大模型加 TP/PP
4. **长序列训练启用 Flash Attention**
5. **微调优先 LoRA/QLoRA**，全量微调需充分评估显存
6. **监控 tokens/s 与 MFU**，GPU util 低先查 DataLoader
7. **用 `max_memory_allocated()` 或 `torch.profiler` 定位峰值**
8. **组合技术前逐项加**：先 AMP，再检查点，再 FSDP，再 TP
9. **定期存 training checkpoint**（model + optimizer + step），异步写入
10. **在真实数据上验证 loss 曲线与下游指标**

### 常见陷阱

| 陷阱 | 后果 | 对策 |
|------|------|------|
| 忘记 `model.train()` | 检查点不生效 | 训练 loop 开头显式 `model.train()` |
| AMP 无 Loss Scaling | 梯度下溢、不收敛 | 使用 `GradScaler` |
| 测量前未 `empty_cache()` | 对比实验失真 | 每场景重建 model 并清 cache |
| 只盯显存不看吞吐 | 训练 wall-clock 暴增 | 同时 benchmark step time |
| 检查点 + 复杂自定义 forward | 重算结果不一致 | 对含随机性的 op 使用 `use_reentrant=False` |

### 运行 Demo

```bash
# 概念 demo（CPU 即可）
python3 basic/chapter_13_memory_efficient_training.py

# CUDA 四场景显存对比
python3 basic/chapter_13_memory_efficient_training.py --gpu

# Problem_4 规模 + step 耗时基准
python3 basic/chapter_13_memory_efficient_training.py --gpu --full

# 原版详细演示（含中文输出）
python3 openAI/Problem_4_openAI_memory_efficient_training.py
```

依赖：`torch >= 1.9.0`，CUDA 环境。无 GPU 时脚本会提示退出（训练内存优化主要针对 GPU）。

---

## 13.19 本章小结

| 概念 | 一句话 |
|------|--------|
| **训练显存四要素** | 参数、梯度、优化器状态、激活值 |
| **混合精度 (AMP)** | FP16/BF16 forward + FP32 更新 + Loss Scaling，首选、几乎默认开 |
| **梯度检查点** | 少存激活、backward 重算，用 ~20–30% 时间换 50–80% 激活内存 |
| **DDP** | 每卡完整模型，AllReduce 梯度，扩吞吐不减单卡显存 |
| **TP / PP** | 切层内权重 / 切层间 depth，解决单卡放不下大模型 |
| **ZeRO / FSDP** | 分片优化器/梯度/参数，多卡显存 ~4P/N |
| **梯度累积** | 小 micro-batch 模拟大 batch，峰值激活不变 |
| **tokens/s & MFU** | 衡量训练吞吐与算力利用率 |
| **Flash Attention** | 训练长序列时激活 O(T²)→O(T) |
| **LoRA / QLoRA** | 只训适配器，大幅省优化器与梯度内存 |
| **DataLoader** | GPU 低 util 时先查 IO，再查模型 |
| **FSDP / DeepSpeed** | PyTorch 原生 ZeRO-3 vs 配置驱动；超大模型 DeepSpeed 3D 更常见 |
| **Megatron 1F1B** | PP 调度减 bubble；M 大 bubble 小但激活多 |
| **3D 并行** | DP × TP × PP，工业界训 100B+ 标配 |

---

## 13.20 思考题与参考答案

### 思考题 1

一个 7B 参数模型用 Adam 做 **FP32 全精度**训练，仅「参数 + 梯度 + Adam 状态」理论至少需要多少 GB 显存？（忽略激活与碎片）

**参考答案**：

Adam 需存：参数 P + 梯度 P + 一阶矩 P + 二阶矩 P ≈ **4P**。

```
7 × 10⁹ × 4 bytes × 4 ≈ 112 GB
```

尚未计入激活、通信 buffer、CUDA 上下文——实际远超此数，故 7B 全 FP32 Adam 单卡几乎不可行，必须 AMP + 检查点 + 分片。

### 思考题 2

AMP 已开启，训练 loss 正常但部分层梯度长期为 0，如何排查？

**参考答案**：

1. 检查是否正确使用 `GradScaler`（`scale` → `step` → `update`）
2. 查看 scaler 是否频繁 skip step（动态 loss scale 过小）
3. 尝试 **BF16**（动态范围更大，常无需 loss scaling）
4. 对 particularly small 的梯度层，考虑该层保持 FP32 或提高 loss scale 初始值
5. 用 `torch.autograd.set_detect_anomaly(True)` 定位具体层

### 思考题 3

梯度检查点与 INT8 量化都能「省内存」，训练场景应选哪个？

**参考答案**：

- **训练**：优先 **AMP + 梯度检查点**；INT8 训练需 QAT 或 careful PTQ，实现复杂、精度风险高，非首选。
- **推理**（第 7 章）：**INT8/FP16 量化**是主力，无 backward，无需检查点。
- 二者解决的问题不同：检查点减**激活**；量化减**权重存储与带宽**。

### 思考题 4

DDP 8 卡训 13B 模型，每卡仍 OOM。你会按什么顺序尝试哪些手段？

**参考答案**：

1. **BF16 AMP** + 减小 micro_batch / seq_len
2. **Gradient checkpointing** + **gradient accumulation** 恢复 global batch
3. **FSDP / ZeRO-2/3**（8 卡可将 Adam 状态分到 ~1/8）
4. 仍 OOM → **TP**（单层权重或激活过大）或 **LoRA**（若是微调而非预训练）
5. 长序列 → **Flash Attention**
6. 用 profiler 确认是 **激活** 还是 **优化器** 占主导，避免盲目上 PP（PP 有 bubble，MFU 可能更低）

### 思考题 5

global batch=1024，seq=4096，step time=4s，7B 参数，单 A100 BF16 峰值 312 TFLOPS。粗算 tokens/s 和 MFU（单卡视角）。

**参考答案**：

```
tokens/s = 1024 × 4096 / 4 = 1,048,576
FLOPs/s ≈ 6 × 7×10⁹ × 1.05×10⁶ ≈ 4.4×10¹⁶
MFU ≈ 4.4×10¹⁶ / 3.12×10¹⁴ ≈ 141%  （>100% 说明多卡或粗算未除 world_size；面试中应说明需除以 GPU 数）
```

单卡时 MFU 不可能 >100%；若 8 卡则每卡 MFU ≈ 141%/8 ≈ **18%**——偏低，应查数据加载或通信。

### 思考题 6

LoRA rank r=16，hidden dim d=4096，有多少可训参数（单层 Linear 的 LoRA）？

**参考答案**：

每层 LoRA：A `[d, r]` + B `[r, d]` → `2 × d × r = 2 × 4096 × 16 = 131,072` 参数/层（不含 bias）。全模型对所有 Linear 加 LoRA 时乘层数；仍远小于 7B 全量。

### 思考题 7

PP=4 stages，microbatch M=8。GPipe 与 1F1B 的 bubble 比例各是多少？

**参考答案**：

```
GPipe:  (P-1)/(P+M-1) = 3/11 ≈ 27.3%
1F1B:   (P-1)/(M+2(P-1)) = 3/14 ≈ 21.4%
```

1F1B 有效算力更高；若 M 增到 16，1F1B bubble → 3/26 ≈ 11.5%，但激活内存随 M 线性增。

---

## 13.21 AI Infra 面试速查（训练向）

| 问题 | 要点 |
|------|------|
| 训练显存哪几块？ | 参数、梯度、优化器、激活 |
| DDP vs ZeRO？ | DDP 每卡 4P；ZeRO-3 约 4P/N |
| TP vs PP？ | TP 切层内、通信 AllReduce；PP 切层间、有 bubble |
| 长序列 OOM？ | FlashAttn、checkpoint、减 seq、TP |
| 想增大 batch OOM？ | grad accum，不是盲目加卡 |
| GPU util 低？ | DataLoader / tokenize / IO |
| MFU 低？ | 通信、PP bubble、checkpoint、小 batch |
| 微调省内存？ | LoRA / QLoRA，冻结基座 |
| BF16 vs FP16？ | 训练优先 BF16（A100+），FP16 要 loss scaling |
| 梯度检查点 vs 存 checkpoint？ | 前者省激活重算；后者是磁盘快照恢复 |
| FSDP vs DeepSpeed？ | FSDP 原生 PyTorch；DeepSpeed 3D/offload 更强 |
| PP bubble 怎么减？ | 1F1B 调度 + 增大 microbatch M |
| ZeRO stage 怎么选？ | 显存紧 stage 3；吞吐优先 stage 2 |

---

## 相关资源

- **本章全部代码（一个文件）**：[`basic/chapter_13_memory_efficient_training.py`](../basic/chapter_13_memory_efficient_training.py) — 运行 `python3 basic/chapter_13_memory_efficient_training.py`
- 完整演示（Problem_4）：[`openAI/Problem_4_openAI_memory_efficient_training.py`](../openAI/Problem_4_openAI_memory_efficient_training.py)
- 面试题梳理：[`openAI/openAI_questions.md`](../openAI/openAI_questions.md) — Problem 4 章节
- 推理优化起点：[`document/chapter_07_model_quantization.md`](chapter_07_model_quantization.md) — 训练完成后如何压缩部署
- 分布式扩展：[`document/chapter_10_distributed_inference.md`](chapter_10_distributed_inference.md) — 单卡放不下时的 TP/PP（推理视角，思想与 FSDP 相通）

---

*本章覆盖 AI Infra 面试训练侧核心知识点：从单卡 AMP/检查点，到多卡 DDP/ZeRO/FSDP/TP/PP，再到吞吐、数据流水线与 OOM 排查。掌握后可与第 7 章推理优化、第 10 章模型并行对照，形成完整「训推一体」视角。*

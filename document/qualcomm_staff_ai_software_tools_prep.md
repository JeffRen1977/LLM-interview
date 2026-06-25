# Qualcomm Staff/Sr. Staff · AI Software Tools 面试准备指南

> **岗位**：Staff / Sr. Staff Software Engineer, AI Software Tools (Onsite, San Diego)  
> **团队**：Qualcomm AI Stack SDK Software — QAIRT、ONNX Runtime / ExecuTorch / TFLite delegates  
> **核心使命**：在 **Snapdragon NPU** 上交付 **GenAI 推理**（LLM / LVM / LMM）的转换、优化、运行时与工具链

本文档将 JD 拆解为 **知识地图 → 本仓库对齐 → 缺口补强 → 面试题型 → 4 周准备计划**。

**动手实验**：[`qualcomm/`](../qualcomm/README.md) 目录含 10 个可运行 Lab（ONNX / ORT / 量化 / 图划分 / LLM / MoE / LoRA / debug / 端到端）+ C++ QNN stub，对应下文各 § 技能点。

---

## 1. 岗位本质（面试前先建立心智模型）

这个岗位 **不是** 纯 ML Research，也 **不是** 纯 App 开发，而是：

```
PyTorch / ONNX 模型
        ↓  convert + graph transform + quantize
中间表示（IR）/ 计算图
        ↓  graph lowering + fusion + delegate
QAIRT / QNN / ORT-EP / ExecuTorch / LiteRT
        ↓
Snapdragon NPU / GPU / CPU
        ↓
低延迟 · 低功耗 · 可部署的 GenAI 推理
```

**Staff 级额外期望**：

- 端到端 **feature ownership**（设计 → 实现 → QA → 客户问题）
- 跨层 **debug**（模型、runtime、compiler、OS、硬件）
- **mentor** 初级工程师、design/code review
- 与 Research / HW / PM / QA **跨时区协作**

面试中要展现：**深度（推理优化）+ 广度（系统栈）+ 领导力（技术决策与推动）**。

---

## 2. JD → 知识技能对照表

| JD 关键词 | 你需要掌握什么 | 优先级 | 本仓库资源 |
|-------------|----------------|--------|------------|
| **Model optimization** | 量化、剪枝、蒸馏、算子融合 | ⭐⭐⭐ | [chapter_07](chapter_07_model_quantization.md)、[chapter_09](chapter_09_flash_attention_operator_fusion.md) |
| **Quantization** | PTQ/QAT、INT8/INT4、per-tensor vs per-channel、校准 | ⭐⭐⭐ | [chapter_07](chapter_07_model_quantization.md)、`Problem_3` |
| **Graph transformations / lowering** | 子图匹配、constant folding、算子融合、layout、delegate 切分 | ⭐⭐⭐ | [chapter_09](chapter_09_flash_attention_operator_fusion.md)；**需补** ONNX graph、compiler 基础 |
| **Runtime execution** | ORT、ExecuTorch、TFLite 执行流、EP/delegate | ⭐⭐⭐ | [chapter_08](chapter_08_inference_pipeline.md)；**需补** 各 runtime 官方文档 |
| **LLM inference** | Prefill/Decode、KV Cache、Continuous Batching | ⭐⭐⭐ | [chapter_08](chapter_08_inference_pipeline.md) |
| **Attention** | MHA、GQA、FlashAttention、长上下文 | ⭐⭐⭐ | [chapter_09](chapter_09_flash_attention_operator_fusion.md)、`Problem_5` |
| **Speculative Decoding** | draft/verify、acceptance rate | ⭐⭐ | [chapter_11](chapter_11_speculative_decoding.md) |
| **LoRA / PEFT** | 低秩适配、部署时 merge / 侧路 | ⭐⭐ | [chapter_13](chapter_13_memory_efficient_training.md) §13.13 |
| **MoE** | routing、expert 并行、内存 | ⭐⭐ | **§11**、[chapter_16_moe_inference.py](../basic/chapter_16_moe_inference.py) |
| **PyTorch + ONNX** | export、dynamic axes、opset、失败排查 | ⭐⭐⭐ | `Problem_3`；**需补** `torch.onnx.export` / `torch.export` |
| **Python + C/C++** | 生产级 SDK、性能热点、ABI | ⭐⭐⭐ | **需补** C++ 刷题 + 读过 QNN C API 示例 |
| **NPU / Snapdragon** | HTP/DSP、fixed-point、memory hierarchy | ⭐⭐⭐ | **§10** QAIRT 专章 |
| **Debug 跨层** | 数值对不齐、性能回退、crash | ⭐⭐⭐ | 准备 2–3 个 **war story** |
| **Monitoring / SLA** | TTFT、TPOT、P99、线上指标 | ⭐⭐ | [chapter_12](chapter_12_inference_monitoring_sla.md) |
| **Evaluation** | 自动指标 + 人工 + 线上 | ⭐ | [chapter_15](chapter_15_llm_evaluation_framework.md) |
| **Android / QNX** | 端侧部署、JNI、RTOS 约束 | ⭐⭐ | **需补** |
| **Mentoring / Staff** | 技术方案、优先级、跨团队推进 | ⭐⭐⭐ | 准备 **leadership stories** |

---

## 3. 按模块深入：考什么 & 怎么准备

### 3.1 模型优化与量化（Must Have · 高频）

**面试会问**：

- INT8 动态量化 vs 静态量化 vs QAT 区别与选型？
- 量化后精度掉太多怎么排查？（校准集、敏感层 FP16、per-channel）
- LLM 权重量化（GPTQ/AWQ）与激活量化的差异？
- 端侧 NPU 为何偏爱 **INT8/INT4** 和 **对称量化**？

#### 面试参考答案

##### Q1. INT8 动态量化 vs 静态量化 vs QAT 区别与选型？

三者都属于 **量化**，但 **何时算 scale、要不要校准、要不要重训** 不同：

| | **动态量化（Dynamic PTQ）** | **静态量化（Static PTQ）** | **QAT（量化感知训练）** |
|--|----------------------------|---------------------------|------------------------|
| **时机** | 训练后，开箱即用 | 训练后 + **校准（calibration）** | **训练过程中** 模拟量化 |
| **权重** | 离线转 INT8 | 离线转 INT8 | 学习适应量化误差 |
| **激活 scale** | 推理时 **现算**（per-tensor 动态） | 校准集上 **预先统计** 固定 scale | 训练中学出更稳的 scale |
| **需要校准集** | 否 | **是**（几百～几千条代表性样本） | 是（训练集） |
| **实现成本** | 最低 | 中等 | 最高（要重训或微调） |
| **精度** | 一般够用 | 通常优于动态 | **最好** |
| **加速场景** | 传统 CV、小模型、**CPU** | NPU/GPU 固定图、端侧部署 | 精度敏感、量化难收敛的模型 |
| **典型 API** | `torch.quantization.quantize_dynamic` | ONNX Runtime QDQ、AIMET PTQ | AIMET QAT、PyTorch QAT |

**量化公式（对称 INT8，最常见）**：

```
quantized = round( float_value / scale )   # scale 由 max(abs) 或校准统计得到
dequant   = quantized * scale
```

**选型决策树（面试可直接画）**：

```
有训练资源且精度要求高？
  ├─ 是 → QAT
  └─ 否 → 部署目标是谁？
           ├─ NPU / 固定 INT8 图 → Static PTQ + 校准集 + QDQ
           ├─ 快速试 CPU 推理 → Dynamic PTQ
           └─ LLM 大模型 → 权重量化 GPTQ/AWQ（见 Q3），激活另议
```

**与 PTQ 的关系**：动态、静态都是 **PTQ（Post-Training Quantization，训练后量化）**；QAT 不是 PTQ，要动训练图。

---

##### Q2. 量化后精度掉太多怎么排查？

按 **从外到内、从易到难** 的顺序排查：

**Step 1 — 确认基准与评测**

- FP32/FP16 **基线指标** 是否可靠（同一验证集、同一预处理）
- 看 **整体指标**（Accuracy/F1/PPL）还是 **逐层 max diff**（`assert_allclose` 阈值）

**Step 2 — 校准集（Static PTQ / LLM 权重量化）**

| 问题 | 处理 |
|------|------|
| 校准集太小或不代表真实分布 | 换 **覆盖业务场景** 的数据（几百～2k 条常见） |
| 校准集与线上分布偏移 | 用 **生产日志脱敏样本** 或合成边界 case |
| 只校准了中间层激活、没覆盖极端 token | LLM 用 **多样化 prompt 长度与主题** |

**Step 3 — 粒度：per-tensor vs per-channel**

| | **per-tensor**（整层一个 scale） | **per-channel**（每输出通道一个 scale） |
|--|----------------------------------|----------------------------------------|
| 优点 | 实现简单、NPU 友好 | **精度明显更好**（卷积/Linear 权重量化常用） |
| 缺点 | 通道间动态范围差异大时误差大 | 略增 metadata、部分老硬件只支持 per-tensor |

→ 精度掉在 **Conv/Linear 权重** 上时，优先试 **per-channel weight quant**。

**Step 4 — 敏感层保持 FP16（混合精度）**

常见 **不量化或保留 FP16** 的层：

- **Embedding**、**LayerNorm / RMSNorm**
- **Softmax** 前后、**Attention** 累加路径
- 输出头 **lm_head**（LLM 对 logits 敏感）
- 量化后 **单层 max diff 突增** 的层（逐层对比定位）

策略：**Quantize + FP16 fallback 混合图**（QNN/ORT 支持部分层高精度）。

**Step 5 — 对称 vs 非对称、INT8 vs INT4**

- 激活分布 **明显非对称**（大量正值、ReLU 后）→ 试 **非对称量化**（zero-point ≠ 0）
- INT4 再掉精度 → 回退 **INT8** 或 **敏感层 INT8 + 其余 INT4**

**Step 6 — 部署链数值对不齐（Qualcomm 常考）**

```
PyTorch FP32  →  ONNX  →  ORT CPU  →  QNN HTP INT8
```

- 若 ORT CPU INT8 准、NPU 不准 → 查 **QDQ 节点、scale 导出、layout（NCHW/NHWC）**
- 若从第一步就偏 → 查 **export 算子替换、fusion 改变计算顺序**

**排查口诀**：**校准集 → per-channel → 敏感层 FP16 → 降 bit 或换 QAT → 逐层 golden 对比**。

---

##### Q3. LLM 权重量化（GPTQ/AWQ）与激活量化的差异？

LLM 推理里 **权重巨大、激活随 batch/seq 变化** — 业界常 **先重度量化权重**，激活策略分开考虑。

| | **权重量化（Weight-only）** | **激活量化（Activation quant）** |
|--|---------------------------|----------------------------------|
| **量化对象** | 主要 **W**（Attention、FFN 的 Linear） | 每层 **中间激活**（matmul 输入/输出） |
| **典型方法** | **GPTQ**、**AWQ**、GGUF Q4_K | SmoothQuant、静态 PTQ、W8A8 |
| **是否需要校准** | 需要（GPTQ/AWQ 用少量样本估 Hessian 或 saliency） | 静态激活量化 **必须校准** |
| **内存收益** | **极大**（7B FP16 ~14GB → INT4 ~4GB） | 主要减 **运行时激活 buffer**（相对权重次要） |
| **精度影响** | 做得好 PPL 掉很少 | 更难，易掉精度（outlier 激活） |
| **Decode 特点** | 权重 **反复读** → 权重量化直接减 **带宽** | 激活每步变化 → 动态范围难估 |

**GPTQ vs AWQ（权重量化内部对比）**：

| | **GPTQ** | **AWQ** |
|--|----------|---------|
| **思路** | 用 Hessian 近似，逐列量化并误差补偿 | 认为 **少量 salient 权重** 更重要，保护 + 缩放激活 |
| **优点** | 成熟、工具链多 | 往往 **同样 bit 下精度更好** |
| **场景** | 通用 W4 权重 | 端侧/推理库常见（与 vLLM、llama.cpp 生态结合多） |

**常见组合（面试常问）**：

| 方案 | 说明 |
|------|------|
| **W4A16**（GPTQ/AWQ + FP16 激活） | **最常用**：权重 INT4，激活 FP16；实现简单、精度好 |
| **W8A8** | 权重+激活都 INT8；更快更省，但 LLM 激活 outlier 多，要校准/SmoothQuant |
| **W4A8** | 更激进，端侧极致压缩 |

**一句话**：

> **GPTQ/AWQ 解决「模型太大装不下、Decode 读权重太慢」；激活量化解决「矩阵乘算子能否走 INT8 管线」— LLM 优先权重量化，激活量化是进阶选项。**

---

##### Q4. 端侧 NPU 为何偏爱 INT8/INT4 和对称量化？

**为何 INT8 / INT4？**

| 原因 | 说明 |
|------|------|
| **硬件电路** | HTP/Hexagon NPU 有 **定点 MAC（乘累加）** 单元，INT8/INT4 吞吐远高于 FP16 |
| **功耗** | 低位宽 → 更少晶体管翻转、更低 **DRAM 带宽**（手机电池/发热硬约束） |
| **内存** | 7B 模型 FP16 ~14GB，手机装不下；INT4 ~3.5–4GB 才可部署 |
| **已验证生态** | QNN/AIMET/AI Hub 默认路径就是 **INT8 权重量化 + HTP** |

**为何常选对称量化（Symmetric）？**

对称：量化范围 **关于 0 对称**，`zero_point = 0`：

```
scale = max(abs(tensor)) / 127     # INT8
q = round( x / scale )
x ≈ q * scale
```

| | **对称（Symmetric）** | **非对称（Asymmetric）** |
|--|----------------------|-------------------------|
| **zero-point** | 固定为 0 | 可非 0，覆盖 [min, max] |
| **硬件** | **只需乘 scale**，无 zero-point 偏移项 → NPU 电路更简单 | 多一项 offset，略复杂 |
| **适用** | 权重分布近似对称、经过 Norm 后的激活 | ReLU 后全非负等偏态分布 |
| **端侧偏好** | **HTP 默认/优先路径** | 可用但支持度、性能可能不如对称 |

**NPU 还偏爱什么（可一并讲）**：

- **per-channel 权重量化** + **per-tensor 激活**（W8A8 常见组合）
- **离线 compile**：scale 固定进 graph，runtime **不做动态搜 max**
- **算子融合后的 QDQ**：`Q → DQ → MatMul` 融合进一个 HTP kernel

**端侧 vs 云端对比（Qualcomm 差异化）**：

| | 云端 GPU | 端侧 HTP |
|--|----------|----------|
| FP16 Tensor Core | 成熟高效 | 不如 INT 矩阵单元划算 |
| 动态量化 | 有时可行 | **更喜静态 scale、固定 shape** |
| 目标 | 吞吐 | **功耗 + 内存 + 延迟** |

**白板一句话**：

> **NPU 为 INT 矩阵乘而生；INT8/INT4 省内存和带宽；对称量化省硬件开销、与 HTP 定点 datapath 最合拍。**

**手算例题（面试可能问）**：

```
7B 参数 × FP16 (2 bytes) ≈ 14 GB 权重
7B 参数 × INT8 (1 byte)  ≈ 7 GB
7B 参数 × INT4 (0.5 byte) ≈ 3.5 GB
```

---

**准备动作**：

1. 精读 [chapter_07](chapter_07_model_quantization.md)，能讲清 **基准 → 量化 → 四项 ratio**
2. 跑 `basic/chapter_07_model_quantization.py` 和 `Problem_3`
3. 手写估算：7B FP16 权重多少 GB？INT8 多少？
4. 了解 **Qualcomm AI Model Efficiencies (AIMET)** 与 QAIRT 量化工具链（官方文档浏览即可）

**可讲的一句话**：推理优化先在 **可接受精度下** 测 FP32 基线，再选 PTQ/QAT；端侧重点看 **权重带宽 + KV 显存**。

---

### 3.2 LLM 推理流水线（Must Have · 高频）

**面试会问**：

- Prefill 和 Decode 瓶颈有何不同？
- KV Cache 是什么？不用的话复杂度怎样？
- Continuous Batching vs 静态 batch？
- PagedAttention 解决什么问题？
- TTFT、TPOT、E2E 延迟如何估算？

#### 面试参考答案

##### Q1. Prefill 和 Decode 瓶颈有何不同？

LLM 自回归推理分 **两阶段**，瓶颈类型完全不同：

| | **Prefill（提示词阶段）** | **Decode（生成阶段）** |
|--|--------------------------|------------------------|
| **输入** | 用户 prompt 的 **全部 token**（长度 L） | 每步 **1 个新 token** |
| **在算什么** | 对 L 个 token 做完整 forward；建立 KV Cache | 只算新 token 的 Q/K/V；Attention 要读 **全部历史 KV** |
| **瓶颈类型** | **Compute-bound**（计算密集） | **Memory-bound**（内存带宽密集） |
| **为何** | 大矩阵乘（QKV、FFN），GPU/NPU **算力**打满 | 每步矩阵很小，但要 **读越来越长的 KV Cache**；算力空闲等 HBM/DRAM |
| **主导指标** | **TTFT**（首 token 时间） | **TPOT**（每输出 token 时间） |
| **典型优化** | FlashAttention、大 batch prefill、Chunked Prefill、TP | KV Cache、Continuous Batching、GQA、算子融合、Speculative Decoding |
| **Attention 复杂度** | O(L²)（整段 prompt 互相关注） | 每步 O(L) 读 cache，L 随生成变长 |

```
用户 prompt: "请写一首关于春天的诗"
      │
      ▼ Prefill（一次吃完 L 个 token）
  大矩阵乘 · 算力瓶颈 · 决定 TTFT
      │  写出 KV Cache
      ▼ Decode（每次 1 token：春 → 风 → 拂 → …）
  小矩阵乘 + 读全部 KV · 带宽瓶颈 · 决定 TPOT
```

**白板一句话**：

> **Prefill 是「一口气算完 prompt」，吃算力；Decode 是「每步只算 1 token 但要翻整本历史 KV」，吃带宽。**

**端侧（Qualcomm）补充**：手机没有 80GB HBM，**Decode 时 DRAM 读 KV + 读权重** 更致命；长 context 时 Prefill 还会 **一次性占满算力和内存**。Genie 部署要同时盯 **TTFT（prefill）** 和 **TPOT（decode）+ KV 占用**。

---

##### Q2. KV Cache 是什么？不用的话复杂度怎样？

**KV Cache**：把每一层 Attention 里，历史 token 算好的 **Key、Value** 缓存在内存里；Decode 每步 **只算新 token 的 Q/K/V**，Attention 时新 Q 与 **缓存的全部 K** 做内积。

**没有 KV Cache**（朴素自回归）：

```
生成第 t 个 token 时，输入 [tok₁…tok_{t-1}]，对全部 t-1 个 token 重新 forward
→ 每步重复算历史 token 的 Q/K/V
→ 生成 T 个 token 总计算量 ~ O(T²)（每层 Attention 近似）
```

**有 KV Cache**：

```
Prefill：算 prompt 的 K/V 并缓存
Decode 每步：只算 1 个新 token 的 Q/K/V，追加进 cache；Attention 读 cache
→ 生成 T 个 token 总计算量 ~ O(T)（每步读长度为 L 的 cache，L 线性增长）
```

| | 无 KV Cache | 有 KV Cache |
|--|-------------|-------------|
| Decode 每步 | 重算全部历史 | 只算 1 个新 token |
| 时间复杂度（生成 T token） | ~O(T²) | ~O(T)（但每步要读 O(L) 的 KV） |
| 代价 | 计算爆炸 | **内存**：每层每 token 存 K/V，随序列变长线性涨 |

**KV 显存粗算**（单层、单请求）：

```
KV_bytes ≈ 2 × num_layers × seq_len × num_kv_heads × head_dim × bytes_per_elem
```

（因子 2 = K 和 V；GQA 用 `num_kv_heads` 而不是 `num_heads`。）

---

##### Q3. Continuous Batching vs 静态 batch？

| | **静态批处理（Static Batching）** | **连续批处理（Continuous Batching）** |
|--|-----------------------------------|---------------------------------------|
| **组 batch 时机** | 请求到达时凑一批 | **每个 decode step** 重新组 batch |
| **请求完成后** | 等整批 **最慢** 的请求结束才释放 | 完成即退出，slot 立刻给新请求 |
| **Decode 问题** | 各请求生成长度不同 → **早完成的空等** | 每步动态进出，GPU 利用率高 |
| **Padding** | 按最长序列 pad，短请求浪费算力 | Iteration-level，decode 时通常无 pad |
| **实现** | 简单 | 复杂（调度器 + KV 管理） |
| **代表** | 早期 Triton | **vLLM**、TGI、TensorRT-LLM |

**类比**：静态 batch = 等满一班车再开；Continuous Batching = 地铁每站可上下人。

**注意**：Continuous Batching 主要优化 **Decode 吞吐**；Prefill 仍可单独 batch 或 Chunked Prefill。

---

##### Q4. PagedAttention 解决什么问题？

**问题**：Continuous Batching 下，每个请求 KV Cache 长度 **动态变化**，若预分配「最大长度」连续内存 → **碎片严重、浪费显存**，并发数上不去。

**PagedAttention**（vLLM）：把 KV Cache 切成固定大小 **block**（类似 OS 虚拟内存页），用 **block table** 映射逻辑位置 → 物理块，按需分配/回收。

| 没有 Paging | 有 PagedAttention |
|-------------|-------------------|
| 每请求占一段连续 max_len 空间 | 按实际长度分配 block |
| 长短请求交错 → 碎片 | 块级复用，显存利用率高 |
| 并发 slot 少 | **同卡跑更多并发请求** |

**一句话**：KV Cache 的 **内存管理器**，不是新 Attention 算法；解决 **碎片 + 动态长度** 下的显存效率。

---

##### Q5. TTFT、TPOT、E2E 延迟如何估算？

| 指标 | 含义 | 主要受谁影响 |
|------|------|--------------|
| **TTFT** | 请求发出 → **第一个输出 token** | 排队 + **Prefill** + 调度 |
| **TPOT** | 相邻输出 token 的平均间隔 | **Decode**、KV 带宽、batch 大小 |
| **E2E** | 请求发出 → **最后一个 token** | TTFT + (输出长度 − 1) × TPOT |

**粗算公式**：

```
E2E ≈ TTFT + (num_output_tokens − 1) × TPOT

# 若还要加排队（服务端）
TTFT ≈ queue_wait + prefill_time + decode_first_token_overhead
```

**数量级直觉**（7B 级、单卡 A100、中等 prompt，仅作面试量级感）：

- Prefill 1000 token：几十～几百 ms 级（与 FlashAttention、batch 有关）
- Decode TPOT：几十 ms/token 级（memory-bound，batch 越大 often 越低）

**Decode 吞吐粗算**：

```
throughput (tok/s) ≈ batch_size / TPOT    # 同一 decode step 的有效 batch
```

**用户体感**：

- 「第一个字慢」→ 查 **TTFT**（Prefill、排队、超长 prompt）
- 「后面一个字一个字慢」→ 查 **TPOT**（Decode、KV、Speculative）

---

**准备动作**：

1. 精读 [chapter_08](chapter_08_inference_pipeline.md)
2. 跑 `basic/chapter_08_inference_pipeline.py`，能口述 scheduler 与 PagedAttention demo
3. 能白板画：**无 cache vs 有 cache** 的 decode 步骤

**与 Qualcomm 的关联**：Genie / QAIRT 在手机上跑 LLM，**内存（KV）和功耗** 是核心 KPI，不是数据中心吞吐。

---

### 3.3 Attention 与算子层优化（Must Have）

**面试会问**：

- 标准 Attention 的内存瓶颈？FlashAttention 核心思想？
- 算子融合（fusion）为何能提速？（减少 kernel launch、HBM 读写）
- GQA / MQA 如何省 KV Cache？
- 长上下文在端侧的挑战？

**准备动作**：

1. [chapter_09](chapter_09_flash_attention_operator_fusion.md) + `basic/chapter_09_flash_attention_operator_fusion.py`
2. `Problem_5` attention variants（至少能讲 Sliding Window、Linear Attention 动机）
3. 了解 QNN 是否支持 **SDPA / custom op**（扫 QAIRT release notes）

---

### 3.4 Graph 转换、Lowering、Delegates（Must Have · Staff 核心）

JD 明确：**graph transformations, graph lowering, ONNX Runtime, ExecuTorch, QAIRT**。

**你需要建立的概念栈**：

```
PyTorch nn.Module
    → torch.export / ONNX graph (nodes, initializers)
    → graph optimization (fold, fuse, eliminate dead code)
    → lowering (high-level op → backend-specific ops)
    → QNN context / ORT Execution Provider / ExecuTorch delegate
    → NPU binary
```

**面试会问**：

- ONNX 里 `MatMul + Add + Relu` 融合成一个 op 的好处？
- 什么是 **delegate / EP**？与 whole-graph offload 的关系？
- 某个 op 在 NPU 不支持时如何 **fallback** 到 CPU？
- Dynamic shape（变长 prompt）对 compile 的影响？

**准备动作**（本仓库缺口，需外部补强）：

| 主题 | 建议资源 |
|------|----------|
| ONNX IR | [ONNX 官方 spec](https://onnx.ai/onnx/intro/) — graph、node、attribute |
| ONNX Runtime EP | ORT docs — QNN EP / QDQ 格式 |
| ExecuTorch | Meta docs — delegate、partitioner |
| QAIRT / QNN | Qualcomm Developer Network — QAIRT getting started、QNN API |
| Graph opt | 读一篇 **TVM** 或 **TorchInductor** pass 介绍（理解 pass 思维即可） |

**实操建议**（选 1–2 个做深）：

```bash
# 导出小模型到 ONNX
python -c "
import torch
m = torch.nn.Linear(128, 128)
torch.onnx.export(m, torch.randn(1,128), 'linear.onnx', opset_version=17)
"

# 用 Netron 可视化 graph
# 尝试 ONNX Runtime 推理 + 可选 QNN EP（若有 Snapdragon 设备或 SDK）
```

---

### 3.5 GenAI 进阶：LoRA、MoE、Speculative Decoding（Preferred）

| 技术 | 面试角度 | 准备 |
|------|----------|------|
| **LoRA** | 训练省显存；部署时 merge weights 或 runtime adapter | [chapter_13](chapter_13_memory_efficient_training.md) §13.13 |
| **MoE** | routing、top-k、负载均衡、端侧权重存储 | **§11** 专章 |
| **Speculative Decoding** | draft 小模型 + 大模型 verify，何时有效 | [chapter_11](chapter_11_speculative_decoding.md) |

---

### 3.6 PyTorch / ONNX 工程（Must Have）

**面试会问**：

- `torch.onnx.export` vs `torch.export`（Dynamo）？
- 动态 batch、动态 sequence length 如何 export？
- ONNX 常见失败：control flow、unsupported op、dtype 不一致
- 如何 debug **PyTorch vs ONNX 输出 max diff**？

**准备清单**：

- [ ] 成功 export 一个 **小型 Transformer block** 或 **Llama 单层** 到 ONNX
- [ ] 用 `onnxruntime` 跑通并 `numpy.testing.assert_allclose` 对比
- [ ] 知道 **ONNX Runtime Quantization**（QDQ 节点格式）

---

### 3.7 C/C++ 与系统软件（Must Have · Staff）

JD 要求 **Python + C/C++ 生产级**。SDK 团队大量工作：

- QNN C API、backend 插件
- 性能关键路径、内存池
- CMake 构建、CI、跨平台（Android/QNX/Linux）

**准备动作**：

- 复习 C++：**智能指针、move、多线程、内存对齐**
- 读 QAIRT 示例里的 **C++ inference sample**
- 准备 1 个 story：**Python 原型 → C++ 性能优化** 的经历

---

### 3.8 端侧与 Snapdragon NPU（Preferred · 差异化）

详见 **§10 QAIRT 专章**。此处保留速记：

| 术语 | 含义 |
|------|------|
| **QAIRT** | Qualcomm AI Runtime — 统一 AI 推理工具与 runtime 套件 |
| **QNN** | Qualcomm AI Engine Direct — 细粒度控制每 op 如何跑在指定处理器 |
| **SNPE** | 更简单的多处理器 API，文件可能更大、控制粒度较粗 |
| **Genie** | GENIE SDK — 面向 LLM/GenAI，构建在 QNN 之上 |
| **HTP** | Hexagon Tensor Processor — Hexagon NPU 上的主力 backend |
| **AI Hub** | 云端 compile 模型为 QNN context binary，再下发设备 |

**与云端推理的差异**（必答）：功耗/热设计、内存小、异构 fallback、offline compile 权重大。

---

### 3.9 Debug 与跨层问题（Staff 必考）

准备 **2–3 个 STAR 故事**，覆盖：

1. **数值不一致**：PyTorch vs ONNX vs NPU 输出 diff → 定位 op / scale / layout
2. **性能回退**：fusion 失效、错误 EP、dynamic shape 导致 recompile
3. **OOM / 延迟**：KV 过大、batch 策略、量化未生效

**调试方法论**（可主动讲）：

```
复现 → 缩小模型/graph → 单层对比 golden → 查 log（ORT verbose / QNN log）
→ 隔离硬件（CPU only）→  fix op / quant param → 回归 benchmark
```

---

### 3.10 Staff / Sr. Staff 软实力

JD 明确要求 **mentor、design review、跨团队、Director 沟通**。

准备 **5–8 个行为面试故事**（各 2 分钟）：

| 主题 | 举例方向 |
|------|----------|
| **Technical leadership** | 主导一条 inference 优化 feature 从 0 到上线 |
| **Mentoring** | 帮 junior 排查量化精度；code review 文化 |
| **Conflict / prioritization** | Research 要新 op，产品要稳定性，如何取舍 |
| **Cross-team** | 与 HW 团队对齐 NPU 限制；与 QA 建 benchmark |
| **Failure** | 一次上线回滚；学到了什么 |

---

## 4. 知识缺口清单（本仓库外必补）

按 **投入产出比** 排序（§10 / §11 已在本文档补全正文，以下为动手项）：

1. **QAIRT SDK 安装 + 跑通 1 个 Genie sample**（§10.6）
2. **ONNX export + ORT inference + 可选 QNN EP**（3–5 天）
3. **ExecuTorch delegate 概念** + partitioner 读文档（1 天）
4. **MoE**：读 §11 + 跑 `basic/chapter_16_moe_inference.py`（半天）
5. **C++ 复习** + 读 QNN C sample（持续）
6. **Android / QNX** 若有端侧经验可整理成 story（按需）

---

## 5. 面试题型预测

### 5.1 技术深度（60%）

- 讲解一次你做的 **模型推理优化** 全流程
- KV Cache + Continuous Batching 白板
- INT8 量化流程与精度问题
- FlashAttention vs 标准 Attention
- Graph fusion 例子
- LoRA 部署选项
- Speculative decoding 适用场景
- MoE routing 与端侧部署难点（§11）
- QAIRT：SNPE vs QNN vs Genie（§10）

### 5.2 系统 / SDK（25%）

- ONNX → backend 的 lowering  pipeline
- 不支持 op 怎么办？
- 如何设计一个 **delegate** 接口？
- 端侧 vs 云端推理 trade-off

### 5.3 编程（10%）

- Python：图遍历、tensor 操作、简单 optimizer
- C++：智能指针、vector、基础多线程
- 本仓库 [coding/](.) 题风格：图、堆、状态机（与 runtime 调度相关）

### 5.4 行为 / Staff（15%）

- Mentor 经历
- 最难的 cross-layer bug
- 如何推动跨团队 feature
- 为何离开 / 为何 Qualcomm / 为何端侧 AI

---

## 6. 与本仓库的学习路径（4 周示例）

### Week 1 — 推理核心（对齐 JD Must Have）

| 天 | 内容 | 产出 |
|----|------|------|
| 1–2 | [chapter_07](chapter_07_model_quantization.md) + `Problem_3` | 能讲量化 pipeline + 手算压缩比 |
| 3–4 | [chapter_08](chapter_08_inference_pipeline.md) + demo | 能画 KV Cache / Continuous batch |
| 5 | [chapter_09](chapter_09_flash_attention_operator_fusion.md) | 能讲 fusion + FlashAttention |
| 6–7 | ONNX export 小模型 + ORT 跑通 | 笔记：export 踩坑列表 |

### Week 2 — GenAI + 进阶优化

| 天 | 内容 | 产出 |
|----|------|------|
| 1 | [chapter_11](chapter_11_speculative_decoding.md) | 2 分钟讲清 speculative |
| 2 | [chapter_13](chapter_13_memory_efficient_training.md) LoRA 节 | 训练 vs 推理优化区分 |
| 3 | [chapter_10](chapter_10_distributed_inference.md) TP 基础 | 端侧多核/NPU 可类比 |
| 4–5 | **§11 MoE** + `chapter_16_moe_inference.py` | MoE 一页纸 + 端侧难点话术 |
| 6–7 | **§10 QAIRT** + Genie tutorial | 跑通 1 个 sample 或看完 build 流程 |

### Week 3 — Qualcomm 栈 + 系统

| 天 | 内容 | 产出 |
|----|------|------|
| 1–3 | **§10** QAIRT、Genie、ORT-QNN EP 文档 | 画一张 Qualcomm 软件栈图 |
| 4 | ExecuTorch delegate 文档 | 与 ORT EP 对比表 |
| 5 | [chapter_12](chapter_12_inference_monitoring_sla.md) | TTFT/TPOT 与线上监控 |
| 6–7 | C++ 复习 + QNN C API 浏览 | 10 道 C++ 小题 |

### Week 4 — 模拟面试 + 故事

| 天 | 内容 | 产出 |
|----|------|------|
| 1–2 | 白板：LLM serving 全栈 | 15 分钟无笔记讲解 |
| 3 | [chapter_15](chapter_15_llm_evaluation_framework.md) 浏览 | 评估与 benchmark 话术 |
| 4 | 准备 8 个 STAR stories | 写下来练口述 |
| 5 | `openAI/Problem_*` 过一遍 | 查漏补缺 |
| 6–7 | Mock interview / 复习缺口 | 弱项回炉 |

---

## 7. 一页纸速记（面试前夜）

```
【岗位】Snapdragon 上 GenAI 推理 SDK：convert → optimize → runtime

【必讲三板斧】
  1. 量化：PTQ/QAT、INT8、精度排查
  2. LLM 推理：Prefill/Decode、KV Cache、PagedAttn、Continuous batch
  3. Graph：ONNX → fusion/lowering → QNN/ORT EP/delegate

【端侧关键词】功耗、内存、异构、offline compile、NPU HTP

【Staff】E2E ownership · 跨层 debug · mentor · design review

【差异化】§10 QAIRT 栈 + §11 MoE + ONNX 一条链 + 2 个 war stories
```

---

## 8. 推荐外部资源

| 资源 | 用途 |
|------|------|
| [Qualcomm AI Developer](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk) | QAIRT / QNN 官方 |
| [Qualcomm AI Hub](https://aihub.qualcomm.com/) | 预优化模型与设备 benchmark |
| ONNX + ONNX Runtime docs | Graph、EP、量化 |
| [ExecuTorch](https://pytorch.org/executorch/) | Meta 端侧 runtime |
| vLLM paper / blog | Continuous batching、PagedAttention 原文 |
| AWQ / GPTQ papers | LLM 权重量化 |

---

## 9. 本仓库快速索引

| 主题 | 文档 | 代码 |
|------|------|------|
| **Qualcomm 动手 Lab** | [qualcomm/README.md](../qualcomm/README.md) | `qualcomm/lab_01` … `lab_10`, `run_all.py` |
| 量化 | [chapter_07](chapter_07_model_quantization.md) | `basic/chapter_07_model_quantization.py` |
| 推理流水线 | [chapter_08](chapter_08_inference_pipeline.md) | `basic/chapter_08_inference_pipeline.py` |
| FlashAttn / Fusion | [chapter_09](chapter_09_flash_attention_operator_fusion.md) | `basic/chapter_09_flash_attention_operator_fusion.py` |
| 分布式 | [chapter_10](chapter_10_distributed_inference.md) | `basic/chapter_10_distributed_inference.py` |
| Speculative | [chapter_11](chapter_11_speculative_decoding.md) | `basic/chapter_11_speculative_decoding.py` |
| 监控 SLA | [chapter_12](chapter_12_inference_monitoring_sla.md) | `basic/chapter_12_inference_monitoring_sla.py` |
| 训练 / LoRA | [chapter_13](chapter_13_memory_efficient_training.md) | `basic/chapter_13_memory_efficient_training.py` |
| 评估 | [chapter_15](chapter_15_llm_evaluation_framework.md) | `basic/chapter_15_llm_evaluation_framework.py` |
| MoE 概念 | **§11**（本文档） | `basic/chapter_16_moe_inference.py` |
| OpenAI 面试题 | — | `openAI/openAI_questions.md` |

---

## 10. QAIRT 专章（面试深度版）

> 官方：[Qualcomm AI Runtime (QAIRT) Overview](https://docs.qualcomm.com/doc/80-63442-10/topic/general_overview.html)  
> 岗位 JD 直接点名 **QAIRT、graph lowering、ORT/ExecuTorch delegates** — 本节把软件栈讲透。

### 10.1 QAIRT 是什么？

**QAIRT（Qualcomm AI Runtime）** 是高通向开发者交付的 **统一 AI 端侧套件**：从 host 上训练好的模型，到 target device（手机、PC、汽车 SoC）上可运行的 binary + runtime，中间 **convert、quantize、compile、execute** 的工具链都在里面。

可以记成三层：

```
┌─────────────────────────────────────────────────────────────┐
│  Application (Android / QNX / Linux on Snapdragon)          │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  SDK 层（选其一为主路径）                                      │
│  · SNPE  — 简单 API，多处理器自动调度，控制粒度粗               │
│  · QNN   — AI Engine Direct，逐 op 控制、按 backend 编译      │
│  · Genie — 扩展 QNN，专用于 LLM / GenAI（prefill/decode/KV）  │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  QAIRT API（底层 runtime）+ Delegates / EP                   │
│  · ONNX Runtime QNN EP                                      │
│  · TFLite Delegate                                          │
│  · ExecuTorch QNN delegate（Meta 栈接入）                    │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  Backends / Processors                                      │
│  CPU · Adreno GPU · HTP (Hexagon NPU) · DSP · …              │
└─────────────────────────────────────────────────────────────┘
```

**与岗位的关系**：AI Software Tools 团队维护的正是这条链路上的 **converter、graph pass、runtime、delegate、文档与 sample** — 不是只写 App，而是让 **别人的模型** 能在 Snapdragon 上 **跑对、跑快、跑省**。

### 10.2 SNPE vs QNN vs Genie — 何时用哪个？

| SDK | 定位 | 优点 | 代价 / 限制 |
|-----|------|------|-------------|
| **SNPE** | 传统 CV / 通用 DL，快速上手 | API 简单；自动多处理器 | 产物可能更大；难细调单 op |
| **QNN** | 需要精细控制、定制 backend | 按 HTP/GPU/CPU 分别 compile；适合工具链开发 | 学习曲线陡；workflow 步骤多 |
| **Genie** | **LLM on device** | 内置 text-to-text、KV、context binary 加载 | 依赖 QNN；模型需 Hub/工具链预编译 |

**面试话术**：

- 「CV 或小模型先试 **SNPE**；要做 **fusion、量化、backend 选型** 或写 **delegate**，深入 **QNN**。」
- 「**Genie** 是 QNN 上的 GenAI 运行时 — 类似 vLLM 在云端，但面向 **端侧内存与 HTP**。」
- 「JD 里的 ORT/ExecuTorch 是 **接入 QNN backend 的框架层**，不是与 QAIRT 竞争，而是 **partition 后 offload 到 HTP**。」

### 10.3 端到端 Workflow（Host → Device）

典型 **QNN / Genie LLM** 路径（与 [AI Hub](https://aihub.qualcomm.com/) 教程一致）：

```
1. Host：PyTorch / ONNX 模型
        ↓
2. Convert：QNN converter（ONNX / PyTorch / TFLite）
        ↓
3. Quantize：AIMET 或 QNN quantizer（INT8/INT4 等）
        ↓
4. Compile：针对 target SoC + backend（如 HTP v73+）生成 context binary
        ↓
5. Package：.bin + metadata + tokenizer（LLM）
        ↓
6. Device：Genie / QNN runtime 加载 binary，执行 inference
```

**关键 artifact 名词**（听到要能反应）：

| 名词 | 含义 |
|------|------|
| **DLC** | SNPE 容器格式（老栈常见） |
| **QNN context binary** | 针对特定芯片/backend **离线编译** 后的图 + 权重 |
| **HTP backend** | Hexagon Tensor Processor — NPU 主力 |
| **Offline prepare** | 在 PC 上 compile，手机只 load binary（端侧不做 JIT 编译） |
| **AIMET** | Qualcomm 量化工具（与 QAIRT 配套，训练后/PTQ） |

**Genie LLM 部署约束**（来自公开教程，面试可提）：

- Hexagon **v73+**；Android 15+ 常见于新旗舰
- **7B + 长上下文** 往往要 **16GB+** 设备内存；3B 约 **12GB+**
- Context length 与 SDK 版本强相关 — 需查 release notes

### 10.4 硬件 Backend：HTP / GPU / CPU

| Backend | 角色 | 面试要点 |
|---------|------|----------|
| **HTP** | NPU，INT 矩阵乘主力 | 偏爱 **静态 shape**、**融合图**；fixed-point 友好 |
| **GPU (Adreno)** | 部分 op fallback 或 FP16 | 适合部分 attention / elementwise |
| **CPU** | 兜底、小 op、dynamic 部分 | 延迟高但兼容性好 |

**异构执行**：

- 图 partition：**支持的子图 → HTP**，其余 **CPU/GPU**
- 与 ORT **Execution Provider**、ExecuTorch **delegate** 同构：框架负责切图，QNN 负责执行 HTP 子图

**Staff 级追问**：「为什么不全放 HTP？」→ 部分 op 未实现、dynamic shape、数值要求 FP32、或 fusion 失败导致性能差。

### 10.5 Graph Lowering 与 Delegate（与 JD 对齐）

把 §3.4 的概念 **落到 Qualcomm 栈**：

```
ONNX Graph (MatMul, Softmax, …)
    → ORT graph optimization (constant fold, QDQ)
    → ORT QNN EP partition：子图 → QNN compiled graph
    → HTP runtime

或

torch.export / ExecuTorch program
    → partitioner 标记 QNN delegate 子图
    → ahead-of-time compile → .pte + QNN binary
    → device 上 delegate 调 QNN
```

**面试常问 & 答法**：

| 问题 | 参考答法 |
|------|----------|
| Delegate / EP 是什么？ | 把图中 **一段子图** 委托给专用 backend 执行；接口层在 ORT/ET，实现层在 QNN |
| 不支持 op 怎么办？ | **CPU fallback**、换等价 op、改 export、向 QNN 注册 custom op |
| Dynamic shape 影响？ | 可能 **无法 offline compile** 或触发 **recompile**；LLM 常固定 max_seq + padding 策略 |
| QDQ 格式？ | ONNX QuantizeLinear-DequantizeLinear 对，便于 ORT 与 NPU 对齐 scale/zero-point |

### 10.6 动手清单（无真机也能准备）

```bash
# 1. 从 Qualcomm Software Center 下载 QAIRT SDK（需账号）
# 2. Python 3.10 环境；按 SDK 内文档 source envsetup
# 3. 浏览 docs/QNN、docs/Genie、docs/SNPE

# Host 侧：ONNX 小模型（不依赖 QAIRT 也能练）
python -c "
import torch, onnxruntime as ort
import numpy as np
m = torch.nn.Linear(128, 128)
x = torch.randn(1, 128)
torch.onnx.export(m, x, '/tmp/linear.onnx', opset_version=17)
sess = ort.InferenceSession('/tmp/linear.onnx')
out = sess.run(None, {'input': x.numpy()})[0]
print('max diff', np.abs(out - m(x).detach().numpy()).max())
"

# 有 SDK + Snapdragon 设备时：
# · AI Hub 下载预编译 Llama context binary
# · genie-t2t-run 跑 text-to-text
# · 打开 QNN/Genie verbose log 对照 CPU baseline
```

**建议产出**（Week 3 对应）：

- [ ] 手绘 **QAIRT 软件栈** 图（本节 10.1）
- [ ] 说清 **SNPE / QNN / Genie** 三选一决策
- [ ] 走通 **ONNX → ORT**；若有条件加 **QNN EP**
- [ ] 读 1 份 **release notes**，记住 2 个 limitation（op / shape / SDK version）

### 10.7 调试与跨层问题（QAIRT 语境）

```
PyTorch golden
    ↔ ONNXRuntime (CPU)
    ↔ ORT + QNN EP (HTP)
    ↔ Genie end-to-end
```

| 症状 | 可能原因 | 排查 |
|------|----------|------|
| 输出 max diff 大 | 量化 scale、per-channel 未对齐、layout NHWC vs NCHW | 逐层 dump；关量化对比 |
| HTP 比 CPU 慢 | fusion 未生效、错误 backend、频繁 CPU fallback | QNN profiling；看 partition 日志 |
| Load binary 失败 | SDK 版本与 compile 版本不匹配、SoC 不对 | 核对 AI Hub target device |
| LLM 长 prompt OOM | KV + 权重超设备 RAM | 减 context；更小模型；INT4 |

### 10.8 QAIRT 面试题速查

1. **QAIRT 和 QNN 什么关系？** — QAIRT 是总套件；QNN 是其中 **Engine Direct** SDK；底层共用 QAIRT runtime API。
2. **Genie 解决什么问题？** — LLM **decode 循环、KV、多模态 GenAI** 在 QNN 上的高层 runtime，不是替代 QNN。
3. **为何端侧要 offline compile？** — NPU 上无重型 JIT；提前针对 **固定芯片微架构** 做 fusion 与指令选择。
4. **AIMET 做什么？** — 量化感知训练 / PTQ 校准，产出 QNN 可消费的量化权重。
5. **与 CUDA + TensorRT 类比？** — QNN≈TensorRT builder/runtime；Genie≈LLM serving runtime；HTP≈NPU engine。

---

## 11. MoE 专章（面试深度版）

> 代码 demo：`basic/chapter_16_moe_inference.py`  
> JD **Preferred**：*MoE, routing, expert parallelism* — 端侧 MoE 是 **研究热点 + 部署难点**。

### 11.1 为什么需要 MoE？

**Mixture of Experts (MoE)**：把 FFN 层拆成 **多个并行 expert**，每 token 只激活 **top-k 个**（常用 k=2）。

```
                    ┌─ Expert 0 ─┐
Token → Router ────┼─ Expert 1 ─┼── weighted sum → output
  (gate)           ├─ Expert 2 ─┤
                    └─ Expert 7 ─┘
                         ↑
              只算被选中的 k 个 expert
```

**动机**：

| 维度 | Dense 7B | MoE（如 8×7B 类） |
|------|----------|-------------------|
| 总参数量 | ~7B | 可达 **数十 B**（多 expert 叠加） |
| 每 token 计算 | 全参数 FFN | 仅 **k 个 expert** ≈ 接近小 FFN |
| 效果 | 基准 | **更大容量**，训练/推理算力可控 |

**一句话**：**参数量 ↑，激活计算量 ≈ 可控** — 适合「要大模型效果、但推理预算有限」的场景。

### 11.2 结构拆解（以 Mixtral 8×7B 为参照）

典型 **Sparse MoE block** 替换 **单层 Dense FFN**：

1. **Router / Gate**：`hidden → num_experts` logits（线性层）
2. **Softmax + Top-k**：选 expert 索引与权重
3. **Expert FFN**：每个 expert 是独立 SwiGLU FFN（与 Llama 同构）
4. **加权求和**：`output = Σ weight_i * Expert_i(x)`

**与 Attention 的关系**：MoE 只改 **FFN 子层**；Attention（含 GQA）结构通常不变。

**参数量直觉**（跑 demo 可见具体数字）：

- Dense FFN：`3 × hidden × ffn_dim`（gate/up/down）
- MoE 总权重：`num_experts ×` 上述
- **每 token 激活**：router + `k ×` 单 expert FFN

### 11.3 Routing 机制

```python
# 概念代码（非生产）
logits = x @ W_gate          # [batch*seq, num_experts]
probs = softmax(logits)
weights, indices = topk(probs, k=2)
# 对每个 token，只 forward 选中的 experts，再按 weights 合并
```

**训练期额外机制**：

| 机制 | 作用 |
|------|------|
| **Load balancing aux loss** | 防止少数 expert 过热、其余闲置 |
| **Expert capacity / token dropping** | 限制每 expert 最多处理多少 token（训练稳定） |
| **Noise / jitter** | 促进探索、改善路由 |

**推理期**：无 aux loss；路由 **完全由输入决定** → **负载不均仍会发生**（batch 内不同 token 走不同 expert）。

### 11.4 Expert Parallelism vs Tensor Parallelism

| 策略 | 切什么 | 典型场景 |
|------|--------|----------|
| **TP** | 单层权重切到多 GPU | Dense 大模型单层太大 |
| **EP** | **不同 expert 放不同设备** | MoE 训练/推理 |
| **DP** | 数据并行 replica | 通用 |

**MoE 推理数据流**（多卡）：

1. Token 到 **router**（常 replicate 或 shard hidden）
2. 按 expert id **dispatch** token 到对应卡
3. 各卡算本地 experts
4. **combine** 回原始 batch 顺序

**面试点**：EP 引入 **all-to-all 通信**；小 batch 时通信占比高。端侧通常 **单 SoC**，无 NVLink — **所有 expert 权重都要在本地存储**。

### 11.5 训练 vs 推理关注点

| | 训练 | 推理（云端） | 推理（端侧 / Snapdragon） |
|--|------|--------------|---------------------------|
| 目标 | 收敛、负载均衡 | 吞吐、延迟 | **内存、功耗、确定性** |
| 内存 | 常 **存全部 expert** + ZeRO/EP | KV + active experts | **必须存全部 expert 权重** |
| 计算 | 反向需更多显存 | batch 大时 EP 高效 | **routing 动态** → NPU 难优化 |
| 量化 | 按 expert 校准 | GPTQ/AWQ per expert | INT4 权重仍 **体积巨大** |

### 11.6 端侧 / QAIRT 部署难点（差异化答案）

这是 **Qualcomm 面试** 最该讲的一段：

1. **存储**：Mixtral 级 MoE 全 expert INT4 仍可能 **数 GB～10GB+**，超手机可用 RAM → 需 **小 MoE、剪枝、distill 到 dense** 或 **partial expert cache**。
2. **动态 routing**：每 token expert 不同 → **算子稀疏、内存访问不规则**；HTP 擅长 **大矩阵定长 batch**，不擅长 **频繁分支**。
3. **无法只加载 active expert**：云端可 **按需换入**；手机闪存带宽与延迟使 **on-demand expert swap** 很贵。
4. **图编译**：QNN 喜欢 **静态子图**；MoE 可能需要 **custom op**（`MoE_dispatch` + 多 FFN binary）或 **展开为条件执行**（性能差）。
5. **功耗**：router + 多 expert 内存读取 → **DRAM 带宽** 成为瓶颈（与 [chapter_08](chapter_08_inference_pipeline.md) KV 争内存）。

**可主动说的 industry 方向**（不需背论文）：expert 聚类、共享 expert、**early fusion routing**、把 MoE **distill 成小 dense** 再上架 Genie。

### 11.7 与其他「稀疏」概念区分

| 概念 | 稀疏性 | 说明 |
|------|--------|------|
| **MoE** | 结构稀疏 — 整层 expert 二选一 | 路由在 **FFN** |
| **Sparse Attention** | 注意力矩阵稀疏 | 见 chapter_09 / Problem_5 |
| **Pruning** | 权重稀疏 | 剪枝后仍可能是 dense 执行 |
| **Switch Transformer** | top-1 routing | k=1，通信更简单 |

### 11.8 MoE 面试题速查

| 问题 | 要点 |
|------|------|
| MoE 为何能放大参数却不线性增加计算？ | 每 token 只激活 **k 个 expert** |
| top-k=2 含义？ | 两个 expert 输出按 gate 权重混合 |
| 训练时 expert collapse？ | aux loss 鼓励均匀使用 |
| MoE 和 ensemble 区别？ | MoE **输入相关路由**；ensemble 固定多模型 |
| 端侧怎么部署 MoE？ | 坦诚：**难**；优先小模型 / dense / 工具链 custom op；全 expert 驻留内存 |
| 量化 MoE 注意什么？ | **per-expert** scale 可能不同；敏感 expert 可 FP16 |
| 你知道哪些 MoE 模型？ | Mixtral、DeepSeek-MoE、Qwen-MoE、Switch Transformer |

### 11.9 与仓库其他章节的关系

| 章节 | 关联 |
|------|------|
| [chapter_08](chapter_08_inference_pipeline.md) | Decode 内存、KV 与 expert 权重 **抢 RAM** |
| [chapter_07](chapter_07_model_quantization.md) | 权重量化是端侧 MoE 能否落地的关键 |
| [chapter_10](chapter_10_distributed_inference.md) | EP 与 TP/PP 对比 |
| [chapter_13](chapter_13_memory_efficient_training.md) | 训练期 ZeRO + EP；LoRA 可只训 router/expert 子集 |

---

*祝面试顺利。Staff 岗的核心信号：**你能独立把「模型在 Snapdragon 上跑快、跑稳、跑省」这件事从图到硬件讲清楚，并证明你做过或能很快上手 QAIRT 栈。***

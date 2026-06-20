# 第 12 章 · 推理监控与 SLA：度量、告警与容量规划

> **本章导读**：第 7–11 章解决了「怎么把模型推快、推稳」。上线后还要回答：**够快吗？够稳吗？要加多少卡？花多少钱？** 本章从生产视角讲解 LLM 推理的 **核心指标**（TTFT、TPOT、P99、吞吐、成本）、**SLA 如何定义**、**怎么埋点与告警**，以及 **容量规划** 的粗算方法——把前面各章优化手段与可观测性闭环。**全部可运行代码见一个文件**：[`basic/chapter_12_inference_monitoring_sla.py`](../basic/chapter_12_inference_monitoring_sla.py)

---

## 12.1 为什么监控是推理优化的最后一环

```
优化（第 7–11 章）          监控（本章）
     │                         │
     ▼                         ▼
  改配置 / 改代码    ←────   指标是否达标？
     │                         │
     └──── 未达标 → 定位瓶颈 ───┘
```

没有测量就没有优化（第 7 章原则）。线上同样：**没有 SLI（指标），就没有 SLO/SLA，也无法做容量决策。**

| 阶段 | 问题 | 依赖监控 |
|------|------|----------|
| 上线前 | 压测是否满足延迟预算 | TTFT / TPOT / P99 |
| 上线后 | 用户投诉「慢」 | 分 Prefill / Decode 拆解 |
| 扩容 | 再加几张 GPU | 吞吐 vs GPU 利用率 |
| 降本 | 能否减卡 | 成本 / 1M tokens |

---

## 12.2 核心指标一览

### 12.2.1 延迟类（用户最关心）

| 指标 | 英文 | 定义 | 主要受哪阶段影响 |
|------|------|------|----------------|
| **首 token 延迟** | TTFT | 请求发出 → 收到第一个输出 token | **Prefill**、排队、调度 |
| **每 token 耗时** | TPOT / ITL | 相邻两个输出 token 的时间差（平均） | **Decode**、Speculative |
| **端到端延迟** | E2E Latency | 请求发出 → 最后一个 token 返回 | TTFT + TPOT × 输出长度 |
| **分位延迟** | P50 / P95 / P99 | 50%/95%/99% 请求的延迟 | 尾延迟、排队、超长 prompt |

**关系（粗算）**：

```
E2E ≈ TTFT + TPOT × (output_tokens - 1)
```

例：TTFT=200ms，TPOT=30ms，生成 100 token：

```
E2E ≈ 200 + 30 × 99 ≈ 3170 ms ≈ 3.2 s
```

### 12.2.2 吞吐类（系统容量）

| 指标 | 定义 | 单位 |
|------|------|------|
| **Output Throughput** | 每秒生成的 token 数 | tokens/s（系统级） |
| **Request Throughput** | 每秒完成的请求数 | req/s |
| **Goodput** | 满足 SLA 的有效吞吐 | req/s 或 tokens/s |

```
系统 tokens/s ≈ 并发 decode 数 / 平均 TPOT
```

### 12.2.3 资源与可靠性

| 指标 | 含义 |
|------|------|
| **GPU 利用率** | SM / 显存带宽是否喂饱 |
| **KV Cache 使用率** | PagedAttention block 占用比例 |
| **队列长度 / 等待时间** | 调度层排队 |
| **OOM 率 / 错误率** | 稳定性 |
| **Prefill 占比** | 时间花在 Prefill vs Decode |

---

## 12.3 TTFT 与 TPOT：分开看，分开优化

第 8 章两阶段推理，**监控也必须两阶段拆解**：

```
时间线 ──────────────────────────────────────────────→

|-- 排队 --|-- Prefill --|-- Decode --|-- Decode --|…
           ↑ TTFT 结束    ↑ TPOT      ↑ TPOT
           第一个 token
```

| 用户抱怨 | 先看 | 常见原因 | 对应优化（前文章节） |
|----------|------|----------|---------------------|
| 「第一个字慢」 | **TTFT** | 超长 prompt、排队、Prefill 占满 GPU | Chunked Prefill、FlashAttention |
| 「后面一个字一个字慢」 | **TPOT** | Decode memory-bound、batch 小 | KV Cache、算子融合、Speculative |
| 「有时特别慢」 | **P99** | 尾延迟、突发流量、个别超长请求 | Continuous Batching、限流、容量 |

**切忌只盯平均延迟**：10 个请求 100ms、1 个 10s，平均 ~1s，但 P99 体验极差。

---

## 12.4 分位延迟：P50 / P95 / P99

```
100 个请求按延迟排序:
  P50 = 第 50 个的延迟   （中位数，典型体验）
  P95 = 第 95 个的延迟   （较差 5%）
  P99 = 第 99 个的延迟   （尾延迟，SLA 常用）
```

| 分位 | 用途 |
|------|------|
| **P50** | 日常体验、A/B 对比 |
| **P95** | 内部 SLO 常用 |
| **P99** | 对外 SLA、容量红线 |

**LLM 特有**：同一服务 TTFT 与 TPOT 的 P99 可能由**不同请求类型**主导——RAG 长文档拉高 TTFT P99，高并发拉高 TPOT P99。

---

## 12.5 SLA / SLO：如何定义

**SLI**（Indicator）= 可测量的指标，如 P99 TTFT  
**SLO**（Objective）= 内部目标，如「P99 TTFT < 500ms」  
**SLA**（Agreement）= 对客户的承诺 + 违约后果

### 12.5.1 示例 SLO（聊天 API）

| SLI | SLO 示例 |
|-----|----------|
| TTFT P99 | < 800 ms（prompt ≤ 4K tokens） |
| TPOT P99 | < 50 ms |
| E2E P99 | < 30 s（输出 ≤ 512 tokens） |
| 可用性 | 99.9% / 月 |
| 错误率 | < 0.1%（非 4xx 客户端错误） |

### 12.5.2 Error Budget（错误预算）

```
SLO 99.9% 可用 → 每月约 43 分钟不可用预算
超出预算 → 冻结新功能，优先修稳定性 / 扩容
```

### 12.5.3 按场景差异化

| 场景 | TTFT | TPOT | 吞吐 |
|------|------|------|------|
| 聊天 | 严格 | 严格 | 中等 |
| 代码补全 | **极严格** | 严格 | 高 |
| 离线批处理 | 宽松 | 宽松 | **极高** |
| RAG | TTFT 常是瓶颈 | 中等 | 中等 |

---

## 12.6 如何埋点与度量

### 12.6.1 请求生命周期时间戳

```python
# 每个请求记录（示意）
request_metrics = {
    "request_id": "...",
    "t_arrive":      t0,   # 进入队列
    "t_prefill_start": t1,
    "t_first_token": t2,   # TTFT = t2 - t0
    "t_token_i":     [...], # 用于算 TPOT
    "t_done":        tN,   # E2E = tN - t0
    "prompt_tokens": 1024,
    "output_tokens": 256,
}
```

### 12.6.2 指标计算

```
TTFT     = t_first_token - t_arrive
TPOT_i   = t_token_{i+1} - t_token_i
TPOT_avg = mean(TPOT_i)
E2E      = t_done - t_arrive
tokens/s = sum(output_tokens) / window_seconds   # 滑动窗口
```

### 12.6.3 常用观测栈

| 组件 | 作用 |
|------|------|
| **Prometheus** | 拉取 vLLM / TGI `/metrics` |
| **Grafana** | TTFT / TPOT / QPS 仪表盘 |
| **OpenTelemetry** | 分布式 trace（Prefill vs Decode span） |
| **vLLM metrics** | `vllm:time_to_first_token_seconds`, `vllm:time_per_output_token_seconds` 等 |

### 12.6.4 仪表盘建议（最小集）

```
┌─────────────────────────────────────────────────┐
│  QPS / tokens/s    GPU 利用率    KV cache %      │
├─────────────────────────────────────────────────┤
│  TTFT  P50 / P95 / P99     （按 prompt 长度分桶） │
│  TPOT  P50 / P95 / P99                           │
│  E2E   P50 / P95 / P99                           │
├─────────────────────────────────────────────────┤
│  队列长度    OOM 次数    5xx 错误率               │
└─────────────────────────────────────────────────┘
```

**按 prompt 长度分桶**（如 0–1K / 1K–4K / 4K+）可避免「平均值掩盖长 prompt 问题」。

---

## 12.7 告警规则

| 告警 | 条件示例 | 可能动作 |
|------|----------|----------|
| TTFT P99 过高 | > SLO 持续 5 min | 查 Prefill 队列、超长 prompt 比例 |
| TPOT P99 过高 | > 80 ms 持续 5 min | 查 GPU 利用率、是否 decode 瓶颈 |
| 队列积压 | queue_depth > 100 | 扩容、限流 |
| GPU KV 满 | cache_usage > 90% | 降 `max_num_seqs` 或加卡 |
| OOM  spike | OOM > 0 / 5 min | 查并发上限、序列长度 |
| 错误率 | 5xx > 1% | 回滚、查模型加载 |

**告警原则**：

- 对 **P99 / 错误率** 告警，不对平均值 alone
- 带 **持续时长**（避免单次毛刺）
- 关联 **prompt 长度、模型版本、副本数** 标签

---

## 12.8 容量规划

### 12.8.1 需要多少 GPU？

**Decode 吞吐粗算**：

```
每卡 tokens/s ≈ 1 / TPOT_single_request   （单请求时）
实际          ≈ Continuous Batching 效率 × 上式 × 并发因子
```

**显存约束**（第 8、10 章）：

```
可并发请求数 ≈ (GPU_mem - weight) / KV_per_request
总吞吐       ≈ 可并发数 / 平均生成长度 × 平均 TPOT
```

**示例**：24 GB 卡，7B FP16 权重 14 GB，每请求 KV 0.5 GB，剩余 ~10 GB：

```
理论并发 ≈ 20 请求
若 TPOT=40ms → 每请求 ~25 tokens/s → 系统 ~500 tokens/s（理想上限，实际更低）
```

### 12.8.2 扩容决策树

```
P99 TTFT 超标 + Prefill 队列长  →  加卡 / Chunked Prefill / 限 prompt 长度
P99 TPOT 超标 + GPU 利用率低   →  查 batch 调度，非单纯加卡
P99 TPOT 超标 + GPU 利用率满   →  加卡 / Speculative / 量化
KV cache 满                   →  PagedAttention 调参或加卡
成本优先                      →  量化减显存 → 提高单卡并发
```

### 12.8.3 压测清单（上线前）

| 步骤 | 内容 |
|------|------|
| 1 | 固定模型与 `max_num_seqs`，扫 prompt 长度 |
| 2 | 记录 TTFT / TPOT P50/P99 |
| 3 | 逐步升 QPS 直到 P99 触 SLO |
| 4 | 记录拐点 QPS = 单副本容量 |
| 5 | 目标 QPS / 单副本容量 = 所需副本数 (+ 冗余) |

---

## 12.9 成本度量

| 指标 | 公式 / 含义 |
|------|------------|
| **$/1M tokens** | 月 GPU 成本 / 月输出 token 数 × 10⁶ |
| **GPU-hour / 1M tokens** | 硬件效率 |
| **成本 vs SLA** | 降 TPOT（Speculative）或提并发（量化）→ 同样 QPS 更少卡 |

```
月成本 ≈ GPU 数量 × 单价/卡/月
$/1M output tokens ≈ 月成本 / (月 output tokens / 1e6)
```

优化手段对成本的影响：

| 优化 | 对成本的影响 |
|------|-------------|
| 量化（第 7 章） | 单卡更多并发 → 少卡 |
| Continuous Batching（第 8 章） | 提高 tokens/s → 少卡 |
| Speculative（第 11 章） | 降 TPOT → 同样延迟下更高吞吐 |
| TP 多卡（第 10 章） | 单请求成本升，大模型才能跑 |

---

## 12.10 与第 7–11 章的指标映射

| 优化 | 主要改善指标 | 监控验证 |
|------|-------------|----------|
| INT8 量化 | 显存 ↓、并发 ↑ | KV 使用率、tokens/s |
| KV Cache + Continuous Batching | TPOT、吞吐 | TPOT P99、queue |
| FlashAttention | TTFT（Prefill） | TTFT 分桶（长 prompt） |
| PagedAttention | 并发上限 | cache_usage、OOM |
| TP / 多卡 | 大模型 TTFT/TPOT | 单卡 vs 系统吞吐 |
| Speculative Decoding | **TPOT** | `time_per_output_token` 下降 |

上线任何优化后，**用本章指标做 before/after 对比**，避免「感觉变快了」却无数据支撑。

---

## 12.11 本章小结

| 概念 | 一句话 |
|------|--------|
| **TTFT** | 首 token 时间，Prefill + 排队 |
| **TPOT** | 每输出 token 时间，Decode 体验 |
| **P99** | 尾延迟，SLA 常用 |
| **E2E** | TTFT + TPOT × 输出长度 |
| **tokens/s** | 系统容量核心 |
| **SLA / SLO** | 指标 + 阈值 + 错误预算 |
| **容量规划** | 压测拐点 × 冗余 = GPU 数 |

---

## 12.12 思考题与参考答案

### 思考题 1

TTFT P99 达标，但用户仍抱怨「生成慢」，应查哪个指标？

**参考答案**：

查 **TPOT P99** 和 **E2E P99**。TTFT 只覆盖「第一个字」；流式输出体验主要由 **TPOT** 决定。若输出很长，E2E = TTFT + TPOT × (N-1) 仍可能很大。还需看 **output_tokens 分布**——抱怨可能来自生成了很长回答。

### 思考题 2

GPU 利用率 30%，但 TPOT P99 很高，可能是什么原因？

**参考答案**：

Decode **Memory-bound**：GPU 计算单元空闲，在等 HBM 读 KV Cache——利用率数字可能不高但延迟仍大。也可能是 **batch 过小**、调度空转、或 **CPU 瓶颈**（tokenizer、调度）。不应简单加卡；应查 batch 大小、Continuous Batching 是否生效、是否可用 Speculative / 融合 Kernel 降 TPOT。

### 思考题 3

目标 1000 req/s，单副本压测饱和于 200 req/s（P99 刚达标），至少几个副本？

**参考答案**：

```
1000 / 200 = 5 副本（理论最小）
生产通常 +20%–50% 冗余 → 6–8 副本
```

还需考虑 **DP 负载均衡** 与健康检查；副本间模型版本需一致。

---

## 相关资源

- 第 7 章：[`document/chapter_07_model_quantization.md`](chapter_07_model_quantization.md) — 基准测量原则
- 第 8 章：[`document/chapter_08_inference_pipeline.md`](chapter_08_inference_pipeline.md) — TTFT / TPOT 定义
- 第 11 章：[`document/chapter_11_speculative_decoding.md`](chapter_11_speculative_decoding.md) — 降低 TPOT
- **本章全部代码（一个文件）**：[`basic/chapter_12_inference_monitoring_sla.py`](../basic/chapter_12_inference_monitoring_sla.py) — 运行 `python3 basic/chapter_12_inference_monitoring_sla.py`
- vLLM metrics：[Production Metrics](https://docs.vllm.ai/en/latest/serving/metrics.html)
- Google SRE：[SLI / SLO / SLA](https://sre.google/sre-book/service-level-objectives/)

---

*至此，第 7–12 章构成完整的 LLM 推理优化与生产闭环：压缩 → 调度 → Kernel → 分布式 → 投机解码 → 监控 SLA。*

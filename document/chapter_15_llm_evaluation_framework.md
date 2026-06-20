# 第 15 章 · 生成模型评估框架：自动指标、人工评估与线上监控

> **本章导读**：OpenAI 面试题 Problem 7 要求设计一套 **生成模型（LLM / 文生图）评估框架**——「好」是多维且主观的，不能只看一个 BLEU。本章以 `Problem_7_openAI_evaluation_framework.py` 为主线，讲解 **三层评估体系**（自动化 / 人工 / 线上）、核心指标取舍、可扩展框架设计，并补充 **LLM 面试高频** 的 LLM-as-Judge、基准集、事实性与统计显著性。**全部可运行代码见一个文件**：[`basic/chapter_15_llm_evaluation_framework.py`](../basic/chapter_15_llm_evaluation_framework.py)

---

## 15.1 为什么生成模型难评估？

| 分类任务 | 生成任务（LLM） |
|----------|----------------|
| 答案对错明确 | 同一问题多种合理回答 |
| Accuracy 够用 | 字面匹配指标失效 |
| 离线可完全复现 | 用户满意度、安全性难量化 |

**关键认知**：评估框架必须是 **分层的、多维的**——开发用自动指标迭代，发布前人工/红队，上线后看真实用户信号。

```
                    ┌─────────────────────────────┐
                    │     生成模型评估金字塔        │
                    └─────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   自动化指标              人工评估               线上评估
   PPL/BLEU/ROUGE         A/B / Likert           点赞/复制/留存
   BERTScore               Red Team               线上 A/B
   Benchmark (MMLU…)       标注一致性              业务 KPI
        │                     │                     │
   快、便宜、可复现          准、贵、主观            真实、有延迟
```

---

## 15.2 Problem 7 代码架构

`Problem_7_openAI_evaluation_framework.py` 采用 **可插拔 Metric + 统一 Framework**：

```
EvaluationMetric (ABC)
    ├── PerplexityMetric
    ├── BLEUMetric
    ├── ROUGEMetric
    └── BERTScoreMetric

EvaluationFramework
    ├── evaluate_automated_metrics()
    ├── evaluate_human_metrics()      → HumanEvaluationSimulator
    ├── evaluate_online_metrics()     → OnlineEvaluationSimulator
    ├── comprehensive_evaluation()
    └── generate_report()
```

### 15.2.1 统一结果容器

```python
@dataclass
class EvaluationResult:
    metric_name: str
    score: float
    confidence: Optional[float] = None
    metadata: Optional[Dict] = None
```

每个 metric 实现 `compute(predictions, references) -> EvaluationResult`，便于 **注册新指标、生成报告、做 CI 回归**。

### 15.2.2 依赖降级（Graceful Fallback）

代码检测 `evaluate`、`bert_score` 是否安装；缺失时用 **简化 word-overlap** 实现，保证 demo 可跑——生产环境应装完整库。

---

## 15.3 第一层：自动化指标

### 15.3.1 Perplexity（PPL）

\[
\text{PPL} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log P(x_i \mid x_{<i})\right)
\]

| 优点 | 缺点 |
|------|------|
| 无需参考答案 | **不能**衡量事实性、有用性 |
| 算得快，预训练常用 | 对 chat 模型 / 指令微调 **参考价值有限** |
| 越低 → 语言建模越好 | Problem 7 中为 **模拟 PPL**（真实需 held-out loss） |

**面试答法**：PPL 适合 **base LM 预训练**；对齐后的助手模型更看 **任务 benchmark + 人工 + 线上**。

### 15.3.2 BLEU / ROUGE（n-gram 重叠）

| 指标 | 侧重 | 典型场景 |
|------|------|----------|
| **BLEU** | n-gram **精确率**（预测中有多少在参考里） | 机器翻译 |
| **ROUGE-L** | **召回率** / 最长公共子序列 | 摘要 |

Problem 7 示例：

```python
predictions = ["The cat is sitting on the mat."]
references  = ["There is a cat on the mat."]

bleu = BLEUMetric().compute(predictions, references)
rouge = ROUGEMetric().compute(predictions, references)
```

**缺点（必考）**：

- 「同义不同词」得分低（cat vs feline）
- 无法判断 **幻觉 / 事实错误**
- 对 **开放式对话** 几乎无效

### 15.3.3 BERTScore（语义相似）

用 contextual embedding 算 token 级相似度，输出 P/R/F1：

```python
# Problem 7: BERTScoreMetric + bert_score 库
P, R, F1 = bert_score(predictions, references, lang="en")
```

| vs BLEU/ROUGE | |
|---------------|--|
| ✅ 捕捉 paraphrase | ❌ 仍依赖 reference，开放式生成 reference 难定 |
| ✅ 与人工判断相关性更好 | ❌ 计算更贵 |

### 15.3.4 指标选型速查

| 指标 | 需要 reference | 测什么 | LLM chat 适用度 |
|------|----------------|--------|----------------|
| PPL | 否（需 token 流） | 流畅 / 建模 | 预训练 △ |
| BLEU/ROUGE | 是 | 字面重叠 | 低（开放对话） |
| BERTScore | 是 | 语义相似 | 中（有标答任务） |
| **LLM-as-Judge** | 可选 | 帮助性/安全/事实 | 高（需注意 judge 偏见） |
| **Benchmark** | 固定题库 | 知识/推理 | 高（MMLU、GSM8K、HumanEval） |

### 15.3.5 图像生成（面试扩展）

| 指标 | 含义 | 方向 |
|------|------|------|
| **FID** | 生成分布 vs 真实分布（Inception 特征） | 越低越好 |
| **IS** | 清晰度 + 多样性 | 越高越好（不与真图比，易 hack） |

Problem 7 注释中列出 FID/IS；文本主线以 LLM 为主。

---

## 15.4 第二层：人工评估

Problem 7 的 `HumanEvaluationSimulator` 模拟三类方法：

### 15.4.1 A/B 测试（Pairwise Comparison）

```python
ab_results = human_evaluator.ab_test(model_a_outputs, model_b_outputs)
# model_a_win_rate, model_b_win_rate, tie_rate
```

- 评估者看 **同一 prompt 下两个回答**，选更好
- **最直接** 比较两个 checkpoint / 两个模型
- 需足够样本 + **统计显著性**（见 15.7）

### 15.4.2 Likert 量表（1–5 分）

Problem 7 六维度：

| 维度 | 英文 | 考察 |
|------|------|------|
| 流畅度 | Fluency | 语法、自然度 |
| 连贯性 | Coherence | 逻辑一致 |
| 事实性 | Factuality | 是否胡编 |
| 帮助性 | Helpfulness | 是否解决问题 |
| 无害性 | Harmlessness | 安全 |
| 创造性 | Creativity | 新颖度 |

```python
likert = human_evaluator.likert_rating(outputs, dimension="helpfulness")
# average, std, distribution
```

**标注质量**：报告 **inter-annotator agreement**（Cohen's κ / Krippendorff's α）。

### 15.4.3 Red Teaming（红队）

专家 **主动诱导** 有害/越狱输出——测 **安全上限**，不是平均质量。

| vs Likert | |
|-----------|--|
| 面向 **worst-case** | 通过平均 Likert 可能 **掩盖** 严重安全漏洞 |
| 发布前必做 | 与第 14 章公平性/偏见评估互补 |

---

## 15.5 第三层：线上评估

`OnlineEvaluationSimulator` 模拟 **隐式信号**：

| 信号 | 含义 |
|------|------|
| 👍 / 👎 | 显式反馈 |
| **复制回答** | 用户认为有用 |
| **会话时长** |  engagement（需结合任务，不是越长越好） |
| **追问** | 可能表示回答不清楚 |

```python
signals = online_evaluator.collect_implicit_signals(outputs)
# satisfaction_rate, copied_responses, avg_session_duration
```

**线上 A/B**：小流量切分（如 5% vs 95%），看 **满意度、留存、业务 KPI**——比离线 BLEU 更接近真实价值。

**注意**：线上指标受 **产品 UI、用户群体、季节** 影响，需与离线指标 **交叉验证**。

---

## 15.6 Problem 7 完整流水线

```
① 准备 predictions + references
         ↓
② evaluate_automated_metrics()  → BLEU, ROUGE, BERTScore, PPL
         ↓
③ evaluate_human_metrics()      → A/B, Likert 六维度
         ↓
④ evaluate_online_metrics()     → 隐式信号模拟
         ↓
⑤ generate_report()             → evaluation_report.md
```

**原则**：`comprehensive_evaluation()` 一次跑三层，报告 **不只看一个数**。

---

## 15.7 统计显著性与样本量（面试常考）

A/B 或 Likert 对比时，要问：**差异是噪声还是真改进？**

| 概念 | 用途 |
|------|------|
| **置信区间** | 指标不确定范围 |
| **p-value / 置换检验** | 两模型差异是否显著 |
| **Bonferroni 校正** | 多指标同时检验时控 false positive |

经验：人工 A/B 常需 **数百～数千** pairwise 判断才有稳定结论；线上 A/B 看 **power analysis** 定流量与时长。

---

## 15.8 LLM 评估扩展（面试高频）

Problem 7 以通用文本 metric 为主；大模型面试还会问：

### 15.8.1 标准 Benchmark

| Benchmark | 测什么 |
|-----------|--------|
| **MMLU** | 多学科知识 |
| **GSM8K** | 数学推理 |
| **HumanEval** | 代码生成 |
| **MT-Bench / Chatbot Arena** | 对话质量（人工/模型裁判） |

### 15.8.2 LLM-as-Judge

用强模型（GPT-4 等）按 rubric 打分或 pairwise 比较。

| 优点 | 缺点 |
|------|------|
| 可扩展、成本低 | **Position bias、verbosity bias、自偏好** |
| 多维 rubric | 需与 **人工子集** 校准 |

### 15.8.3 事实性 / 幻觉检测

- **FActScore**、检索增强对比、**引用溯源**
- n-gram 指标 **检测不了** 「流畅但错误」

### 15.8.4 Reference-free 指标

无标准答案时用：**Self-BLEU**（多样性）、**Perplexity on domain**、**win-rate vs baseline**。

---

## 15.9 分层评估策略（Best Practice）

| 阶段 | 主要手段 | 目标 |
|------|----------|------|
| **开发/迭代** | PPL、BLEU/BERTScore、小 benchmark | 快速回归 |
| **发布前** | 人工 Likert、A/B、Red Team、全量 benchmark | 质量与安全门禁 |
| **生产** | 隐式信号、线上 A/B、SLA（见第 12 章） | 真实价值与 drift |

```
开发 ──► 离线 benchmark 回归 ──► 人工/红队门禁 ──► 小流量线上 A/B ──► 全量
         ↑ 每次 PR/commit              ↑ 发版前           ↑ 持续
```

**不要**：只用 BLEU 决定是否上线 LLM chat 产品。

---

## 15.10 生产 Checklist

1. **多指标 dashboard**：自动 + 人工 + 线上  
2. **固定 eval set 版本化**（data + prompt 模板）  
3. **新模型必须过** regression + 安全集  
4. **报告 confidence / 样本量**，不只报 point estimate  
5. **LLM-as-Judge** 定期与人工校准  
6. 与第 12 章 **SLI/SLO** 对齐（延迟、错误率、满意度）

### 运行 Demo

```bash
# 概念 demo（numpy only，word-overlap metrics）
python3 basic/chapter_15_llm_evaluation_framework.py

# 完整框架（evaluate / bert-score 可选）
python3 openAI/Problem_7_openAI_evaluation_framework.py
```

---

## 15.11 本章小结

| 概念 | 一句话 |
|------|--------|
| **三层评估** | 自动 / 人工 / 线上，各解决不同问题 |
| **PPL** | 建模好坏，不衡量事实与有用 |
| **BLEU/ROUGE** | n-gram 重叠，翻译/摘要可用，开放对话弱 |
| **BERTScore** | 语义相似，比 BLEU 鲁棒，仍要 reference |
| **A/B 测试** | pairwise 比模型，最直接 |
| **Likert** | 多维度 1–5 分，需标注一致性 |
| **Red Team** | 测安全 worst-case |
| **隐式信号** | 点赞、复制、时长——线上真实反馈 |
| **LLM-as-Judge** | 可扩展但要防 bias、需校准 |

---

## 15.12 思考题与参考答案

### 思考题 1

BLEU 很高但用户满意度低，可能原因？

**参考答案**：

- 回答 **字面贴近 reference 但语义错误或无用**  
- Reference 本身质量差或与用户目标不符  
- BLEU **不衡量** 帮助性、安全、个性化  
- 应用 **人工评估 + 线上信号 + 任务 benchmark** 补充

### 思考题 2

为什么 chat 模型发布不能只看 MMLU？

**参考答案**：

MMLU 测 **短答知识题**，不覆盖 **多轮对话、指令遵循、工具调用、安全拒绝、长文生成**；需 MT-Bench、人工 A/B、红队、线上 A/B 组合。

### 思考题 3

线上 A/B 新模型满意度 +2%，是否立即全量？

**参考答案**：

检查：**统计显著性**、样本量、是否牺牲延迟/成本、**长尾安全 case** 是否变差、不同用户群是否均衡受益；必要时 **分阶段 rollout + 回滚预案**（第 12 章）。

---

## 15.13 面试速查

| 问题 | 要点 |
|------|------|
| 生成模型为何难 eval？ | 开放答案、多维质量、主观性 |
| BLEU 缺点？ | 字面匹配、同义词惩罚、不查事实 |
| BERTScore vs BLEU？ | 语义 embedding，更鲁棒，更贵 |
| PPL 适用？ | 预训练 LM；chat 辅助参考 |
| 人工 A/B vs Likert？ | A/B 比两模型；Likert 单模型多维分 |
| Red Team 作用？ | 安全/越狱 worst-case |
| 线上隐式信号？ | 点赞、复制、留存、追问 |
| LLM-as-Judge 风险？ | position/verbosity bias，需人工校准 |
| 评估框架设计？ | Metric 插件化 + 分层 + 报告 + 显著性 |

---

## 相关资源

- **本章全部代码（一个文件）**：[`basic/chapter_15_llm_evaluation_framework.py`](../basic/chapter_15_llm_evaluation_framework.py)
- 完整框架：[`openAI/Problem_7_openAI_evaluation_framework.py`](../openAI/Problem_7_openAI_evaluation_framework.py)
- 面试题梳理：[`openAI/openAI_questions.md`](../openAI/openAI_questions.md) — Problem 7 章节
- 线上监控：[`document/chapter_12_inference_monitoring_sla.md`](chapter_12_inference_monitoring_sla.md)
- 偏见/安全：[`document/chapter_14_bias_fairness_dataset.md`](chapter_14_bias_fairness_dataset.md)

---

*本章建立「自动 → 人工 → 线上」评估闭环；开发时用 Problem 7 框架快速回归，发版前用 benchmark + 红队门禁，上线后用第 12 章 SLA 持续监控。*

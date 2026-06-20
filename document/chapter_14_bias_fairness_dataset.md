# 第 14 章 · 数据集偏见与公平性：检测、缓解与 LLM 扩展

> **本章导读**：OpenAI 面试题 Problem 6 要求分析并修复**有偏见的数据集**——表面是信用违约的类别不平衡，底层考察的是 **ML 公平性全流程**：偏见从哪来、怎么量、怎么修、怎么在线上持续监控。本章以 `Problem_6_openAI_fix_bias_dataset.py` 为主线（95% vs 5% 违约样本 → SMOTE → 模型对比），并扩展到 **大模型时代** 常考的 RLHF 偏见、毒性、表征偏差与 red teaming。**全部可运行代码见一个文件**：[`basic/chapter_14_bias_fairness_dataset.py`](../basic/chapter_14_bias_fairness_dataset.py)

---

## 14.1 两类「偏」：别混淆

面试里「bias」可能指不同东西，先分清：

| 概念 | 含义 | Problem 6 主要演示 |
|------|------|-------------------|
| **类别不平衡（Class Imbalance）** | 正负样本数量悬殊 | ✅ 950 vs 50（19:1） |
| **公平性偏见（Fairness Bias）** | 模型对不同 **敏感属性群体**（性别、种族、地域）表现不均 | 代码未显式建 sensitive attribute，但方法论可迁移 |
| **LLM 偏见** | 预训练语料、对齐目标导致的刻板印象 / 毒性 | 14.8 节扩展 |

**关键认知**：SMOTE 解决的是 **「少数类样本太少、模型学不到」**；若偏见来自 **敏感群体在特征或标签上的系统性差异**，还需要 **分组评估 + 公平性约束 + 数据治理**。

---

## 14.2 偏见从哪来：四类来源

| 类型 | 定义 | 示例 |
|------|------|------|
| **采样偏见（Sampling）** | 收集方式导致子群体比例失真 | 人脸数据几乎全是某一 demographic |
| **社会偏见（Societal）** | 数据反映历史/文化中的不平等 | 古文本里「医生→男、护士→女」 |
| **测量偏见（Measurement）** | 采集工具或流程系统性误差 | 不同地区摄像头质量不同 |
| **算法偏见（Algorithmic）** | 目标函数或优化过程放大偏差 | 只优化点击率 → 推送极端内容 |

Problem 6 的合成数据主要是 **采样偏见 + 代表性不足**：

```python
data = {
    'feature1': np.random.rand(1000) * 10,
    'feature2': np.random.rand(1000) * 5,
    'target': [0] * 950 + [1] * 50   # 19:1 严重不平衡
}
```

特征与标签独立随机——现实中违约样本往往 **特征分布也不同**；面试中应说明：真实项目要先做 **按群组的 EDA**，再决定用重采样还是改特征/标签定义。

---

## 14.3 Problem 6 完整流水线

```
┌─────────────────────────────────────────────────────────┐
│  ① 构造 / 加载数据集，统计类别比例                         │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│  ② 偏见检测：countplot、不平衡比、特征 boxplot             │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│  ③ 基线模型：LogisticRegression on 原始 train              │
│     → accuracy 高但 minority F1 差（accuracy trap）       │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│  ④ 数据层修复：SMOTE 平衡训练集（仅 train，不动 test）       │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│  ⑤ 重训 + 对比：F1、confusion matrix、classification_report│
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│  ⑥ 总结 + 算法层/后处理/监控建议                           │
└─────────────────────────────────────────────────────────┘
```

**原则：先量化偏见，再选手段；test 集必须反映真实分布，不能用 SMOTE 污染 test。**

---

## 14.4 检测：别被 Accuracy 骗了

### 14.4.1 Accuracy Trap（准确率陷阱）

Problem 6 中，模型可以 **几乎全预测「非违约」** 仍获得 ~95% accuracy：

```
950 个负样本全对 + 50 个正样本全错 → accuracy ≈ 95%
但对业务关键的「违约」类 recall ≈ 0
```

| 指标 | 关注什么 | 不平衡场景 |
|------|----------|-----------|
| **Accuracy** | 整体对错 | ❌ 易被多数类主导 |
| **Precision** | 预测为正中真正为正的比例 | 少样本类不稳定 |
| **Recall** | 真正为正中被找出的比例 | ✅ .catch 少数类能力 |
| **F1** | P 与 R 的调和平均 | ✅ Problem 6 主对比指标 |
| **AUC-PR** | PR 曲线下面积 | 不平衡分类常用 |

```python
from sklearn.metrics import classification_report, f1_score

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['非违约', '违约']))
f1 = f1_score(y_test, y_pred)   # 关注 minority 的 f1-score
```

### 14.4.2 不平衡比与严重度

```python
class_counts = y.value_counts()
imbalance_ratio = class_counts[0] / class_counts[1]   # Problem 6: 19:1

# 经验阈值（演示用，非绝对标准）
# > 10:1 严重  |  > 5:1 中等  |  else 轻微
```

### 14.4.3 按敏感属性分组评估（公平性检测）

Problem 6 未建 `gender` / `region` 列，但面试必会：

```python
def detect_bias_by_group(y_true, y_pred, sensitive_attr):
    results = {}
    for value in np.unique(sensitive_attr):
        mask = sensitive_attr == value
        results[value] = {
            "accuracy": accuracy_score(y_true[mask], y_pred[mask]),
            "f1": f1_score(y_true[mask], y_pred[mask], zero_division=0),
            "count": mask.sum(),
        }
    return results
```

若某群体 F1 系统性低于其他群体 → **公平性风险**，不能只看 overall metrics。

---

## 14.5 公平性指标（面试常考）

在 **二分类 + 敏感属性 A**（如性别）下，设预测为 \(\hat{Y}\)，真实为 \(Y\)：

| 指标 | 公式/intuition | 含义 |
|------|----------------|------|
| **Demographic Parity（人口统计平等）** | \(P(\hat{Y}=1 \mid A=a) \approx P(\hat{Y}=1 \mid A=b)\) | 各群体 **正例率** 相近 |
| **Equalized Odds（均等化机会）** | 各群体 **TPR、FPR** 都相近 | 对正负例都公平 |
| **Disparate Impact（ disparate impact ）** | 少数群体正例率 / 多数群体正例率 ≥ 0.8（四分之五法则） | 美国就业场景常用启发式 |

**Trade-off（必答）**：

- 追求 demographic parity 可能 **牺牲 overall accuracy**
- 公平性与效用 **often Pareto 前沿**，需产品/法务参与定目标
- 单一指标不够，应 **多指标 + 分组 confusion matrix**

---

## 14.6 解决方案：三层框架

业界常分 **Pre / In / Post processing**（与 Problem 6 + openAI_questions 一致）：

```
        Pre-processing          In-processing           Post-processing
        （改数据）               （改训练）               （改预测）
            │                       │                       │
     SMOTE / 欠采样          class_weight /           按群体调阈值
     数据增强 / 合成          fair loss 约束            校准输出
     收集更多少数类           adversarial debiasing
```

### 14.6.1 数据层（Problem 6 核心：SMOTE）

**只在训练集上** 重采样：

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
# 950:50 → 约 665:665（对少数类合成样本）
```

**SMOTE 原理**：在少数类样本的 k 近邻之间 **线性插值** 生成合成点。

| 方法 | 做法 | 优点 | 缺点 |
|------|------|------|------|
| **Random Oversampling** | 复制少数类 | 简单 | 过拟合风险 |
| **SMOTE** | 合成新样本 | 多样性更好 | 高维 / 稀疏特征可能生成噪声 |
| **Random Undersampling** | 删多数类 | 快 | 丢信息 |
| **SMOTETomek** | SMOTE + 清理边界 | 平衡质量 | 更慢 |

**禁忌**：❌ 对 **test 集** 做 SMOTE → 指标虚高、上线失真。

### 14.6.2 算法层（不改数据分布）

**类别权重**（sklearn 一行，无 imblearn 依赖）：

```python
LogisticRegression(class_weight='balanced')  # 反比于类别频率
```

**公平性损失**（概念代码）：

```python
def fair_loss(predictions, targets, sensitive_attr, lambda_fair=1.0):
    base_loss = F.cross_entropy(predictions, targets)
    # 约束不同群体预测均值接近
    g0 = predictions[sensitive_attr == 0].mean()
    g1 = predictions[sensitive_attr == 1].mean()
    fairness_loss = (g0 - g1) ** 2
    return base_loss + lambda_fair * fairness_loss
```

### 14.6.3 后处理层

- **按群体调整分类阈值**，使 TPR/FPR 更均衡  
- **校准（Calibration）**：Platt scaling / isotonic regression  
- LLM 场景：**拒绝回答、改写、安全分类器** 作为 post-filter

---

## 14.7 Problem 6 实验解读

| 阶段 | 典型现象 |
|------|----------|
| 偏见模型 | Accuracy ~0.95，违约类 **recall 极低**，F1 低 |
| SMOTE 后 | Accuracy 可能 **略降**，违约类 recall/F1 **显著升** |
| 混淆矩阵 | 偏见模型对违约几乎全判错；修复后 TP 增加 |

**面试答法**：「我们不用 accuracy 判断成功，看 **minority 的 recall/F1** 和 **confusion matrix**；SMOTE 只作用于 train，test 保持真实分布。」

---

## 14.8 扩展到大模型（LLM 面试高频）

传统表格分类的 SMOTE **不能直接套** 到 GPT，但 **问题结构相同**：数据从哪来 → 怎么量 → 怎么修 → 怎么监控。

| 维度 | 传统 ML（本章） | LLM |
|------|----------------|-----|
| **数据偏见** | 类别 / 群体样本量 | 预训练语料 **地域、语言、性别** 占比 |
| **标签偏见** | 违约标签定义 | RLHF 标注者偏好、reward model 偏差 |
| **检测** | 分组 F1、公平性指标 | **Bias benchmarks**（BBQ、CrowS-Pairs）、毒性检测 |
| **缓解** | SMOTE、class_weight | **数据过滤、平衡采样、DPO/RLHF 约束、system prompt** |
| **后处理** | 阈值调整 | **Moderation API、guardrails、refusal** |
| **监控** | 分组指标 dashboard | **red teaming、在线毒性率、用户举报** |

### 14.8.1 常考 LLM 偏见题型

1. **「模型为什么会对某群体刻板印象？」**  
   → 预训练语料频率 + 对齐数据不足 + 评估未覆盖该群体。

2. **「RLHF 会引入什么偏见？」**  
   → 标注员 demographic 单一、reward hacking、过度拒绝（false refusal）。

3. **「如何评估 LLM 公平性？」**  
   → 分组 prompt 集、对比不同 demographic 下有害/刻板输出率、人工 + 自动 judge。

4. **「SMOTE 能用于 LLM 吗？」**  
   → 不直接适用；类比手段是 **curated 平衡语料、 rejection sampling、合成 data with human review**。

---

## 14.9 生产 Checklist

### 数据与实验

1. **train/test 分层**（`stratify=y`），test 保持真实分布  
2. **报告 overall + 分组** 指标，不只 accuracy  
3. **重采样仅 train**；记录版本与 random seed  
4. **文档化** 敏感属性定义与合规边界（GDPR、四分之五法则等）

### 上线与监控

1. **持续监控** 各群体 F1 / 正例率 drift  
2. **A/B 测试** 公平性约束前后对业务与公平指标的影响  
3. LLM：**定期 red team**，毒性/偏见 benchmark 回归  
4. **人在回路**：高风险场景人工复核

### 运行 Demo

```bash
# 概念 demo（numpy + sklearn，无需 imblearn）
python3 basic/chapter_14_bias_fairness_dataset.py

# 完整 SMOTE + 可视化（需 imblearn、matplotlib）
python3 openAI/Problem_6_openAI_fix_bias_dataset.py
```

依赖（完整版）：`pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`, `matplotlib`, `seaborn`

---

## 14.10 本章小结

| 概念 | 一句话 |
|------|--------|
| **采样偏见** | 某类/某群体样本过少或过多 |
| **Accuracy trap** | 不平衡数据上 accuracy 误导，看 F1/recall |
| **SMOTE** | 训练集少数类合成过采样，test 不可用 |
| **Pre/In/Post** | 改数据 / 改损失 / 改预测三层缓解 |
| **Demographic parity** | 各群体正例率接近 |
| **Equalized odds** | 各群体 TPR、FPR 接近 |
| **分组评估** | 公平性必须拆 sensitive attribute |
| **LLM 扩展** | 语料+对齐+监控，SMOTE 不直接套用 |

---

## 14.11 思考题与参考答案

### 思考题 1

950 负、50 正，模型全预测负，accuracy 多少？对正类的 recall？

**参考答案**：

- Accuracy = 950/1000 = **95%**
- Recall（正类）= 0/50 = **0%** —— 典型 accuracy trap

### 思考题 2

为什么 SMOTE 只能用在训练集？

**参考答案**：

Test 应模拟 **真实部署分布**。对 test 过采样会 **虚增少数类频率**，指标乐观但无法反映线上对 rare 事件的真实检出能力；且造成 **数据泄露** 式的评估偏差。

### 思考题 3

Demographic parity 与 Equalized odds 能同时满足吗？

**参考答案**：

一般 **不能同时严格满足**（除非 perfect prediction 或群体 base rate 相同）。实践中选其一或加 relax 约束，并在 accuracy 与公平性间做 **显式 trade-off**。

### 思考题 4

LLM 指令微调后，英语 benchmark 升、低资源语言降，算哪种偏见？怎么缓解？

**参考答案**：

- **采样/表征偏见**：微调语料以英语为主  
- 缓解：增加低资源语言数据、**按语言分组 eval**、多语言混合 batch、DPO 加语言公平约束；上线按 locale 监控

---

## 14.12 面试速查

| 问题 | 要点 |
|------|------|
| 不平衡 vs 公平偏见？ | 前者类比例；后者 **群体间表现差** |
| 为什么不看 accuracy？ | 多数类 dominant → **accuracy trap** |
| SMOTE 是什么？ | 少数类 kNN 间 **插值合成** |
| 三层缓解？ | Pre 采样 / In 损失 / Post 阈值 |
| Demographic parity？ | 各群体 **正例率** 接近 |
| Equalized odds？ | 各群体 **TPR、FPR** 接近 |
| LLM 怎么评偏见？ | BBQ、CrowS-Pairs、red team、分组 prompt |
| RLHF 偏见来源？ | 标注 demographic、reward 设计、过度对齐 |

---

## 相关资源

- **本章全部代码（一个文件）**：[`basic/chapter_14_bias_fairness_dataset.py`](../basic/chapter_14_bias_fairness_dataset.py)
- 完整流水线（SMOTE + 可视化）：[`openAI/Problem_6_openAI_fix_bias_dataset.py`](../openAI/Problem_6_openAI_fix_bias_dataset.py)
- 面试题梳理：[`openAI/openAI_questions.md`](../openAI/openAI_questions.md) — Problem 6 章节
- 训练数据治理：[`document/chapter_13_memory_efficient_training.md`](chapter_13_memory_efficient_training.md) — 数据流水线
- 线上监控：[`document/chapter_12_inference_monitoring_sla.md`](chapter_12_inference_monitoring_sla.md) — SLA 与指标

---

*本章从表格分类的类别不平衡出发，建立「检测 → 缓解 → 监控」公平性思维，并映射到大模型语料与对齐场景——AI 安全与 Responsible AI 面试的通用模板。*

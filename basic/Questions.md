# 基础机器学习面试问题集

本文件包含了机器学习面试中的核心基础问题，每个问题都配有详细的解释、数学原理和完整的代码实现。

## 📚 目录

1. [Transformer 注意力机制详解](#problem-1-transformer-注意力机制详解) - [📁 transformer_implementation.py](transformer_implementation.py)
2. [反向传播数学原理与实现](#problem-2-反向传播数学原理与实现) - [📁 backpropagation.py](backpropagation.py) | [📁 backpropagation_improved.py](backpropagation_improved.py) | [📁 backprogagation.py](backprogagation.py)

---

## Problem 1: Transformer 注意力机制详解

### 🎯 问题概述

这是一个非常经典且重要的机器学习面试问题，旨在考察你对 Transformer 核心组件的深入理解和代码实现能力。

**考察重点：**
- 自注意力机制的数学原理
- 多头注意力的实现细节
- 复杂度分析（时间和空间）
- 从零开始的代码实现能力

### 🧠 核心概念

#### 1. 自注意力机制 (Self-Attention)

自注意力机制允许模型在处理一个序列时，为序列中的每个单词计算"注意力分数"，决定在编码当前单词时应该对其他单词投入多少关注度。

**三个关键角色：**
- **Query (Q)**: 代表当前正在处理的单词，会去"查询"序列中所有其他单词
- **Key (K)**: 代表序列中可以被"查询"的单词，Query 会和每一个 Key 计算相似度
- **Value (V)**: 代表序列中单词的实际内容，注意力分数会作用在 Value 上

#### 2. 缩放点积注意力 (Scaled Dot-Product Attention)

**计算步骤：**
1. **计算相似度分数**: `Q · K^T` (通过矩阵乘法高效实现)
2. **缩放**: 除以 `√d_k` 防止梯度消失
3. **掩码** (可选): 在解码器中防止看到未来信息
4. **Softmax**: 转换为概率分布
5. **加权求和**: 注意力权重与 Value 矩阵相乘

**数学公式：**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

#### 3. 多头注意力 (Multi-Head Attention)

**实现过程：**
1. **线性投影**: 将 Q, K, V 通过 h 个独立的线性层投影
2. **并行计算**: 对 h 组 Q, K, V 分别执行注意力计算
3. **拼接**: 将 h 个输出在特征维度上拼接
4. **最终变换**: 通过线性层得到最终输出

**数学公式：**
```
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
```

### 📊 复杂度分析

**时间复杂度: O(n² · d)**
- 主要瓶颈：注意力分数计算 `Q · K^T`
- 序列长度 n 通常大于嵌入维度 d
- 每个头的复杂度：O(n² · d_k)，总复杂度：O(n² · d)

**空间复杂度: O(n²)**
- 主要存储：注意力权重矩阵 (h, n, n)
- 这是 Transformer 难以处理长序列的主要原因

### 💻 代码实现

**实现文件**: [transformer_implementation.py](transformer_implementation.py)

**实现特点：**
- 从零开始实现，只使用基础张量操作
- 包含 PyTorch 和 NumPy 两个版本
- 详细的注释和数学公式对应
- 完整的测试用例和可视化

**核心功能：**
- `scaled_dot_product_attention()`: 缩放点积注意力
- `multi_head_attention()`: 多头注意力机制
- `positional_encoding()`: 位置编码
- `transformer_block()`: 完整的 Transformer 块

---

## Problem 2: 反向传播数学原理与实现

### 🎯 问题概述

反向传播是深度学习的核心算法，通过链式法则计算梯度，实现神经网络的参数更新。

**考察重点：**
- 链式法则的数学原理
- 梯度计算的推导过程
- 不同激活函数的导数
- 实际代码实现和调试

### 🧮 数学基础

#### 1. 前向传播

神经网络每一层可以表示为：
```
a^(l) = f^(l)(z^(l))
z^(l) = W^(l) a^(l-1) + b^(l)
```

其中：
- `a^(l)`: 第 l 层的输出（激活值）
- `W^(l), b^(l)`: 权重和偏置
- `f^(l)`: 激活函数

#### 2. 损失函数

以均方误差 (MSE) 为例：
```
L = (1/2) Σ(y_i - ŷ_i)²
```

#### 3. 参数更新

通过梯度下降更新参数：
```
W^(l) ← W^(l) - η ∂L/∂W^(l)
b^(l) ← b^(l) - η ∂L/∂b^(l)
```

### 🔑 链式法则

#### 梯度计算

使用链式法则计算梯度：
```
∂L/∂W^(l) = ∂L/∂a^(l) · ∂a^(l)/∂z^(l) · ∂z^(l)/∂W^(l)
```

#### 误差项

定义误差项 δ^(l)：
```
δ^(l) = ∂L/∂z^(l) = (W^(l+1))^T δ^(l+1) ⊙ f'(z^(l))
```

### 💻 代码实现

**实现文件**: 
- [backpropagation.py](backpropagation.py) - 基础实现
- [backpropagation_improved.py](backpropagation_improved.py) - 改进版本
- [backprogagation.py](backprogagation.py) - 备用实现

**实现特点：**
- 从零开始实现反向传播算法
- 支持多种激活函数（Sigmoid, Tanh, ReLU）
- 包含梯度检查功能
- 完整的训练循环和可视化

**核心功能：**
- `forward_pass()`: 前向传播
- `backward_pass()`: 反向传播
- `update_parameters()`: 参数更新
- `gradient_check()`: 梯度验证

### 📈 训练示例

**XOR 问题训练结果：**
```
Epoch 0, Loss: 0.3330
Epoch 1000, Loss: 0.2486
Epoch 2000, Loss: 0.2447
...
Epoch 9000, Loss: 0.0098
```

**预测结果：**
```
[[0.021]  # 0 XOR 0 = 0
 [0.981]  # 0 XOR 1 = 1
 [0.981]  # 1 XOR 0 = 1
 [0.018]] # 1 XOR 1 = 0
```

---

## 🚀 使用指南

### 运行代码

1. **环境准备**:
   ```bash
   pip install torch numpy matplotlib
   ```

2. **运行 Transformer 实现**:
   ```bash
   python transformer_implementation.py
   ```

3. **运行反向传播实现**:
   ```bash
   python backpropagation.py
   python backpropagation_improved.py
   ```

### 学习建议

1. **理论理解**: 先理解数学原理，再查看代码实现
2. **代码实践**: 运行代码，观察输出结果
3. **参数调试**: 尝试修改超参数，观察对训练的影响
4. **扩展实现**: 基于现有代码实现更复杂的功能

### 面试准备

1. **概念掌握**: 能够清晰解释每个概念
2. **数学推导**: 能够推导关键公式
3. **代码实现**: 能够从零开始实现核心算法
4. **复杂度分析**: 能够分析时间和空间复杂度
5. **实际应用**: 能够讨论在实际项目中的应用

---

## 📝 总结

这两个问题是机器学习面试中的经典题目，涵盖了：

- **Transformer**: 现代深度学习的核心架构
- **反向传播**: 深度学习的基础算法

通过深入理解这些概念和实现，你将具备：
- 扎实的理论基础
- 强大的代码实现能力
- 清晰的数学思维
- 解决实际问题的能力

**继续学习建议**: 可以进一步学习注意力机制的变体（如稀疏注意力、线性注意力）和更高级的优化算法（如 Adam、RMSprop）。
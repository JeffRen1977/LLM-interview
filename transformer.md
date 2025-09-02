# Transformer 注意力机制详解

好的，这是一个非常经典且重要的机器学习面试问题。它旨在考察你对 Transformer 核心组件的深入理解和代码实现能力。

下面我将为你提供详细的解释、从零开始的代码实现（PyTorch 和 NumPy 版本）以及复杂度分析，完全遵循面试题目的要求。

## 详细解释：Transformer 注意力机制
Transformer 的核心是 自注意力机制（Self-Attention），它允许模型在处理一个序列（如一个句子）时，为序列中的每个单词（token）都计算一个“注意力分数”，这个分数决定了在编码当前单词时，应该对序列中其他单词投入多少关注度。
这个过程主要涉及三个关键的角色，它们都是从同一个输入向量（词嵌入）通过不同的线性变换得到的：
 * Query (Q)：代表当前正在处理的单词，它会去“查询”序列中所有其他的单词。
 * Key (K)：代表序列中可以被“查询”的单词。Query 会和每一个 Key 计算相似度。
 * Value (V)：代表序列中单词的实际内容。计算出的注意力分数会作用在 Value 上，用来加权求和，得到最终的输出。
1. 缩放点积注意力 (Scaled Dot-Product Attention)
这是注意力机制最核心的计算部分。它的计算过程分为以下几步：
 * 计算相似度分数：将每个 Query 向量与所有 Key 向量进行点积运算。这可以高效地通过矩阵乘法 Q \\cdot K^T 实现。
 * 缩放 (Scaling)：为了防止点积结果过大导致梯度消失（尤其是在 softmax 函数中），需要将点积结果除以一个缩放因子。这个因子通常是 Key 向量维度 $d_k$ 的平方根，即 $\frac{1}{\sqrt{d_k}}$。
 * 可选的掩码 (Masking)：在某些场景下（例如，在解码器中防止看到未来的信息），需要应用一个掩码，将特定位置的分数设置为一个非常小的负数（如 -1e9），这样在经过 softmax 后，这些位置的权重会趋近于 0。
 * 计算注意力权重：对缩放后的分数应用 Softmax 函数，将其转换为概率分布，得到每个单词相对于当前单词的注意力权重。
 * 加权求和：将计算出的注意力权重矩阵与 Value 矩阵相乘，得到最终的加权输出。
其数学公式为：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
2. 多头注意力 (Multi-Head Attention)
为了让模型能够从不同的表示子空间中共同学习信息，Transformer 引入了“多头”机制。它不是只计算一次注意力，而是将 Query、Key 和 Value 通过不同的线性变换（权重矩阵）投影多次，然后并行地对每一次投影的结果执行缩放点积注意力。
其过程如下：
 * 线性投影：将输入的 Q, K, V 分别通过 h 个独立的线性层（$W_i^Q$, $W_i^K$, $W_i^V$）进行投影，得到 h 组不同的 Q, K, V。这里的 h 就是"头的数量"。
 * 并行计算注意力：对这 h 组 Q, K, V 分别执行缩放点积注意力计算，得到 h 个输出矩阵 $\text{head}_i$。
   $$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$
 * 拼接 (Concatenate)：将这 h 个输出矩阵在特征维度上拼接起来。
 * 最终线性变换：将拼接后的矩阵通过最后一个线性层（$W^O$）进行变换，得到最终的多头注意力输出。
这种机制的好处在于，每个“头”可以学习到不同方面的注意力关系。例如，一个头可能关注语法关系，另一个头可能关注语义上的近义词关系。
## 代码实现

我们将严格遵循"从零实现"的要求，只使用基础的 torch.Tensor 或 numpy.array 操作。

### PyTorch 实现

在 PyTorch 中实现更为常见，因为其自动求导功能和神经网络模块化的特性。
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    """
    实现缩放点积注意力机制
    
    这是 Transformer 的核心组件，计算序列中每个位置对其他位置的注意力权重
    """
    def __init__(self, dropout_rate=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, q, k, v, mask=None):
        """
        前向传播计算注意力
        
        Args:
            q: Queries, 形状 [batch_size, n_heads, seq_len, d_k]
            k: Keys, 形状 [batch_size, n_heads, seq_len, d_k]
            v: Values, 形状 [batch_size, n_heads, seq_len, d_v]
            mask: 掩码, 形状 [batch_size, 1, 1, seq_len] 或 [batch_size, 1, seq_len, seq_len]
        
        Returns:
            output: 注意力输出, 形状 [batch_size, n_heads, seq_len, d_v]
            attention_weights: 注意力权重, 形状 [batch_size, n_heads, seq_len, seq_len]
        """
        d_k = k.size(-1)  # 获取 key 的维度
        
        # 1. 计算 Q 和 K^T 的点积 (注意力分数)
        # (batch, h, seq_len, d_k) @ (batch, h, d_k, seq_len) -> (batch, h, seq_len, seq_len)
        attention_scores = torch.matmul(q, k.transpose(-2, -1))

        # 2. 缩放：除以 sqrt(d_k) 防止梯度消失
        attention_scores = attention_scores / math.sqrt(d_k)

        # 3. (可选) 应用掩码，将不需要的位置设为负无穷
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # 4. 计算 Softmax 得到注意力权重 (概率分布)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 5. 权重与 V 相乘得到最终输出
        # (batch, h, seq_len, seq_len) @ (batch, h, seq_len, d_v) -> (batch, h, seq_len, d_v)
        output = torch.matmul(attention_weights, v)

        return output, attention_weights
```


```python
class MultiHeadAttention(nn.Module):
    """
    实现多头注意力机制
    
    通过多个并行的注意力头来捕获不同类型的关系
    """
    def __init__(self, d_model, n_heads, dropout_rate=0.1):
        """
        初始化多头注意力
        
        Args:
            d_model: 模型的总维度 (embedding dimension)
            n_heads: 注意力头的数量
            dropout_rate: Dropout 概率
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度

        # 定义线性投影层：为每个头创建独立的 Q, K, V 投影
        self.W_q = nn.Linear(d_model, d_model)  # Query 投影
        self.W_k = nn.Linear(d_model, d_model)  # Key 投影
        self.W_v = nn.Linear(d_model, d_model)  # Value 投影
        self.W_o = nn.Linear(d_model, d_model)  # 输出投影

        self.attention = ScaledDotProductAttention(dropout_rate)

    def forward(self, q, k, v, mask=None):
        """
        前向传播计算多头注意力
        
        Args:
            q, k, v: 输入张量, 形状 [batch_size, seq_len, d_model]
            mask: 可选的掩码张量
        
        Returns:
            output: 多头注意力输出, 形状 [batch_size, seq_len, d_model]
            attention_weights: 注意力权重, 形状 [batch_size, n_heads, seq_len, seq_len]
        """
        batch_size = q.size(0)

        # 1. 线性投影：将输入投影到 Q, K, V 空间
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model]
        q, k, v = self.W_q(q), self.W_k(k), self.W_v(v)

        # 2. 拆分成多个头：重塑张量以支持多头并行计算
        # [batch_size, seq_len, d_model] -> [batch_size, n_heads, seq_len, d_k]
        q = q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 3. 计算注意力：对每个头并行计算注意力
        # output: [batch_size, n_heads, seq_len, d_k]
        # attention_weights: [batch_size, n_heads, seq_len, seq_len]
        output, attention_weights = self.attention(q, k, v, mask)

        # 4. 拼接头：将所有头的输出拼接起来
        # [batch_size, n_heads, seq_len, d_k] -> [batch_size, seq_len, d_model]
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 5. 最终的线性变换：通过输出投影层
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model]
        output = self.W_o(output)

        return output, attention_weights
```

#### 使用示例

```python
# 设置参数
d_model = 512      # 模型维度
n_heads = 8        # 注意力头数
seq_len = 10       # 序列长度
batch_size = 64    # 批次大小

# 创建多头注意力模块
mha = MultiHeadAttention(d_model, n_heads)

# 创建随机输入数据
q = torch.randn(batch_size, seq_len, d_model)  # Query
k = torch.randn(batch_size, seq_len, d_model)  # Key
v = torch.randn(batch_size, seq_len, d_model)  # Value

# 前向传播
output, attn_weights = mha(q, k, v)

# 打印形状信息
print("Input shape:", q.shape)
print("Output shape:", output.shape)
print("Attention weights shape:", attn_weights.shape)

# 输出结果:
# Input shape: torch.Size([64, 10, 512])
# Output shape: torch.Size([64, 10, 512])
# Attention weights shape: torch.Size([64, 8, 10, 10])
```

### NumPy 实现

用 NumPy 实现需要我们手动管理权重矩阵，并且没有自动梯度。这纯粹是为了考察对底层数学运算的理解。

```python
import numpy as np

def softmax(x):
    """
    实现数值稳定的 softmax 函数
    
    Args:
        x: 输入数组
    
    Returns:
        softmax 输出
    """
    # 减去最大值防止数值溢出
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def scaled_dot_product_attention(q, k, v, mask=None):
    """
    实现缩放点积注意力 (NumPy 版本)
    
    Args:
        q: Query 矩阵
        k: Key 矩阵
        v: Value 矩阵
        mask: 可选的掩码
    
    Returns:
        output: 注意力输出
        attention_weights: 注意力权重
    """
    d_k = q.shape[-1]
    
    # 1. 计算 Q @ K.T (注意力分数)
    attention_scores = np.matmul(q, k.swapaxes(-2, -1))
    
    # 2. 缩放：除以 sqrt(d_k)
    attention_scores = attention_scores / np.sqrt(d_k)
    
    # 3. 应用掩码 (如果提供)
    if mask is not None:
        attention_scores += (mask * -1e9)
    
    # 4. 计算 Softmax 得到注意力权重
    attention_weights = softmax(attention_scores)
    
    # 5. 计算最终输出
    output = np.matmul(attention_weights, v)
    
    return output, attention_weights

class MultiHeadAttentionNumpy:
    """
    使用 NumPy 实现的多头注意力机制
    
    这个实现展示了底层数学运算，不依赖深度学习框架
    """
    def __init__(self, d_model, n_heads):
        """
        初始化多头注意力
        
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # 初始化权重矩阵 (在实际训练中，这些权重会被学习)
        # 使用 Xavier 初始化
        self.W_q = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_k = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_v = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_o = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)

    def forward(self, q, k, v, mask=None):
        """
        前向传播计算多头注意力
        
        Args:
            q, k, v: 输入矩阵, 形状 [batch_size, seq_len, d_model]
            mask: 可选的掩码
        
        Returns:
            output: 多头注意力输出
            attn_weights: 注意力权重
        """
        batch_size = q.shape[0]
        seq_len = q.shape[1]

        # 1. 线性投影：将输入投影到 Q, K, V 空间
        q = np.dot(q, self.W_q)
        k = np.dot(k, self.W_k)
        v = np.dot(v, self.W_v)

        # 2. 拆分头：重塑张量以支持多头并行计算
        q = q.reshape(batch_size, seq_len, self.n_heads, self.d_k).swapaxes(1, 2)
        k = k.reshape(batch_size, seq_len, self.n_heads, self.d_k).swapaxes(1, 2)
        v = v.reshape(batch_size, seq_len, self.n_heads, self.d_k).swapaxes(1, 2)

        # 3. 计算注意力：对每个头并行计算注意力
        output, attn_weights = scaled_dot_product_attention(q, k, v, mask)

        # 4. 拼接头：将所有头的输出拼接起来
        output = output.swapaxes(1, 2).reshape(batch_size, seq_len, self.d_model)

        # 5. 最终线性变换：通过输出投影层
        output = np.dot(output, self.W_o)

        return output, attn_weights

#### 使用示例

```python
# 设置参数
d_model_np = 512    # 模型维度
n_heads_np = 8      # 注意力头数
seq_len_np = 10     # 序列长度
batch_size_np = 64  # 批次大小

# 创建多头注意力模块
mha_np = MultiHeadAttentionNumpy(d_model_np, n_heads_np)

# 创建随机输入数据
q_np = np.random.randn(batch_size_np, seq_len_np, d_model_np)
k_np = np.random.randn(batch_size_np, seq_len_np, d_model_np)
v_np = np.random.randn(batch_size_np, seq_len_np, d_model_np)

# 前向传播
output_np, attn_weights_np = mha_np.forward(q_np, k_np, v_np)

# 打印形状信息
print("Input shape:", q_np.shape)
print("Output shape:", output_np.shape)
print("Attention weights shape:", attn_weights_np.shape)

# 输出结果:
# Input shape: (64, 10, 512)
# Output shape: (64, 10, 512)
# Attention weights shape: (64, 8, 10, 10)
```

## 复杂度分析

假设序列长度为 $n$，输入和输出的嵌入维度为 $d$。在多头注意力中，每个头的维度 $d_k = d / h$。

### 时间复杂度: $O(n^2 \cdot d)$
我们来分解计算瓶颈：

1. **初始线性投影**：将形状为 $(n, d)$ 的 Q, K, V 输入乘以形状为 $(d, d)$ 的权重矩阵。这个操作的复杂度是 $3 \times O(n \cdot d^2)$。

2. **计算注意力分数**：这是最关键的部分。我们计算 $Q \cdot K^T$。在拆分到多头后，Q 和 K 的形状变为 $(h, n, d_k)$。
   - $Q \cdot K^T$ 的计算是 $(n, d_k)$ 与 $(d_k, n)$ 的矩阵相乘，结果是 $(n, n)$。这个操作的复杂度是 $O(n^2 \cdot d_k)$。
   - 由于有 $h$ 个头，总复杂度是 $O(h \cdot n^2 \cdot d_k)$。
   - 因为 $h \cdot d_k = d$，所以这一步的复杂度是 $O(n^2 \cdot d)$。

3. **权重乘以V**：将形状为 $(n, n)$ 的注意力权重矩阵与形状为 $(n, d_v)$ 的 V 矩阵相乘（这里 $d_v=d_k$）。
   - 每个头的复杂度是 $O(n^2 \cdot d_v)$。
   - $h$ 个头的总复杂度是 $O(h \cdot n^2 \cdot d_v) = O(n^2 \cdot d)$。

4. **最终线性投影**：将拼接后的输出（形状为 $(n, d)$）乘以最终的权重矩阵（形状为 $(d, d)$），复杂度为 $O(n \cdot d^2)$。

**总结**：
总的时间复杂度为 $O(n \cdot d^2) + O(n^2 \cdot d)$。在典型的 Transformer 应用中，序列长度 $n$ 通常会大于嵌入维度 $d$（例如 $n=512, d=768$ 或者 $n=2048, d=1024$）。因此，主要的时间复杂度瓶颈是 $O(n^2 \cdot d)$。
### 空间复杂度: $O(n^2)$

1. **主要存储**：最大的中间产物是注意力权重矩阵，其形状为 $(h, n, n)$。因此，它占用的空间是 $O(h \cdot n^2)$。即使 $h$ 很大，它也是一个常数，所以通常简化为 $O(n^2)$。

2. **其他存储**：输入的 Q, K, V 以及输出的存储都是 $O(n \cdot d)$。

3. **模型参数**：权重矩阵 $W_q, W_k, W_v, W_o$ 的大小都是 $d \times d$，总共是 $4 \cdot d^2$，这是一个与序列长度 $n$ 无关的常数。

**总结**：
空间复杂度的瓶颈在于存储注意力分数矩阵，为 $O(n^2)$。这也是为什么标准 Transformer 难以处理非常长序列的原因。

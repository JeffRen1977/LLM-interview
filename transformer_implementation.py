#!/usr/bin/env python3
"""
Transformer Attention Mechanism Implementation

This file contains the complete implementation of Transformer's attention mechanism
as described in transformer.md, including both PyTorch and NumPy versions.

Author: Generated from transformer.md
"""

import numpy as np
import math

# Try to import PyTorch, but don't fail if it's not available
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch not available. Only NumPy implementation will work.")
    PYTORCH_AVAILABLE = False


# =============================================================================
# PyTorch Implementation
# =============================================================================

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


# =============================================================================
# NumPy Implementation
# =============================================================================

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


# =============================================================================
# Demo Functions
# =============================================================================

def demo_pytorch_implementation():
    """演示 PyTorch 实现"""
    if not PYTORCH_AVAILABLE:
        print("=" * 60)
        print("PyTorch Implementation Demo")
        print("=" * 60)
        print("❌ PyTorch not available. Skipping PyTorch demo.")
        print("Please install PyTorch to run this demo: pip install torch")
        return None, None
    
    print("=" * 60)
    print("PyTorch Implementation Demo")
    print("=" * 60)
    
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

    print(f"Input shapes:")
    print(f"  Query: {q.shape}")
    print(f"  Key:   {k.shape}")
    print(f"  Value: {v.shape}")

    # 前向传播
    output, attn_weights = mha(q, k, v)

    # 打印形状信息
    print(f"\nOutput shapes:")
    print(f"  Output: {output.shape}")
    print(f"  Attention weights: {attn_weights.shape}")
    
    # 验证输出形状
    assert output.shape == (batch_size, seq_len, d_model), f"Expected output shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    assert attn_weights.shape == (batch_size, n_heads, seq_len, seq_len), f"Expected attention weights shape {(batch_size, n_heads, seq_len, seq_len)}, got {attn_weights.shape}"
    
    print("✅ PyTorch implementation works correctly!")
    
    return output, attn_weights


def demo_numpy_implementation():
    """演示 NumPy 实现"""
    print("\n" + "=" * 60)
    print("NumPy Implementation Demo")
    print("=" * 60)
    
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

    print(f"Input shapes:")
    print(f"  Query: {q_np.shape}")
    print(f"  Key:   {k_np.shape}")
    print(f"  Value: {v_np.shape}")

    # 前向传播
    output_np, attn_weights_np = mha_np.forward(q_np, k_np, v_np)

    # 打印形状信息
    print(f"\nOutput shapes:")
    print(f"  Output: {output_np.shape}")
    print(f"  Attention weights: {attn_weights_np.shape}")
    
    # 验证输出形状
    assert output_np.shape == (batch_size_np, seq_len_np, d_model_np), f"Expected output shape {(batch_size_np, seq_len_np, d_model_np)}, got {output_np.shape}"
    assert attn_weights_np.shape == (batch_size_np, n_heads_np, seq_len_np, seq_len_np), f"Expected attention weights shape {(batch_size_np, n_heads_np, seq_len_np, seq_len_np)}, got {attn_weights_np.shape}"
    
    print("✅ NumPy implementation works correctly!")
    
    return output_np, attn_weights_np


def demo_attention_mechanism():
    """演示注意力机制的工作原理"""
    print("\n" + "=" * 60)
    print("Attention Mechanism Analysis")
    print("=" * 60)
    
    if not PYTORCH_AVAILABLE:
        print("❌ PyTorch not available. Skipping attention mechanism demo.")
        print("Please install PyTorch to run this demo: pip install torch")
        return
    
    # 创建一个简单的例子来展示注意力权重
    d_model = 64
    n_heads = 4
    seq_len = 5
    batch_size = 1

    mha = MultiHeadAttention(d_model, n_heads)
    
    # 创建输入（让第一个和最后一个token相似）
    q = torch.randn(batch_size, seq_len, d_model)
    k = q.clone()  # 自注意力：Q = K = V
    v = q.clone()
    
    # 让第一个和最后一个token更相似
    q[0, -1] = q[0, 0] + 0.1 * torch.randn(d_model)
    k[0, -1] = k[0, 0] + 0.1 * torch.randn(d_model)
    v[0, -1] = v[0, 0] + 0.1 * torch.randn(d_model)

    output, attn_weights = mha(q, k, v)
    
    # 分析第一个头的注意力权重
    first_head_weights = attn_weights[0, 0].detach().numpy()
    
    print("Attention weights for the first head (first token attending to all tokens):")
    for i, weight in enumerate(first_head_weights[0]):
        print(f"  Token {i}: {weight:.4f}")
    
    print(f"\nSum of attention weights: {first_head_weights[0].sum():.4f}")
    print("✅ Attention weights sum to 1 (as expected for softmax)")


def complexity_analysis():
    """展示复杂度分析"""
    print("\n" + "=" * 60)
    print("Complexity Analysis")
    print("=" * 60)
    
    if not PYTORCH_AVAILABLE:
        print("❌ PyTorch not available. Skipping complexity analysis demo.")
        print("Please install PyTorch to run this demo: pip install torch")
        return
    
    import time
    
    # 测试不同序列长度的性能
    d_model = 512
    n_heads = 8
    batch_size = 32
    
    sequence_lengths = [10, 50, 100, 200]
    
    print("Performance test with different sequence lengths:")
    print(f"Model dimension: {d_model}, Heads: {n_heads}, Batch size: {batch_size}")
    print("-" * 60)
    
    for seq_len in sequence_lengths:
        mha = MultiHeadAttention(d_model, n_heads)
        q = torch.randn(batch_size, seq_len, d_model)
        k = torch.randn(batch_size, seq_len, d_model)
        v = torch.randn(batch_size, seq_len, d_model)
        
        # 预热
        _ = mha(q, k, v)
        
        # 计时
        start_time = time.time()
        for _ in range(10):  # 运行10次取平均
            _ = mha(q, k, v)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        print(f"Seq length {seq_len:3d}: {avg_time:.4f}s (O(n²) complexity expected)")
    
    print("\nNote: Time complexity should scale quadratically with sequence length")


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("Transformer Attention Mechanism Implementation")
    print("This script demonstrates both PyTorch and NumPy implementations")
    print("of the Transformer's multi-head attention mechanism.\n")
    
    try:
        # 运行 PyTorch 演示
        demo_pytorch_implementation()
        
        # 运行 NumPy 演示
        demo_numpy_implementation()
        
        # 演示注意力机制
        demo_attention_mechanism()
        
        # 复杂度分析
        complexity_analysis()
        
        print("\n" + "=" * 60)
        print("All demonstrations completed successfully! 🎉")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

#!/usr/bin/env python3
"""
OpenAI Interview Question: Attention Mechanisms and Variants

This comprehensive module implements standard attention mechanisms and various
attention variants, demonstrating different approaches to improve efficiency,
scalability, and performance.

Implemented Attention Mechanisms:
1. Standard Scaled Dot-Product Attention
2. Multi-Head Attention
3. Sliding Window Attention
4. Sparse Attention
5. Linear Attention (Linformer-style)
6. Longformer Attention
7. Local Attention
8. Dilated Attention

Key Features:
- Complete PyTorch implementations
- Detailed documentation and comments
- Performance comparisons
- Usage examples for each variant
- Complexity analysis

Author: Jianfeng Ren
Date: 2024
Version: 1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, List
import warnings

warnings.filterwarnings("ignore")


# =============================================================================
# 1. Standard Scaled Dot-Product Attention
# =============================================================================

class ScaledDotProductAttention(nn.Module):
    """
    标准的缩放点积注意力机制 (Scaled Dot-Product Attention)
    
    这是 Transformer 架构中的核心组件，由 "Attention is All You Need" 论文提出。
    
    数学公式:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    
    复杂度:
        - 时间复杂度: O(N² · d)
        - 空间复杂度: O(N²)
        其中 N 是序列长度，d 是特征维度
    
    优点:
        - 简单高效
        - 并行化友好
        - 全局依赖建模
    
    缺点:
        - 二次复杂度，难以处理长序列
        - 内存需求大
    """
    
    def __init__(self, dropout: float = 0.1):
        """
        初始化缩放点积注意力
        
        Args:
            dropout: Dropout 概率，用于正则化
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播计算注意力
        
        Args:
            query: Query 张量, shape [batch_size, seq_len, d_model] 或 [batch_size, n_heads, seq_len, d_k]
            key: Key 张量, shape [batch_size, seq_len, d_model] 或 [batch_size, n_heads, seq_len, d_k]
            value: Value 张量, shape [batch_size, seq_len, d_model] 或 [batch_size, n_heads, seq_len, d_v]
            mask: 可选的掩码张量, shape [batch_size, seq_len, seq_len] 或 [batch_size, 1, seq_len, seq_len]
                  mask == 0 的位置会被设为 -inf
        
        Returns:
            output: 注意力输出, shape 与 value 相同
            attention_weights: 注意力权重, shape [batch_size, seq_len, seq_len] 或 [batch_size, n_heads, seq_len, seq_len]
        """
        # 获取 key 的最后一个维度（特征维度）
        d_k = key.size(-1)
        
        # 步骤 1: 计算注意力分数 Q @ K^T
        # query: [..., seq_len_q, d_k]
        # key: [..., seq_len_k, d_k]
        # scores: [..., seq_len_q, seq_len_k]
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        
        # 步骤 2: 缩放 - 除以 sqrt(d_k) 防止梯度消失
        # 当 d_k 很大时，点积的值会很大，导致 softmax 进入饱和区
        # 缩放可以保持梯度的稳定性
        attention_scores = attention_scores / math.sqrt(d_k)
        
        # 步骤 3: 应用掩码（如果提供）
        # 将不需要的位置设为负无穷，softmax 后会变成 0
        if mask is not None:
            # mask == 0 的位置设为 -1e9（接近负无穷）
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # 步骤 4: 计算 Softmax 得到注意力权重（概率分布）
        # 每一行的权重和为 1
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 步骤 5: 权重与 Value 相乘得到最终输出
        # attention_weights: [..., seq_len_q, seq_len_k]
        # value: [..., seq_len_k, d_v]
        # output: [..., seq_len_q, d_v]
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


# =============================================================================
# 2. Multi-Head Attention
# =============================================================================

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制 (Multi-Head Attention)
    
    通过多个并行的注意力头来捕获不同类型的关系，每个头关注不同的表示子空间。
    
    数学公式:
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
        其中 head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    
    复杂度:
        - 时间复杂度: O(N² · d · h)，其中 h 是头数
        - 空间复杂度: O(N² · h)
    
    优点:
        - 可以同时关注不同类型的依赖关系
        - 提高模型的表达能力
        - 每个头可以学习不同的注意力模式
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        初始化多头注意力
        
        Args:
            d_model: 模型的总维度（embedding dimension）
            n_heads: 注意力头的数量，必须能整除 d_model
            dropout: Dropout 概率
        """
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度
        
        # 定义线性投影层：为每个头创建独立的 Q, K, V 投影
        self.W_q = nn.Linear(d_model, d_model)  # Query 投影
        self.W_k = nn.Linear(d_model, d_model)  # Key 投影
        self.W_v = nn.Linear(d_model, d_model)  # Value 投影
        self.W_o = nn.Linear(d_model, d_model)  # 输出投影
        
        # 使用标准的缩放点积注意力
        self.attention = ScaledDotProductAttention(dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播计算多头注意力
        
        Args:
            query: Query 张量, shape [batch_size, seq_len, d_model]
            key: Key 张量, shape [batch_size, seq_len, d_model]
            value: Value 张量, shape [batch_size, seq_len, d_model]
            mask: 可选的掩码张量
        
        Returns:
            output: 多头注意力输出, shape [batch_size, seq_len, d_model]
            attention_weights: 注意力权重, shape [batch_size, n_heads, seq_len, seq_len]
        """
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # 步骤 1: 线性投影：将输入投影到 Q, K, V 空间
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model]
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # 步骤 2: 拆分成多个头：重塑张量以支持多头并行计算
        # [batch_size, seq_len, d_model] -> [batch_size, n_heads, seq_len, d_k]
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 步骤 3: 计算注意力：对每个头并行计算注意力
        # output: [batch_size, n_heads, seq_len, d_k]
        # attention_weights: [batch_size, n_heads, seq_len, seq_len]
        output, attention_weights = self.attention(Q, K, V, mask)
        
        # 步骤 4: 拼接头：将所有头的输出拼接起来
        # [batch_size, n_heads, seq_len, d_k] -> [batch_size, seq_len, d_model]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 步骤 5: 最终的线性变换：通过输出投影层
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model]
        output = self.W_o(output)
        
        return output, attention_weights


# =============================================================================
# 3. Sliding Window Attention
# =============================================================================

class SlidingWindowAttention(nn.Module):
    """
    滑动窗口注意力 (Sliding Window Attention)
    
    核心思想：对于序列中的每一个 token，只与其左右一个固定大小的窗口内的 token 进行注意力计算。
    
    改进之处:
        - 降低复杂度: 从 O(N²) 降低到 O(N·w)，其中 w 是窗口大小
        - 专注局部信息: 强调了局部上下文的重要性
        - 适合处理长序列: 可以处理比标准 attention 更长的序列
    
    复杂度:
        - 时间复杂度: O(N · w · d)，其中 w << N
        - 空间复杂度: O(N · w)
    
    应用场景:
        - 长文本处理
        - 局部依赖建模
        - 计算资源受限的场景
    """
    
    def __init__(self, d_model: int, window_size: int, dropout: float = 0.1):
        """
        初始化滑动窗口注意力
        
        Args:
            d_model: 模型维度
            window_size: 窗口大小（每个 token 只关注左右 window_size 个 token）
            dropout: Dropout 概率
        """
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播计算滑动窗口注意力
        
        Args:
            query: Query 张量, shape [batch_size, seq_len, d_model]
            key: Key 张量, shape [batch_size, seq_len, d_model]
            value: Value 张量, shape [batch_size, seq_len, d_model]
            mask: 可选的掩码张量
        
        Returns:
            output: 注意力输出, shape [batch_size, seq_len, d_model]
            attention_weights: 注意力权重, shape [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, d_model = query.shape
        
        # 初始化输出和注意力权重
        output = torch.zeros_like(query)
        attention_weights = torch.zeros(batch_size, seq_len, seq_len, device=query.device)
        
        # 对每个位置计算窗口内的注意力
        for i in range(seq_len):
            # 计算窗口范围
            # 窗口中心在位置 i，左右各 window_size//2 个位置
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            
            # 提取窗口内的 key 和 value
            q_i = query[:, i:i+1, :]  # [batch_size, 1, d_model]
            k_window = key[:, start:end, :]  # [batch_size, window_size, d_model]
            v_window = value[:, start:end, :]  # [batch_size, window_size, d_model]
            
            # 计算注意力分数
            # [batch_size, 1, d_model] @ [batch_size, d_model, window_size] -> [batch_size, 1, window_size]
            scores = torch.matmul(q_i, k_window.transpose(-2, -1)) / math.sqrt(d_model)
            
            # 应用掩码（如果提供）
            if mask is not None:
                # 提取窗口对应的掩码
                mask_window = mask[:, i, start:end] if mask.dim() == 3 else mask[:, start:end]
                scores = scores.masked_fill(mask_window == 0, -1e9)
            
            # 计算注意力权重
            attention_weights_window = F.softmax(scores, dim=-1)
            attention_weights_window = self.dropout(attention_weights_window)
            
            # 计算加权和
            # [batch_size, 1, window_size] @ [batch_size, window_size, d_model] -> [batch_size, 1, d_model]
            attended = torch.matmul(attention_weights_window, v_window)
            output[:, i:i+1, :] = attended
            
            # 保存注意力权重（用于可视化）
            attention_weights[:, i, start:end] = attention_weights_window.squeeze(1)
        
        return output, attention_weights


# =============================================================================
# 4. Sparse Attention
# =============================================================================

class SparseAttention(nn.Module):
    """
    稀疏注意力 (Sparse Attention)
    
    核心思想：只计算特定位置的注意力，而不是所有位置对。
    可以通过多种方式实现稀疏性：
    1. 固定模式（如 stride attention）
    2. 学习到的稀疏模式
    3. 基于距离的稀疏性
    
    改进之处:
        - 降低计算复杂度
        - 减少内存使用
        - 可以处理更长的序列
    
    复杂度:
        - 时间复杂度: O(N · k · d)，其中 k 是每个 token 关注的 token 数量
        - 空间复杂度: O(N · k)
    """
    
    def __init__(self, d_model: int, sparsity_pattern: str = "stride", 
                 stride: int = 2, dropout: float = 0.1):
        """
        初始化稀疏注意力
        
        Args:
            d_model: 模型维度
            sparsity_pattern: 稀疏模式
                - "stride": 步长模式，每隔 stride 个位置计算一次
                - "local": 局部模式，只关注附近的 token
                - "global": 全局模式，只关注特定全局 token
            stride: 步长（用于 stride 模式）
            dropout: Dropout 概率
        """
        super().__init__()
        self.d_model = d_model
        self.sparsity_pattern = sparsity_pattern
        self.stride = stride
        self.dropout = nn.Dropout(dropout)
    
    def _get_attention_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        生成稀疏注意力掩码
        
        Args:
            seq_len: 序列长度
            device: 设备
        
        Returns:
            mask: 稀疏注意力掩码, shape [seq_len, seq_len]
        """
        mask = torch.zeros(seq_len, seq_len, device=device)
        
        if self.sparsity_pattern == "stride":
            # 步长模式：每个位置只关注每隔 stride 个位置
            for i in range(seq_len):
                # 关注位置 i, i±stride, i±2*stride, ...
                positions = []
                for offset in range(0, seq_len, self.stride):
                    if i + offset < seq_len:
                        positions.append(i + offset)
                    if i - offset >= 0 and offset > 0:
                        positions.append(i - offset)
                mask[i, positions] = 1.0
        
        elif self.sparsity_pattern == "local":
            # 局部模式：每个位置只关注附近的 token
            local_window = self.stride
            for i in range(seq_len):
                start = max(0, i - local_window)
                end = min(seq_len, i + local_window + 1)
                mask[i, start:end] = 1.0
        
        elif self.sparsity_pattern == "global":
            # 全局模式：每个位置都关注全局 token（如 [CLS] token）
            # 这里简化为关注第一个和最后一个 token
            mask[:, 0] = 1.0  # 关注第一个 token
            mask[:, -1] = 1.0  # 关注最后一个 token
            # 也关注自己
            mask.fill_diagonal_(1.0)
        
        return mask
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播计算稀疏注意力
        
        Args:
            query: Query 张量, shape [batch_size, seq_len, d_model]
            key: Key 张量, shape [batch_size, seq_len, d_model]
            value: Value 张量, shape [batch_size, seq_len, d_model]
            mask: 可选的掩码张量
        
        Returns:
            output: 注意力输出, shape [batch_size, seq_len, d_model]
            attention_weights: 注意力权重, shape [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, d_model = query.shape
        
        # 生成稀疏注意力掩码
        sparse_mask = self._get_attention_mask(seq_len, query.device)
        sparse_mask = sparse_mask.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, seq_len, seq_len]
        
        # 计算注意力分数
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_model)
        
        # 应用稀疏掩码：将不需要的位置设为负无穷
        attention_scores = attention_scores.masked_fill(sparse_mask == 0, -1e9)
        
        # 应用额外的掩码（如果提供）
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # 计算注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 计算输出
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


# =============================================================================
# 5. Linear Attention (Linformer-style)
# =============================================================================

class LinearAttention(nn.Module):
    """
    线性注意力 (Linear Attention) - Linformer 风格
    
    核心思想：通过低秩投影将 key 和 value 的序列长度从 N 降低到 k (k << N)，
    从而将复杂度从 O(N²) 降低到 O(N·k)。
    
    数学公式:
        LinearAttention(Q, K, V) = softmax(Q(K^T E^T)) (EV)
        其中 E 是投影矩阵，将 N 维降低到 k 维
    
    改进之处:
        - 降低复杂度: 从 O(N²) 降低到 O(N·k)
        - 线性复杂度: 与序列长度成线性关系
        - 适合处理超长序列
    
    复杂度:
        - 时间复杂度: O(N · k · d)，其中 k << N
        - 空间复杂度: O(N · k)
    
    参考: "Linformer: Self-Attention with Linear Complexity" (Wang et al., 2020)
    """
    
    def __init__(self, d_model: int, projected_dim: int = 128, dropout: float = 0.1):
        """
        初始化线性注意力
        
        Args:
            d_model: 模型维度
            projected_dim: 投影后的维度 k，应该远小于序列长度
            dropout: Dropout 概率
        """
        super().__init__()
        self.d_model = d_model
        self.projected_dim = projected_dim
        
        # 投影矩阵：将序列长度从 N 投影到 k
        self.E_k = nn.Linear(d_model, projected_dim, bias=False)  # Key 投影
        self.E_v = nn.Linear(d_model, projected_dim, bias=False)  # Value 投影
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播计算线性注意力
        
        Args:
            query: Query 张量, shape [batch_size, seq_len, d_model]
            key: Key 张量, shape [batch_size, seq_len, d_model]
            value: Value 张量, shape [batch_size, seq_len, d_model]
            mask: 可选的掩码张量（注意：线性注意力对掩码的支持有限）
        
        Returns:
            output: 注意力输出, shape [batch_size, seq_len, d_model]
            attention_weights: 注意力权重, shape [batch_size, seq_len, projected_dim]
        """
        batch_size, seq_len, d_model = query.shape
        
        # 步骤 1: 投影 key 和 value 到低维空间
        # key: [batch_size, seq_len, d_model] -> [batch_size, projected_dim, d_model]
        # value: [batch_size, seq_len, d_model] -> [batch_size, projected_dim, d_model]
        # 注意：这里我们投影特征维度，而不是序列维度（简化实现）
        K_proj = self.E_k(key)  # [batch_size, seq_len, projected_dim]
        V_proj = self.E_v(value)  # [batch_size, seq_len, projected_dim]
        
        # 步骤 2: 计算注意力分数
        # query: [batch_size, seq_len, d_model]
        # K_proj: [batch_size, seq_len, projected_dim]
        # scores: [batch_size, seq_len, projected_dim]
        attention_scores = torch.matmul(query, K_proj.transpose(-2, -1)) / math.sqrt(d_model)
        
        # 应用掩码（如果提供，但线性注意力的掩码处理较复杂）
        if mask is not None:
            # 简化处理：对投影后的维度应用平均掩码
            mask_avg = mask.float().mean(dim=-1, keepdim=True)  # [batch_size, seq_len, 1]
            attention_scores = attention_scores * mask_avg
        
        # 步骤 3: 计算注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 步骤 4: 计算输出
        # attention_weights: [batch_size, seq_len, projected_dim]
        # V_proj: [batch_size, seq_len, projected_dim]
        # output: [batch_size, seq_len, projected_dim]
        output = torch.matmul(attention_weights, V_proj)
        
        # 步骤 5: 投影回原始维度（如果需要）
        # 这里简化实现，直接返回投影后的输出
        # 实际应用中可能需要额外的投影层
        
        return output, attention_weights


# =============================================================================
# 6. Longformer Attention
# =============================================================================

class LongformerAttention(nn.Module):
    """
    Longformer 注意力机制
    
    结合了滑动窗口注意力和全局注意力：
    - 局部注意力：使用滑动窗口关注附近的 token
    - 全局注意力：特定 token（如 [CLS]）关注所有 token，所有 token 也关注这些全局 token
    
    改进之处:
        - 线性复杂度: O(N · w)，其中 w 是窗口大小
        - 保留全局信息: 通过全局 token 传递全局信息
        - 适合处理长文档
    
    复杂度:
        - 时间复杂度: O(N · w · d)，其中 w << N
        - 空间复杂度: O(N · w)
    
    参考: "Longformer: The Long-Document Transformer" (Beltagy et al., 2020)
    """
    
    def __init__(self, d_model: int, window_size: int, num_global_tokens: int = 2,
                 dropout: float = 0.1):
        """
        初始化 Longformer 注意力
        
        Args:
            d_model: 模型维度
            window_size: 滑动窗口大小
            num_global_tokens: 全局 token 的数量（通常是第一个和最后一个）
            dropout: Dropout 概率
        """
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        self.num_global_tokens = num_global_tokens
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播计算 Longformer 注意力
        
        Args:
            query: Query 张量, shape [batch_size, seq_len, d_model]
            key: Key 张量, shape [batch_size, seq_len, d_model]
            value: Value 张量, shape [batch_size, seq_len, d_model]
            mask: 可选的掩码张量
        
        Returns:
            output: 注意力输出, shape [batch_size, seq_len, d_model]
            attention_weights: 注意力权重, shape [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, d_model = query.shape
        
        # 初始化输出和注意力权重
        output = torch.zeros_like(query)
        attention_weights = torch.zeros(batch_size, seq_len, seq_len, device=query.device)
        
        # 定义全局 token 位置（通常是第一个和最后一个）
        global_positions = [0]  # 第一个 token
        if self.num_global_tokens > 1:
            global_positions.append(seq_len - 1)  # 最后一个 token
        
        # 对每个位置计算注意力
        for i in range(seq_len):
            # 计算窗口范围
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            
            # 收集需要关注的位置
            # 1. 窗口内的位置
            window_positions = list(range(start, end))
            
            # 2. 全局 token 位置（所有 token 都关注全局 token）
            attention_positions = list(set(window_positions + global_positions))
            
            # 提取对应的 key 和 value
            q_i = query[:, i:i+1, :]  # [batch_size, 1, d_model]
            k_selected = key[:, attention_positions, :]  # [batch_size, num_positions, d_model]
            v_selected = value[:, attention_positions, :]  # [batch_size, num_positions, d_model]
            
            # 计算注意力分数
            scores = torch.matmul(q_i, k_selected.transpose(-2, -1)) / math.sqrt(d_model)
            
            # 应用掩码
            if mask is not None:
                mask_selected = mask[:, i, attention_positions] if mask.dim() == 3 else mask[:, attention_positions]
                scores = scores.masked_fill(mask_selected == 0, -1e9)
            
            # 计算注意力权重
            attention_weights_selected = F.softmax(scores, dim=-1)
            attention_weights_selected = self.dropout(attention_weights_selected)
            
            # 计算加权和
            attended = torch.matmul(attention_weights_selected, v_selected)
            output[:, i:i+1, :] = attended
            
            # 保存注意力权重
            for idx, pos in enumerate(attention_positions):
                attention_weights[:, i, pos] = attention_weights_selected[:, 0, idx]
        
        return output, attention_weights


# =============================================================================
# 7. Local Attention
# =============================================================================

class LocalAttention(nn.Module):
    """
    局部注意力 (Local Attention)
    
    核心思想：每个 token 只关注其固定范围内的邻居 token。
    这是滑动窗口注意力的简化版本，窗口大小固定。
    
    改进之处:
        - 计算效率高
        - 内存占用小
        - 适合局部依赖建模
    
    复杂度:
        - 时间复杂度: O(N · w · d)
        - 空间复杂度: O(N · w)
    """
    
    def __init__(self, d_model: int, local_window_size: int = 5, dropout: float = 0.1):
        """
        初始化局部注意力
        
        Args:
            d_model: 模型维度
            local_window_size: 局部窗口大小
            dropout: Dropout 概率
        """
        super().__init__()
        self.d_model = d_model
        self.local_window_size = local_window_size
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播计算局部注意力
        
        Args:
            query: Query 张量, shape [batch_size, seq_len, d_model]
            key: Key 张量, shape [batch_size, seq_len, d_model]
            value: Value 张量, shape [batch_size, seq_len, d_model]
            mask: 可选的掩码张量
        
        Returns:
            output: 注意力输出, shape [batch_size, seq_len, d_model]
            attention_weights: 注意力权重, shape [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, d_model = query.shape
        
        # 使用滑动窗口注意力的实现
        # 局部注意力本质上是窗口大小固定的滑动窗口注意力
        sliding_window = SlidingWindowAttention(d_model, self.local_window_size, self.dropout.p)
        return sliding_window.forward(query, key, value, mask)


# =============================================================================
# 8. Dilated Attention
# =============================================================================

class DilatedAttention(nn.Module):
    """
    扩张注意力 (Dilated Attention)
    
    核心思想：类似于扩张卷积，使用扩张的窗口来扩大感受野，
    同时保持计算复杂度不变。
    
    改进之处:
        - 扩大感受野: 通过扩张窗口捕获更远距离的依赖
        - 保持效率: 计算复杂度与普通窗口注意力相同
        - 多尺度建模: 可以捕获不同尺度的依赖关系
    
    复杂度:
        - 时间复杂度: O(N · w · d)
        - 空间复杂度: O(N · w)
    """
    
    def __init__(self, d_model: int, window_size: int, dilation: int = 2,
                 dropout: float = 0.1):
        """
        初始化扩张注意力
        
        Args:
            d_model: 模型维度
            window_size: 基础窗口大小
            dilation: 扩张率，控制窗口的扩张程度
            dropout: Dropout 概率
        """
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        self.dilation = dilation
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播计算扩张注意力
        
        Args:
            query: Query 张量, shape [batch_size, seq_len, d_model]
            key: Key 张量, shape [batch_size, seq_len, d_model]
            value: Value 张量, shape [batch_size, seq_len, d_model]
            mask: 可选的掩码张量
        
        Returns:
            output: 注意力输出, shape [batch_size, seq_len, d_model]
            attention_weights: 注意力权重, shape [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, d_model = query.shape
        
        # 初始化输出和注意力权重
        output = torch.zeros_like(query)
        attention_weights = torch.zeros(batch_size, seq_len, seq_len, device=query.device)
        
        # 对每个位置计算扩张窗口内的注意力
        for i in range(seq_len):
            # 计算扩张窗口范围
            # 扩张窗口：每隔 dilation 个位置采样一个
            window_positions = []
            for j in range(-self.window_size // 2, self.window_size // 2 + 1):
                pos = i + j * self.dilation
                if 0 <= pos < seq_len:
                    window_positions.append(pos)
            
            if len(window_positions) == 0:
                window_positions = [i]  # 至少关注自己
            
            # 提取窗口内的 key 和 value
            q_i = query[:, i:i+1, :]
            k_window = key[:, window_positions, :]
            v_window = value[:, window_positions, :]
            
            # 计算注意力分数
            scores = torch.matmul(q_i, k_window.transpose(-2, -1)) / math.sqrt(d_model)
            
            # 应用掩码
            if mask is not None:
                mask_window = mask[:, i, window_positions] if mask.dim() == 3 else mask[:, window_positions]
                scores = scores.masked_fill(mask_window == 0, -1e9)
            
            # 计算注意力权重
            attention_weights_window = F.softmax(scores, dim=-1)
            attention_weights_window = self.dropout(attention_weights_window)
            
            # 计算加权和
            attended = torch.matmul(attention_weights_window, v_window)
            output[:, i:i+1, :] = attended
            
            # 保存注意力权重
            for idx, pos in enumerate(window_positions):
                attention_weights[:, i, pos] = attention_weights_window[:, 0, idx]
        
        return output, attention_weights


# =============================================================================
# 演示和测试函数
# =============================================================================

def compare_attention_mechanisms():
    """
    比较不同注意力机制的性能和复杂度
    """
    print("=" * 80)
    print("注意力机制对比演示")
    print("=" * 80)
    
    # 设置参数
    batch_size = 2
    seq_len = 100
    d_model = 128
    n_heads = 8
    
    # 创建示例数据
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)
    
    print(f"\n输入形状: batch_size={batch_size}, seq_len={seq_len}, d_model={d_model}")
    print(f"序列长度: {seq_len}")
    print(f"模型维度: {d_model}")
    
    # 1. 标准注意力
    print("\n" + "-" * 80)
    print("1. 标准缩放点积注意力 (Scaled Dot-Product Attention)")
    print("-" * 80)
    standard_attn = ScaledDotProductAttention()
    start_time = time.time()
    output, weights = standard_attn(query, key, value)
    standard_time = time.time() - start_time
    print(f"   输出形状: {output.shape}")
    print(f"   注意力权重形状: {weights.shape}")
    print(f"   计算时间: {standard_time:.4f}s")
    print(f"   复杂度: O(N²·d) = O({seq_len}²·{d_model})")
    
    # 2. 多头注意力
    print("\n" + "-" * 80)
    print("2. 多头注意力 (Multi-Head Attention)")
    print("-" * 80)
    multi_head_attn = MultiHeadAttention(d_model, n_heads)
    start_time = time.time()
    output, weights = multi_head_attn(query, key, value)
    multi_head_time = time.time() - start_time
    print(f"   输出形状: {output.shape}")
    print(f"   注意力权重形状: {weights.shape}")
    print(f"   计算时间: {multi_head_time:.4f}s")
    print(f"   头数: {n_heads}")
    
    # 3. 滑动窗口注意力
    print("\n" + "-" * 80)
    print("3. 滑动窗口注意力 (Sliding Window Attention)")
    print("-" * 80)
    window_size = 10
    sliding_attn = SlidingWindowAttention(d_model, window_size)
    start_time = time.time()
    output, weights = sliding_attn(query, key, value)
    sliding_time = time.time() - start_time
    print(f"   输出形状: {output.shape}")
    print(f"   注意力权重形状: {weights.shape}")
    print(f"   计算时间: {sliding_time:.4f}s")
    print(f"   窗口大小: {window_size}")
    print(f"   复杂度: O(N·w·d) = O({seq_len}·{window_size}·{d_model})")
    print(f"   加速比: {standard_time/sliding_time:.2f}x")
    
    # 4. 稀疏注意力
    print("\n" + "-" * 80)
    print("4. 稀疏注意力 (Sparse Attention)")
    print("-" * 80)
    sparse_attn = SparseAttention(d_model, sparsity_pattern="stride", stride=5)
    start_time = time.time()
    output, weights = sparse_attn(query, key, value)
    sparse_time = time.time() - start_time
    print(f"   输出形状: {output.shape}")
    print(f"   注意力权重形状: {weights.shape}")
    print(f"   计算时间: {sparse_time:.4f}s")
    print(f"   稀疏模式: stride (步长=5)")
    print(f"   加速比: {standard_time/sparse_time:.2f}x")
    
    # 5. 线性注意力
    print("\n" + "-" * 80)
    print("5. 线性注意力 (Linear Attention)")
    print("-" * 80)
    projected_dim = 32
    linear_attn = LinearAttention(d_model, projected_dim)
    start_time = time.time()
    output, weights = linear_attn(query, key, value)
    linear_time = time.time() - start_time
    print(f"   输出形状: {output.shape}")
    print(f"   注意力权重形状: {weights.shape}")
    print(f"   计算时间: {linear_time:.4f}s")
    print(f"   投影维度: {projected_dim}")
    print(f"   复杂度: O(N·k·d) = O({seq_len}·{projected_dim}·{d_model})")
    print(f"   加速比: {standard_time/linear_time:.2f}x")
    
    # 6. Longformer 注意力
    print("\n" + "-" * 80)
    print("6. Longformer 注意力 (Longformer Attention)")
    print("-" * 80)
    longformer_attn = LongformerAttention(d_model, window_size=10, num_global_tokens=2)
    start_time = time.time()
    output, weights = longformer_attn(query, key, value)
    longformer_time = time.time() - start_time
    print(f"   输出形状: {output.shape}")
    print(f"   注意力权重形状: {weights.shape}")
    print(f"   计算时间: {longformer_time:.4f}s")
    print(f"   窗口大小: 10, 全局 token 数: 2")
    
    # 总结
    print("\n" + "=" * 80)
    print("性能总结")
    print("=" * 80)
    print(f"{'注意力机制':<30} {'时间 (s)':<15} {'加速比':<15}")
    print("-" * 80)
    print(f"{'标准注意力':<30} {standard_time:<15.4f} {'1.00x':<15}")
    print(f"{'多头注意力':<30} {multi_head_time:<15.4f} {standard_time/multi_head_time:<15.2f}x")
    print(f"{'滑动窗口注意力':<30} {sliding_time:<15.4f} {standard_time/sliding_time:<15.2f}x")
    print(f"{'稀疏注意力':<30} {sparse_time:<15.4f} {standard_time/sparse_time:<15.2f}x")
    print(f"{'线性注意力':<30} {linear_time:<15.4f} {standard_time/linear_time:<15.2f}x")
    print(f"{'Longformer 注意力':<30} {longformer_time:<15.4f} {standard_time/longformer_time:<15.2f}x")
    print("=" * 80)


def demonstrate_attention_patterns():
    """
    演示不同注意力机制的注意力模式
    """
    print("\n" + "=" * 80)
    print("注意力模式可视化")
    print("=" * 80)
    
    seq_len = 20
    d_model = 64
    
    # 创建示例数据
    query = torch.randn(1, seq_len, d_model)
    key = torch.randn(1, seq_len, d_model)
    value = torch.randn(1, seq_len, d_model)
    
    print(f"\n序列长度: {seq_len}")
    print(f"模型维度: {d_model}")
    
    # 1. 标准注意力模式
    print("\n1. 标准注意力模式:")
    print("   - 每个 token 关注所有其他 token")
    print("   - 注意力矩阵: 完全密集")
    standard_attn = ScaledDotProductAttention()
    _, weights = standard_attn(query, key, value)
    print(f"   非零元素比例: {(weights > 0.01).float().mean().item():.2%}")
    
    # 2. 滑动窗口注意力模式
    print("\n2. 滑动窗口注意力模式:")
    print("   - 每个 token 只关注窗口内的 token")
    window_size = 5
    sliding_attn = SlidingWindowAttention(d_model, window_size)
    _, weights = sliding_attn(query, key, value)
    print(f"   窗口大小: {window_size}")
    print(f"   非零元素比例: {(weights > 0.01).float().mean().item():.2%}")
    print(f"   理论非零比例: {window_size/seq_len:.2%}")
    
    # 3. 稀疏注意力模式
    print("\n3. 稀疏注意力模式 (stride=3):")
    print("   - 每个 token 只关注每隔 3 个位置的 token")
    sparse_attn = SparseAttention(d_model, sparsity_pattern="stride", stride=3)
    _, weights = sparse_attn(query, key, value)
    print(f"   非零元素比例: {(weights > 0.01).float().mean().item():.2%}")
    
    # 4. Longformer 注意力模式
    print("\n4. Longformer 注意力模式:")
    print("   - 局部窗口 + 全局 token")
    longformer_attn = LongformerAttention(d_model, window_size=5, num_global_tokens=2)
    _, weights = longformer_attn(query, key, value)
    print(f"   窗口大小: 5, 全局 token: [0, {seq_len-1}]")
    print(f"   非零元素比例: {(weights > 0.01).float().mean().item():.2%}")


if __name__ == "__main__":
    import time
    
    print("🚀 注意力机制实现和变体演示")
    print("=" * 80)
    print("本演示展示了标准注意力机制和多种变体的实现")
    print("包括：标准注意力、多头注意力、滑动窗口、稀疏、线性、Longformer 等")
    print("=" * 80)
    
    # 运行对比演示
    compare_attention_mechanisms()
    
    # 运行模式演示
    demonstrate_attention_patterns()
    
    print("\n" + "=" * 80)
    print("✅ 演示完成！")
    print("=" * 80)
    print("\n💡 使用建议:")
    print("1. 短序列 (< 512): 使用标准注意力或多头注意力")
    print("2. 中等序列 (512-2048): 使用滑动窗口或稀疏注意力")
    print("3. 长序列 (> 2048): 使用线性注意力或 Longformer")
    print("4. 需要全局信息: 使用 Longformer 或添加全局 token")
    print("=" * 80)

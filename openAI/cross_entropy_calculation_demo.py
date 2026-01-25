#!/usr/bin/env python3
"""
详细演示 F.cross_entropy 的计算过程
展示每一步的数学计算和中间结果
"""

import torch
import torch.nn.functional as F
import numpy as np

def manual_cross_entropy_step_by_step(inputs, targets):
    """
    手动逐步计算交叉熵损失，展示每一步的中间结果
    """
    print("=" * 80)
    print("交叉熵损失计算详解")
    print("=" * 80)
    
    print(f"\n1. 输入数据:")
    print(f"   inputs (logits):\n   {inputs}")
    print(f"   shape: {inputs.shape}")
    print(f"   targets (真实类别索引): {targets}")
    print(f"   shape: {targets.shape}")
    
    # 步骤 1: 计算 log_softmax
    print(f"\n2. 计算 log_softmax (数值稳定的方式):")
    print(f"   公式: log_softmax(x_i) = x_i - log(Σⱼ exp(x_j))")
    
    # 手动计算 log_softmax
    max_vals = inputs.max(dim=1, keepdim=True)[0]
    print(f"\n   2.1 找到每行的最大值 (用于数值稳定性):")
    print(f"   max_vals = {max_vals.squeeze()}")
    
    inputs_shifted = inputs - max_vals
    print(f"\n   2.2 减去最大值 (x - max):")
    print(f"   inputs_shifted =\n   {inputs_shifted}")
    
    exp_shifted = torch.exp(inputs_shifted)
    print(f"\n   2.3 计算 exp(x - max):")
    print(f"   exp_shifted =\n   {exp_shifted}")
    
    sum_exp = exp_shifted.sum(dim=1, keepdim=True)
    print(f"\n   2.4 计算每行的和 Σ exp(x_j - max):")
    print(f"   sum_exp = {sum_exp.squeeze()}")
    
    log_sum_exp = torch.log(sum_exp)
    print(f"\n   2.5 计算 log(Σ exp(x_j - max)):")
    print(f"   log_sum_exp = {log_sum_exp.squeeze()}")
    
    log_probs_manual = inputs_shifted - log_sum_exp
    print(f"\n   2.6 计算 log_softmax = (x - max) - log(Σ exp(x_j - max)):")
    print(f"   log_probs_manual =\n   {log_probs_manual}")
    
    # 使用 PyTorch 的 log_softmax 验证
    log_probs_pytorch = F.log_softmax(inputs, dim=1)
    print(f"\n   2.7 使用 PyTorch F.log_softmax 验证:")
    print(f"   log_probs_pytorch =\n   {log_probs_pytorch}")
    print(f"   是否相等: {torch.allclose(log_probs_manual, log_probs_pytorch)}")
    
    # 步骤 2: 提取对应类别的 log 概率
    print(f"\n3. 提取每个样本对应真实类别的 log 概率:")
    print(f"   使用 gather 操作从 log_probs 中提取 targets 指定的位置")
    
    targets_expanded = targets.unsqueeze(1)
    print(f"   targets.unsqueeze(1) = {targets_expanded.squeeze()}")
    
    selected_log_probs = log_probs_pytorch.gather(1, targets_expanded).squeeze(1)
    print(f"\n   对于每个样本 i，取 log_probs[i][targets[i]]:")
    for i in range(len(targets)):
        print(f"   样本 {i}: log_probs[{i}][{targets[i]}] = {log_probs_pytorch[i][targets[i]].item():.6f}")
    print(f"\n   selected_log_probs = {selected_log_probs}")
    
    # 步骤 3: 取负号得到损失
    print(f"\n4. 计算交叉熵损失 (取负号):")
    print(f"   公式: CE_loss = -log(p_correct)")
    ce_loss_manual = -selected_log_probs
    print(f"   ce_loss = -selected_log_probs = {ce_loss_manual}")
    
    # 步骤 4: 使用 PyTorch 的 cross_entropy 验证
    print(f"\n5. 使用 PyTorch F.cross_entropy 验证:")
    ce_loss_pytorch = F.cross_entropy(inputs, targets, reduction='none')
    print(f"   F.cross_entropy(inputs, targets, reduction='none') = {ce_loss_pytorch}")
    print(f"   是否相等: {torch.allclose(ce_loss_manual, ce_loss_pytorch)}")
    
    # 步骤 5: 验证概率
    print(f"\n6. 验证: 从损失反推概率:")
    probs_from_loss = torch.exp(-ce_loss_pytorch)
    print(f"   p_correct = exp(-ce_loss) = {probs_from_loss}")
    
    # 计算 softmax 概率验证
    probs = F.softmax(inputs, dim=1)
    print(f"\n   从 softmax 直接计算概率:")
    print(f"   probs =\n   {probs}")
    for i in range(len(targets)):
        prob_correct = probs[i][targets[i]].item()
        print(f"   样本 {i}: p[{targets[i]}] = {prob_correct:.6f}, -log(p) = {-np.log(prob_correct):.6f}")
    
    return ce_loss_pytorch


def demonstrate_reduction_modes():
    """
    演示不同的 reduction 模式
    """
    print("\n" + "=" * 80)
    print("Reduction 模式对比")
    print("=" * 80)
    
    inputs = torch.tensor([
        [2.0, 1.0, 0.5],
        [0.5, 2.5, 1.0],
        [1.0, 0.5, 2.0]
    ])
    targets = torch.tensor([0, 1, 2], dtype=torch.long)
    
    # reduction='none': 返回每个样本的损失
    loss_none = F.cross_entropy(inputs, targets, reduction='none')
    print(f"\nreduction='none':")
    print(f"  返回每个样本的损失: {loss_none}")
    print(f"  shape: {loss_none.shape}")
    
    # reduction='mean': 返回平均值
    loss_mean = F.cross_entropy(inputs, targets, reduction='mean')
    print(f"\nreduction='mean':")
    print(f"  返回平均损失: {loss_mean.item():.6f}")
    print(f"  计算: {loss_none.mean().item():.6f} (应该相等)")
    
    # reduction='sum': 返回总和
    loss_sum = F.cross_entropy(inputs, targets, reduction='sum')
    print(f"\nreduction='sum':")
    print(f"  返回总损失: {loss_sum.item():.6f}")
    print(f"  计算: {loss_none.sum().item():.6f} (应该相等)")


if __name__ == "__main__":
    # 示例 1: 详细计算过程
    print("\n" + "=" * 80)
    print("示例 1: 详细计算过程")
    print("=" * 80)
    
    inputs = torch.tensor([
        [2.0, 1.0, 0.5, 0.1],  # 样本 0
        [0.5, 2.5, 1.0, 0.3],  # 样本 1
        [1.0, 0.5, 2.0, 0.8]   # 样本 2
    ], dtype=torch.float32)
    
    targets = torch.tensor([0, 1, 2], dtype=torch.long)
    
    ce_loss = manual_cross_entropy_step_by_step(inputs, targets)
    
    # 示例 2: Reduction 模式
    demonstrate_reduction_modes()
    
    # 示例 3: 数值稳定性对比
    print("\n" + "=" * 80)
    print("示例 3: 数值稳定性对比")
    print("=" * 80)
    
    # 使用较大的 logits 值来展示数值稳定性
    large_inputs = torch.tensor([
        [100.0, 99.0, 98.0],
        [50.0, 52.0, 51.0]
    ])
    large_targets = torch.tensor([0, 1], dtype=torch.long)
    
    print(f"\n使用较大的 logits 值:")
    print(f"inputs = {large_inputs}")
    
    # 方法 1: 直接计算 softmax + log (可能不稳定)
    try:
        probs = F.softmax(large_inputs, dim=1)
        log_probs_unsafe = torch.log(probs.gather(1, large_targets.unsqueeze(1)).squeeze(1))
        loss_unsafe = -log_probs_unsafe
        print(f"\n方法 1 (softmax + log): {loss_unsafe}")
    except Exception as e:
        print(f"\n方法 1 失败: {e}")
    
    # 方法 2: 使用 cross_entropy (数值稳定)
    loss_safe = F.cross_entropy(large_inputs, large_targets, reduction='none')
    print(f"方法 2 (cross_entropy): {loss_safe}")
    print(f"\n✓ cross_entropy 内部使用 log_softmax，数值更稳定")

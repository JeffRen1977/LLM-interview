#!/usr/bin/env python3
"""
详细演示 one-hot 编码转换过程
展示 F.one_hot() 和 permute() 的每一步
"""

import torch
import torch.nn.functional as F

print("=" * 80)
print("One-Hot 编码转换详解")
print("=" * 80)

# 原始 targets
targets = torch.tensor([
    [1, 0, 0, 3],  # 样本 0: [B-PER, O, O, B-LOC]
    [0, 0, 0, 0]   # 样本 1: [O, O, O, O]
], dtype=torch.long)

num_classes = 4

print("\n1. 原始 targets:")
print(f"   targets = {targets}")
print(f"   shape: {targets.shape}")
print(f"   含义: (batch_size=2, sequence_length=4)")
print(f"   每个值表示该位置的类别索引 (0-3)")

# 步骤 1: F.one_hot()
print("\n" + "-" * 80)
print("步骤 1: F.one_hot(targets, num_classes)")
print("-" * 80)

targets_one_hot_step1 = F.one_hot(targets, num_classes)
print(f"\n   输出 shape: {targets_one_hot_step1.shape}")
print(f"   含义: (batch_size=2, sequence_length=4, num_classes=4)")
print(f"\n   详细内容:")
print(f"   样本 0:")
print(f"      位置 0 (类别索引=1): {targets_one_hot_step1[0, 0].tolist()}")
print(f"         → 类别 1 的位置是 1，其他是 0")
print(f"      位置 1 (类别索引=0): {targets_one_hot_step1[0, 1].tolist()}")
print(f"         → 类别 0 的位置是 1，其他是 0")
print(f"      位置 2 (类别索引=0): {targets_one_hot_step1[0, 2].tolist()}")
print(f"      位置 3 (类别索引=3): {targets_one_hot_step1[0, 3].tolist()}")

print(f"\n   完整矩阵 (样本 0):")
print(f"   {targets_one_hot_step1[0]}")
print(f"   解释:")
print(f"     行 = 位置 (0, 1, 2, 3)")
print(f"     列 = 类别 (0, 1, 2, 3)")
print(f"     值 = 1 表示该位置属于该类别，0 表示不属于")

print(f"\n   完整矩阵 (样本 1):")
print(f"   {targets_one_hot_step1[1]}")

# 步骤 2: permute(0, 2, 1)
print("\n" + "-" * 80)
print("步骤 2: .permute(0, 2, 1)")
print("-" * 80)
print("   将维度从 (N, L, C) 转换为 (N, C, L)")
print("   即: (batch_size, sequence_length, num_classes)")
print("    → (batch_size, num_classes, sequence_length)")

targets_one_hot_step2 = targets_one_hot_step1.permute(0, 2, 1)
print(f"\n   输出 shape: {targets_one_hot_step2.shape}")
print(f"   含义: (batch_size=2, num_classes=4, sequence_length=4)")

print(f"\n   详细内容 (样本 0):")
print(f"   类别 0: {targets_one_hot_step2[0, 0].tolist()}")
print(f"      → 位置 [0,1,2,3] 上类别 0 的分布: [0, 1, 1, 0]")
print(f"      → 位置 1 和 2 是类别 0")
print(f"   类别 1: {targets_one_hot_step2[0, 1].tolist()}")
print(f"      → 位置 [0,1,2,3] 上类别 1 的分布: [1, 0, 0, 0]")
print(f"      → 位置 0 是类别 1")
print(f"   类别 2: {targets_one_hot_step2[0, 2].tolist()}")
print(f"      → 位置 [0,1,2,3] 上类别 2 的分布: [0, 0, 0, 0]")
print(f"      → 没有位置是类别 2")
print(f"   类别 3: {targets_one_hot_step2[0, 3].tolist()}")
print(f"      → 位置 [0,1,2,3] 上类别 3 的分布: [0, 0, 0, 1]")
print(f"      → 位置 3 是类别 3")

print(f"\n   完整矩阵 (样本 0):")
print(f"   {targets_one_hot_step2[0]}")
print(f"   解释:")
print(f"     行 = 类别 (0, 1, 2, 3)")
print(f"     列 = 位置 (0, 1, 2, 3)")
print(f"     值 = 1 表示该类别在该位置出现，0 表示不出现")

# 步骤 3: .float()
print("\n" + "-" * 80)
print("步骤 3: .float()")
print("-" * 80)

targets_one_hot_final = targets_one_hot_step2.float()
print(f"   输出 shape: {targets_one_hot_final.shape}")
print(f"   数据类型: {targets_one_hot_final.dtype}")
print(f"   将整数转换为浮点数，用于后续计算")

# 完整的一行代码
print("\n" + "=" * 80)
print("完整的一行代码:")
print("=" * 80)
print("targets_one_hot = F.one_hot(targets, num_classes).permute(0, 2, 1).float()")
print("\n最终结果:")
print(targets_one_hot_final)

# 可视化对比
print("\n" + "=" * 80)
print("可视化对比")
print("=" * 80)

print("\n原始 targets (类别索引):")
print("样本 0: [1, 0, 0, 3]")
print("        │  │  │  │")
print("        │  │  │  └─ 位置 3: 类别 3")
print("        │  │  └──── 位置 2: 类别 0")
print("        │  └─────── 位置 1: 类别 0")
print("        └────────── 位置 0: 类别 1")

print("\nOne-hot 编码后 (permute 前):")
print("样本 0:")
print("位置 0: [0, 1, 0, 0]  ← 类别 1")
print("位置 1: [1, 0, 0, 0]  ← 类别 0")
print("位置 2: [1, 0, 0, 0]  ← 类别 0")
print("位置 3: [0, 0, 0, 1]  ← 类别 3")

print("\nPermute 后 (最终结果):")
print("样本 0:")
print("类别 0: [0, 1, 1, 0]  ← 位置 1,2 是类别 0")
print("类别 1: [1, 0, 0, 0]  ← 位置 0 是类别 1")
print("类别 2: [0, 0, 0, 0]  ← 没有位置是类别 2")
print("类别 3: [0, 0, 0, 1]  ← 位置 3 是类别 3")

# 为什么需要 permute？
print("\n" + "=" * 80)
print("为什么需要 permute？")
print("=" * 80)
print("""
在 Dice Loss 中，我们需要：
- inputs shape: (N, C, L) = (batch_size, num_classes, sequence_length)
- targets_one_hot shape: (N, C, L) = (batch_size, num_classes, sequence_length)

这样两个张量的形状匹配，可以：
1. 直接相乘: inputs * targets_one_hot
2. 在相同的维度上求和: sum(dim=(0, 2))

F.one_hot() 默认输出 (N, L, C)，所以需要 permute 转换为 (N, C, L)
""")

# 验证形状匹配
print("\n" + "=" * 80)
print("验证形状匹配")
print("=" * 80)

inputs = torch.randn(2, 4, 4)  # (N, C, L)
targets_one_hot = F.one_hot(targets, num_classes).permute(0, 2, 1).float()  # (N, C, L)

print(f"inputs shape: {inputs.shape}")
print(f"targets_one_hot shape: {targets_one_hot.shape}")
print(f"形状匹配: {inputs.shape == targets_one_hot.shape}")

# 可以相乘
print("\n可以执行元素级乘法:")
intersection = inputs * targets_one_hot
print(f"intersection shape: {intersection.shape}")

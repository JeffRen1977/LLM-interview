#!/usr/bin/env python3
"""
Memory-Efficient Training Implementation

This implementation demonstrates memory-efficient training techniques including
mixed precision training and gradient checkpointing, extracted from 
openAI_memory_efficient_training.md.

Author: Extracted from openAI_memory_efficient_training.md
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint  # 导入梯度检查点
from torch.cuda.amp import GradScaler, autocast  # 导入混合精度工具
import time


# --- 1. 定义一个模拟的大型模型 ---
# 包含多个内存消耗大的 Block
class MemoryIntensiveBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(4 * hidden_dim, hidden_dim)
    
    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


class LargeModel(nn.Module):
    def __init__(self, num_layers=32, hidden_dim=1024, use_checkpointing=False):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        self.layers = nn.ModuleList(
            [MemoryIntensiveBlock(hidden_dim) for _ in range(num_layers)]
        )
    
    def forward(self, x):
        for layer in self.layers:
            if self.use_checkpointing and self.training:
                # 只在训练时使用 checkpoint
                x = checkpoint(layer, x)
            else:
                x = layer(x)
        return x


# 辅助函数：运行一个训练步骤并报告峰值内存
def run_training_step(model, optimizer, use_amp=False):
    scaler = GradScaler() if use_amp else None
    
    # 模拟输入数据
    input_data = torch.randn(16, 1024, 1024).cuda()  # (batch, seq_len, hidden)
    model.train()
    optimizer.zero_grad()
    
    # --- 核心：混合精度 ---
    # autocast 会自动将操作转为 FP16
    with autocast(enabled=use_amp):
        output = model(input_data)
        loss = output.mean()

    if use_amp:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:  # 标准 FP32 训练
        loss.backward()
        optimizer.step()
        
    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
    torch.cuda.reset_peak_memory_stats()  # 重置统计
    return peak_memory_gb


def compare_memory_usage():
    """Compare memory usage across different optimization techniques"""
    
    print("🚀 Memory-Efficient Training Comparison")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("❌ 需要 CUDA GPU 来运行此示例。")
        return
    
    results = {}
    
    # 场景 1: 标准训练 (可能会 OOM)
    print("--- 场景 1: 标准 FP32 训练 ---")
    model_base = LargeModel(use_checkpointing=False).to(device)
    optimizer_base = torch.optim.Adam(model_base.parameters())
    try:
        mem_base = run_training_step(model_base, optimizer_base, use_amp=False)
        results['Standard FP32'] = mem_base
        print(f"✅ 峰值内存占用: {mem_base:.2f} GB")
    except RuntimeError as e:
        results['Standard FP32'] = "OOM"
        print(f"❌ 内存不足 (OOM): {e}")
    del model_base, optimizer_base
    torch.cuda.empty_cache()

    # 场景 2: 仅使用混合精度
    print("\n--- 场景 2: 使用混合精度 (AMP) ---")
    model_amp = LargeModel(use_checkpointing=False).to(device)
    optimizer_amp = torch.optim.Adam(model_amp.parameters())
    mem_amp = run_training_step(model_amp, optimizer_amp, use_amp=True)
    results['Mixed Precision'] = mem_amp
    print(f"✅ 峰值内存占用: {mem_amp:.2f} GB")
    del model_amp, optimizer_amp
    torch.cuda.empty_cache()

    # 场景 3: 仅使用梯度检查点
    print("\n--- 场景 3: 使用梯度检查点 ---")
    model_cp = LargeModel(use_checkpointing=True).to(device)
    optimizer_cp = torch.optim.Adam(model_cp.parameters())
    mem_cp = run_training_step(model_cp, optimizer_cp, use_amp=False)
    results['Gradient Checkpointing'] = mem_cp
    print(f"✅ 峰值内存占用: {mem_cp:.2f} GB")
    del model_cp, optimizer_cp
    torch.cuda.empty_cache()

    # 场景 4: 结合两者
    print("\n--- 场景 4: 混合精度 + 梯度检查点 ---")
    model_both = LargeModel(use_checkpointing=True).to(device)
    optimizer_both = torch.optim.Adam(model_both.parameters())
    mem_both = run_training_step(model_both, optimizer_both, use_amp=True)
    results['Combined'] = mem_both
    print(f"✅ 峰值内存占用: {mem_both:.2f} GB")
    del model_both, optimizer_both
    torch.cuda.empty_cache()

    # Print summary
    print("\n" + "=" * 60)
    print("📊 Memory Usage Summary")
    print("=" * 60)
    for technique, memory in results.items():
        if memory == "OOM":
            print(f"{technique:25} | OOM (Out of Memory)")
        else:
            print(f"{technique:25} | {memory:.2f} GB")
    
    return results


def benchmark_performance():
    """Benchmark training speed with different techniques"""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("❌ CUDA GPU required for performance benchmarking.")
        return
    
    print("\n🏃 Performance Benchmarking")
    print("=" * 60)
    
    num_iterations = 10
    results = {}
    
    # Test different configurations
    configs = [
        ("Standard FP32", False, False),
        ("Mixed Precision", True, False),
        ("Gradient Checkpointing", False, True),
        ("Combined", True, True),
    ]
    
    for config_name, use_amp, use_checkpointing in configs:
        print(f"\nTesting {config_name}...")
        
        model = LargeModel(use_checkpointing=use_checkpointing).to(device)
        optimizer = torch.optim.Adam(model.parameters())
        
        # Warmup
        for _ in range(3):
            run_training_step(model, optimizer, use_amp=use_amp)
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_iterations):
            run_training_step(model, optimizer, use_amp=use_amp)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_iterations
        results[config_name] = avg_time
        
        print(f"  Average time per iteration: {avg_time:.4f}s")
        
        del model, optimizer
        torch.cuda.empty_cache()
    
    # Print performance summary
    print("\n" + "=" * 60)
    print("⏱️  Performance Summary")
    print("=" * 60)
    baseline_time = results.get("Standard FP32", 1.0)
    
    for technique, time_taken in results.items():
        speedup = baseline_time / time_taken if time_taken > 0 else 0
        print(f"{technique:25} | {time_taken:.4f}s | {speedup:.2f}x speedup")
    
    return results


def demonstrate_techniques():
    """Demonstrate the memory-efficient training techniques"""
    
    print("🧠 Memory-Efficient Training Techniques Demo")
    print("Extracted from openAI_memory_efficient_training.md")
    print("=" * 60)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        print(f"   Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️  CUDA not available. Some features may not work.")
    
    # Run memory comparison
    memory_results = compare_memory_usage()
    
    # Run performance benchmark
    performance_results = benchmark_performance()
    
    print("\n" + "=" * 60)
    print("🎉 All demonstrations completed!")
    print("=" * 60)


def main():
    """Main function to run the memory-efficient training demonstration"""
    
    print("💾 Memory-Efficient Training Implementation")
    print("This demonstrates mixed precision training and gradient checkpointing")
    print("as described in openAI_memory_efficient_training.md")
    print("=" * 60)
    
    try:
        demonstrate_techniques()
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint # 导入梯度检查点
from torch.cuda.amp import GradScaler, autocast # 导入混合精度工具

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
    input_data = torch.randn(16, 1024, 1024).cuda() # (batch, seq_len, hidden)
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
    else: # 标准 FP32 训练
        loss.backward()
        optimizer.step()
        
    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
    torch.cuda.reset_peak_memory_stats() # 重置统计
    return peak_memory_gb

# --- 2. 实验对比 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("需要 CUDA GPU 来运行此示例。")
else:
    # 场景 1: 标准训练 (可能会 OOM)
    print("--- 场景 1: 标准 FP32 训练 ---")
    model_base = LargeModel(use_checkpointing=False).to(device)
    optimizer_base = torch.optim.Adam(model_base.parameters())
    try:
        mem_base = run_training_step(model_base, optimizer_base, use_amp=False)
        print(f"峰值内存占用: {mem_base:.2f} GB\n")
    except RuntimeError as e:
        print(f"内存不足 (OOM): {e}\n")
    del model_base, optimizer_base

    # 场景 2: 仅使用混合精度
    print("--- 场景 2: 使用混合精度 (AMP) ---")
    model_amp = LargeModel(use_checkpointing=False).to(device)
    optimizer_amp = torch.optim.Adam(model_amp.parameters())
    mem_amp = run_training_step(model_amp, optimizer_amp, use_amp=True)
    print(f"峰值内存占用: {mem_amp:.2f} GB\n")
    del model_amp, optimizer_amp

    # 场景 3: 仅使用梯度检查点
    print("--- 场景 3: 使用梯度检查点 ---")
    model_cp = LargeModel(use_checkpointing=True).to(device)
    optimizer_cp = torch.optim.Adam(model_cp.parameters())
    mem_cp = run_training_step(model_cp, optimizer_cp, use_amp=False)
    print(f"峰值内存占用: {mem_cp:.2f} GB\n")
    del model_cp, optimizer_cp

    # 场景 4: 结合两者
    print("--- 场景 4: 混合精度 + 梯度检查点 ---")
    model_both = LargeModel(use_checkpointing=True).to(device)
    optimizer_both = torch.optim.Adam(model_both.parameters())
    mem_both = run_training_step(model_both, optimizer_both, use_amp=True)
    print(f"峰值内存占用: {mem_both:.2f} GB\n")
    del model_both, optimizer_both
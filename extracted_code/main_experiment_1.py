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
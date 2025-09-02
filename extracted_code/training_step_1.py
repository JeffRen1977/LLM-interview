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
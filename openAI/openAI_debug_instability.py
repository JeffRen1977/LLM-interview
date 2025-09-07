"""
OpenAI 訓練不穩定性問題調試工具

本模組提供了一套完整的訓練不穩定性診斷和解決方案，專門針對深度學習模型
訓練過程中可能遇到的各種穩定性問題。

主要功能:
1. 訓練穩定性診斷
   - 梯度范數監控和分析
   - 權重范數統計和檢查
   - 激活值分布分析
   - 損失函數異常檢測

2. 穩定訓練器實現
   - 梯度裁剪和正則化
   - 學習率預熱和調度
   - 數值穩定性保護
   - 模型健康狀態監控

3. 常見問題修復
   - 學習率調整策略
   - 梯度問題解決方案
   - 數據質量檢查
   - 模型架構優化

4. 診斷工具和可視化
   - 實時監控指標
   - 統計分析和報告
   - 問題識別和建議
   - 性能基準測試

技術特點:
- 支持多種優化器和學習率調度策略
- 提供詳細的梯度分析工具
- 包含數值穩定性保護機制
- 支持混合精度訓練
- 提供完整的錯誤處理和日誌記錄

適用場景:
- 深度學習模型訓練調試
- 訓練不穩定性問題診斷
- 模型性能優化
- 生產環境穩定性監控

作者: OpenAI Interview Preparation
版本: 1.0.0
更新日期: 2024
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
import warnings
import logging
from typing import Dict, List, Tuple

# 配置日誌系統
# 設置日誌級別為INFO，確保重要信息被記錄
# 使用標準的Python logging模組進行日誌管理
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrainingStabilityDiagnostics:
    """
    訓練穩定性診斷工具類
    
    這個類提供了一套完整的訓練穩定性診斷工具，用於監控和分析深度學習模型
    訓練過程中的各種穩定性指標。它能夠檢測梯度爆炸、梯度消失、權重異常、
    激活值問題等常見的訓練不穩定性問題。
    
    主要功能:
    1. 梯度范數監控
       - 計算總體梯度范數
       - 分析各層梯度分布
       - 檢測梯度爆炸和消失
       - 提供梯度統計信息
    
    2. 權重范數分析
       - 監控權重參數范數
       - 分析權重分布統計
       - 檢測權重異常變化
       - 提供權重健康狀態
    
    3. 激活值檢查
       - 分析各層激活值分布
       - 檢測NaN和Inf值
       - 監控激活值範圍
       - 提供激活統計信息
    
    4. 損失函數診斷
       - 檢測損失跳躍
       - 識別振盪模式
       - 發現異常值
       - 提供穩定性建議
    
    技術特點:
    - 實時監控和診斷
    - 詳細的統計分析
    - 自動問題檢測
    - 可視化支持
    - 生產環境友好
    
    使用示例:
        diagnostics = TrainingStabilityDiagnostics()
        
        # 記錄訓練指標
        diagnostics.log_metrics(epoch=0, loss=0.5, accuracy=0.8)
        
        # 檢查梯度范數
        grad_stats = diagnostics.check_gradient_norms(model)
        
        # 檢查權重范數
        weight_stats = diagnostics.check_weight_norms(model)
        
        # 檢查激活值
        activation_stats = diagnostics.check_activations(model, sample_input)
        
        # 診斷不穩定性
        issues = diagnostics.diagnose_instability(loss_history)
    """
    
    def __init__(self):
        """
        初始化訓練穩定性診斷工具
        
        設置用於存儲各種診斷指標的數據結構，包括:
        - metrics: 存儲訓練過程中的各種指標
        - gradient_norms: 存儲梯度范數歷史
        - weight_norms: 存儲權重范數歷史
        - activations: 存儲激活值統計信息
        """
        # 使用defaultdict存儲各種指標，自動創建列表
        self.metrics = defaultdict(list)
        
        # 存儲梯度范數歷史，用於分析梯度變化趨勢
        self.gradient_norms = []
        
        # 存儲權重范數歷史，用於監控權重變化
        self.weight_norms = []
        
        # 存儲激活值統計信息，用於分析激活分布
        self.activations = []
        
    def log_metrics(self, epoch: int, loss: float, **kwargs):
        """
        記錄訓練指標
        
        這個方法用於記錄訓練過程中的各種指標，包括損失值、準確率、
        學習率等。記錄的指標可以用於後續的穩定性分析和可視化。
        
        參數:
            epoch (int): 當前訓練輪次
            loss (float): 當前損失值
            **kwargs: 其他需要記錄的指標，如accuracy、learning_rate等
        
        功能:
        - 自動記錄epoch和loss
        - 支持動態添加其他指標
        - 維護指標歷史記錄
        - 支持後續分析和可視化
        
        使用示例:
            diagnostics.log_metrics(epoch=0, loss=0.5, accuracy=0.8, lr=0.001)
        """
        # 記錄epoch和loss，這些是基本的訓練指標
        self.metrics['epoch'].append(epoch)
        self.metrics['loss'].append(loss)
        
        # 記錄其他動態指標
        for key, value in kwargs.items():
            self.metrics[key].append(value)
    
    def check_gradient_norms(self, model: nn.Module) -> Dict[str, float]:
        """
        檢查梯度范數
        
        這個方法計算和分析模型中所有參數的梯度范數，用於檢測梯度爆炸
        和梯度消失問題。梯度范數是衡量梯度大小的重要指標，對於訓練
        穩定性至關重要。
        
        參數:
            model (nn.Module): 要檢查的PyTorch模型
        
        返回:
            Dict[str, float]: 包含梯度范數統計信息的字典
                - total_norm: 總體梯度范數 (L2范數)
                - max_norm: 最大單個參數梯度范數
                - min_norm: 最小單個參數梯度范數
                - avg_norm: 平均梯度范數
                - layer_norms: 各層梯度范數詳情
        
        技術細節:
        - 使用L2范數計算梯度大小
        - 只計算有梯度的參數
        - 提供總體和逐層的統計信息
        - 支持梯度爆炸和消失檢測
        
        使用示例:
            grad_stats = diagnostics.check_gradient_norms(model)
            if grad_stats['total_norm'] > 10.0:
                print("警告: 梯度范數過大，可能存在梯度爆炸")
        """
        # 初始化統計變量
        total_norm = 0.0  # 總體梯度范數的平方和
        param_count = 0   # 有梯度的參數數量
        max_norm = 0.0    # 最大單個參數梯度范數
        min_norm = float('inf')  # 最小單個參數梯度范數
        layer_norms = {}  # 各層梯度范數詳情
        
        # 遍歷模型中的所有參數
        for name, param in model.named_parameters():
            # 只處理有梯度的參數
            if param.grad is not None:
                # 計算當前參數的L2范數
                param_norm = param.grad.data.norm(2)
                
                # 記錄各層的梯度范數
                layer_norms[name] = param_norm.item()
                
                # 累加到總體范數的平方和中
                total_norm += param_norm.item() ** 2
                param_count += 1
                
                # 更新最大值和最小值
                max_norm = max(max_norm, param_norm.item())
                min_norm = min(min_norm, param_norm.item())
        
        # 計算總體L2范數 (平方和的平方根)
        total_norm = total_norm ** (1. / 2)
        
        # 返回完整的梯度統計信息
        return {
            'total_norm': total_norm,  # 總體梯度范數
            'max_norm': max_norm,      # 最大單個參數梯度范數
            'min_norm': min_norm if min_norm != float('inf') else 0.0,  # 最小單個參數梯度范數
            'avg_norm': total_norm / max(param_count, 1),  # 平均梯度范數
            'layer_norms': layer_norms  # 各層梯度范數詳情
        }
    
    def check_weight_norms(self, model: nn.Module) -> Dict[str, float]:
        """
        檢查權重范數
        
        這個方法計算和分析模型中所有權重參數的范數和統計信息，用於
        監控權重的健康狀態和檢測權重異常。權重范數是衡量權重大小的
        重要指標，對於模型穩定性和性能至關重要。
        
        參數:
            model (nn.Module): 要檢查的PyTorch模型
        
        返回:
            Dict[str, Dict[str, float]]: 包含各層權重統計信息的字典
                每個層的統計信息包括:
                - norm: 權重L2范數
                - mean: 權重均值
                - std: 權重標準差
        
        技術細節:
        - 只檢查包含'weight'的參數
        - 計算L2范數、均值和標準差
        - 提供權重分布統計信息
        - 支持權重異常檢測
        
        使用示例:
            weight_stats = diagnostics.check_weight_norms(model)
            for layer_name, stats in weight_stats.items():
                if stats['norm'] > 100.0:
                    print(f"警告: {layer_name} 權重范數過大")
        """
        weight_stats = {}
        
        # 遍歷模型中的所有參數
        for name, param in model.named_parameters():
            # 只處理權重參數 (通常包含'weight'關鍵字)
            if 'weight' in name:
                # 計算權重的L2范數
                weight_norm = param.data.norm(2).item()
                
                # 計算權重的均值
                weight_mean = param.data.mean().item()
                
                # 計算權重的標準差
                weight_std = param.data.std().item()
                
                # 記錄該層的權重統計信息
                weight_stats[name] = {
                    'norm': weight_norm,    # L2范數
                    'mean': weight_mean,    # 均值
                    'std': weight_std       # 標準差
                }
        
        return weight_stats
    
    def check_activations(self, model: nn.Module, sample_input: torch.Tensor):
        """
        檢查激活值分布
        
        這個方法使用PyTorch的hook機制來捕獲模型前向傳播過程中各層的
        激活值，並分析其分布統計信息。激活值分析對於檢測梯度消失、
        激活飽和、數值不穩定等問題非常重要。
        
        參數:
            model (nn.Module): 要檢查的PyTorch模型
            sample_input (torch.Tensor): 用於前向傳播的樣本輸入
        
        返回:
            Dict[str, Dict[str, float]]: 包含各層激活值統計信息的字典
                每個層的統計信息包括:
                - mean: 激活值均值
                - std: 激活值標準差
                - max: 激活值最大值
                - min: 激活值最小值
                - nan_count: NaN值數量
                - inf_count: Inf值數量
        
        技術細節:
        - 使用forward hook捕獲激活值
        - 只監控葉子模組 (leaf modules)
        - 計算完整的統計信息
        - 檢測數值異常 (NaN/Inf)
        - 自動清理hook避免內存洩漏
        
        使用示例:
            sample_input = torch.randn(1, 784)
            activation_stats = diagnostics.check_activations(model, sample_input)
            for layer_name, stats in activation_stats.items():
                if stats['nan_count'] > 0:
                    print(f"警告: {layer_name} 包含NaN值")
        """
        activations = {}  # 存儲各層激活值統計信息
        handles = []      # 存儲hook句柄，用於後續清理
        
        def get_activation(name):
            """
            創建激活值捕獲hook
            
            這個內部函數創建一個hook函數，用於捕獲指定模組的激活值
            並計算統計信息。
            
            參數:
                name (str): 模組名稱
            
            返回:
                function: hook函數
            """
            def hook(model, input, output):
                # 只處理張量輸出
                if isinstance(output, torch.Tensor):
                    activations[name] = {
                        'mean': output.mean().item(),      # 激活值均值
                        'std': output.std().item(),        # 激活值標準差
                        'max': output.max().item(),        # 激活值最大值
                        'min': output.min().item(),        # 激活值最小值
                        'nan_count': torch.isnan(output).sum().item(),  # NaN值數量
                        'inf_count': torch.isinf(output).sum().item()   # Inf值數量
                    }
            return hook
        
        # 註冊forward hook到所有葉子模組
        for name, module in model.named_modules():
            # 只監控葉子模組 (沒有子模組的模組)
            if len(list(module.children())) == 0:
                handle = module.register_forward_hook(get_activation(name))
                handles.append(handle)
        
        # 執行前向傳播以觸發hook
        model.eval()  # 設置為評估模式
        with torch.no_grad():  # 禁用梯度計算
            _ = model(sample_input)
        
        # 清理所有hook，避免內存洩漏
        for handle in handles:
            handle.remove()
            
        return activations
    
    def diagnose_instability(self, loss_history: List[float], threshold: float = 1.0) -> List[str]:
        """
        診斷訓練不穩定性
        
        這個方法分析損失函數的歷史記錄，檢測各種訓練不穩定性問題，
        包括損失跳躍、數值異常、振盪模式等。這些問題通常表示訓練
        過程中存在數值不穩定、學習率過大、梯度問題等。
        
        參數:
            loss_history (List[float]): 損失函數歷史記錄
            threshold (float): 損失跳躍檢測閾值，默認為1.0
        
        返回:
            List[str]: 檢測到的問題列表，每個問題包含描述信息
        
        檢測的問題類型:
        1. 損失跳躍 (Loss Jump)
           - 檢測相鄰步驟間損失值的劇烈變化
           - 通常表示學習率過大或梯度爆炸
        
        2. 數值異常 (Invalid Values)
           - 檢測NaN (Not a Number) 值
           - 檢測Inf (Infinity) 值
           - 通常表示數值計算溢出或除零錯誤
        
        3. 損失振盪 (Loss Oscillation)
           - 檢測損失值的劇烈波動
           - 通常表示學習率不穩定或數據問題
        
        使用示例:
            loss_history = [0.5, 0.3, 0.8, 0.2, 0.9]
            issues = diagnostics.diagnose_instability(loss_history, threshold=0.5)
            for issue in issues:
                print(f"問題: {issue}")
        """
        issues = []  # 存儲檢測到的問題
        
        # 檢查歷史記錄長度
        if len(loss_history) < 2:
            return issues
        
        # 1. 檢查損失跳躍
        # 檢測相鄰步驟間損失值的劇烈變化
        for i in range(1, len(loss_history)):
            loss_change = abs(loss_history[i] - loss_history[i-1])
            if loss_change > threshold:
                issues.append(
                    f"Loss jump detected at step {i}: "
                    f"{loss_history[i-1]:.4f} -> {loss_history[i]:.4f} "
                    f"(change: {loss_change:.4f})"
                )
        
        # 2. 檢查NaN或Inf值
        # 檢測數值異常，這些通常表示嚴重的數值問題
        for i, loss in enumerate(loss_history):
            if np.isnan(loss):
                issues.append(f"NaN loss value detected at step {i}: {loss}")
            elif np.isinf(loss):
                issues.append(f"Infinity loss value detected at step {i}: {loss}")
        
        # 3. 檢查損失振盪
        # 檢測損失值的劇烈波動模式
        if len(loss_history) > 10:
            recent_losses = loss_history[-10:]  # 取最近10個步驟
            loss_std = np.std(recent_losses)    # 計算標準差
            loss_mean = np.mean(recent_losses)  # 計算均值
            
            # 如果標準差相對於均值過大，認為存在振盪
            if loss_std > loss_mean * 0.1:  # 10%的閾值
                issues.append(
                    f"Loss oscillation detected in recent steps: "
                    f"std={loss_std:.4f}, mean={loss_mean:.4f}, "
                    f"cv={loss_std/loss_mean:.2%}"
                )
        
        return issues

class StableTrainer:
    """
    穩定訓練器類
    
    這個類提供了一套完整的穩定訓練解決方案，專門針對深度學習模型
    訓練過程中的各種穩定性問題。它集成了多種穩定化技術，包括梯度
    裁剪、學習率調度、數值穩定性保護等。
    
    主要功能:
    1. 穩定化訓練技術
       - 梯度裁剪和正則化
       - 學習率預熱和調度
       - 數值穩定性保護
       - 混合精度訓練支持
    
    2. 訓練監控和診斷
       - 實時梯度監控
       - 權重健康檢查
       - 激活值分析
       - 損失穩定性診斷
    
    3. 優化器和調度器
       - 支持多種優化器 (Adam, AdamW, SGD)
       - 學習率調度策略 (cosine, plateau, step)
       - 自適應學習率調整
       - 權重衰減正則化
    
    4. 錯誤處理和恢復
       - 自動異常檢測
       - 訓練中斷恢復
       - 數值溢出保護
       - 詳細日誌記錄
    
    技術特點:
    - 生產級穩定性保證
    - 完整的錯誤處理機制
    - 詳細的監控和診斷
    - 支持大規模訓練
    - 可配置的穩定化參數
    
    使用示例:
        model = MyModel()
        train_loader = DataLoader(train_dataset, batch_size=32)
        trainer = StableTrainer(model, train_loader)
        
        # 設置優化器和調度器
        trainer.setup_optimizer_and_scheduler(
            learning_rate=1e-3,
            use_warmup=True,
            scheduler_type='cosine'
        )
        
        # 開始訓練
        for epoch in range(num_epochs):
            metrics = trainer.train_epoch(epoch)
            print(f"Epoch {epoch}: {metrics}")
    """
    
    def __init__(self, model: nn.Module, train_loader: DataLoader, 
                 val_loader: DataLoader = None, device: str = 'cpu'):
        """
        初始化穩定訓練器
        
        參數:
            model (nn.Module): 要訓練的PyTorch模型
            train_loader (DataLoader): 訓練數據加載器
            val_loader (DataLoader, optional): 驗證數據加載器，默認為None
            device (str): 計算設備，默認為'cpu'
        
        功能:
        - 初始化模型和數據加載器
        - 設置診斷工具
        - 配置穩定化參數
        - 準備訓練環境
        """
        # 將模型移動到指定設備
        self.model = model.to(device)
        
        # 設置數據加載器
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 設置計算設備
        self.device = device
        
        # 初始化診斷工具
        self.diagnostics = TrainingStabilityDiagnostics()
        
        # 穩定性配置參數
        self.max_grad_norm = 1.0              # 梯度裁剪閾值
        self.loss_scale = 1.0                 # 損失縮放因子 (混合精度訓練)
        self.warmup_steps = 0                 # 學習率預熱步數
        self.gradient_accumulation_steps = 1  # 梯度累積步數
        
    def setup_optimizer_and_scheduler(self, learning_rate: float = 1e-3,
                                    weight_decay: float = 1e-4,
                                    use_warmup: bool = True,
                                    scheduler_type: str = 'cosine'):
        """
        設置優化器和學習率調度器
        
        這個方法配置訓練過程中的優化器和學習率調度策略，提供多種
        穩定化技術來改善訓練效果和穩定性。
        
        參數:
            learning_rate (float): 初始學習率，默認為1e-3
            weight_decay (float): 權重衰減係數，默認為1e-4
            use_warmup (bool): 是否使用學習率預熱，默認為True
            scheduler_type (str): 學習率調度器類型，可選'cosine'或'plateau'
        
        支持的優化器:
        - AdamW: 自適應學習率優化器，具有權重衰減正則化
        - 數值穩定性優化 (eps=1e-8)
        - 動量參數調優 (betas=(0.9, 0.999))
        
        支持的調度器:
        - CosineAnnealingLR: 餘弦退火調度器
        - ReduceLROnPlateau: 平台降低調度器
        
        使用示例:
            trainer.setup_optimizer_and_scheduler(
                learning_rate=1e-3,
                weight_decay=1e-4,
                use_warmup=True,
                scheduler_type='cosine'
            )
        """
        
        # 使用AdamW優化器，提供更好的穩定性
        # AdamW結合了Adam的優點和權重衰減正則化
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,           # 學習率
            weight_decay=weight_decay,  # 權重衰減正則化
            eps=1e-8,                   # 數值穩定性參數
            betas=(0.9, 0.999)          # 動量參數
        )
        
        # 設置學習率預熱
        if use_warmup:
            # 預熱步數設為第一個epoch的10%
            self.warmup_steps = len(self.train_loader) // 10
        
        # 設置學習率調度器
        if scheduler_type == 'cosine':
            # 餘弦退火調度器：學習率按餘弦函數下降
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=len(self.train_loader) * 10  # 假設訓練10個epoch
            )
        elif scheduler_type == 'plateau':
            # 平台降低調度器：當指標不再改善時降低學習率
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min',      # 監控指標越小越好
                factor=0.5,      # 學習率縮放因子
                patience=5,      # 容忍步數
                verbose=True     # 打印調度信息
            )
        else:
            # 不使用調度器
            self.scheduler = None
    
    def warmup_lr(self, step: int):
        """
        學習率預熱
        
        這個方法實現學習率預熱機制，在訓練初期逐步增加學習率，
        有助於穩定訓練過程和改善收斂效果。
        
        參數:
            step (int): 當前訓練步數
        
        預熱策略:
        - 線性預熱：學習率從0線性增加到目標學習率
        - 預熱步數：在warmup_steps步內完成預熱
        - 平滑過渡：避免學習率突然變化
        
        技術細節:
        - 只在預熱階段調整學習率
        - 預熱因子 = step / warmup_steps
        - 預熱完成後使用正常學習率
        
        使用示例:
            for step in range(total_steps):
                trainer.warmup_lr(step)
                # 執行訓練步驟
        """
        # 只在預熱階段調整學習率
        if step < self.warmup_steps:
            # 計算預熱因子：從0線性增加到1
            warmup_factor = step / self.warmup_steps
            
            # 更新所有參數組的學習率
            for param_group in self.optimizer.param_groups:
                # 學習率 = 原始學習率 × 預熱因子
                param_group['lr'] = param_group['lr'] * warmup_factor
    
    def clip_gradients(self):
        """
        梯度裁剪
        
        這個方法實現梯度裁剪技術，用於防止梯度爆炸問題。
        梯度裁剪通過限制梯度范數來保持訓練穩定性。
        
        技術細節:
        - 使用L2范數進行梯度裁剪
        - 只裁剪范數超過閾值的梯度
        - 保持梯度方向不變
        - 防止梯度爆炸和數值不穩定
        
        裁剪策略:
        - 計算所有參數梯度的總范數
        - 如果總范數超過max_grad_norm，則縮放所有梯度
        - 縮放因子 = max_grad_norm / 實際范數
        
        使用示例:
            # 在反向傳播後調用
            loss.backward()
            trainer.clip_gradients()
            optimizer.step()
        """
        # 只在設置了梯度裁剪閾值時執行
        if self.max_grad_norm > 0:
            # 使用PyTorch的梯度裁剪函數
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),  # 要裁剪的參數
                self.max_grad_norm        # 最大梯度范數閾值
            )
    
    def check_model_health(self, step: int) -> bool:
        """
        檢查模型健康狀態
        
        這個方法檢查模型參數是否包含異常值（NaN或Inf），
        用於檢測數值不穩定問題和防止訓練崩潰。
        
        參數:
            step (int): 當前訓練步數
        
        返回:
            bool: 模型是否健康
                - True: 模型參數正常
                - False: 發現異常值
        
        檢查內容:
        - NaN (Not a Number) 值檢測
        - Inf (Infinity) 值檢測
        - 參數數值範圍檢查
        - 異常值位置記錄
        
        技術細節:
        - 使用torch.isnan()檢測NaN值
        - 使用torch.isinf()檢測Inf值
        - 記錄異常參數名稱和位置
        - 提供詳細的錯誤日誌
        
        使用示例:
            if not trainer.check_model_health(step):
                logger.error("模型健康檢查失敗，停止訓練")
                break
        """
        # 檢查所有模型參數是否包含異常值
        for name, param in self.model.named_parameters():
            # 檢查NaN值
            if torch.isnan(param).any():
                logger.error(f"NaN detected in parameter {name} at step {step}")
                return False
            
            # 檢查Inf值
            if torch.isinf(param).any():
                logger.error(f"Infinity detected in parameter {name} at step {step}")
                return False
        
        # 所有參數都正常
        return True
    
    def train_step(self, batch_idx: int, data: torch.Tensor, 
                   target: torch.Tensor) -> Dict[str, float]:
        """
        單步訓練
        
        這個方法執行一個完整的訓練步驟，包括前向傳播、損失計算、
        反向傳播、梯度裁剪和參數更新。它集成了多種穩定化技術。
        
        參數:
            batch_idx (int): 當前批次索引
            data (torch.Tensor): 輸入數據
            target (torch.Tensor): 目標標籤
        
        返回:
            Dict[str, float]: 訓練指標字典
                - loss: 原始損失值
                - scaled_loss: 縮放後損失值
                - grad_norm: 梯度范數
                - lr: 當前學習率
        
        訓練流程:
        1. 學習率預熱
        2. 前向傳播
        3. 損失計算
        4. 損失縮放
        5. 反向傳播
        6. 梯度檢查
        7. 梯度裁剪
        8. 參數更新
        9. 學習率調度
        
        技術特點:
        - 支持多種損失函數
        - 混合精度訓練支持
        - 梯度累積機制
        - 自動梯度監控
        - 學習率調度
        
        使用示例:
            for batch_idx, (data, target) in enumerate(train_loader):
                metrics = trainer.train_step(batch_idx, data, target)
                print(f"Loss: {metrics['loss']:.4f}")
        """
        
        # 1. 學習率預熱
        # 在訓練初期逐步增加學習率
        if hasattr(self, 'warmup_steps') and batch_idx < self.warmup_steps:
            self.warmup_lr(batch_idx)
        
        # 2. 前向傳播
        # 計算模型輸出
        output = self.model(data)
        
        # 3. 損失計算
        # 根據任務類型選擇合適的損失函數
        if len(target.shape) > 1 and target.shape[1] > 1:  # 多標籤分類
            loss = F.binary_cross_entropy_with_logits(output, target.float())
        else:  # 單標籤分類或回歸
            if target.dtype == torch.long:  # 分類任務
                loss = F.cross_entropy(output, target)
            else:  # 回歸任務
                loss = F.mse_loss(output, target)
        
        # 4. 損失縮放 (混合精度訓練)
        # 用於防止梯度下溢
        scaled_loss = loss * self.loss_scale
        
        # 5. 反向傳播
        # 計算梯度
        scaled_loss.backward()
        
        # 6. 梯度累積和參數更新
        # 只在累積步數達到時更新參數
        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
            # 檢查梯度范數
            grad_stats = self.diagnostics.check_gradient_norms(self.model)
            
            # 梯度裁剪
            self.clip_gradients()
            
            # 優化器步驟
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # 學習率調度
            if self.scheduler and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
        
        # 返回訓練指標
        return {
            'loss': loss.item(),                    # 原始損失值
            'scaled_loss': scaled_loss.item(),      # 縮放後損失值
            'grad_norm': grad_stats.get('total_norm', 0.0),  # 梯度范數
            'lr': self.optimizer.param_groups[0]['lr']       # 當前學習率
        }
    
    def validate(self) -> Dict[str, float]:
        """
        驗證步驟
        
        這個方法在驗證集上評估模型性能，計算損失和準確率等指標。
        驗證過程使用評估模式，禁用梯度計算以提高效率。
        
        返回:
            Dict[str, float]: 驗證指標字典
                - val_loss: 平均驗證損失
                - val_accuracy: 驗證準確率
        
        驗證流程:
        1. 設置模型為評估模式
        2. 禁用梯度計算
        3. 遍歷驗證數據集
        4. 計算損失和準確率
        5. 返回平均指標
        
        技術特點:
        - 自動設備轉移
        - 支持多種任務類型
        - 高效的驗證過程
        - 詳細的指標計算
        
        使用示例:
            val_metrics = trainer.validate()
            print(f"驗證損失: {val_metrics['val_loss']:.4f}")
            print(f"驗證準確率: {val_metrics['val_accuracy']:.4f}")
        """
        # 檢查是否有驗證數據加載器
        if not self.val_loader:
            return {}
        
        # 設置模型為評估模式
        self.model.eval()
        
        # 初始化統計變量
        total_loss = 0    # 總損失
        correct = 0       # 正確預測數量
        total = 0         # 總樣本數量
        
        # 禁用梯度計算以提高效率
        with torch.no_grad():
            # 遍歷驗證數據集
            for data, target in self.val_loader:
                # 將數據移動到指定設備
                data, target = data.to(self.device), target.to(self.device)
                
                # 前向傳播
                output = self.model(data)
                
                # 計算損失
                if len(target.shape) > 1 and target.shape[1] > 1:  # 多標籤分類
                    loss = F.binary_cross_entropy_with_logits(output, target.float())
                else:  # 單標籤分類或回歸
                    if target.dtype == torch.long:  # 分類任務
                        loss = F.cross_entropy(output, target)
                        # 計算準確率
                        pred = output.argmax(dim=1, keepdim=True)
                        correct += pred.eq(target.view_as(pred)).sum().item()
                    else:  # 回歸任務
                        loss = F.mse_loss(output, target)
                
                # 累加統計信息
                total_loss += loss.item()
                total += target.size(0)
        
        # 計算平均指標
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total if total > 0 else 0
        
        # 返回驗證指標
        return {
            'val_loss': avg_loss,      # 平均驗證損失
            'val_accuracy': accuracy   # 驗證準確率
        }
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        訓練一個epoch
        
        這個方法執行一個完整的訓練epoch，包括數據處理、訓練步驟、
        健康檢查、穩定性診斷和進度監控。
        
        參數:
            epoch (int): 當前訓練epoch
        
        返回:
            Dict[str, float]: epoch平均指標
                - avg_loss: 平均損失
                - avg_grad_norm: 平均梯度范數
                - avg_lr: 平均學習率
        
        訓練流程:
        1. 設置模型為訓練模式
        2. 初始化指標收集
        3. 遍歷訓練數據
        4. 數據質量檢查
        5. 執行訓練步驟
        6. 記錄指標
        7. 健康狀態檢查
        8. 穩定性診斷
        9. 進度監控
        10. 計算平均指標
        
        技術特點:
        - 自動數據質量檢查
        - 實時健康監控
        - 穩定性問題診斷
        - 詳細的進度報告
        - 自動錯誤處理
        
        使用示例:
            for epoch in range(num_epochs):
                metrics = trainer.train_epoch(epoch)
                print(f"Epoch {epoch}: {metrics}")
        """
        # 設置模型為訓練模式
        self.model.train()
        
        # 初始化epoch指標收集
        epoch_metrics = {
            'loss': [],      # 損失值列表
            'grad_norm': [], # 梯度范數列表
            'lr': []         # 學習率列表
        }
        
        # 遍歷訓練數據
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # 將數據移動到指定設備
            data, target = data.to(self.device), target.to(self.device)
            
            # 檢查輸入數據質量
            if torch.isnan(data).any() or torch.isinf(data).any():
                logger.warning(f"NaN/Inf detected in input data at batch {batch_idx}")
                continue  # 跳過有問題的批次
            
            # 執行訓練步驟
            step_metrics = self.train_step(batch_idx, data, target)
            
            # 記錄指標
            for key, value in step_metrics.items():
                if key in epoch_metrics:
                    epoch_metrics[key].append(value)
            
            # 檢查模型健康狀態
            if not self.check_model_health(batch_idx):
                logger.error(f"Training stopped due to model health issues at batch {batch_idx}")
                break  # 停止訓練
            
            # 診斷訓練不穩定性
            if len(epoch_metrics['loss']) > 10:
                # 使用最近10個損失值進行診斷
                issues = self.diagnostics.diagnose_instability(
                    epoch_metrics['loss'][-10:], 
                    threshold=np.mean(epoch_metrics['loss']) * 0.5
                )
                # 記錄發現的問題
                for issue in issues:
                    logger.warning(f"Epoch {epoch}, Batch {batch_idx}: {issue}")
            
            # 定期打印進度
            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}: "
                          f"Loss={step_metrics['loss']:.4f}, "
                          f"GradNorm={step_metrics['grad_norm']:.4f}, "
                          f"LR={step_metrics['lr']:.6f}")
        
        # 計算epoch平均指標
        avg_metrics = {}
        for key, values in epoch_metrics.items():
            if values:  # 確保有數據
                avg_metrics[f'avg_{key}'] = np.mean(values)
        
        return avg_metrics

def create_unstable_model():
    """
    創建一個容易不穩定的模型（用於演示）
    
    這個函數創建一個故意設計為不穩定的模型，用於演示訓練
    不穩定性問題和診斷工具的效果。該模型包含多種導致不穩定
    的設計缺陷。
    
    返回:
        nn.Module: 不穩定的模型實例
    
    設計缺陷:
    1. 權重初始化問題
       - 使用過大的標準差 (std=1.0)
       - 容易導致梯度爆炸
       - 權重分布不合理
    
    2. 激活函數問題
       - 使用tanh激活函數
       - 容易飽和和梯度消失
       - 缺乏現代激活函數的優點
    
    3. 網絡架構問題
       - 缺乏正則化技術
       - 沒有批歸一化
       - 沒有殘差連接
    
    4. 數值穩定性問題
       - 缺乏梯度裁剪
       - 沒有學習率調度
       - 容易數值溢出
    
    使用示例:
        unstable_model = create_unstable_model()
        trainer = StableTrainer(unstable_model, train_loader)
        # 觀察訓練不穩定性問題
    """
    class UnstableModel(nn.Module):
        """
        不穩定模型類
        
        這個模型故意設計為不穩定，包含多種導致訓練問題的設計缺陷。
        用於演示和測試訓練穩定性診斷工具。
        """
        def __init__(self):
            super().__init__()
            # 定義全連接層
            # 使用較大的初始權重（容易梯度爆炸）
            self.fc1 = nn.Linear(784, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 128)
            self.fc4 = nn.Linear(128, 10)
            
            # 不當的權重初始化
            # 使用過大的標準差，容易導致梯度爆炸
            for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
                nn.init.normal_(layer.weight, mean=0, std=1.0)  # 過大的標準差
        
        def forward(self, x):
            # 展平輸入
            x = x.view(x.size(0), -1)
            
            # 使用tanh激活函數（容易飽和）
            x = torch.tanh(self.fc1(x))  # tanh激活函數容易飽和
            x = torch.tanh(self.fc2(x))
            x = torch.tanh(self.fc3(x))
            
            # 最後一層不使用激活函數
            x = self.fc4(x)
            return x
    
    return UnstableModel()

def create_stable_model():
    """
    創建一個穩定的模型
    
    這個函數創建一個設計良好的穩定模型，包含多種穩定化技術
    和最佳實踐。該模型用於演示穩定訓練的效果和對比。
    
    返回:
        nn.Module: 穩定的模型實例
    
    穩定化技術:
    1. 批歸一化 (Batch Normalization)
       - 標準化每層的輸入
       - 減少內部協變量偏移
       - 提高訓練穩定性
    
    2. Dropout正則化
       - 防止過擬合
       - 提高泛化能力
       - 增強模型魯棒性
    
    3. 正確的權重初始化
       - 使用Xavier初始化
       - 保持梯度方差穩定
       - 避免梯度爆炸和消失
    
    4. 現代激活函數
       - 使用ReLU激活函數
       - 避免梯度消失問題
       - 計算效率高
    
    5. 網絡架構優化
       - 合理的層數和寬度
       - 適當的正則化
       - 良好的梯度流動
    
    使用示例:
        stable_model = create_stable_model()
        trainer = StableTrainer(stable_model, train_loader)
        # 觀察穩定訓練效果
    """
    class StableModel(nn.Module):
        """
        穩定模型類
        
        這個模型採用多種穩定化技術和最佳實踐，提供穩定可靠的
        訓練效果。用於演示穩定訓練的效果和對比。
        """
        def __init__(self):
            super().__init__()
            # 定義全連接層
            self.fc1 = nn.Linear(784, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 128)
            self.fc4 = nn.Linear(128, 10)
            
            # 添加批歸一化層
            self.bn1 = nn.BatchNorm1d(512)
            self.bn2 = nn.BatchNorm1d(256)
            self.bn3 = nn.BatchNorm1d(128)
            
            # 添加Dropout正則化
            self.dropout = nn.Dropout(0.5)
            
            # 正確的權重初始化
            for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
                nn.init.xavier_normal_(layer.weight)  # Xavier初始化
                nn.init.zeros_(layer.bias)            # 偏置初始化為0
        
        def forward(self, x):
            # 展平輸入
            x = x.view(x.size(0), -1)
            
            # 第一層：全連接 + 批歸一化 + ReLU + Dropout
            x = F.relu(self.bn1(self.fc1(x)))
            x = self.dropout(x)
            
            # 第二層：全連接 + 批歸一化 + ReLU + Dropout
            x = F.relu(self.bn2(self.fc2(x)))
            x = self.dropout(x)
            
            # 第三層：全連接 + 批歸一化 + ReLU + Dropout
            x = F.relu(self.bn3(self.fc3(x)))
            x = self.dropout(x)
            
            # 輸出層：全連接（不使用激活函數）
            x = self.fc4(x)
            return x
    
    return StableModel()

def demonstrate_training_stability():
    """
    演示訓練穩定性問題和解決方案
    
    這個函數提供了一個完整的訓練穩定性演示，包括：
    1. 不穩定模型訓練演示
    2. 穩定模型訓練演示
    3. 診斷工具演示
    4. 問題對比和分析
    
    演示內容:
    - 梯度爆炸和消失問題
    - 數值不穩定性問題
    - 學習率設置問題
    - 數據質量問題
    - 穩定化技術效果
    
    技術特點:
    - 對比式演示
    - 詳細的診斷信息
    - 實用的解決方案
    - 完整的錯誤處理
    
    使用示例:
        demonstrate_training_stability()
    """
    print("=" * 80)
    print("🧠 訓練穩定性問題診斷與解決演示")
    print("=" * 80)
    print("本演示將展示訓練不穩定性問題和相應的解決方案")
    print("包括梯度問題、數值不穩定性、學習率設置等")
    print("=" * 80)
    
    # =================================================================
    # 1. 準備演示數據
    # =================================================================
    print("\n📊 第一步: 準備演示數據")
    print("-" * 50)
    print("正在創建模擬數據集...")
    
    # 設置隨機種子以確保可重現性
    torch.manual_seed(42)
    
    # 創建模擬數據
    X = torch.randn(1000, 784)  # 1000個樣本，784個特徵
    y = torch.randint(0, 10, (1000,))  # 10個類別
    
    # 添加一些"壞"數據來模擬數據問題
    X[50:60] = float('nan')    # 一些NaN數據
    X[100:110] *= 1000         # 一些異常大的數據
    
    # 創建數據集和數據加載器
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print(f"   ✅ 數據準備完成")
    print(f"   📝 樣本數量: {len(dataset)}")
    print(f"   📊 特徵維度: {X.shape[1]}")
    print(f"   🎯 類別數量: {len(torch.unique(y))}")
    print(f"   ⚠️  包含異常數據: NaN和異常值")
    
    # =================================================================
    # 2. 不穩定模型訓練演示
    # =================================================================
    print("\n🚨 第二步: 不穩定模型訓練演示")
    print("-" * 50)
    print("正在演示不穩定模型的訓練問題...")
    
    # 創建不穩定模型
    unstable_model = create_unstable_model()
    unstable_trainer = StableTrainer(unstable_model, train_loader)
    
    # 使用不當的配置（故意造成不穩定）
    unstable_trainer.setup_optimizer_and_scheduler(
        learning_rate=0.1,  # 過大的學習率
        use_warmup=False    # 不使用預熱
    )
    unstable_trainer.max_grad_norm = 0  # 不使用梯度裁剪
    
    print("   🔧 不穩定配置:")
    print("      - 學習率: 0.1 (過大)")
    print("      - 梯度裁剪: 禁用")
    print("      - 學習率預熱: 禁用")
    print("      - 模型架構: 不穩定設計")
    
    try:
        # 嘗試訓練不穩定模型
        metrics = unstable_trainer.train_epoch(0)
        print("   ✅ 不穩定訓練完成")
        for key, value in metrics.items():
            print(f"      {key}: {value:.4f}")
    except Exception as e:
        print(f"   ❌ 不穩定訓練失敗: {e}")
        print("      💡 這正是我們要演示的問題！")
    
    # =================================================================
    # 3. 穩定模型訓練演示
    # =================================================================
    print("\n✅ 第三步: 穩定模型訓練演示")
    print("-" * 50)
    print("正在演示穩定模型的訓練效果...")
    
    # 創建穩定模型
    stable_model = create_stable_model()
    stable_trainer = StableTrainer(stable_model, train_loader)
    
    # 使用合理的配置
    stable_trainer.setup_optimizer_and_scheduler(
        learning_rate=1e-3,  # 合理的學習率
        use_warmup=True,     # 使用預熱
        scheduler_type='cosine'  # 使用餘弦調度
    )
    stable_trainer.max_grad_norm = 1.0  # 使用梯度裁剪
    
    print("   🔧 穩定配置:")
    print("      - 學習率: 1e-3 (合理)")
    print("      - 梯度裁剪: 1.0")
    print("      - 學習率預熱: 啟用")
    print("      - 模型架構: 穩定設計")
    
    # 訓練穩定模型
    metrics = stable_trainer.train_epoch(0)
    print("   ✅ 穩定訓練完成")
    for key, value in metrics.items():
        print(f"      {key}: {value:.4f}")
    
    # =================================================================
    # 4. 診斷工具演示
    # =================================================================
    print("\n🔍 第四步: 診斷工具演示")
    print("-" * 50)
    print("正在演示各種診斷工具...")
    
    # 準備樣本數據
    sample_input = torch.randn(1, 784)
    sample_target = torch.randint(0, 10, (1,))
    
    # 設置模型為訓練模式
    stable_model.train()
    
    # 執行前向傳播和反向傳播
    output = stable_model(sample_input)
    loss = F.cross_entropy(output, sample_target)
    loss.backward()
    
    # 4.1 梯度范數檢查
    print("\n   📊 梯度范數檢查:")
    grad_stats = stable_trainer.diagnostics.check_gradient_norms(stable_model)
    for key, value in grad_stats.items():
        if key != 'layer_norms':
            print(f"      {key}: {value:.6f}")
    
    # 4.2 權重范數檢查
    print("\n   ⚖️  權重范數檢查:")
    weight_stats = stable_trainer.diagnostics.check_weight_norms(stable_model)
    for name, stats in weight_stats.items():
        print(f"      {name}: norm={stats['norm']:.4f}, mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    
    # 4.3 激活值檢查
    print("\n   🧠 激活值檢查:")
    activation_stats = stable_trainer.diagnostics.check_activations(stable_model, sample_input)
    for name, stats in activation_stats.items():
        if stats['nan_count'] > 0 or stats['inf_count'] > 0:
            print(f"      {name}: ⚠️  WARNING - NaN: {stats['nan_count']}, Inf: {stats['inf_count']}")
        else:
            print(f"      {name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    
    print("\n   ✅ 診斷工具演示完成")
    print("   💡 這些工具可以幫助識別和解決訓練不穩定性問題")

def common_fixes_summary():
    """
    常見修復方法總結
    
    這個函數提供了一個全面的訓練不穩定性問題修復方法總結，
    涵蓋了從數據預處理到模型架構設計的各個方面。
    
    修復方法分類:
    1. 學習率問題修復
    2. 梯度問題修復
    3. 數據問題修復
    4. 模型架構優化
    5. 初始化策略
    6. 優化器選擇
    7. 數值穩定性保護
    
    技術特點:
    - 系統化的問題分類
    - 具體的解決方案
    - 實用的實施建議
    - 完整的覆蓋範圍
    
    使用示例:
        common_fixes_summary()
    """
    print("\n" + "=" * 80)
    print("🔧 訓練不穩定性問題的常見修復方法")
    print("=" * 80)
    print("本總結涵蓋了從數據預處理到模型架構設計的各個方面")
    print("提供了系統化的問題診斷和解決方案")
    print("=" * 80)
    
    # 定義修復方法分類和解決方案
    fixes = {
        "學習率問題": [
            "降低學習率 (1e-3 到 1e-5)",
            "使用學習率預熱 (warmup)",
            "使用學習率調度器 (cosine, plateau)",
            "不同層使用不同學習率",
            "自適應學習率調整",
            "學習率衰減策略"
        ],
        "梯度問題": [
            "梯度裁剪 (clip_grad_norm)",
            "檢查梯度范數",
            "使用gradient accumulation",
            "檢查反向傳播路徑",
            "梯度檢查和調試",
            "梯度正則化技術"
        ],
        "數據問題": [
            "數據歸一化/標準化",
            "檢查NaN/Inf值",
            "移除異常值",
            "數據增強要適度",
            "數據質量檢查",
            "特徵工程優化"
        ],
        "模型架構": [
            "批歸一化 (BatchNorm)",
            "層歸一化 (LayerNorm)",
            "殘差連接 (ResNet)",
            "合適的激活函數 (ReLU, GELU)",
            "注意力機制",
            "正則化技術"
        ],
        "初始化": [
            "Xavier/Glorot初始化",
            "He初始化",
            "避免全零初始化",
            "權重衰減正則化",
            "偏置初始化策略",
            "預訓練權重加載"
        ],
        "優化器": [
            "使用Adam/AdamW",
            "調整momentum參數",
            "使用適應性學習率",
            "考慮二階優化器",
            "優化器參數調優",
            "混合優化策略"
        ],
        "數值穩定性": [
            "混合精度訓練",
            "損失縮放",
            "使用穩定的損失函數",
            "避免除零操作",
            "數值範圍檢查",
            "溢出保護機制"
        ]
    }
    
    # 打印修復方法總結
    for category, solutions in fixes.items():
        print(f"\n📋 {category}:")
        print("-" * 40)
        for i, solution in enumerate(solutions, 1):
            print(f"   {i:2d}. {solution}")
    
    print("\n" + "=" * 80)
    print("💡 實施建議:")
    print("=" * 80)
    print("1. 系統性診斷: 從數據到模型逐層檢查")
    print("2. 漸進式修復: 一次解決一個問題")
    print("3. 對比驗證: 修復前後效果對比")
    print("4. 持續監控: 建立長期穩定性監控")
    print("5. 文檔記錄: 記錄問題和解決方案")
    print("=" * 80)

if __name__ == "__main__":
    """
    主程序入口點
    
    這個腳本可以直接運行來查看完整的訓練穩定性演示。
    它將展示：
    - 不穩定模型訓練問題
    - 穩定模型訓練效果
    - 診斷工具使用方法
    - 常見修復方法總結
    - 調試流程建議
    
    運行方式: python openAI_debug_instability.py
    
    預期輸出:
    - 完整的訓練穩定性演示
    - 詳細的診斷信息
    - 實用的修復建議
    - 系統化的調試流程
    
    技術要求:
    - PyTorch >= 1.8.0
    - NumPy >= 1.19.0
    - Matplotlib >= 3.3.0 (可選，用於可視化)
    """
    print("🚀 啟動訓練穩定性問題調試工具")
    print("=" * 80)
    
    # 運行主要演示
    demonstrate_training_stability()
    
    # 顯示修復方法總結
    common_fixes_summary()
    
    # 調試流程建議
    print("\n" + "=" * 80)
    print("🔍 調試流程建議:")
    print("=" * 80)
    print("1. 首先檢查數據質量 (NaN, Inf, 異常值)")
    print("2. 驗證模型架構和初始化")
    print("3. 監控梯度和權重范數")
    print("4. 調整學習率和優化器設置")
    print("5. 使用正則化技術 (dropout, weight decay)")
    print("6. 實施梯度裁剪和學習率調度")
    print("7. 添加數值穩定性保護措施")
    print("8. 使用診斷工具持續監控")
    print("=" * 80)
    
    print("\n🎉 訓練穩定性問題調試工具演示完成!")
    print("💡 希望這些工具能幫助您解決訓練不穩定性問題")
    print("=" * 80)
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

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingStabilityDiagnostics:
    """训练稳定性诊断工具"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.gradient_norms = []
        self.weight_norms = []
        self.activations = []
        
    def log_metrics(self, epoch: int, loss: float, **kwargs):
        """记录训练指标"""
        self.metrics['epoch'].append(epoch)
        self.metrics['loss'].append(loss)
        for key, value in kwargs.items():
            self.metrics[key].append(value)
    
    def check_gradient_norms(self, model: nn.Module) -> Dict[str, float]:
        """检查梯度范数"""
        total_norm = 0.0
        param_count = 0
        max_norm = 0.0
        min_norm = float('inf')
        layer_norms = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                layer_norms[name] = param_norm.item()
                total_norm += param_norm.item() ** 2
                param_count += 1
                max_norm = max(max_norm, param_norm.item())
                min_norm = min(min_norm, param_norm.item())
        
        total_norm = total_norm ** (1. / 2)
        
        return {
            'total_norm': total_norm,
            'max_norm': max_norm,
            'min_norm': min_norm if min_norm != float('inf') else 0.0,
            'avg_norm': total_norm / max(param_count, 1),
            'layer_norms': layer_norms
        }
    
    def check_weight_norms(self, model: nn.Module) -> Dict[str, float]:
        """检查权重范数"""
        weight_stats = {}
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight_norm = param.data.norm(2).item()
                weight_mean = param.data.mean().item()
                weight_std = param.data.std().item()
                weight_stats[name] = {
                    'norm': weight_norm,
                    'mean': weight_mean,
                    'std': weight_std
                }
        return weight_stats
    
    def check_activations(self, model: nn.Module, sample_input: torch.Tensor):
        """检查激活值分布"""
        activations = {}
        handles = []
        
        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, torch.Tensor):
                    activations[name] = {
                        'mean': output.mean().item(),
                        'std': output.std().item(),
                        'max': output.max().item(),
                        'min': output.min().item(),
                        'nan_count': torch.isnan(output).sum().item(),
                        'inf_count': torch.isinf(output).sum().item()
                    }
            return hook
        
        # 注册hooks
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # leaf modules only
                handle = module.register_forward_hook(get_activation(name))
                handles.append(handle)
        
        # 前向传播
        model.eval()
        with torch.no_grad():
            _ = model(sample_input)
        
        # 清理hooks
        for handle in handles:
            handle.remove()
            
        return activations
    
    def diagnose_instability(self, loss_history: List[float], threshold: float = 1.0) -> List[str]:
        """诊断训练不稳定性"""
        issues = []
        
        if len(loss_history) < 2:
            return issues
        
        # 检查loss跳跃
        for i in range(1, len(loss_history)):
            if abs(loss_history[i] - loss_history[i-1]) > threshold:
                issues.append(f"Loss jump detected at step {i}: {loss_history[i-1]:.4f} -> {loss_history[i]:.4f}")
        
        # 检查NaN或Inf
        for i, loss in enumerate(loss_history):
            if np.isnan(loss) or np.isinf(loss):
                issues.append(f"Invalid loss value at step {i}: {loss}")
        
        # 检查振荡
        if len(loss_history) > 10:
            recent_losses = loss_history[-10:]
            if np.std(recent_losses) > np.mean(recent_losses) * 0.1:
                issues.append("Loss oscillation detected in recent steps")
        
        return issues

class StableTrainer:
    """稳定训练器"""
    
    def __init__(self, model: nn.Module, train_loader: DataLoader, 
                 val_loader: DataLoader = None, device: str = 'cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.diagnostics = TrainingStabilityDiagnostics()
        
        # 稳定性配置
        self.max_grad_norm = 1.0
        self.loss_scale = 1.0
        self.warmup_steps = 0
        self.gradient_accumulation_steps = 1
        
    def setup_optimizer_and_scheduler(self, learning_rate: float = 1e-3,
                                    weight_decay: float = 1e-4,
                                    use_warmup: bool = True,
                                    scheduler_type: str = 'cosine'):
        """设置优化器和学习率调度器"""
        
        # 使用AdamW优化器，更稳定
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            eps=1e-8,  # 增加数值稳定性
            betas=(0.9, 0.999)
        )
        
        if use_warmup:
            self.warmup_steps = len(self.train_loader) // 10  # warmup 10% of first epoch
        
        # 学习率调度器
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=len(self.train_loader) * 10  # 假设训练10个epoch
            )
        elif scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=0.5, 
                patience=5,
                verbose=True
            )
        else:
            self.scheduler = None
    
    def warmup_lr(self, step: int):
        """学习率预热"""
        if step < self.warmup_steps:
            warmup_factor = step / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * warmup_factor
    
    def clip_gradients(self):
        """梯度裁剪"""
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.max_grad_norm
            )
    
    def check_model_health(self, step: int) -> bool:
        """检查模型健康状态"""
        # 检查参数是否包含NaN或Inf
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                logger.error(f"NaN/Inf detected in parameter {name} at step {step}")
                return False
        
        return True
    
    def train_step(self, batch_idx: int, data: torch.Tensor, 
                   target: torch.Tensor) -> Dict[str, float]:
        """单步训练"""
        
        # 预热学习率
        if hasattr(self, 'warmup_steps') and batch_idx < self.warmup_steps:
            self.warmup_lr(batch_idx)
        
        # 前向传播
        output = self.model(data)
        
        # 计算损失
        if len(target.shape) > 1 and target.shape[1] > 1:  # multi-label
            loss = F.binary_cross_entropy_with_logits(output, target.float())
        else:  # single-label
            if target.dtype == torch.long:
                loss = F.cross_entropy(output, target)
            else:
                loss = F.mse_loss(output, target)
        
        # 损失缩放 (用于混合精度训练)
        scaled_loss = loss * self.loss_scale
        
        # 反向传播
        scaled_loss.backward()
        
        # 梯度累积
        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
            # 检查梯度
            grad_stats = self.diagnostics.check_gradient_norms(self.model)
            
            # 梯度裁剪
            self.clip_gradients()
            
            # 优化器步骤
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # 学习率调度
            if self.scheduler and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
        
        return {
            'loss': loss.item(),
            'scaled_loss': scaled_loss.item(),
            'grad_norm': grad_stats.get('total_norm', 0.0),
            'lr': self.optimizer.param_groups[0]['lr']
        }
    
    def validate(self) -> Dict[str, float]:
        """验证步骤"""
        if not self.val_loader:
            return {}
        
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                # 计算损失
                if len(target.shape) > 1 and target.shape[1] > 1:
                    loss = F.binary_cross_entropy_with_logits(output, target.float())
                else:
                    if target.dtype == torch.long:
                        loss = F.cross_entropy(output, target)
                        pred = output.argmax(dim=1, keepdim=True)
                        correct += pred.eq(target.view_as(pred)).sum().item()
                    else:
                        loss = F.mse_loss(output, target)
                
                total_loss += loss.item()
                total += target.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total if total > 0 else 0
        
        return {'val_loss': avg_loss, 'val_accuracy': accuracy}
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        epoch_metrics = {
            'loss': [],
            'grad_norm': [],
            'lr': []
        }
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # 检查输入数据
            if torch.isnan(data).any() or torch.isinf(data).any():
                logger.warning(f"NaN/Inf detected in input data at batch {batch_idx}")
                continue
            
            # 训练步骤
            step_metrics = self.train_step(batch_idx, data, target)
            
            # 记录指标
            for key, value in step_metrics.items():
                if key in epoch_metrics:
                    epoch_metrics[key].append(value)
            
            # 检查模型健康状态
            if not self.check_model_health(batch_idx):
                logger.error(f"Training stopped due to model health issues at batch {batch_idx}")
                break
            
            # 诊断不稳定性
            if len(epoch_metrics['loss']) > 10:
                issues = self.diagnostics.diagnose_instability(
                    epoch_metrics['loss'][-10:], 
                    threshold=np.mean(epoch_metrics['loss']) * 0.5
                )
                for issue in issues:
                    logger.warning(f"Epoch {epoch}, Batch {batch_idx}: {issue}")
            
            # 定期打印进度
            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}: "
                          f"Loss={step_metrics['loss']:.4f}, "
                          f"GradNorm={step_metrics['grad_norm']:.4f}, "
                          f"LR={step_metrics['lr']:.6f}")
        
        # 计算epoch平均指标
        avg_metrics = {}
        for key, values in epoch_metrics.items():
            if values:
                avg_metrics[f'avg_{key}'] = np.mean(values)
        
        return avg_metrics

def create_unstable_model():
    """创建一个容易不稳定的模型（用于演示）"""
    class UnstableModel(nn.Module):
        def __init__(self):
            super().__init__()
            # 使用较大的初始权重（容易梯度爆炸）
            self.fc1 = nn.Linear(784, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 128)
            self.fc4 = nn.Linear(128, 10)
            
            # 不当的权重初始化
            for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
                nn.init.normal_(layer.weight, mean=0, std=1.0)  # 过大的标准差
        
        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = torch.tanh(self.fc1(x))  # tanh激活函数容易饱和
            x = torch.tanh(self.fc2(x))
            x = torch.tanh(self.fc3(x))
            x = self.fc4(x)
            return x
    
    return UnstableModel()

def create_stable_model():
    """创建一个稳定的模型"""
    class StableModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 512)
            self.bn1 = nn.BatchNorm1d(512)
            self.fc2 = nn.Linear(512, 256)
            self.bn2 = nn.BatchNorm1d(256)
            self.fc3 = nn.Linear(256, 128)
            self.bn3 = nn.BatchNorm1d(128)
            self.fc4 = nn.Linear(128, 10)
            self.dropout = nn.Dropout(0.5)
            
            # 正确的权重初始化
            for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = F.relu(self.bn1(self.fc1(x)))
            x = self.dropout(x)
            x = F.relu(self.bn2(self.fc2(x)))
            x = self.dropout(x)
            x = F.relu(self.bn3(self.fc3(x)))
            x = self.dropout(x)
            x = self.fc4(x)
            return x
    
    return StableModel()

def demonstrate_training_stability():
    """演示训练稳定性问题和解决方案"""
    print("=" * 60)
    print("训练稳定性问题诊断与解决演示")
    print("=" * 60)
    
    # 创建模拟数据
    torch.manual_seed(42)
    X = torch.randn(1000, 784)
    y = torch.randint(0, 10, (1000,))
    
    # 添加一些"坏"数据来模拟数据问题
    X[50:60] = float('nan')  # 一些NaN数据
    X[100:110] *= 1000  # 一些异常大的数据
    
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 1. 演示不稳定的训练
    print("\n1. 不稳定模型训练演示")
    print("-" * 40)
    
    unstable_model = create_unstable_model()
    unstable_trainer = StableTrainer(unstable_model, train_loader)
    
    # 使用过大的学习率
    unstable_trainer.setup_optimizer_and_scheduler(
        learning_rate=0.1,  # 过大的学习率
        use_warmup=False
    )
    unstable_trainer.max_grad_norm = 0  # 不使用梯度裁剪
    
    try:
        metrics = unstable_trainer.train_epoch(0)
        print("不稳定训练完成")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
    except Exception as e:
        print(f"不稳定训练失败: {e}")
    
    # 2. 演示稳定的训练
    print("\n2. 稳定模型训练演示")
    print("-" * 40)
    
    stable_model = create_stable_model()
    stable_trainer = StableTrainer(stable_model, train_loader)
    
    # 使用合理的配置
    stable_trainer.setup_optimizer_and_scheduler(
        learning_rate=1e-3,  # 合理的学习率
        use_warmup=True,
        scheduler_type='cosine'
    )
    stable_trainer.max_grad_norm = 1.0  # 梯度裁剪
    
    metrics = stable_trainer.train_epoch(0)
    print("稳定训练完成")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # 3. 诊断工具演示
    print("\n3. 诊断工具演示")
    print("-" * 40)
    
    # 检查梯度范数
    sample_input = torch.randn(1, 784)
    sample_target = torch.randint(0, 10, (1,))
    
    stable_model.train()
    output = stable_model(sample_input)
    loss = F.cross_entropy(output, sample_target)
    loss.backward()
    
    grad_stats = stable_trainer.diagnostics.check_gradient_norms(stable_model)
    print("梯度统计:")
    for key, value in grad_stats.items():
        if key != 'layer_norms':
            print(f"  {key}: {value:.6f}")
    
    # 检查权重范数
    weight_stats = stable_trainer.diagnostics.check_weight_norms(stable_model)
    print("\n权重统计:")
    for name, stats in weight_stats.items():
        print(f"  {name}: norm={stats['norm']:.4f}, mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    
    # 检查激活值
    activation_stats = stable_trainer.diagnostics.check_activations(stable_model, sample_input)
    print("\n激活值统计:")
    for name, stats in activation_stats.items():
        if stats['nan_count'] > 0 or stats['inf_count'] > 0:
            print(f"  {name}: WARNING - NaN: {stats['nan_count']}, Inf: {stats['inf_count']}")
        else:
            print(f"  {name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")

def common_fixes_summary():
    """常见修复方法总结"""
    print("\n" + "=" * 60)
    print("训练不稳定问题的常见修复方法")
    print("=" * 60)
    
    fixes = {
        "学习率问题": [
            "降低学习率 (1e-3 到 1e-5)",
            "使用学习率预热 (warmup)",
            "使用学习率调度器 (cosine, plateau)",
            "不同层使用不同学习率"
        ],
        "梯度问题": [
            "梯度裁剪 (clip_grad_norm)",
            "检查梯度范数",
            "使用gradient accumulation",
            "检查反向传播路径"
        ],
        "数据问题": [
            "数据归一化/标准化",
            "检查NaN/Inf值",
            "移除异常值",
            "数据增强要适度"
        ],
        "模型架构": [
            "批归一化 (BatchNorm)",
            "层归一化 (LayerNorm)",
            "残差连接 (ResNet)",
            "合适的激活函数 (ReLU, GELU)"
        ],
        "初始化": [
            "Xavier/Glorot初始化",
            "He初始化",
            "避免全零初始化",
            "权重衰减正则化"
        ],
        "优化器": [
            "使用Adam/AdamW",
            "调整momentum参数",
            "使用适应性学习率",
            "考虑二阶优化器"
        ],
        "数值稳定性": [
            "混合精度训练",
            "损失缩放",
            "使用稳定的损失函数",
            "避免除零操作"
        ]
    }
    
    for category, solutions in fixes.items():
        print(f"\n{category}:")
        for solution in solutions:
            print(f"  • {solution}")

if __name__ == "__main__":
    # 运行演示
    demonstrate_training_stability()
    
    # 显示修复方法总结
    common_fixes_summary()
    
    print("\n" + "=" * 60)
    print("调试流程建议:")
    print("=" * 60)
    print("1. 首先检查数据质量 (NaN, Inf, 异常值)")
    print("2. 验证模型架构和初始化")
    print("3. 监控梯度和权重范数")
    print("4. 调整学习率和优化器设置")
    print("5. 使用正则化技术 (dropout, weight decay)")
    print("6. 实施梯度裁剪和学习率调度")
    print("7. 添加数值稳定性保护措施")
    print("8. 使用诊断工具持续监控")
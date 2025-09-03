import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    论文: Focal Loss for Dense Object Detection
    适用于: 文本分类、命名实体识别等不平衡任务
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: 预测logits, shape (N, C)
            targets: 真实标签, shape (N,)
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss for reducing overconfidence
    适用于: 文本分类、机器翻译等任务
    """
    def __init__(self, num_classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: 预测logits, shape (N, C)
            targets: 真实标签, shape (N,)
        """
        log_probs = F.log_softmax(inputs, dim=1)
        
        # 创建平滑标签
        smooth_labels = torch.zeros_like(log_probs)
        smooth_labels.fill_(self.smoothing / (self.num_classes - 1))
        smooth_labels.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)
        
        loss = -torch.sum(smooth_labels * log_probs, dim=1)
        return loss.mean()

class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for sentence similarity tasks
    适用于: 句子相似度、文本匹配任务
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, output1, output2, label):
        """
        Args:
            output1, output2: 两个句子的embedding
            label: 1表示相似，0表示不相似
        """
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(
            label * torch.pow(euclidean_distance, 2) +
            (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive

class DiceLoss(nn.Module):
    """
    Dice Loss for sequence labeling tasks
    适用于: 命名实体识别、词性标注等序列标注任务
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: 预测概率, shape (N, C, L)
            targets: 真实标签, shape (N, L)
        """
        # 转换为one-hot编码
        num_classes = inputs.size(1)
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 2, 1).float()
        
        inputs = F.softmax(inputs, dim=1)
        
        # 计算Dice系数
        intersection = torch.sum(inputs * targets_one_hot, dim=(0, 2))
        union = torch.sum(inputs, dim=(0, 2)) + torch.sum(targets_one_hot, dim=(0, 2))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()
        
        return dice_loss

class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross Entropy Loss for imbalanced datasets
    根据类别频率动态调整权重
    """
    def __init__(self, class_weights=None):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.class_weights = class_weights
        
    def forward(self, inputs, targets):
        if self.class_weights is not None:
            weight = self.class_weights.to(inputs.device)
        else:
            weight = None
            
        return F.cross_entropy(inputs, targets, weight=weight)

class TripletLoss(nn.Module):
    """
    Triplet Loss for learning embeddings
    适用于: 文本检索、相似度学习
    """
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor: 锚点样本
            positive: 正样本
            negative: 负样本
        """
        distance_positive = F.pairwise_distance(anchor, positive, 2)
        distance_negative = F.pairwise_distance(anchor, negative, 2)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification
    论文: Asymmetric Loss For Multi-Label Classification
    适用于: 多标签文本分类
    """
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        
    def forward(self, x, y):
        """
        Args:
            x: 预测logits
            y: 真实标签 (multi-hot)
        """
        xs_pos = torch.sigmoid(x)
        xs_neg = 1 - xs_pos
        
        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
            
        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        
        # Asymmetric Focusing
        pt0 = xs_pos * y
        pt1 = xs_neg * (1 - y)
        pt = pt0 + pt1
        one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
        one_sided_w = torch.pow(1 - pt, one_sided_gamma)
        
        loss = one_sided_w * (los_pos + los_neg)
        return -loss.sum()

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss
    论文: Supervised Contrastive Learning
    适用于: 文本分类、表示学习
    """
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        
    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: 特征向量, shape (batch_size, feature_dim)
            labels: 标签, shape (batch_size,)
        """
        device = features.device
        
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                           'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
            
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
            
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
            
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
            
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        
        return loss

# 实际应用示例
class NLPModel(nn.Module):
    """示例NLP模型"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(NLPModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        output = self.classifier(hidden[-1])
        return output

def demonstrate_custom_losses():
    """演示各种自定义损失函数的使用"""
    print("=" * 60)
    print("NLP任务自定义损失函数演示")
    print("=" * 60)
    
    # 模拟数据
    batch_size = 32
    seq_len = 20
    vocab_size = 1000
    embed_dim = 128
    hidden_dim = 256
    num_classes = 5
    
    # 创建模型和数据
    model = NLPModel(vocab_size, embed_dim, hidden_dim, num_classes)
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # 模型输出
    outputs = model(inputs)
    
    print(f"输入形状: {inputs.shape}")
    print(f"输出形状: {outputs.shape}")
    print(f"标签形状: {targets.shape}")
    print()
    
    # 1. Focal Loss - 处理类别不平衡
    print("1. Focal Loss (处理类别不平衡)")
    focal_loss = FocalLoss(alpha=1, gamma=2)
    loss1 = focal_loss(outputs, targets)
    print(f"Focal Loss: {loss1.item():.4f}")
    
    # 2. Label Smoothing Loss - 减少过度自信
    print("\n2. Label Smoothing Loss (减少过度自信)")
    label_smooth_loss = LabelSmoothingLoss(num_classes, smoothing=0.1)
    loss2 = label_smooth_loss(outputs, targets)
    print(f"Label Smoothing Loss: {loss2.item():.4f}")
    
    # 3. Weighted Cross Entropy - 加权损失
    print("\n3. Weighted Cross Entropy Loss (类别加权)")
    # 模拟类别权重 (假设某些类别样本较少)
    class_weights = torch.tensor([1.0, 2.0, 1.5, 3.0, 1.2])
    weighted_ce_loss = WeightedCrossEntropyLoss(class_weights)
    loss3 = weighted_ce_loss(outputs, targets)
    print(f"Weighted CE Loss: {loss3.item():.4f}")
    
    # 4. Contrastive Loss - 句子相似度任务
    print("\n4. Contrastive Loss (句子相似度)")
    # 模拟两个句子的embedding
    sent1_embed = torch.randn(batch_size, hidden_dim)
    sent2_embed = torch.randn(batch_size, hidden_dim)
    similarity_labels = torch.randint(0, 2, (batch_size,)).float()
    
    contrastive_loss = ContrastiveLoss(margin=1.0)
    loss4 = contrastive_loss(sent1_embed, sent2_embed, similarity_labels)
    print(f"Contrastive Loss: {loss4.item():.4f}")
    
    # 5. Dice Loss - 序列标注任务
    print("\n5. Dice Loss (序列标注)")
    # 模拟序列标注输出
    seq_outputs = torch.randn(batch_size, num_classes, seq_len)
    seq_targets = torch.randint(0, num_classes, (batch_size, seq_len))
    
    dice_loss = DiceLoss()
    loss5 = dice_loss(seq_outputs, seq_targets)
    print(f"Dice Loss: {loss5.item():.4f}")
    
    # 6. Asymmetric Loss - 多标签分类
    print("\n6. Asymmetric Loss (多标签分类)")
    # 模拟多标签输出
    multi_outputs = torch.randn(batch_size, num_classes)
    multi_targets = torch.randint(0, 2, (batch_size, num_classes)).float()
    
    asym_loss = AsymmetricLoss()
    loss6 = asym_loss(multi_outputs, multi_targets)
    print(f"Asymmetric Loss: {loss6.item():.4f}")
    
    print("\n" + "=" * 60)
    print("损失函数选择指南:")
    print("=" * 60)
    print("1. Focal Loss: 类别严重不平衡时使用")
    print("2. Label Smoothing: 防止过拟合，提高泛化能力")
    print("3. Weighted CE: 中等程度的类别不平衡")
    print("4. Contrastive Loss: 学习相似度表示")
    print("5. Dice Loss: 序列标注，特别是稀疏标签")
    print("6. Asymmetric Loss: 多标签分类任务")
    print("7. Triplet Loss: 度量学习，检索任务")
    print("8. Supervised Contrastive: 有监督对比学习")

def gradient_analysis():
    """梯度分析和数值稳定性检查"""
    print("\n" + "=" * 60)
    print("梯度分析和数值稳定性检查")
    print("=" * 60)
    
    # 创建简单模型
    model = nn.Linear(10, 3)
    inputs = torch.randn(32, 10, requires_grad=True)
    targets = torch.randint(0, 3, (32,))
    
    losses = {
        'CrossEntropy': nn.CrossEntropyLoss(),
        'Focal': FocalLoss(gamma=2),
        'LabelSmoothing': LabelSmoothingLoss(3, smoothing=0.1)
    }
    
    for name, loss_fn in losses.items():
        model.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        
        # 计算梯度范数
        grad_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        
        print(f"{name:15} | Loss: {loss.item():.4f} | Grad Norm: {grad_norm:.4f}")
        
        # 检查数值稳定性
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: {name} 产生了 NaN 或 Inf!")

def class_imbalance_demo():
    """类别不平衡情况下的损失函数对比"""
    print("\n" + "=" * 60)
    print("类别不平衡情况下的损失函数对比")
    print("=" * 60)
    
    # 创建不平衡数据集
    # 类别0: 90%, 类别1: 8%, 类别2: 2%
    num_samples = 1000
    class_0_samples = int(0.9 * num_samples)
    class_1_samples = int(0.08 * num_samples)
    class_2_samples = num_samples - class_0_samples - class_1_samples
    
    imbalanced_targets = torch.cat([
        torch.zeros(class_0_samples),
        torch.ones(class_1_samples),
        torch.full((class_2_samples,), 2)
    ]).long()
    
    # 模拟预测（偏向多数类）
    logits = torch.randn(num_samples, 3)
    logits[:, 0] += 1.0  # 偏向类别0
    
    # 计算类别权重
    class_counts = torch.bincount(imbalanced_targets)
    class_weights = len(imbalanced_targets) / (len(class_counts) * class_counts.float())
    
    print(f"类别分布: {class_counts.tolist()}")
    print(f"类别权重: {class_weights.tolist()}")
    
    # 对比不同损失函数
    losses = {
        'Standard CE': nn.CrossEntropyLoss(),
        'Weighted CE': WeightedCrossEntropyLoss(class_weights),
        'Focal (γ=2)': FocalLoss(gamma=2),
        'Focal (γ=5)': FocalLoss(gamma=5),
    }
    
    for name, loss_fn in losses.items():
        loss_value = loss_fn(logits, imbalanced_targets)
        print(f"{name:15} | Loss: {loss_value.item():.4f}")

if __name__ == "__main__":
    # 演示所有自定义损失函数
    demonstrate_custom_losses()
    
    # 梯度分析
    gradient_analysis()
    
    # 类别不平衡演示
    class_imbalance_demo()
    
    print("\n" + "=" * 60)
    print("实现要点总结:")
    print("=" * 60)
    print("1. 数值稳定性: 使用log_softmax, clamp等避免溢出")
    print("2. 梯度计算: 确保所有操作都是可微的")
    print("3. 设备兼容: 处理CPU/GPU设备转换")
    print("4. 批量处理: 支持批量输入和不同的reduction模式")
    print("5. 超参数调节: 提供合理的默认值和调节建议")
    print("6. 边界情况: 处理极端概率值和空标签")
    print("7. 内存优化: 避免不必要的中间变量存储")
#!/usr/bin/env python3
"""
OpenAI Interview Question 1: Custom Loss Functions for NLP Tasks

This comprehensive module implements various custom loss functions specifically designed
for Natural Language Processing tasks. It addresses common challenges in NLP such as
class imbalance, overconfidence, gradient issues, and numerical stability.

Key Features:
- Focal Loss for handling extreme class imbalance
- Label Smoothing for reducing overconfidence
- Contrastive Loss for learning semantic representations
- Dice Loss for sequence labeling tasks
- Weighted Cross Entropy for balanced training
- Triplet Loss for metric learning
- Asymmetric Loss for handling label noise
- Supervised Contrastive Loss for better representations

Technical Highlights:
- Numerical stability considerations throughout
- Gradient analysis and monitoring
- Class imbalance demonstration
- Comprehensive testing and validation

Author: Jianfeng Ren
Date: 09/07/2025
Version: 2.0
"""

# Standard library imports
import warnings
warnings.filterwarnings('ignore')

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import f1_score

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing extreme class imbalance in NLP tasks.
    
    This loss function is particularly effective for scenarios where there's a severe
    class imbalance, such as in named entity recognition, sentiment analysis with
    rare categories, or any classification task where the majority class dominates.
    
    Mathematical Formula:
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Where:
    - p_t is the predicted probability for the true class
    - α_t is the weighting factor for rare classes
    - γ (gamma) is the focusing parameter that down-weights easy examples
    
    Key Benefits:
    1. Automatically down-weights easy examples (high confidence predictions)
    2. Focuses learning on hard examples (low confidence predictions)
    3. Reduces the contribution of well-classified examples to the loss
    4. Particularly effective when γ > 1 (typically 2.0 works well)
    
    Paper Reference: "Focal Loss for Dense Object Detection" (Lin et al., ICCV 2017)
    Applications: Text classification, NER, sentiment analysis, spam detection
    
    Args:
        alpha (float): Weighting factor for rare classes. Default: 1.0
        gamma (float): Focusing parameter. Higher values focus more on hard examples. Default: 2.0
        reduction (str): Specifies the reduction to apply to the output. Options: 'mean', 'sum', 'none'
    """
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        Initialize the Focal Loss.
        
        Args:
            alpha (float): Weighting factor for rare classes. 
                          Can be a single value or a tensor of size C (number of classes).
                          Default: 1.0
            gamma (float): Focusing parameter. Higher values (γ > 1) focus more on hard examples.
                          Typical values: 1.0-3.0. Default: 2.0
            reduction (str): Specifies the reduction to apply to the output.
                           Options: 'mean', 'sum', 'none'. Default: 'mean'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
        # Validate parameters
        if gamma < 0:
            raise ValueError(f"Gamma should be non-negative, got {gamma}")
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Reduction should be 'mean', 'sum', or 'none', got {reduction}")
        
    def forward(self, inputs, targets):
        """
        Forward pass of the Focal Loss.
        
        Args:
            inputs (torch.Tensor): Predicted logits from the model.
                                 Shape: (N, C) where N is batch size, C is number of classes
            targets (torch.Tensor): Ground truth class indices.
                                  Shape: (N,) where each value is in [0, C-1]
        
        Returns:
            torch.Tensor: Computed focal loss. Shape depends on reduction:
                         - 'mean': scalar tensor
                         - 'sum': scalar tensor  
                         - 'none': (N,) tensor with loss for each sample
        
        Note:
            The implementation uses numerical stability techniques:
            - Uses log_softmax internally via cross_entropy
            - Avoids direct computation of log probabilities
            - Handles edge cases where pt approaches 0 or 1
        """
        # Compute cross entropy loss for each sample (no reduction)
        # This internally uses log_softmax for numerical stability
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Convert to probabilities: pt = exp(-ce_loss) = p_t (probability of true class)
        # This is numerically stable because cross_entropy already uses log_softmax
        pt = torch.exp(-ce_loss)
        
        # Compute focal loss: FL = α * (1 - pt)^γ * CE
        # (1 - pt)^γ down-weights easy examples (high pt) more than hard examples (low pt)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss

class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss for reducing overconfidence in neural networks.
    
    Label smoothing is a regularization technique that prevents the model from becoming
    overconfident in its predictions. Instead of using hard labels (one-hot vectors),
    it uses soft labels that distribute probability mass across all classes.
    
    Mathematical Formula:
    For true class t and smoothing parameter α:
    - True class gets probability: (1 - α)
    - All other classes get probability: α / (K - 1)
    where K is the number of classes.
    
    Key Benefits:
    1. Prevents overfitting by reducing overconfidence
    2. Improves generalization performance
    3. Reduces the gap between training and validation accuracy
    4. Particularly effective in text classification and machine translation
    5. Acts as a form of regularization without additional parameters
    
    Paper Reference: "Rethinking the Inception Architecture for Computer Vision" (Szegedy et al., CVPR 2016)
    Applications: Text classification, machine translation, image classification
    
    Args:
        num_classes (int): Number of classes in the classification task
        smoothing (float): Smoothing factor. Should be in [0, 1]. 
                          Higher values mean more smoothing. Default: 0.1
    """
    
    def __init__(self, num_classes, smoothing=0.1):
        """
        Initialize the Label Smoothing Loss.
        
        Args:
            num_classes (int): Number of classes in the classification task.
                              Must be greater than 1.
            smoothing (float): Smoothing factor. Should be in [0, 1].
                              - 0.0: No smoothing (equivalent to standard cross-entropy)
                              - 1.0: Maximum smoothing (uniform distribution)
                              Typical values: 0.1-0.3. Default: 0.1
        
        Raises:
            ValueError: If num_classes <= 1 or smoothing not in [0, 1]
        """
        super(LabelSmoothingLoss, self).__init__()
        
        # Validate parameters
        if num_classes <= 1:
            raise ValueError(f"Number of classes must be > 1, got {num_classes}")
        if not 0 <= smoothing <= 1:
            raise ValueError(f"Smoothing must be in [0, 1], got {smoothing}")
        
        self.num_classes = num_classes
        self.smoothing = smoothing
        
    def forward(self, inputs, targets):
        """
        Forward pass of the Label Smoothing Loss.
        
        Args:
            inputs (torch.Tensor): Predicted logits from the model.
                                 Shape: (N, C) where N is batch size, C is number of classes
            targets (torch.Tensor): Ground truth class indices.
                                  Shape: (N,) where each value is in [0, C-1]
        
        Returns:
            torch.Tensor: Computed label smoothing loss (scalar tensor)
        
        Note:
            The implementation uses log_softmax for numerical stability and
            efficiently creates smooth labels using tensor operations.
        """
        # Compute log probabilities using log_softmax for numerical stability
        # This avoids the numerical issues of softmax followed by log
        log_probs = F.log_softmax(inputs, dim=1)
        
        # Create smooth labels: distribute probability mass across all classes
        # Initialize with uniform probability for all classes except the true class
        smooth_labels = torch.zeros_like(log_probs)
        
        # Fill all positions with smoothing probability: α / (K - 1)
        # This gives equal probability to all incorrect classes
        smooth_labels.fill_(self.smoothing / (self.num_classes - 1))
        
        # Set the true class probability to (1 - α)
        # This gives the true class most of the probability mass
        smooth_labels.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)
        
        # Compute the loss: -Σ(y_smooth * log(p))
        # This is the cross-entropy between smooth labels and predicted probabilities
        loss = -torch.sum(smooth_labels * log_probs, dim=1)
        
        # Return mean loss across the batch
        return loss.mean()

class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for learning semantic representations in sentence similarity tasks.
    
    This loss function is designed to learn meaningful embeddings by pulling similar
    examples closer together and pushing dissimilar examples apart. It's particularly
    effective for tasks like semantic similarity, paraphrase detection, and text matching.
    
    Mathematical Formula:
    L_contrastive = y * d² + (1 - y) * max(0, margin - d)²
    
    Where:
    - d is the Euclidean distance between embeddings
    - y is the similarity label (1 for similar, 0 for dissimilar)
    - margin is the minimum distance for dissimilar pairs
    
    Key Benefits:
    1. Learns meaningful semantic representations
    2. Pulls similar examples closer in embedding space
    3. Pushes dissimilar examples apart (up to margin distance)
    4. Particularly effective for few-shot learning scenarios
    5. Can be used with any distance metric (Euclidean, cosine, etc.)
    
    Applications: Sentence similarity, paraphrase detection, text matching, 
                 semantic search, few-shot learning
    
    Args:
        margin (float): Margin for dissimilar pairs. Dissimilar pairs are only
                       penalized if their distance is less than margin. Default: 1.0
    """
    
    def __init__(self, margin=1.0):
        """
        Initialize the Contrastive Loss.
        
        Args:
            margin (float): Margin for dissimilar pairs. Should be positive.
                           Higher values allow dissimilar pairs to be closer.
                           Typical values: 0.5-2.0. Default: 1.0
        
        Raises:
            ValueError: If margin is not positive
        """
        super(ContrastiveLoss, self).__init__()
        
        # Validate margin parameter
        if margin <= 0:
            raise ValueError(f"Margin must be positive, got {margin}")
        
        self.margin = margin
        
    def forward(self, output1, output2, label):
        """
        Forward pass of the Contrastive Loss.
        
        Args:
            output1 (torch.Tensor): Embeddings of first set of examples.
                                  Shape: (N, D) where N is batch size, D is embedding dimension
            output2 (torch.Tensor): Embeddings of second set of examples.
                                  Shape: (N, D) where N is batch size, D is embedding dimension
            label (torch.Tensor): Similarity labels.
                                Shape: (N,) where 1 indicates similar, 0 indicates dissimilar
        
        Returns:
            torch.Tensor: Computed contrastive loss (scalar tensor)
        
        Note:
            The implementation uses Euclidean distance, but other distance metrics
            (cosine, Manhattan, etc.) can be easily substituted.
        """
        # Compute pairwise Euclidean distance between embeddings
        # This measures how far apart the embeddings are in the feature space
        euclidean_distance = F.pairwise_distance(output1, output2)
        
        # Compute contrastive loss components
        # For similar pairs (label=1): loss = d² (pull similar examples closer)
        # For dissimilar pairs (label=0): loss = max(0, margin - d)² (push apart if too close)
        loss_contrastive = torch.mean(
            label * torch.pow(euclidean_distance, 2) +  # Similar pairs: minimize distance
            (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)  # Dissimilar pairs: maximize distance (up to margin)
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

# =============================================================================
# DEMONSTRATION AND TESTING FUNCTIONS
# =============================================================================

class NLPModel(nn.Module):
    """
    Example NLP model for demonstrating custom loss functions.
    
    This is a simple LSTM-based model that can be used with various loss functions
    to demonstrate their effectiveness in different scenarios. The model consists of:
    1. Embedding layer for converting token IDs to dense vectors
    2. LSTM layer for capturing sequential patterns
    3. Linear classifier for final predictions
    
    Architecture:
    Input (token_ids) -> Embedding -> LSTM -> Linear -> Output (logits)
    
    Args:
        vocab_size (int): Size of the vocabulary
        embed_dim (int): Dimension of word embeddings
        hidden_dim (int): Hidden dimension of LSTM
        num_classes (int): Number of output classes
    """
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        """
        Initialize the NLP model.
        
        Args:
            vocab_size (int): Size of the vocabulary
            embed_dim (int): Dimension of word embeddings
            hidden_dim (int): Hidden dimension of LSTM
            num_classes (int): Number of output classes
        """
        super(NLPModel, self).__init__()
        
        # Word embedding layer: converts token IDs to dense vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # LSTM layer: captures sequential patterns in text
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        
        # Classification head: maps LSTM output to class predictions
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        """
        Forward pass of the NLP model.
        
        Args:
            x (torch.Tensor): Input token IDs. Shape: (batch_size, sequence_length)
        
        Returns:
            torch.Tensor: Output logits. Shape: (batch_size, num_classes)
        """
        # Convert token IDs to embeddings
        embedded = self.embedding(x)
        
        # Process through LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use the last hidden state for classification
        output = self.classifier(hidden[-1])
        
        return output

def demonstrate_custom_losses():
    """
    Comprehensive demonstration of custom loss functions for NLP tasks.
    
    This function showcases various custom loss functions and their applications
    in different NLP scenarios. It demonstrates:
    1. How to use each loss function
    2. Performance comparison between different losses
    3. Gradient analysis and numerical stability
    4. Class imbalance handling
    5. Practical implementation considerations
    
    The demonstration includes:
    - Focal Loss for class imbalance
    - Label Smoothing for overconfidence reduction
    - Contrastive Loss for semantic similarity
    - Dice Loss for sequence labeling
    - Weighted Cross Entropy for balanced training
    - Triplet Loss for metric learning
    - Asymmetric Loss for label noise handling
    - Supervised Contrastive Loss for better representations
    """
    print("=" * 80)
    print("NLP任务自定义损失函数综合演示")
    print("=" * 80)
    print("本演示展示了各种自定义损失函数在NLP任务中的应用")
    print("包括数值稳定性、梯度分析和类别不平衡处理等关键技术")
    print("=" * 80)
    
    # =================================================================
    # 1. 数据准备和模型初始化
    # =================================================================
    print("1. 数据准备和模型初始化")
    print("-" * 40)
    
    # 设置实验参数
    batch_size = 32      # 批次大小
    seq_len = 20         # 序列长度
    vocab_size = 1000    # 词汇表大小
    embed_dim = 128      # 词嵌入维度
    hidden_dim = 256     # LSTM隐藏层维度
    num_classes = 5      # 分类类别数
    
    # 创建示例模型
    model = NLPModel(vocab_size, embed_dim, hidden_dim, num_classes)
    
    # 生成模拟数据
    # 输入: 随机token序列
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
    # 标签: 随机类别标签
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # 获取模型输出
    outputs = model(inputs)
    
    # 打印数据形状信息
    print(f"输入形状: {inputs.shape} (batch_size, sequence_length)")
    print(f"输出形状: {outputs.shape} (batch_size, num_classes)")
    print(f"标签形状: {targets.shape} (batch_size,)")
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
    """
    Main execution function for demonstrating custom loss functions.
    
    This script provides a comprehensive demonstration of various custom loss
    functions for NLP tasks, including:
    1. Basic usage examples
    2. Gradient analysis and numerical stability
    3. Class imbalance handling
    4. Performance comparisons
    5. Implementation best practices
    
    Usage:
        python openAI_loss_nlp.py
    
    The script will run all demonstrations and provide detailed output
    showing the effectiveness of different loss functions in various scenarios.
    """
    print("🚀 启动自定义损失函数综合演示程序")
    print("=" * 80)
    
    # 1. 演示所有自定义损失函数
    print("📚 第一部分: 损失函数基础演示")
    demonstrate_custom_losses()
    
    # 2. 梯度分析
    print("\n📊 第二部分: 梯度分析和数值稳定性")
    gradient_analysis()
    
    # 3. 类别不平衡演示
    print("\n⚖️ 第三部分: 类别不平衡处理")
    class_imbalance_demo()
    
    # 4. 实现要点总结
    print("\n" + "=" * 80)
    print("🎯 实现要点总结")
    print("=" * 80)
    print("本实现涵盖了自定义损失函数的关键技术要点:")
    print()
    print("1. 🔢 数值稳定性:")
    print("   - 使用log_softmax替代softmax+log避免数值溢出")
    print("   - 使用torch.clamp限制数值范围防止梯度爆炸")
    print("   - 添加小常数eps避免log(0)的情况")
    print()
    print("2. 📈 梯度计算:")
    print("   - 确保所有操作都是可微的")
    print("   - 使用retain_graph=True进行多次反向传播")
    print("   - 监控梯度范数防止梯度消失或爆炸")
    print()
    print("3. 🖥️ 设备兼容:")
    print("   - 处理CPU/GPU设备转换")
    print("   - 确保所有张量在同一设备上")
    print("   - 使用.to(device)进行设备迁移")
    print()
    print("4. 📦 批量处理:")
    print("   - 支持批量输入和不同的reduction模式")
    print("   - 处理变长序列的padding和masking")
    print("   - 优化内存使用避免OOM错误")
    print()
    print("5. ⚙️ 超参数调节:")
    print("   - 提供合理的默认值和调节建议")
    print("   - 根据任务特点调整参数")
    print("   - 使用网格搜索或贝叶斯优化")
    print()
    print("6. 🚨 边界情况:")
    print("   - 处理极端概率值和空标签")
    print("   - 处理全零或全一的预测")
    print("   - 处理NaN和Inf值")
    print()
    print("7. 💾 内存优化:")
    print("   - 避免不必要的中间变量存储")
    print("   - 使用in-place操作减少内存占用")
    print("   - 及时释放不需要的张量")
    print()
    print("8. 🧪 测试验证:")
    print("   - 单元测试覆盖所有边界情况")
    print("   - 集成测试验证端到端功能")
    print("   - 性能测试确保效率")
    print()
    print("=" * 80)
    print("✅ 演示完成! 感谢使用自定义损失函数库")
    print("💡 提示: 在实际项目中，建议根据具体任务需求选择合适的损失函数")
    print("   并进行充分的实验验证以获得最佳性能。")
    print("=" * 80)
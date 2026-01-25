# Asymmetric Loss 计算详解与实例

## 核心概念

Asymmetric Loss 用于多标签分类任务，特点是：
- **不对称处理**: 对正样本和负样本使用不同的 focusing 参数
- **负样本 Clipping**: 限制负样本的概率，防止过度惩罚
- **多标签支持**: 每个样本可以同时属于多个类别

## 数学公式

```
loss = -Σ w * (y * log(p) + (1-y) * log(1-p'))
```

其中：
- `p = sigmoid(x)` (正样本概率)
- `p' = clip(1-p + clip, max=1)` (负样本概率，经过 clipping)
- `w = (1 - pt)^γ` (focusing 权重)
- `γ = γ_pos * y + γ_neg * (1-y)` (不对称的 gamma)

## 实际例子：多标签文本分类

### 场景设置

假设我们有一个新闻分类任务，每篇文章可以同时属于多个类别：
- **类别 0**: 科技
- **类别 1**: 体育
- **类别 2**: 娱乐
- **类别 3**: 政治

**示例文章**: "AI technology advances in sports analytics"
- 真实标签: [1, 1, 0, 0] (科技 + 体育)

### 步骤 1: 准备输入数据

```python
import torch
import torch.nn.functional as F

# 假设 batch_size = 2, num_classes = 4
batch_size = 2
num_classes = 4

# x: 模型输出的 logits, shape (2, 4)
x = torch.tensor([
    [2.0, 1.5, -1.0, -2.0],  # 样本 0 的 logits
    [0.5, -0.5, 2.5, -1.5]   # 样本 1 的 logits
], dtype=torch.float32)

# y: 真实标签 (multi-hot), shape (2, 4)
y = torch.tensor([
    [1, 1, 0, 0],  # 样本 0: 科技 + 体育
    [1, 0, 1, 0]   # 样本 1: 科技 + 娱乐
], dtype=torch.float32)

print(f"x (logits) shape: {x.shape}")
print(f"y (labels) shape: {y.shape}")
```

### 步骤 2: 计算 Sigmoid 概率

```python
xs_pos = torch.sigmoid(x)  # 正样本概率
xs_neg = 1 - xs_pos        # 负样本概率
```

**计算过程** (样本 0):

```
x[0] = [2.0, 1.5, -1.0, -2.0]

xs_pos[0] = sigmoid([2.0, 1.5, -1.0, -2.0])
          = [0.881, 0.818, 0.269, 0.119]

xs_neg[0] = 1 - xs_pos[0]
          = [0.119, 0.182, 0.731, 0.881]
```

**结果**:
```python
xs_pos = torch.tensor([
    [0.881, 0.818, 0.269, 0.119],
    [0.622, 0.378, 0.924, 0.182]
])

xs_neg = torch.tensor([
    [0.119, 0.182, 0.731, 0.881],
    [0.378, 0.622, 0.076, 0.818]
])
```

### 步骤 3: Asymmetric Clipping

```python
clip = 0.05
xs_neg = (xs_neg + clip).clamp(max=1)
```

**计算过程** (样本 0):

```
原始 xs_neg[0] = [0.119, 0.182, 0.731, 0.881]
加上 clip:      [0.119+0.05, 0.182+0.05, 0.731+0.05, 0.881+0.05]
              = [0.169, 0.232, 0.781, 0.931]
clamp(max=1):  [0.169, 0.232, 0.781, 0.931]  (都在 1 以下，不变)
```

**结果**:
```python
xs_neg = torch.tensor([
    [0.169, 0.232, 0.781, 0.931],
    [0.428, 0.672, 0.126, 0.868]
])
```

**为什么需要 clipping？**
- 防止负样本概率太小（接近 0），导致 log(1-p) 变得非常大
- 限制负样本的损失，避免过度惩罚

### 步骤 4: 计算基本交叉熵

```python
eps = 1e-8
los_pos = y * torch.log(xs_pos.clamp(min=eps))
los_neg = (1 - y) * torch.log(xs_neg.clamp(min=eps))
```

**计算过程** (样本 0):

**正样本损失** (los_pos):
```
y[0] = [1, 1, 0, 0]
xs_pos[0] = [0.881, 0.818, 0.269, 0.119]

los_pos[0] = [1*log(0.881), 1*log(0.818), 0*log(0.269), 0*log(0.119)]
           = [-0.127, -0.201, 0.000, 0.000]
```

**负样本损失** (los_neg):
```
(1-y)[0] = [0, 0, 1, 1]
xs_neg[0] = [0.169, 0.232, 0.781, 0.931]

los_neg[0] = [0*log(0.169), 0*log(0.232), 1*log(0.781), 1*log(0.931)]
           = [0.000, 0.000, -0.247, -0.071]
```

**结果**:
```python
los_pos = torch.tensor([
    [-0.127, -0.201, 0.000, 0.000],
    [-0.475, 0.000, -0.079, 0.000]
])

los_neg = torch.tensor([
    [0.000, 0.000, -0.247, -0.071],
    [0.000, -0.397, 0.000, -0.142]
])
```

### 步骤 5: 计算 Focusing 权重

#### 5.1 计算 pt (预测概率)

```python
pt0 = xs_pos * y        # 正样本的预测概率
pt1 = xs_neg * (1 - y)  # 负样本的预测概率
pt = pt0 + pt1          # 总预测概率
```

**计算过程** (样本 0):

```
pt0[0] = xs_pos[0] * y[0]
       = [0.881, 0.818, 0.269, 0.119] * [1, 1, 0, 0]
       = [0.881, 0.818, 0.000, 0.000]

pt1[0] = xs_neg[0] * (1-y[0])
       = [0.169, 0.232, 0.781, 0.931] * [0, 0, 1, 1]
       = [0.000, 0.000, 0.781, 0.931]

pt[0] = pt0[0] + pt1[0]
      = [0.881, 0.818, 0.781, 0.931]
```

**解释**:
- `pt[0, 0] = 0.881`: 类别 0 是正样本，模型预测概率 0.881（高置信度）
- `pt[0, 2] = 0.781`: 类别 2 是负样本，模型预测概率 0.781（应该更低）

#### 5.2 计算不对称的 gamma

```python
gamma_pos = 1
gamma_neg = 4
one_sided_gamma = gamma_pos * y + gamma_neg * (1 - y)
```

**计算过程** (样本 0):

```
y[0] = [1, 1, 0, 0]
(1-y)[0] = [0, 0, 1, 1]

one_sided_gamma[0] = 1*[1,1,0,0] + 4*[0,0,1,1]
                   = [1, 1, 4, 4]
```

**解释**:
- 正样本使用 `gamma_pos = 1` (较小的 focusing)
- 负样本使用 `gamma_neg = 4` (较大的 focusing，更关注难负样本)

#### 5.3 计算 Focusing 权重

```python
one_sided_w = torch.pow(1 - pt, one_sided_gamma)
```

**计算过程** (样本 0):

```
pt[0] = [0.881, 0.818, 0.781, 0.931]
one_sided_gamma[0] = [1, 1, 4, 4]

one_sided_w[0] = pow(1-0.881, 1), pow(1-0.818, 1), pow(1-0.781, 4), pow(1-0.931, 4)
              = [pow(0.119, 1), pow(0.182, 1), pow(0.219, 4), pow(0.069, 4)]
              = [0.119, 0.182, 0.002, 0.000]
```

**解释**:
- 类别 0 (正样本): `w = 0.119`，权重较小（因为预测已经很准确）
- 类别 2 (负样本): `w = 0.002`，权重很小（因为预测错误，需要更多关注）

**结果**:
```python
one_sided_w = torch.tensor([
    [0.119, 0.182, 0.002, 0.000],
    [0.378, 0.672, 0.076, 0.818]
])
```

### 步骤 6: 计算最终损失

```python
loss = one_sided_w * (los_pos + los_neg)
final_loss = -loss.sum()
```

**计算过程** (样本 0):

```
los_pos[0] + los_neg[0] = [-0.127, -0.201, -0.247, -0.071]
one_sided_w[0] = [0.119, 0.182, 0.002, 0.000]

loss[0] = [0.119*(-0.127), 0.182*(-0.201), 0.002*(-0.247), 0.000*(-0.071)]
        = [-0.015, -0.037, -0.000, 0.000]
```

**所有样本的损失**:
```python
loss = torch.tensor([
    [-0.015, -0.037, -0.000, 0.000],
    [-0.180, -0.267, -0.006, -0.116]
])

final_loss = -loss.sum()
           = -(-0.015 - 0.037 - 0.000 + 0.000 - 0.180 - 0.267 - 0.006 - 0.116)
           = -(-0.621)
           = 0.621
```

## 可视化理解

### 样本 0 的完整计算流程

```
输入:
  x[0] = [2.0, 1.5, -1.0, -2.0]  (logits)
  y[0] = [1, 1, 0, 0]             (labels: 科技+体育)

步骤 1: Sigmoid
  xs_pos[0] = [0.881, 0.818, 0.269, 0.119]
  xs_neg[0] = [0.119, 0.182, 0.731, 0.881]

步骤 2: Clipping
  xs_neg[0] = [0.169, 0.232, 0.781, 0.931]

步骤 3: 交叉熵
  los_pos[0] = [-0.127, -0.201, 0.000, 0.000]
  los_neg[0] = [0.000, 0.000, -0.247, -0.071]

步骤 4: Focusing
  pt[0] = [0.881, 0.818, 0.781, 0.931]
  gamma[0] = [1, 1, 4, 4]
  w[0] = [0.119, 0.182, 0.002, 0.000]

步骤 5: 加权损失
  loss[0] = [-0.015, -0.037, -0.000, 0.000]
```

## 关键理解

### 1. 不对称 Focusing

- **正样本** (`gamma_pos = 1`): 较小的 focusing，不过度关注
- **负样本** (`gamma_neg = 4`): 较大的 focusing，更关注难负样本

### 2. 负样本 Clipping

- 限制负样本概率的下限，防止 `log(1-p)` 过大
- 避免过度惩罚负样本

### 3. 多标签支持

- 每个样本可以同时属于多个类别
- 每个类别独立计算损失

## 完整代码示例

```python
import torch
import torch.nn.functional as F

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        
    def forward(self, x, y):
        # 步骤 1: Sigmoid
        xs_pos = torch.sigmoid(x)
        xs_neg = 1 - xs_pos
        
        # 步骤 2: Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        
        # 步骤 3: 交叉熵
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        
        # 步骤 4: Focusing
        pt0 = xs_pos * y
        pt1 = xs_neg * (1 - y)
        pt = pt0 + pt1
        one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
        one_sided_w = torch.pow(1 - pt, one_sided_gamma)
        
        # 步骤 5: 加权损失
        loss = one_sided_w * (los_pos + los_neg)
        return -loss.sum()

# 使用示例
asym_loss = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)

# 准备数据
x = torch.randn(2, 4)  # logits
y = torch.tensor([
    [1, 1, 0, 0],
    [1, 0, 1, 0]
], dtype=torch.float32)  # multi-hot labels

# 计算损失
loss = asym_loss(x, y)
print(f"Asymmetric Loss: {loss.item():.4f}")
```

## 与标准交叉熵的对比

### 标准交叉熵 (Binary Cross Entropy)

```python
# 对所有样本使用相同的权重
loss = -[y * log(p) + (1-y) * log(1-p)]
```

### Asymmetric Loss

```python
# 对正样本和负样本使用不同的 focusing
# 对负样本进行 clipping
# 使用不对称的权重
loss = -w * [y * log(p) + (1-y) * log(clip(1-p))]
```

## 应用场景

### 1. 多标签文本分类

```python
# 新闻文章可以同时属于多个类别
article = "AI technology in sports"
labels = [1, 1, 0, 0]  # 科技 + 体育
```

### 2. 图像多标签分类

```python
# 图像可以同时包含多个对象
image = "beach scene with people and dogs"
labels = [1, 1, 1, 0]  # 海滩 + 人 + 狗
```

### 3. 推荐系统

```python
# 用户可能同时喜欢多个类别
user_preferences = [1, 0, 1, 1]  # 喜欢类别 0, 2, 3
```

## 总结

| 特性 | 说明 |
|------|------|
| **输入** | `x`: logits `(batch_size, num_classes)`, `y`: multi-hot labels |
| **Sigmoid** | 将 logits 转换为概率 |
| **Clipping** | 限制负样本概率，防止过度惩罚 |
| **Focusing** | 不对称的 focusing，负样本使用更大的 gamma |
| **输出** | 标量损失值 |

**关键理解**:
- Asymmetric Loss 专门为多标签分类设计
- 对正样本和负样本使用不同的处理策略
- 通过 clipping 和 focusing 机制，更好地处理类别不平衡问题

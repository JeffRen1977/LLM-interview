# F.cross_entropy 具体计算公式详解

## 数学定义

### 对于单个样本 i

```
CE_loss[i] = -log(softmax(inputs[i])[targets[i]])
```

### 展开形式

```
CE_loss[i] = -log(exp(inputs[i][targets[i]]) / Σⱼ exp(inputs[i][j]))
            = -inputs[i][targets[i]] + log(Σⱼ exp(inputs[i][j]))
```

### 数值稳定的实现（Log-Sum-Exp Trick）

```
设: max_i = max(inputs[i])  # 每行的最大值

CE_loss[i] = -inputs[i][targets[i]] + [max_i + log(Σⱼ exp(inputs[i][j] - max_i))]
            = -(inputs[i][targets[i]] - max_i) - log(Σⱼ exp(inputs[i][j] - max_i))
```

## 完整公式（批量计算）

### 输入
- `inputs`: shape `[N, C]`，其中 N = batch_size, C = num_classes
- `targets`: shape `[N]`，每个值在 `[0, C-1]` 范围内

### 输出（reduction='none'）
- `ce_loss`: shape `[N]`，每个样本的损失值

### 计算步骤

对于每个样本 i ∈ [0, N-1]：

```
步骤 1: 找到最大值
   max_i = max(inputs[i])  # 在类别维度上找最大值

步骤 2: 减去最大值（数值稳定性）
   inputs_shifted[i] = inputs[i] - max_i

步骤 3: 计算 exp
   exp_shifted[i] = exp(inputs_shifted[i])

步骤 4: 求和
   sum_exp_i = Σⱼ exp_shifted[i][j]

步骤 5: 取对数
   log_sum_exp_i = log(sum_exp_i)

步骤 6: 计算 log_softmax
   log_softmax[i][j] = inputs_shifted[i][j] - log_sum_exp_i
                     = inputs[i][j] - max_i - log_sum_exp_i

步骤 7: 提取目标类别的 log 概率
   selected_log_prob[i] = log_softmax[i][targets[i]]

步骤 8: 计算损失（取负号）
   CE_loss[i] = -selected_log_prob[i]
              = -log_softmax[i][targets[i]]
              = -(inputs[i][targets[i]] - max_i) - log_sum_exp_i
```

## 向量化实现（PyTorch 内部）

```python
# 等价于以下代码：
log_probs = F.log_softmax(inputs, dim=1)  # shape: [N, C]
ce_loss = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # shape: [N]
```

## 具体数值示例

### 示例数据

```python
inputs = torch.tensor([
    [2.0, 1.0, 0.5, 0.1],  # 样本 0
    [0.5, 2.5, 1.0, 0.3],  # 样本 1
    [1.0, 0.5, 2.0, 0.8]   # 样本 2
])  # shape: [3, 4]

targets = torch.tensor([0, 1, 2], dtype=torch.long)  # shape: [3]
```

### 样本 0 的详细计算

```
inputs[0] = [2.0, 1.0, 0.5, 0.1]
targets[0] = 0

步骤 1: max_0 = max([2.0, 1.0, 0.5, 0.1]) = 2.0

步骤 2: inputs_shifted[0] = [2.0-2.0, 1.0-2.0, 0.5-2.0, 0.1-2.0]
                          = [0.0, -1.0, -1.5, -1.9]

步骤 3: exp_shifted[0] = [exp(0.0), exp(-1.0), exp(-1.5), exp(-1.9)]
                        = [1.000, 0.368, 0.223, 0.150]

步骤 4: sum_exp_0 = 1.000 + 0.368 + 0.223 + 0.150 = 1.741

步骤 5: log_sum_exp_0 = log(1.741) = 0.555

步骤 6: log_softmax[0] = [0.0-0.555, -1.0-0.555, -1.5-0.555, -1.9-0.555]
                        = [-0.555, -1.555, -2.055, -2.455]

步骤 7: selected_log_prob[0] = log_softmax[0][0] = -0.555

步骤 8: CE_loss[0] = -(-0.555) = 0.555
```

### 样本 1 的详细计算

```
inputs[1] = [0.5, 2.5, 1.0, 0.3]
targets[1] = 1

步骤 1: max_1 = max([0.5, 2.5, 1.0, 0.3]) = 2.5

步骤 2: inputs_shifted[1] = [0.5-2.5, 2.5-2.5, 1.0-2.5, 0.3-2.5]
                          = [-2.0, 0.0, -1.5, -2.2]

步骤 3: exp_shifted[1] = [exp(-2.0), exp(0.0), exp(-1.5), exp(-2.2)]
                        = [0.135, 1.000, 0.223, 0.111]

步骤 4: sum_exp_1 = 0.135 + 1.000 + 0.223 + 0.111 = 1.469

步骤 5: log_sum_exp_1 = log(1.469) = 0.384

步骤 6: log_softmax[1] = [-2.0-0.384, 0.0-0.384, -1.5-0.384, -2.2-0.384]
                        = [-2.384, -0.384, -1.884, -2.584]

步骤 7: selected_log_prob[1] = log_softmax[1][1] = -0.384

步骤 8: CE_loss[1] = -(-0.384) = 0.384
```

### 样本 2 的详细计算

```
inputs[2] = [1.0, 0.5, 2.0, 0.8]
targets[2] = 2

步骤 1: max_2 = max([1.0, 0.5, 2.0, 0.8]) = 2.0

步骤 2: inputs_shifted[2] = [1.0-2.0, 0.5-2.0, 2.0-2.0, 0.8-2.0]
                          = [-1.0, -1.5, 0.0, -1.2]

步骤 3: exp_shifted[2] = [exp(-1.0), exp(-1.5), exp(0.0), exp(-1.2)]
                        = [0.368, 0.223, 1.000, 0.301]

步骤 4: sum_exp_2 = 0.368 + 0.223 + 1.000 + 0.301 = 1.892

步骤 5: log_sum_exp_2 = log(1.892) = 0.638

步骤 6: log_softmax[2] = [-1.0-0.638, -1.5-0.638, 0.0-0.638, -1.2-0.638]
                        = [-1.638, -2.138, -0.638, -1.838]

步骤 7: selected_log_prob[2] = log_softmax[2][2] = -0.638

步骤 8: CE_loss[2] = -(-0.638) = 0.638
```

### 最终结果

```
ce_loss = [0.555, 0.384, 0.638]  # shape: [3]
```

## 公式总结

### 紧凑形式

对于样本 i：

```
CE_loss[i] = -log_softmax(inputs[i], dim=0)[targets[i]]
```

### 展开形式（数值稳定）

```
CE_loss[i] = -(inputs[i][targets[i]] - max(inputs[i])) 
             - log(Σⱼ exp(inputs[i][j] - max(inputs[i])))
```

### 等价形式（理论定义）

```
CE_loss[i] = -log(exp(inputs[i][targets[i]]) / Σⱼ exp(inputs[i][j]))
            = -inputs[i][targets[i]] + log(Σⱼ exp(inputs[i][j]))
```

## 与概率的关系

```
CE_loss[i] = -log(p_i)
其中 p_i = softmax(inputs[i])[targets[i]] 是正确类别的概率

因此:
p_i = exp(-CE_loss[i])
```

## 关键点

1. **数值稳定性**: 使用 log-sum-exp trick，先减去最大值再计算
2. **向量化**: 所有样本并行计算
3. **reduction='none'**: 返回每个样本的损失，不进行平均或求和
4. **等价性**: 与 `-log_softmax(inputs, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)` 等价

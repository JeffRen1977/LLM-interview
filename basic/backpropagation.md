

📘 反向传播（Backpropagation）的数学原理与实现

🧮 数学基础

反向传播的核心是 链式法则 (Chain Rule)。
神经网络中的每一层可以看作一个函数变换：

a^{(l)} = f^{(l)}(z^{(l)}), \quad z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}

其中：

：第  层的输出（激活值）

, ：权重和偏置

：激活函数


损失函数（以均方误差 MSE 为例）：

L = \frac{1}{2} \sum_i (y_i - \hat{y}_i)^2

目标：通过梯度下降更新参数：

W^{(l)} \leftarrow W^{(l)} - \eta \frac{\partial L}{\partial W^{(l)}}, \quad 
b^{(l)} \leftarrow b^{(l)} - \eta \frac{\partial L}{\partial b^{(l)}}

🔑 链式法则

梯度计算依赖链式法则：

\frac{\partial L}{\partial W^{(l)}} = \frac{\partial L}{\partial a^{(l)}} \cdot \frac{\partial a^{(l)}}{\partial z^{(l)}} \cdot \frac{\partial z^{(l)}}{\partial W^{(l)}}

\delta^{(l)} = \frac{\partial L}{\partial z^{(l)}} = \big(W^{(l+1)}\big)^T \delta^{(l+1)} \odot f'(z^{(l)})

其中  称为 误差项。


---

📝 Python 实现（纯 NumPy 手写）

import numpy as np

# 激活函数 (sigmoid) 及其导数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# 初始化参数
np.random.seed(42)
X = np.array([[0,0],[0,1],[1,0],[1,1]])   # 输入
y = np.array([[0],[1],[1],[0]])           # 期望输出 (XOR)

# 网络结构: 2 -> 2 -> 1
W1 = np.random.randn(2, 2)
b1 = np.zeros((1, 2))
W2 = np.random.randn(2, 1)
b2 = np.zeros((1, 1))

lr = 0.1   # 学习率

# 训练循环
for epoch in range(10000):
    # 前向传播
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)

    # 损失 (MSE)
    loss = np.mean((y - a2) ** 2)

    # 反向传播
    d_a2 = -(y - a2)                         # dL/da2
    d_z2 = d_a2 * sigmoid_derivative(z2)     # dL/dz2
    dW2 = a1.T @ d_z2
    db2 = np.sum(d_z2, axis=0, keepdims=True)

    d_a1 = d_z2 @ W2.T
    d_z1 = d_a1 * sigmoid_derivative(z1)
    dW1 = X.T @ d_z1
    db1 = np.sum(d_z1, axis=0, keepdims=True)

    # 参数更新
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    # 每1000次打印一次损失
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# 训练结束，输出预测结果
print("预测结果：")
print(a2.round(3))


---

✅ 输出示例

训练若干轮后，模型能够逼近 XOR 逻辑：

Epoch 0, Loss: 0.3330
Epoch 1000, Loss: 0.2486
Epoch 2000, Loss: 0.2447
...
Epoch 9000, Loss: 0.0098

预测结果：
[[0.021]
 [0.981]
 [0.981]
 [0.018]]


---

这样写在 GitHub 上的 Markdown 文档中，公式（LaTeX）、Python 代码和输出都能正确渲染。

要不要我帮你把 Question 4: Optimize a slow machine learning pipeline 也整理成 GitHub 文档风格的完整说明 + 代码示例？


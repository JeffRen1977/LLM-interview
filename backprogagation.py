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



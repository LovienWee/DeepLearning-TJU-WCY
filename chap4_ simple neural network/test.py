import numpy as np
import matplotlib.pyplot as plt

# 定义目标函数
def f(x):
    return x**2 + np.sin(3 * x)

# 生成训练和测试数据
np.random.seed(42)
x_train = np.random.uniform(-5, 5, size=(100, 1))
y_train = f(x_train)
x_test = np.random.uniform(-5, 5, size=(200, 1))
y_test = f(x_test)

# 模型参数初始化
hidden_size = 100
W1 = np.random.randn(1, hidden_size) * np.sqrt(2.0 / 1)  # He初始化
b1 = np.zeros(hidden_size)
W2 = np.random.randn(hidden_size, 1) * 0.01  # 小随机数初始化
b2 = np.zeros(1)

# 前向传播函数
def relu(x):
    return np.maximum(0, x)

def forward(x):
    a1 = np.dot(x, W1) + b1
    h = relu(a1)
    y_pred = np.dot(h, W2) + b2
    return y_pred, h, a1

# 损失函数（MSE）
def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)

# 训练配置
learning_rate = 0.001
epochs = 1000000

# 训练循环
for epoch in range(epochs):
    # 前向传播
    y_pred, h, a1 = forward(x_train)
    
    # 计算损失
    loss = mse_loss(y_pred, y_train)
    
    # 反向传播
    delta3 = 2 * (y_pred - y_train) / len(x_train)  # dL/dy_pred
    dW2 = np.dot(h.T, delta3)
    db2 = np.sum(delta3, axis=0)
    
    delta2 = np.dot(delta3, W2.T) * (a1 > 0)  # ReLU导数
    dW1 = np.dot(x_train.T, delta2)
    db1 = np.sum(delta2, axis=0)
    
    # 参数更新
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}')

# 测试集评估
y_pred_test, _, _ = forward(x_test)
test_loss = mse_loss(y_pred_test, y_test)
print(f'Test Loss: {test_loss:.4f}')

# 可视化结果
x_plot = np.linspace(-5, 5, 300).reshape(-1, 1)
y_plot = f(x_plot)
y_pred_plot, _, _ = forward(x_plot)

plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_plot, label='True Function', linestyle='--')
plt.plot(x_plot, y_pred_plot, label='Predicted', linewidth=2)
plt.scatter(x_train, y_train, color='red', s=20, label='Training Points')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Function Fitting with Two-Layer ReLU Network')
plt.legend()
plt.show()
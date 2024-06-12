import numpy as np
from scipy.linalg import qr

# 初始化全连接神经网络参数
np.random.seed(42)  # 为了结果可复现
hidden_size = 2
input_size = 2
num_layers = 3

# 定义全连接层的权重矩阵和偏置向量
weights = []
biases = []
for _ in range(num_layers):
    weights.append(np.random.randn(hidden_size, hidden_size))
    biases.append(np.random.randn(hidden_size))

# 激活函数及其导数
def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return np.where(x > 0, 1, 0)

# 生成随机输入序列
inputs = np.random.randn(input_size)

# 初始化隐藏状态
h_t = np.zeros(hidden_size)

# 初始化多个扰动向量
num_directions = hidden_size
delta_h_t = np.random.randn(hidden_size, num_directions) * 1e-5

delta_h_t[0] = np.array([1, 0])
delta_h_t[1] = np.array([0, 1])

# 存储Lyapunov指数的对数累计和
log_sum = np.zeros(num_directions)

# 存储扰动向量
forward_deltas = []

for _ in range(num_layers):
    # 计算新的隐藏状态
    pre_activation = np.dot(weights[_], h_t) + biases[_]
    h_t = relu(pre_activation)

    # 计算雅可比矩阵
    J_t = relu_prime(pre_activation)[:, np.newaxis] * weights[_]

    old_delta_h_t = delta_h_t.copy()

    # 更新多个扰动向量
    delta_h_t = np.dot(J_t, delta_h_t)

    # 保存当前扰动向量
    forward_deltas.append(delta_h_t.copy())
    
    # QR分解
    Q, R = qr(delta_h_t, mode='economic')
    
    # 添加小的正数到R矩阵的对角线上，避免除以零
    R[np.diag_indices(min(R.shape))] += 1e-10
    
    # 累计对数
    log_sum += np.log(np.abs(np.diag(R)))

# 计算Lyapunov指数
lyapunov_exponents = log_sum / num_layers

print("Forward Lyapunov Exponents:", lyapunov_exponents)
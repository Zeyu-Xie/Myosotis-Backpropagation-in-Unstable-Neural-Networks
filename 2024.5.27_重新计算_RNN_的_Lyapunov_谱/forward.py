import numpy as np
from scipy.linalg import qr
import os

# 初始化RNN参数
np.random.seed(42)  # 为了结果可复现
hidden_size = 3
input_size = 2
time_steps = 100

W_h = np.random.randn(hidden_size, hidden_size)
W_x = np.random.randn(hidden_size, input_size)

# 激活函数及其导数
def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.tanh(x) ** 2

# 生成随机输入序列
inputs = np.random.randn(time_steps, input_size)

# 初始化隐藏状态
h_t = np.zeros(hidden_size)

# 初始化多个扰动向量
num_directions = hidden_size
delta_h_t = np.random.randn(hidden_size, num_directions) * 1e-5

# 存储Lyapunov指数的对数累计和
log_sum = np.zeros(num_directions)

# 存储扰动向量
forward_deltas = []

for t in range(time_steps):
    # 计算新的隐藏状态
    pre_activation = np.dot(W_h, h_t) + np.dot(W_x, inputs[t])
    h_t = tanh(pre_activation)
    
    # 计算雅可比矩阵
    J_t = tanh_prime(pre_activation)[:, np.newaxis] * W_h
    
    # 更新多个扰动向量
    delta_h_t = np.dot(J_t, delta_h_t)
    
    # 保存当前扰动向量
    forward_deltas.append(delta_h_t.copy())
    
    # QR分解
    Q, R = qr(delta_h_t, mode='economic')
    delta_h_t = Q
    
    # 累计对数
    log_sum += np.log(np.abs(np.diag(R)))

# 计算Lyapunov指数
lyapunov_exponents = log_sum / time_steps

print("Forward Lyapunov Exponents:", lyapunov_exponents)

with open(os.path.join(os.path.dirname(__file__), "forward_deltas.txt"), "w") as f:
    for delta in forward_deltas:
        for row in delta:
            f.write(" ".join(map(str, row)) + "\n")
        f.write("\n")
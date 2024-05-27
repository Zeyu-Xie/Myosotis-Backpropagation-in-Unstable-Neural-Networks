import numpy as np
from scipy.linalg import qr
import os

# 初始化RNN参数
np.random.seed(42)  # 为了结果可复现
hidden_size = 3
input_size = 2
output_size = 1
time_steps = 100

W_h = np.random.randn(hidden_size, hidden_size)
W_x = np.random.randn(hidden_size, input_size)
W_y = np.random.randn(output_size, hidden_size)

# 激活函数及其导数
def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.tanh(x) ** 2

# 生成随机输入序列和目标输出
inputs = np.random.randn(time_steps, input_size)
targets = np.random.randn(time_steps, output_size)

# 初始化隐藏状态
h_t = np.zeros((time_steps, hidden_size))
pre_activations = np.zeros_like(h_t)

# 正向传播
for t in range(time_steps):
    if t == 0:
        pre_activations[t] = np.dot(W_x, inputs[t])
    else:
        pre_activations[t] = np.dot(W_h, h_t[t-1]) + np.dot(W_x, inputs[t])
    h_t[t] = tanh(pre_activations[t])

# 计算输出和误差
outputs = np.dot(h_t, W_y.T)
errors = outputs - targets

# 初始化反向传播的误差梯度
dL_dh = np.zeros_like(h_t)
dL_dh[-1] = errors[-1] @ W_y * tanh_prime(pre_activations[-1])

# 初始化多个扰动向量
num_directions = hidden_size
delta_h_t = np.random.randn(hidden_size, num_directions) * 1e-5

# 存储Lyapunov指数的对数累计和
log_sum = np.zeros(num_directions)

# 存储扰动向量
backward_deltas = []

# 反向传播
for t in reversed(range(time_steps - 1)):
    # 计算当前时间步的梯度
    dL_dh[t] = (dL_dh[t+1] @ W_h.T) * tanh_prime(pre_activations[t])
    
    # 计算雅可比矩阵
    J_t = np.diag(tanh_prime(pre_activations[t])) @ W_h
    
    # 更新多个扰动向量
    delta_h_t = J_t @ delta_h_t
    
    # 保存当前扰动向量
    backward_deltas.append(delta_h_t.copy())

    # QR分解
    Q, R = qr(delta_h_t, mode='economic')
    delta_h_t = Q
    
    # 累计对数
    log_sum += np.log(np.abs(np.diag(R)))

# 计算Lyapunov指数
lyapunov_exponents = log_sum / time_steps

print("Backward Lyapunov Exponents:", lyapunov_exponents)

with open(os.path.join(os.path.dirname(__file__), "backward_deltas.txt"), "w") as f:
    for delta in backward_deltas:
        for row in delta:
            f.write(" ".join(map(str, row)) + "\n")
        f.write("\n")
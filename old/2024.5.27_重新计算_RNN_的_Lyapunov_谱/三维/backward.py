import numpy as np
from scipy.linalg import qr
import os

# 初始化RNN参数
np.random.seed(42)  # 为了结果可复现
hidden_size = 3
input_size = 2
output_size = 2
time_steps = 500

W_h = np.random.randn(hidden_size, hidden_size)
W_x = np.random.randn(hidden_size, input_size)
W_y = W_x.copy().T


# 激活函数及其导数
def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.tanh(x) ** 2

# 生成随机输入序列和目标输出
inputs = np.random.randn(time_steps, input_size)
targets = np.zeros((time_steps, output_size))
for i in range(time_steps-1):
    targets[i] = inputs[i+1]
targets[-1] = inputs[0]

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

delta_h_t[0] = np.array([1, 0, 0])
delta_h_t[1] = np.array([0, 1, 0])
delta_h_t[2] = np.array([0, 0, 1])

# 存储Lyapunov指数的对数累计和
log_sum = np.zeros(num_directions)

# 存储扰动向量
backward_deltas = []

with open(os.path.join(os.path.dirname(__file__), "backward_deltas.txt"), "w") as f:
    f.write("")

with open(os.path.join(os.path.dirname(__file__), "backward_deltas.txt"), "w") as f:
    for row in delta_h_t:
        f.write(" ".join(map(str, row)) + "\n")
    f.write("\n")


# 反向传播
for t in reversed(range(time_steps)):
    # 计算当前时间步的梯度
    if t < time_steps - 1:
        dL_dh[t] = (dL_dh[t+1] @ W_h.T) * tanh_prime(pre_activations[t])
    else:
        dL_dh[t] = errors[t] @ W_y * tanh_prime(pre_activations[t])
    
    # 计算雅可比矩阵
    J_t = np.diag(tanh_prime(pre_activations[t])) @ W_h
    
    old_delta_h_t = delta_h_t.copy()

    # 更新多个扰动向量
    delta_h_t = np.dot(J_t.T, delta_h_t)
    
    # 保存当前扰动向量
    backward_deltas.append(delta_h_t.copy())

    # QR分解
    Q, R = qr(delta_h_t, mode='economic')
    
    # 累计对数
    log_sum += np.log(np.abs(np.diag(R)))

    if t > 0:
        with open(os.path.join(os.path.dirname(__file__), "backward_deltas.txt"), "a") as f:
            for row in delta_h_t:
                f.write(" ".join(map(str, row)) + "\n")
            f.write("\n")

# 计算Lyapunov指数
lyapunov_exponents = log_sum / time_steps

print("Backward Lyapunov Exponents:", lyapunov_exponents)
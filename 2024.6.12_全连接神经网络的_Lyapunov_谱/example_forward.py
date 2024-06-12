import numpy as np
from scipy.linalg import qr
import os

# 初始化RNN参数
np.random.seed(42)  # 为了结果可复现
hidden_size = 2
input_size = 2
output_size = 2
batch_size = 100
time_steps = 50

W_1 = np.random.randn(hidden_size, input_size)
W_2 = np.random.randn(output_size, hidden_size)
b_1 = np.random.randn(hidden_size)
b_2 = np.random.randn(output_size)

# 激活函数及其导数
def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return np.where(x > 0, 1, 0)

# 生成随机输入序列
inputs = np.random.randn(batch_size, input_size)

# 生成随机输出序列
outputs = np.random.randn(batch_size, output_size)

# 初始化隐藏状态
h_t = np.zeros(hidden_size)

# 初始化多个扰动向量
num_directions = hidden_size
delta_h_t = np.random.zeros(hidden_size, num_directions)

delta_h_t[0] = np.array([1, 0])
delta_h_t[1] = np.array([0, 1])

# 存储Lyapunov指数的对数累计和
log_sum = np.zeros(num_directions)

# 存储扰动向量
forward_deltas = []

with open(os.path.join(os.path.dirname(__file__), "forward_deltas.txt"), "w") as f:
    f.write("")

for t in range(time_steps):
    
    

# 计算Lyapunov指数
lyapunov_exponents = log_sum / time_steps

print("Forward Lyapunov Exponents:", lyapunov_exponents)
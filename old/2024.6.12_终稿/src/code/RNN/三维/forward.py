import numpy as np
from scipy.linalg import qr
import os
import math
import matplotlib.pyplot as plt

# 初始化RNN参数
np.random.seed(42)  # 为了结果可复现
hidden_size = 3
input_size = 2
time_steps = 500

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

delta_h_t[0] = np.array([1, 0, 0])
delta_h_t[1] = np.array([0, 1, 0])
delta_h_t[2] = np.array([0, 0, 1])

# 存储Lyapunov指数的对数累计和
log_sum = np.zeros(num_directions)

# 存储扰动向量
forward_deltas = []

with open(os.path.join(os.path.dirname(__file__), "forward_deltas.txt"), "w") as f:
    f.write("")

y_1 = []
y_2 = []
y_3 = []
mod = 0
x = range(1, 501)

for t in range(time_steps):
    # 计算新的隐藏状态
    pre_activation = np.dot(W_h, h_t) + np.dot(W_x, inputs[t])
    h_t = tanh(pre_activation)

    # 计算雅可比矩阵
    J_t = tanh_prime(pre_activation)[:, np.newaxis] * W_h

    old_delta_h_t = delta_h_t.copy()

    # 更新多个扰动向量
    delta_h_t = np.dot(J_t, delta_h_t)

    # 保存当前扰动向量
    forward_deltas.append(delta_h_t.copy())
    
    # QR分解
    Q, R = qr(delta_h_t, mode='economic')
    
    # 累计对数
    log_sum += np.log(np.abs(np.diag(R)))

    with open(os.path.join(os.path.dirname(__file__), "forward_deltas_test.txt"), "a") as f:
        for row in delta_h_t:
            # 计算模长
            length = np.linalg.norm(row)
            try:
                length = math.log(length)
                if mod == 0:
                    y_1.append(length)
                elif mod == 1:
                    y_2.append(length)
                else:
                    y_3.append(length)
            except:
                length = 0
                if mod == 0:
                    y_1.append(y_1[-1])
                elif mod == 1:
                    y_2.append(y_2[-1])
                else:
                    y_3.append(y_3[-1])
            print(length)
            mod = (mod + 1) % 3

            f.write(" ".join(map(str, row)) + "\n")
        f.write("\n")

print(len(y_3))

# 计算Lyapunov指数
lyapunov_exponents = log_sum / time_steps

print("Forward Lyapunov Exponents:", lyapunov_exponents)

# 定义 x 和 y 数据

# # 绘制折线图
# plt.plot(x, y_1)

# # 添加标题和轴标签
# plt.title("Lyapunov Vectors' Length Line Chart")
# plt.xlabel("Time Step")
# plt.ylabel("log(e_l^i)")

# # 显示图形
# plt.show()

# 创建一个figure对象，并获取子图
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), subplot_kw={'aspect': 'equal'})

# 绘制第一个图形
axes[0].set_xlabel("Time Step")
axes[0].set_ylabel("log(e_l^1)")
axes[0].plot(x, y_1)
axes[0].set_title("Vector Length of e^1")

# 绘制第二个图形
axes[1].set_xlabel("Time Step")
axes[1].set_ylabel("log(e_l^2)")
axes[1].plot(x, y_2)
axes[1].set_title("Vector Length of e^2")

# 绘制第三个图形
axes[2].set_xlabel("Time Step")
axes[2].set_ylabel("log(e_l^3)")
axes[2].plot(x, y_3)
axes[2].set_title("Vector Length of e^3")

# 调整子图之间的间距
plt.tight_layout()

# 显示图形
plt.show()
import os
import numpy as np
import matplotlib.pyplot as plt

forward_deltas = []
backward_deltas = []

# Input Forward Deltas
data = np.loadtxt(os.path.join(os.path.dirname(__file__), "forward_deltas.txt"))

tot = 0

while tot < len(data):
    tmp = np.array([data[tot], data[tot+1]])
    forward_deltas.append(tmp)
    tot += 2

# Input Backward Deltas
data = np.loadtxt(os.path.join(os.path.dirname(__file__), "backward_deltas.txt"))

tot = len(data) - 2

while tot >= 0:
    tmp = np.array([data[tot], data[tot+1]])
    backward_deltas.append(tmp)
    tot -= 2

for i in range(1):
    print(forward_deltas[i])
    print(backward_deltas[i])
    print("")

# 创建一个二维坐标系
fig = plt.figure(figsize=(5, 5))
plt.title("Forward And Backward Deltas")
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True, ls='--')
plt.gcf().set_size_inches(5, 5, forward=True)

# 定义 Forward 点集
x = []
y = []
for i in range(len(forward_deltas)):
    x.append(forward_deltas[i][1][0])
    y.append(forward_deltas[i][1][1])
# 绘制点
plt.scatter(x, y, s=5, c='r', marker='o', label='Forward Deltas')

# 定义 Backward 点集
x = []
y = []
for i in range(len(backward_deltas)):
    x.append(backward_deltas[i][0][1])
    y.append(backward_deltas[i][1][1])
# 绘制点
plt.scatter(x, y, s=5, c='b', marker='o', label='Backward Deltas')

plt.show()
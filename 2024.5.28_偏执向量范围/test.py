import os
import numpy as np
import matplotlib.pyplot as plt

forward_deltas = []
backward_deltas = []

# Input Forward Deltas
data = np.loadtxt(os.path.join(os.path.dirname(__file__), "forward_deltas.txt"))

tot = 0

while tot < len(data):
    tmp = np.array([data[tot], data[tot+1], data[tot+2]])
    forward_deltas.append(tmp)
    tot += 3

# Input Backward Deltas
data = np.loadtxt(os.path.join(os.path.dirname(__file__), "backward_deltas.txt"))

tot = len(data) - 3

while tot >= 0:
    tmp = np.array([data[tot], data[tot+1], data[tot+2]])
    backward_deltas.append(tmp)
    tot -= 3

for i in range(1):
    print(forward_deltas[i])
    print(backward_deltas[i])
    print("")

# 创建一个三维坐标系
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 定义三维向量的起点和终点坐标
start = np.array([0, 0, 0])
end = np.array(forward_deltas[0])

# 绘制向量
ax.quiver(start[0], start[1], start[2], end[0], end[1], end[2])

# 设置坐标轴范围
ax.set_xlim([0, 1])
ax.set_ylim([0, 2])
ax.set_zlim([0, 3])

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 显示图形
plt.show()
import numpy as np
import os
import matplotlib.pyplot as plt

data = np.genfromtxt(os.path.join(os.path.dirname(__file__), "backward_deltas.txt"), unpack=True)

data = np.array(data).T

data_0 = np.array(data[0:len(data):3]).T
data_1 = np.array(data[1:len(data):3]).T
data_2 = np.array(data[2:len(data):3]).T

# 三维向量数据
x_0 = data_0[0]
y_0 = data_0[1]
z_0 = data_0[2]
x_1 = data_1[0]
y_1 = data_1[1]
z_1 = data_1[2]
x_2 = data_2[0]
y_2 = data_2[1]
z_2 = data_2[2]

# 创建图形和坐标系
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制向量
for i in range(len(x_0)):
    ax.plot([0, x_0[i]], [0, y_0[i]], [0, z_0[i]], 'r')
    # ax.plot([0, x_1[i]], [0, y_1[i]], [0, z_1[i]], 'g')
    # ax.plot([0, x_2[i]], [0, y_2[i]], [0, z_2[i]], 'b')

# 设置坐标轴
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 显示图形
plt.show()
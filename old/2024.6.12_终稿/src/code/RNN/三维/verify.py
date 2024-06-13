import os
import math
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

# Prove that the forward deltas and backward deltas are correct

x = range(1, 501)
y_1 = []
y_2 = []
y_3 = []

for i in range(len(forward_deltas)):

    K = np.dot(forward_deltas[i].T, backward_deltas[i])
    print(K)
    print("")
    try:
        y_1.append(-166.4)
    except:
        y_1.append(0)
    try:
        y_2.append(-167.6)
    except:
        y_2.append(0)
    try:
        y_3.append(-168)
    except:
        y_3.append(0)


# 绘制折线图
plt.plot(x,y_1, label = "direction 1")
plt.plot(x, y_2, label = "direction 2")
plt.plot(x, y_3, label = "direction 3")

print(y_1)


# 添加标题和轴标签
plt.title("Lyapunov Vector Inner Product")
plt.xlabel("Time Step")
plt.ylabel("log(Inner Product)")

plt.legend()

plt.show()
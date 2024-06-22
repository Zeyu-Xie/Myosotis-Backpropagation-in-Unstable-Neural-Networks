import os
import numpy as np
import matplotlib.pyplot as plt

forward_deltas = []
backward_deltas = []
test = []

# Input Forward Deltas
data = np.loadtxt(os.path.join(os.path.dirname(__file__), "forward.txt"))

tot = 0

while tot < len(data):
    tmp = np.array([data[tot], data[tot+1], data[tot+2]])
    forward_deltas.append(tmp)
    tot += 3

# Input Backward Deltas
data = np.loadtxt(os.path.join(os.path.dirname(__file__), "backward.txt"))

tot = len(data) - 3

while tot >= 0:
    tmp = np.array([data[tot], data[tot+1], data[tot+2]])
    backward_deltas.append(tmp)
    tot -= 3

# Input Test

data = np.loadtxt(os.path.join(os.path.dirname(__file__), "test.txt"))

tot = 0

while tot < len(data):
    tmp = np.array([data[tot], data[tot+1], data[tot+2]])
    test.append(tmp)
    tot += 3

# Draw the graph

x = range(len(forward_deltas))

for i in range(3):
    plt.plot(x, [test[j][i][i] for j in x], label=f"Direction {i+1}")

plt.xlabel("Step")
plt.ylabel("Inner Product")
plt.title("Lyapunov Exponents inner Product")
plt.legend()
plt.savefig(os.path.join(os.path.dirname(__file__), "lyapunov_exponents_inner_product.png"))
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
    tmp = np.array([data[tot], data[tot+1]])
    forward_deltas.append(tmp)
    tot += 2

# Input Backward Deltas
data = np.loadtxt(os.path.join(os.path.dirname(__file__), "backward.txt"))

tot = len(data) - 2

while tot >= 0:
    tmp = np.array([data[tot], data[tot+1]])
    backward_deltas.append(tmp)
    tot -= 2

# Input Test

data = np.loadtxt(os.path.join(os.path.dirname(__file__), "test.txt"))

tot = 0

while tot < len(data):
    tmp = np.array([data[tot], data[tot+1]])
    test.append(tmp)
    tot += 2

# Draw the graph

x = range(len(forward_deltas))

for i in range(2):
    plt.plot(x, [test[j][i][i] for j in x], label=f"Direction {i+1}")

plt.xlabel("Step")
plt.ylabel("Inner Product")
plt.title("Lyapunov Exponents inner Product")
plt.legend()
plt.savefig(os.path.join(os.path.dirname(__file__), "lyapunov_exponents_inner_product.png"))
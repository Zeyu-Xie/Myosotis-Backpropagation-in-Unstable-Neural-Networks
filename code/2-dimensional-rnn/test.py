import os
import numpy as np

forward_deltas = []
backward_deltas = []

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

# Prove that the forward deltas and backward deltas are correct

for i in range(len(forward_deltas)):
    print(np.dot(forward_deltas[i].T, backward_deltas[i]))
    print("")

n = len(forward_deltas)

with open ("test.txt", "w") as f:
    for i in range(len(forward_deltas)):
        tmp = np.dot(forward_deltas[i], backward_deltas[n-1-i].T)
        f.write(f"{tmp[0][0]} {tmp[0][1]}\n")
        f.write(f"{tmp[1][0]} {tmp[1][1]}\n")
        f.write("\n")
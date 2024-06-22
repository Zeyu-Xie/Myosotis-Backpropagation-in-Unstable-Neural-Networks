import os
import numpy as np

forward_deltas = []
backward_deltas = []

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

# Prove that the forward deltas and backward deltas are correct

n = len(forward_deltas)

for i in range(len(forward_deltas)):
    print(np.dot(forward_deltas[i], backward_deltas[n-1-i].T))
    print("")

with open ("test.txt", "w") as f:
    for i in range(len(forward_deltas)):
        tmp = np.dot(forward_deltas[i], backward_deltas[n-1-i].T)
        f.write(f"{tmp[0][0]} {tmp[0][1]} {tmp[0][2]}\n")
        f.write(f"{tmp[1][0]} {tmp[1][1]} {tmp[1][2]}\n")
        f.write(f"{tmp[2][0]} {tmp[2][1]} {tmp[2][2]}\n")
        f.write("\n")
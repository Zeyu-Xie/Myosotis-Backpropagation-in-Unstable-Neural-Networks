import os
import numpy as np

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

# Prove that the forward deltas and backward deltas are correct

for i in range(len(forward_deltas)):
    print(np.dot(forward_deltas[i].T, backward_deltas[i]))
    print("")
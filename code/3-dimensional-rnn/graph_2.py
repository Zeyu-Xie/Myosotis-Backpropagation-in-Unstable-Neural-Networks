import os
import numpy as np
import matplotlib.pyplot as plt
from forward import forward

time_steps = 300
graph_started_at = 1

ans_forward = [[-1, -1, -1]]
ans_backward = [[-1, -1, -1]]

for i in range(1, time_steps):
    ans_forward.append(forward(i)["exponents"])
    print(i, ans_forward[-1])

for i in range(1, time_steps):
    ans_backward.append(forward(i)["exponents"])
    print(i, ans_backward[-1])

with open(os.path.join(os.path.dirname(__name__), "lyapunov_exponents_forward.txt"), "w") as f:
    for i in range(1, time_steps):
        f.write(f"{ans_forward[i][0]} {ans_forward[i][1]} {ans_forward[i][2]}\n")

with open(os.path.join(os.path.dirname(__name__), "lyapunov_exponents_backward.txt"), "w") as f:
    for i in range(1, time_steps):
        f.write(f"{ans_backward[i][0]} {ans_backward[i][1]} {ans_backward[i][2]}\n")

x = range(graph_started_at, time_steps)
y1 = [ans_forward[j][0] for j in range(time_steps)]
y2 = [ans_forward[j][1] for j in range(time_steps)]
y3 = [ans_forward[j][2] for j in range(time_steps)]

plt.plot(x, y1[graph_started_at:], label="Direction 1")
plt.plot(x, y2[graph_started_at:], label="Direction 2")
plt.plot(x, y3[graph_started_at:], label="Direction 3")
plt.xlabel("Step")
plt.ylabel("Lyapunov Exponents")
plt.title("Forward Lyapunov Exponents")
plt.legend()
plt.savefig(os.path.join(os.path.dirname(__name__), "lyapunov_exponents_forward.png"))

plt.clf()

y1 = [ans_backward[j][0] for j in range(time_steps)]
y2 = [ans_backward[j][1] for j in range(time_steps)]
y3 = [ans_backward[j][2] for j in range(time_steps)]

plt.plot(x, y1[graph_started_at:], label="Direction 1")
plt.plot(x, y2[graph_started_at:], label="Direction 2")
plt.plot(x, y3[graph_started_at:], label="Direction 3")
plt.xlabel("Step")
plt.ylabel("Lyapunov Exponents")
plt.title("Backward Lyapunov Exponents")
plt.legend()
plt.savefig(os.path.join(os.path.dirname(__name__), "lyapunov_exponents_backward.png"))

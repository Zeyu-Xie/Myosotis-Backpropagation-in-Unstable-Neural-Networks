import numpy as np
import matplotlib.pyplot as plt
import os

N = 1000
a = np.arange(0, 1.401, 0.001)
b = 0.3
na = len(a)
LE1 = np.zeros(na)
LE2 = np.zeros(na)
x = 0.2
y = 0.3

def Lyapunov_exponent(f, jac, x, N):
    LCEvector = np.zeros(len(x))
    Q = np.eye(len(x))
    for j in range(N):
        xprev = x
        x = f(x)
        Ji = jac(xprev)
        B = np.dot(Ji, Q)
        Q, R = np.linalg.qr(B)
        LCEvector += np.log(np.abs(np.diag(R)))
    LE = LCEvector / N
    return LE[0], LE[1]

for i in range(na):
    def f (x):
        return np.array([1-a[i]*x[0]**2 + x[1], b*x[0]])
    def jac (x):
        return np.array([[-2*a[i]*x[0], 1], [b, 0]])
    LE1[i], LE2[i] = Lyapunov_exponent(f, jac, np.array([0.2, 0.3]), N)

plt.figure(1)
plt.plot(a, LE1, 'g', linewidth=1)
plt.plot(a, LE2, 'b', linewidth=1)
plt.xlim(0, 1.4)
plt.ylim(-2, 1)
plt.legend(['$\lambda_1$', '$\lambda_2$'])
plt.plot([0, 1.4], [0, 0], 'k--', linewidth=1)
plt.xlabel('a')
plt.ylabel('LE')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.savefig(os.path.join(os.path.dirname(__file__), 'LE.png'))
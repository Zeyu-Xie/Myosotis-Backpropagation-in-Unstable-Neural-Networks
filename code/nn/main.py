import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import qr
import matplotlib.pylab as plt
from mnist import load_mnist  # 假设你已经安装并配置好了 MNIST 数据加载工具
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lyapunov_spectrum import lyapunov_spectrum

# Sigmoid 函数和其梯度
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    x = x - np.max(x)  # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        return y

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads

    def hidden_layer_activations(self, x):
        W1, b1 = self.params['W1'], self.params['b1']
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        return z1

def get_parameters():
    with open("W1.txt", "r") as f:
        W1 = np.array([list(map(float, line.strip().split())) for line in f])
        W1.shape = (784, 50)
    with open("b1.txt", "r") as f:
        b1 = np.array(list(map(float, f)))
    with open("W2.txt", "r") as f:
        W2 = np.array([list(map(float, line.strip().split())) for line in f])
        W2.shape = (50, 10)
    with open("b2.txt", "r") as f:
        b2 = np.array(list(map(float, f)))
    return W1, b1, W2, b2

# 主程序
if __name__ == "__main__":
    # 加载数据
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    # 初始化网络
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    network.params['W1'], network.params['b1'], network.params['W2'], network.params['b2'] = get_parameters()

    W1 = network.params['W1']
    b1 = network.params['b1']
    W2 = network.params['W2']
    b2 = network.params['b2']

    # 定义正向传播函数
    f_1 = lambda x: sigmoid(np.dot(x, W1) + b1)
    f_2 = lambda z: softmax(np.dot(z, W2) + b2)

    # 定义反向传播函数
    def f_1_grad(x, t):
        a = np.dot(x, W1) + b1
        # z = sigmoid(a)
        dz = np.eye(a.shape[1])
        da = np.dot(dz, np.diag(sigmoid_grad(a[0])))
        dx = np.dot(da, W1.T)
        return dx
    def f_2_grad(z, t):
        batch_num = z.shape[0]
        a2 = np.dot(z, W2) + b2
        y = softmax(a2)
        dy = np.diag((y - t).flatten() / batch_num)
        dz = np.dot(dy, W2.T)
        return dz
    
    def f1(x):
        return f_1(x.T).flatten()
    def f2(z):
        return f_2(z.T).flatten()
    def f1_grad(x, t):
        x = x.reshape(1, x.shape[0])
        t = t.reshape(1, t.shape[0])
        return f_1_grad(x, t)
    def f2_grad(z, t):
        z = z.reshape(1, z.shape[0])
        t = t.reshape(1, t.shape[0])
        return f_2_grad(z, t)

    x = x_train[:1].flatten()
    z = f1(x)
    y = f2(z)
    x_grad = f1_grad(x, t_train[:1].flatten())
    z_grad = f2_grad(z, t_train[:1].flatten())

    t_span = (0, 1)
    u0 = x_train[t_span[0]:t_span[1]].flatten()
    m = u0.shape[0]
    M = f1(u0).shape[0]
    K = 1
    delta_t = (t_span[1] - t_span[0]) / K

    lambdas_s = []

    def random_input(time_steps):

        if time_steps == 0:
            return

        # 初始条件
        u0 = np.random.randn(784)
        Q0 = np.linalg.qr(np.random.randn(784, 50))[0]

        # 时间步数量
        K = 2
        # 每个时间步的时间长度
        delta_t = 1.0

        # 存储R的对角元素
        D = []

        # 第一时间步
        u1 = f1(u0)
        _W1 = np.zeros((50, 50))
        for j in range(50):
            w0j = Q0[:, j]
            w1j = f1_grad(u0, t_train[0]) @ w0j
            _W1[:, j] = w1j

        Q1, R1 = qr(_W1)
        D.append(np.diag(R1))

        # 第二时间步
        u2 = f2(u1)
        _W2 = np.zeros((10, 10))
        for j in range(10):
            w1j = Q1[:50, j]
            w2j = f2_grad(u1, t_train[0]) @ w1j  # 使用雅可比矩阵计算梯度
            _W2[:, j] = w2j

        Q2, R2 = qr(_W2)
        D.append(np.diag(R2))

        # 计算李雅普诺夫指数
        lambdas = []
        for j in range(10):
            lambda_j = sum(np.log(np.abs(D[i][j])) for i in range(K)) / (K * delta_t)
            lambdas.append(lambda_j.item())

        lambdas_s.append(lambdas)

        random_input(time_steps-1)

    random_input(10)

    for i in range(len(lambdas_s)):
        print(lambdas_s[i])
        plt.plot(lambdas_s[i], marker="o")
    
    plt.xlabel("Index")
    plt.ylabel("Lyapunov Exponent")
    plt.show()
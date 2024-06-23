import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import qr
import matplotlib.pylab as plt
from mnist import load_mnist
import sys
import os

num = 13
is_forward = False

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

def softmax_grad(x):
    return softmax(x) * (1 - softmax(x))

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
        self.params['W1'] = weight_init_std * \
            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
            np.random.randn(hidden_size, output_size)
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

    dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train")

    with open(os.path.join(dir, "W1.txt"), "r") as f:
        W1 = np.array([list(map(float, line.strip().split())) for line in f])
        W1.shape = (784, 50)
    with open(os.path.join(dir, "b1.txt"), "r") as f:
        b1 = np.array(list(map(float, f)))
    with open(os.path.join(dir, "W2.txt"), "r") as f:
        W2 = np.array([list(map(float, line.strip().split())) for line in f])
        W2.shape = (50, 10)
    with open(os.path.join(dir, "b2.txt"), "r") as f:
        b2 = np.array(list(map(float, f)))

    print(W1.shape, b1.shape, W2.shape, b2.shape)

    return W1, b1, W2, b2


if __name__ == "__main__":

    # Load MNIST data
    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=True, one_hot_label=True)

    # Initialize network
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    network.params['W1'], network.params['b1'], network.params['W2'], network.params['b2'] = get_parameters()

    W1 = network.params['W1']
    b1 = network.params['b1']
    W2 = network.params['W2']
    b2 = network.params['b2']

    def f1(x):
        return sigmoid(np.dot(x.T, W1) + b1).flatten()

    def f2(z):
        return softmax(np.dot(z.T, W2) + b2).flatten()

    def f1_grad(x, dz):
        x = x.reshape(1, x.shape[0])
        # t = t.reshape(1, t.shape[0])
        a = np.dot(x, W1) + b1
        # z = sigmoid(a)
        # dz = np.eye(a.shape[1])
        da = np.dot(dz, np.diag(sigmoid_grad(a[0])))
        dx = np.dot(da, W1.T)
        return dx

    def f2_grad(z):
        # z - value of the hidden layer
        z = z.reshape(1, z.shape[0])
        # t - target value
        # t = t.reshape(1, t.shape[0])
        # batch_num = z.shape[0]

        a2 = np.dot(z, W2) + b2
        # y = softmax(a2)
        dy = np.random.randn(10)
        # dy = np.diag((y - t).flatten() / batch_num)
        dy = dy.reshape(1, dy.shape[0])
        dz = np.dot(softmax_grad(dy), W2.T)
        return dz

    forward_lambdas_s = []
    backward_lambdas_s = []

    for tot in range(num):

        if tot > num - 1:
            break

        idx = np.random.randint(0, 60000)
        x = x_train[idx].flatten()
        Q0 = np.linalg.qr(np.random.randn(784, 50))[0]

        K = 2
        delta_t = 1.0
        D_forward = []
        D_backward = []

        # Forward - 1st time step
        z = f1(x)
        _W1 = np.zeros((50, 50))
        for j in range(50):
            w0j = Q0[:, j]
            w1j = f1(x)
            _W1[:, j] = w1j
        Q1, R1 = qr(_W1)
        D_forward.append(np.diag(R1))

        # Forward - 2nd time step
        y = f2(z)
        _W2 = np.zeros((10, 10))
        for j in range(10):
            w1j = Q1[:50, j]
            w2j = f2(z)
            _W2[:, j] = w2j
        Q2, R2 = qr(_W2)
        D_forward.append(np.diag(R2))

        y_theory = t_train[idx]
        dy = y_theory - y
        Q2 = np.linalg.qr(np.random.randn(50, 10))[0]

        # Backward - 2nd time step
        _W2 = np.zeros((784, 50))
        for j in range(10):
            w2j = Q2[:50, j]
            w1j = f2_grad(z) @ w2j
            _W2[:, j] = w1j
        Q1, R1 = qr(_W2)
        D_backward.append(np.diag(R1))

        dz = _W2

        # Backward - 1st time step
        _W1 = np.zeros((784, 784))
        for j in range(50):
            w1j = Q1[:784, j]
            w0j = f1_grad(x, sigmoid_grad(dz)) @ w1j
            _W1[:, j] = w0j
        Q0, R0 = qr(_W1)
        D_backward.append(np.diag(R0))

        # Lyapunov Exponents
        lambdas = []
        for j in range(10):
            if is_forward:
                D = D_forward
            else:
                D = D_backward
            lambda_j = sum(np.log(np.abs(D[i][j]))
                           for i in range(K)) / (K * delta_t)
            lambdas.append(lambda_j.item())

        if is_forward:
            forward_lambdas_s.append(lambdas)
        else:
            backward_lambdas_s.append(lambdas)

    if is_forward:
        x = range(len(forward_lambdas_s[0]))
        for i in range(len(forward_lambdas_s)):
            print(forward_lambdas_s[i])
            plt.scatter(x, forward_lambdas_s[i], marker=".")
    else:
        x = range(len(backward_lambdas_s[0]))
        for i in range(len(backward_lambdas_s)):
            print(backward_lambdas_s[i])
            plt.scatter(x, backward_lambdas_s[i], marker=".")

    if is_forward:
        plt.title("Forward Lyapunov Exponents")
    else:
        plt.title("Backward Lyapunov Exponents")

    plt.xlabel("Index")
    plt.ylabel("Lyapunov Exponent")

    if is_forward:
        plt.savefig(os.path.join(os.path.dirname(
            __file__), "forward_lyapunov.png"))
    else:
        plt.savefig(os.path.join(os.path.dirname(
            __file__), "backward_lyapunov.png"))

    plt.show()

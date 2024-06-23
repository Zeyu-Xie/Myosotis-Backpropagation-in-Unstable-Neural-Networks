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

# 定义 system 和 jacobian 函数
# def system(t, u, net, x):
#     u = u.reshape(-1, net.params['W1'].shape[1])
#     z1 = net.hidden_layer_activations(x)
#     return z1.flatten()

# def jacobian(u, net, x):
#     W1 = net.params['W1']
#     a1 = np.dot(x, W1) + net.params['b1']
#     grad = sigmoid_grad(a1)
#     return np.diag(grad.flatten())

# def backpropagation_system(t, u, net, x, t_target):
#     u = u.reshape(-1, net.params['W1'].shape[1])
#     grads = net.gradient(x, t_target)
#     return grads['W1'].flatten()

# def backpropagation_jacobian(u, net, x, t_target):
#     W1 = net.params['W1']
#     grad = sigmoid_grad(np.dot(x, W1) + net.params['b1'])
#     return np.diag(grad.flatten())

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
        dy = np.diag(((y - t) / batch_num)[0])
        dz = np.dot(dy, W2.T)
        return dz
    
    # 测试反向传播函数
    x = x_train[:1]
    z = f_1(x)
    t = t_train[:1]
    dz = f_2_grad(z, t)
    print(dz.shape)

    # # 设置参数
    # t_span = (0, 10)
    # u0 = network.hidden_layer_activations(x_train[:1]).flatten()
    # m = u0.size
    # M = m
    # K = 100
    # delta_t = (t_span[1] - t_span[0]) / K

    # # 计算 Lyapunov 谱（正向传播）
    # ans_forward = lyapunov_spectrum(
    #     lambda t, u: system(t, u, network, x_train[:1]),
    #     lambda u: jacobian(u, network, x_train[:1]),
    #     t_span, u0, m, M, K, delta_t
    # )
    # print("Forward Lyapunov Exponents:")
    # tmp = ""
    # for i, exponent in enumerate(ans_forward["exponents"]):
    #     tmp += f"{round(exponent, 4)}"
    #     if (i + 1) % 5 == 0:
    #         tmp += " \\\\ \n"
    #     else :
    #         tmp += " & "
    # print(tmp)
    
    # # 计算 Lyapunov 谱（反向传播）
    # u0_backprop = network.gradient(x_train[:1], t_train[:1])['W1'].flatten()
    # ans_backward = lyapunov_spectrum(
    #     lambda t, u: backpropagation_system(t, u, network, x_train[:1], t_train[:1]),
    #     lambda u: backpropagation_jacobian(u, network, x_train[:1], t_train[:1]),
    #     t_span, u0_backprop, m, M, K, delta_t
    # )
    # print("Backward Lyapunov Exponents:")
    # tmp = ""
    # for i, exponent in enumerate(ans_backward["exponents"]):
    #     tmp += f"{round(exponent, 4)}"
    #     if (i + 1) % 5 == 0:
    #         tmp += " \\\\ \n"
    #     else :
    #         tmp += " & "
    # print(tmp)

    # # 创建一个具有两个子图的图形窗口，1行2列
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
    # y1 = ans_forward["exponents"]
    # y2 = ans_backward["exponents"]
    # x = np.arange(M)

    # # 在ax1子图中绘制图形
    # ax1.plot(x, y1, 'o', color='b', label='Forward')
    # ax1.set_title('Forward Lyapunov Exponents')
    # ax1.set_xlabel("Index")
    # ax1.set_ylabel("Exponents")
    # ax1.legend()

    # # 在ax2子图中绘制图形
    # ax2.plot(x, y2, 'o', color='r', label='Backward')
    # ax2.set_title('Backward Lyapunov Exponents')
    # ax2.set_xlabel("Index")
    # ax2.set_ylabel("Exponents")
    # ax2.legend()

    # # 显示图形
    # plt.savefig(os.path.join(os.path.dirname(__file__), "lyapunov_exponents.png"))

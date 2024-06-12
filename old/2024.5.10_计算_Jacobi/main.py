import matplotlib.pylab as plt
import numpy as np
import os
from mnist import load_mnist  # Load MNIST
from PIL import Image

# Type 1. Basic Functions

# Function - Numerical Differentiation
def num_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)
# Function - Numerical Gradient
def num_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 还原值
        it.iternext()   
        
    return grad
# Function - Gradient Descent
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = num_gradient(f, x)
        x -= lr * grad
    return x
# Function - Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))    
# Function - ReLU
def relu(x):
    return np.maximum(0, x)
# Function - Step Function
def step_function(x):
    return np.array(x > 0, dtype=np.int)
# Function - Sigmoid Gradient
def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

# Type 2. Layers

# Function - Softmax Layer
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 
    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))
# Function - Cross Entropy Error Layer
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

# Class - Two Layer Neural Network
class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重
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
        
    # x:输入数据, t:监督数据
    def loss(self, x, t):
        y = self.predict(x)
        
        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x:输入数据, t:监督数据
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = num_gradient(loss_W, self.params['W1'])
        grads['b1'] = num_gradient(loss_W, self.params['b1'])
        grads['W2'] = num_gradient(loss_W, self.params['W2'])
        grads['b2'] = num_gradient(loss_W, self.params['b2'])
        
        return grads
        
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

def _f(params, learning_rate=0.1, batch_size=100):
    
    global x_train, t_train
    
    # 将参数映射到原有的参数集合中
    W1, b1, W2, b2 = (
        params['W1'], params['b1'], 
        params['W2'], params['b2']
    )
    
    # 计算梯度
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    network.params['W1'] = W1
    network.params['b1'] = b1
    network.params['W2'] = W2
    network.params['b2'] = b2
    grad = network.gradient(x_train, t_train)
    
    # 更新参数
    W1 -= learning_rate * grad['W1']
    b1 -= learning_rate * grad['b1']
    W2 -= learning_rate * grad['W2']
    b2 -= learning_rate * grad['b2']
    
    # 返回更新后的参数集合
    updated_params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return updated_params

def f(params):

    global x_train, t_train

    params_W1 = params[0:784*50].reshape(784, 50)
    params_b1 = params[784*50:784*50+50]
    params_W2 = params[784*50+50:784*50+50+50*10].reshape(50, 10)
    params_b2 = params[784*50+50+50*10:]

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    network.params['W1'] = params_W1
    network.params['b1'] = params_b1
    network.params['W2'] = params_W2
    network.params['b2'] = params_b2

    _params = _f(network.params, learning_rate=0.1, batch_size=100)

    _params_flatten = _params['W1'].flatten()
    _params_flatten = np.append(_params_flatten, _params['b1'].flatten())
    _params_flatten = np.append(_params_flatten, _params['W2'].flatten())
    _params_flatten = np.append(_params_flatten, _params['b2'].flatten())

    return _params_flatten

# 定义Jacobi矩阵的计算函数
def jacobi_matrix(f, x):
    n = len(f(x))  # 函数f的维度
    m = len(x)     # 变量x的维度
    J = np.zeros((n, m))
    h = 1e-5  # 用于计算偏导数的微小增量

    for i in range(n):
        for j in range(m):
            x_plus_h = x.copy()
            x_plus_h[j] += h
            J[i, j] = (f(x_plus_h)[i] - f(x)[i]) / h

    return J

# Main
if __name__ == "__main__":

    # Init Network
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    params_flatten = network.params['W1'].flatten()
    params_flatten = np.append(params_flatten, network.params['b1'].flatten())
    params_flatten = np.append(params_flatten, network.params['W2'].flatten())
    params_flatten = np.append(params_flatten, network.params['b2'].flatten())
    v = params_flatten

    print(jacobi_matrix(f, v))
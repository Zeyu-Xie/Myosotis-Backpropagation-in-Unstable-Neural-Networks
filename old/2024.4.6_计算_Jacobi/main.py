import numpy as np
import time
from mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(
    normalize=True, one_hot_label=True)

x_train = x_train[:100]
t_train = t_train[:100]

x_train_1 = []

for num in range(100):
    tmp = [0] * 196
    for i in range(196):
        tmp[i] = (x_train[num][2*i]+x_train[num][2*i+1]+x_train[num][2*i+28]+x_train[num][2*i+29])/4
    x_train_1.append(tmp)

x_train = np.array(x_train_1)

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
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


def gradient(p):
    W1, W2 = p['W1'], p['W2']
    b1, b2 = p['b1'], p['b2']
    grads = {}

    batch_num = 100

    # forward
    a1 = np.dot(x_train, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    y = softmax(a2)

    # backward
    dy = (y - t_train) / batch_num
    grads['W2'] = np.dot(z1.T, dy)
    grads['b2'] = np.sum(dy, axis=0)

    da1 = np.dot(dy, W2.T)
    dz1 = sigmoid_grad(a1) * da1
    grads['W1'] = np.dot(x_train.T, dz1)
    grads['b1'] = np.sum(dz1, axis=0)

    return grads


def iterate(p):
    grad = gradient(p)
    for key in ('W1', 'b1', 'W2', 'b2'):
        p[key] -= 0.1 * grad[key]
    return p


def flatten(p):
    return np.concatenate([p['W1'].flatten(), p['b1'].flatten(), p['W2'].flatten(), p['b2'].flatten()])


def unflatten(flat):
    p = {}
    p['W1'] = flat[:4900].reshape(196, 25)
    p['b1'] = flat[4900:4925]
    p['W2'] = flat[4925:5175].reshape(25, 10)
    p['b2'] = flat[5175:]
    return p


if __name__ == '__main__':

    time1 = time.time()

    params = {}
    params['W1'] = 0.01 * \
        np.random.randn(196, 25)
    params['b1'] = np.zeros(25)
    params['W2'] = 0.01 * \
        np.random.randn(25, 10)
    params['b2'] = np.zeros(10)

    jacobi = np.zeros((5185, 5185))

    for i in range(196):

        print(i)

        for j in range(25):

            tmp = params['W1'][i][j]
            params['W1'][i][j] = tmp + 1e-6
            params_1 = iterate(params)
            f1 = flatten(params_1)
            params['W1'][i][j] = tmp - 1e-6
            params_2 = iterate(params)
            f2 = flatten(params_2)
            params['W1'][i][j] = tmp
            jacobi[i*25+j] = (f1-f2)/2e-6

    for i in range(25):
        tmp = params['b1'][i]
        params['b1'][i] = tmp + 1e-6
        params_1 = iterate(params)
        f1 = flatten(params_1)
        params['b1'][i] = tmp - 1e-6
        params_2 = iterate(params)
        f2 = flatten(params_2)
        params['b1'][i] = tmp
        jacobi[4900+i] = (f1-f2)/2e-6

    for i in range(25):
        for j in range(10):
            tmp = params['W2'][i][j]
            params['W2'][i][j] = tmp + 1e-6
            params_1 = iterate(params)
            f1 = flatten(params_1)
            params['W2'][i][j] = tmp - 1e-6
            params_2 = iterate(params)
            f2 = flatten(params_2)
            params['W2'][i][j] = tmp
            jacobi[4925+i*10+j] = (f1-f2)/2e-6

    for i in range(10):

        tmp = params['b2'][i]
        params['b2'][i] = tmp + 1e-6
        params_1 = iterate(params)
        f1 = flatten(params_1)
        params['b2'][i] = tmp - 1e-6
        params_2 = iterate(params)
        f2 = flatten(params_2)
        params['b2'][i] = tmp
        jacobi[5175+i] = (f1-f2)/2e-6

    # jacobi = jacobi.T

    for i in range(5185):
        for j in range(5185):
            if np.abs(jacobi[i][j]-0) < 1e-6:
                print(i, j, jacobi[i][j])

    time2 = time.time()

    print(time2-time1)

import numpy as np
from mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(
    normalize=True, one_hot_label=True)

x_train = x_train[:100]
t_train = t_train[:100]


class V:

    def __init__(self, x):
        if type(x) == list:
            self.params = np.array(x)
        elif type(x) == np.ndarray:
            self.params = x
        elif type(x) == dict:
            self.params = np.array(x["W1"].flatten().tolist() +
                                   x["W2"].flatten().tolist()+x["b1"].tolist()+x["b2"].tolist())

    def unstandarize(self):
        return {"W1": self.params[:784*50].reshape(784, 50),
                "W2": self.params[784*50:784*50+50*10].reshape(50, 10),
                "b1": self.params[784*50+50*10:784*50+50*10+50],
                "b2": self.params[784*50+50*10+50:784*50+50*10+50+10]}


v_zero = V([0]*784*50+[0]*50*10+[0]*50+[0]*10)


def v_random():
    ans = V(np.random.randn(784*50+50*10+50+10))
    ans.params[39700:39760] = 0
    return ans


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


def gradient(x):
    W1, W2 = x['W1'], x['W2']
    b1, b2 = x['b1'], x['b2']
    grads = {}

    batch_num = x_train.shape[0]

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


def f(v):
    x = v.unstandarize()
    gra = gradient(x)
    gra_standarized = np.array(gra["W1"].flatten().tolist(
    ) + gra["W2"].flatten().tolist() + gra["b1"].tolist() + gra["b2"].tolist())
    return V(v.params-0.1*gra_standarized)


print(f(f(v_zero)).params)

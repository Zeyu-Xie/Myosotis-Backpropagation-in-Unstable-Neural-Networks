from f import f, TwoLayerNet
import numpy as np


def random_data():
    tln = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    x = {"W1": tln.params['W1'], "W2": tln.params['W2'],
         "b1": tln.params['b1'], "b2": tln.params['b2']}
    return x


zero_data = {"W1": np.zeros((784, 50)), "W2": np.zeros((50, 10)),
             "b1": np.zeros(50), "b2": np.zeros(10)}

def standarize(x):
    y = np.array(x["W1"].flatten().tolist() +
                 x["W2"].flatten().tolist()+x["b1"].tolist()+x["b2"].tolist())
    return y


def unstandarize(y):
    x = {"W1": np.array(y[:784*50]).reshape(784, 50),
         "W2": np.array(y[784*50:784*50+50*10]).reshape(50, 10),
         "b1": np.array(y[784*50+50*10:784*50+50*10+50]),
         "b2": np.array(y[784*50+50*10+50:784*50+50*10+50+10])}
    return x

def random_standarize_data():
    return standarize(random_data())

def zero_standarize_data():
    return standarize(zero_data)

def F(v):
    x = unstandarize(v)
    return standarize(f(x))
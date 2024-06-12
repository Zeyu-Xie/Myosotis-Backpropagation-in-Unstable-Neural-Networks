from fix import TwoLayerNet, F as f, random_standarize_data as random_data, zero_standarize_data as zero_data, unstandarize, standarize
import numpy as np
from mnist import load_mnist  # Load MNIST

# Get Data
(x_train, t_train), (x_test, t_test) = load_mnist(
    normalize=True, one_hot_label=True)

# Sample Data
# choice = np.random.choice(x_train.shape[0], sample_size)
choice = np.array(range(60000))
x_train = x_train[choice]
t_train = t_train[choice]

def jacobi(f, x):
    h = 1e-4
    ans = np.zeros((39760, 39760))
    for i in range(39760):
        for j in range(39760):
            tmp = x[i]
            x[i] = tmp + h
            f1 = f(x)[j]
            x[i] = tmp - h
            f2 = f(x)[j]
            x[i] = tmp
            ans[i][j] = (f1 - f2) / (2 * h)
    return ans

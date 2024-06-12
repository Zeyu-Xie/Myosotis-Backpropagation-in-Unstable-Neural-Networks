from iteration_function import f, random_data, zero_data, jacobi
import numpy as np

# 速度太慢，失败！
print(np.shape(jacobi(f, zero_data())))
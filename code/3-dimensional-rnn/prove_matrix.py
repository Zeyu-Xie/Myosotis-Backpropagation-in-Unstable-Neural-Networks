from forward import forward
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lyapunov_spectrum import lyapunov_spectrum

if __name__ == "__main__":

    # 1. f: The differential equation of the original system (function)
    # 2. jacobian: The Jacobian matrix of the homogeneous tangent system (function)
    # 3. t_span: The time span (tuple), e.g. (0, 1000)
    # 4. u0: The initial condition of the original system (array)
    # 5. m: The dimension of the system
    # 6. M: The number of Lyapunov exponents
    # 7. K: The number of time segments
    # 8. delta_t: The length of each time segment

    # Matrix
    # A = np.array([[1, 2, 3], [2, 2, 3], [4, 3, 2]])
    A = np.random.randn(3, 3)

    # Args
    def f(t, x):
        return A @ x
    def f_jacobian(x):
        return A
    t_span = (0, 80)
    u0 = np.random.randn(3)
    m = 3
    M = 3
    K = 80
    delta_t = 1

    # Calculate the Lyapunov exponents
    ans = lyapunov_spectrum(f, f_jacobian, t_span, u0, m, M, K, delta_t)
    print("Lyapunov Exponents:\n")
    print(ans["exponents"])
    print("")
    print("Eigenvalues:\n")
    print(np.linalg.eigvals(A))
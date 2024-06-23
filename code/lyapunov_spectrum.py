import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import qr


def lyapunov_spectrum(f, jacobian, t_span, u0, m, M, K, delta_t):
    """
    Compute the Lyapunov spectrum and Lyapunov vectors

    Args:
    1. f: The differential equation of the original system (function)
    2. jacobian: The Jacobian matrix of the homogeneous tangent system (function)
    3. t_span: The time span (tuple), e.g. (0, 1000)
    4. u0: The initial condition of the original system (array)
    5. m: The dimension of the system
    6. M: The number of Lyapunov exponents
    7. K: The number of time segments
    8. delta_t: The length of each time segment

    Returns:
    1. lyapunov_exponents: The approximate Lyapunov exponents (array)
    2. lyapunov_vectors: The Lyapunov vectors (list)
    3. intermediate_deltas: All intermediate perturbation vectors (list)
    """

    # Step 1. Compute the original solution
    sol = solve_ivp(f, t_span, u0, t_eval=np.linspace(
        t_span[0], t_span[1], K+1))
    u = sol.y.T

    # Step 2. Generate a random orthogonal matrix Q0
    Q0, _ = np.linalg.qr(np.random.randn(m, M))
    Q0 = np.eye(M)

    Q = Q0
    R_all = []
    W_all = []
    intermediate_deltas = []
    lyapunov_vectors = []

    for i in range(K):
        t_i = sol.t[i]
        t_ip1 = sol.t[i + 1]

        # Step 3. Compute the homogeneous tangent solution
        W = np.zeros((m, M))

        def tangent_eq(t, W_flat):
            W = W_flat.reshape((m, M))
            J = jacobian(u[i])
            dWdt = J @ W
            return dWdt.flatten()

        W0_flat = Q.flatten()
        sol_tangent = solve_ivp(tangent_eq, [t_i, t_ip1], W0_flat)
        W = sol_tangent.y[:, -1].reshape((m, M))
        W_all.append(W)

        # QR decomposition
        Q, R = qr(W)
        R_all.append(R)

        # Save the current step's perturbation vector
        intermediate_deltas.append(Q.copy())

    # Step 4. Compute the Lyapunov exponents
    R_all = np.array(R_all)
    D = np.zeros(M)

    for R in R_all:
        diag_R_inv = np.diag(R)
        D += np.log(np.abs(diag_R_inv))

    lyapunov_exponents = D / (K * delta_t)

    # Step 5. Compute the Lyapunov vectors
    V_t = (W_all)[math.floor(K/2)]

    for i in range(math.floor(K/2)+1, K):
        R_inv = np.linalg.inv(R_all[i])
        V_t = np.dot(V_t, R_inv)

    lyapunov_vectors = V_t

    # Output
    output = {
        "exponents": lyapunov_exponents,
        "vectors": lyapunov_vectors,
        "deltas": intermediate_deltas
    }
    return output
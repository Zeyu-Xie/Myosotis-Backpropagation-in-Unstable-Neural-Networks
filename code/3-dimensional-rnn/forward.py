import numpy as np
import os
from scipy.linalg import qr
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lyapunov_spectrum import lyapunov_spectrum
from record_log import record_log

def forward(time_steps):
    # Initialize the random seed
    np.random.seed(42)
    hidden_size = 3
    input_size = 2
    # time_steps = 1000
    use_input = True

    W_h = np.random.randn(hidden_size, hidden_size)
    W_x = np.random.randn(hidden_size, input_size)

    # Initialization of the activation functions and their derivatives

    def tanh(x):
        return np.tanh(x)

    def tanh_prime(x):
        return 1 - np.tanh(x) ** 2

    # Generate the input sequence
    inputs = np.random.randn(time_steps, input_size)

    # Initialize the hidden state
    h_t = np.zeros(hidden_size)

    # Initialize the perturbation vector
    num_directions = hidden_size
    delta_h_t = np.eye(hidden_size)

    # Initialize the log sum
    log_sum = np.zeros(num_directions)

    # Initialize the forward deltas
    forward_deltas = []

    # Define the RNN system and the Jacobian matrix


    def rnn_system(t, h):
        i = int(t) % time_steps
        x = inputs[i]
        if not use_input:
            return tanh(W_h @ h)
        else:
            return tanh(W_h @ h + W_x @ x)


    def rnn_jacobian(h):
        return W_h * tanh_prime(h)


    # Define the parameters
    t_span = (0, time_steps)
    u0 = h_t
    m = hidden_size
    M = num_directions
    K = time_steps
    delta_t = 1

    # Initialize the intermediate deltas
    ans = lyapunov_spectrum(rnn_system, rnn_jacobian, t_span, u0, m, M, K, delta_t)
    return ans

if __name__ == "__main__":

    ans = forward(300)
    print("Forward Lyapunov Exponents:", ans["exponents"])
    record_log(ans["deltas"], 3, os.path.join(os.path.dirname(__file__), "forward.txt"))


import numpy as np
import os
from scipy.linalg import qr
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lyapunov_spectrum import lyapunov_spectrum
from record_log import record_log

def backward(time_steps):
    # Initialize the random seed
    np.random.seed(42)
    hidden_size = 3
    input_size = 2
    output_size = 2
    # time_steps = 1000
    use_input = True

    W_h = np.random.randn(hidden_size, hidden_size)
    W_x = np.random.randn(hidden_size, input_size)
    W_y = W_x.copy().T

    # Activation functions and their derivatives
    def tanh(x):
        return np.tanh(x)
    def tanh_prime(x):
        return 1 - np.tanh(x) ** 2

    # Generate the input sequence and target outputs
    inputs = np.random.randn(time_steps, input_size)
    targets = np.zeros((time_steps, output_size))
    for i in range(time_steps-1):
        targets[i] = inputs[i+1]
    targets[-1] = inputs[0]

    # Initialize the hidden state
    h_t = np.zeros((time_steps, hidden_size))
    pre_activations = np.zeros_like(h_t)

    # Forward pass
    for t in range(time_steps):
        if t == 0:
            if not use_input:
                pre_activations[t] = np.zeros(hidden_size)
            else:
                pre_activations[t] = np.dot(W_x, inputs[t])
        else:
            if not use_input:
                pre_activations[t] = np.dot(W_h, h_t[t-1])
            else:
                pre_activations[t] = np.dot(W_h, h_t[t-1]) + np.dot(W_x, inputs[t])
        h_t[t] = tanh(pre_activations[t])

    # Calculate the outputs and errors
    outputs = np.dot(h_t, W_y.T)
    errors = outputs - targets

    # Initialize the error gradients for backpropagation
    dL_dh = np.zeros_like(h_t)
    dL_dh[-1] = errors[-1] @ W_y * tanh_prime(pre_activations[-1])

    # Backward pass
    for t in reversed(range(time_steps - 1)):
        dL_dh[t] = (dL_dh[t+1] @ W_h.T + errors[t] @ W_y) * tanh_prime(pre_activations[t])

    # Initialize multiple perturbation vectors
    num_directions = hidden_size
    delta_h_t = np.eye(hidden_size)


    # Store the perturbation vectors
    backward_deltas = []

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
    u0 = h_t[-1]
    m = hidden_size
    M = num_directions
    K = time_steps
    delta_t = 1

    # Call the function to compute the Lyapunov spectrum
    ans = lyapunov_spectrum(rnn_system, rnn_jacobian, t_span, u0, m, M, K, delta_t)
    return ans

if __name__ == "__main__":
    ans = backward(300)
    record_log(ans["deltas"], 3, os.path.join(os.path.dirname(__file__), "backward.txt"))
    print("Backward Lyapunov Exponents:", ans["exponents"])


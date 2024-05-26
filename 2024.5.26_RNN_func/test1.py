import torch
from torch.autograd.functional import jacobian
from main import func as f

def g(h):
    h1 = 3 * h[0]**2 + h[1]**2
    h2 = 2 * h[0]**2 - h[1]**2
    zero = torch.zeros(2)
    zero[0] = h1
    zero[1] = h2
    return zero

if __name__ == "__main__":

    h = torch.ones(2)

    h1 = g(h)

    j = jacobian(g, h)

    print(j)

    print(h1)
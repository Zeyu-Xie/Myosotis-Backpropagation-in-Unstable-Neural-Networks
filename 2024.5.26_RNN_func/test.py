import torch
from torch.autograd.functional import jacobian
from main import func as f

g = f

if __name__ == "__main__":

    h = torch.rand(32, dtype=torch.float64) * 2 - 1
    
    h1 = g(h*0.5)
    j = jacobian(g, h)

    print(h1)
    print(j)
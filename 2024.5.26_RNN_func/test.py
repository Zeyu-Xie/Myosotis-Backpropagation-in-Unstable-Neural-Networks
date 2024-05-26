# jacobian_calculation.py
import torch
from torch.autograd.functional import jacobian
from main import func as f

g = f

if __name__ == "__main__":

    h = torch.tensor([0.2007,  0.0051,  0.4154,  0.4034,  0.3125, -0.0950, -0.8950,
                      -0.7195, -0.0895,  0.3049,  0.9229, -0.3494,  0.2028, -0.7477,
                      -0.2891,  0.0730, -0.9841, -0.1823, -0.8008,  0.3737, -0.4704,
                      0.6192, -0.0636,  0.0086,  0.8629,  0.2117,  0.4237, -0.2888,
                      0.0315,  0.6691, -0.7257, -0.7713], requires_grad=True)

    j = jacobian(g, h)

    print(j)

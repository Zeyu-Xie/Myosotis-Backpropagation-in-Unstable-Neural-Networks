import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import qr

def record_log(source, M, destination):

    with open(destination, "w") as f:
        for deltas in source:
            for delta in deltas:
                f.write(" ".join(map(str, delta)) + "\n")
            f.write("\n")
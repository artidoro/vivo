import torch

import math
import numpy as np
from scipy.special import iv

"""
vMF distribution
p(e(w); µ, κ) = C_m(κ) * exp(κ * (µ @ e(w)))
C_m = κ ** (m/2 - 1) / ((2 * pi) ** (m/2) I_(m/2 - 1)(κ))

p(e_tgt; e_pred) = vMF(e_tgt); e_pred) = C_m(norm(e_pred)) * exp(e_pred @ e_tgt)
NLLvMF = - log(C_m(norm(e_pred))) - e_pred @ e_tgt
"""

def bessel(v, x, n):
    acc = 0
    for i in range(n):
        acc += (x/2)**(2*i+v) / (math.factorial(i) * math.gamma(v+1+i))
    return acc

def log_bessel(v, x, n):
    acc = float('-inf')
    for i in range(n):
        term = math.log(x/2)*(2*i+v) - math.log(math.factorial(i)) - math.lgamma(v+1+i)
        acc = np.logaddexp(acc, term)
    return acc


bessel_coeffs = None
bessel_exps = None

def log_bessel_torch(v, x, n):
    global bessel_coeffs, bessel_exps
    if bessel_coeffs is None:
        bessel_coeffs = np.ndarray(n)
        # term = math.log(x/2)*(2*i+v) - math.log(math.factorial(i)) - math.lgamma(v+1+i)
        for i in range(n):
            bessel_coeffs[i] = -math.log(math.factorial(i)) - math.lgamma(v+1+i)
        bessel_coeffs = torch.Tensor(bessel_coeffs)
        bessel_exps = torch.Tensor([2*i+v for i in range(n)])

    x = torch.Tensor([x])
    res = torch.logsumexp(torch.log(x / 2) * bessel_exps + bessel_coeffs, 0)
    return res.item()

def compare(x, f, g):
    fx = f(x)
    gx = g(x)
    error = abs(fx-gx) / gx
    print(f"{x:.3f}\t({error})\t{fx}\t{gx}")


if __name__ == '__main__':
    m = 300
    r = 10
    scale = 50
    n = 10
    f = lambda x: log_bessel_torch(m/2 - 1, x, n)
    g = lambda x: log_bessel(m/2 - 1, x, n)
    # g = lambda x: math.log(iv(m/2 - 1, x))
    for x in range(r):
        x = 1 - (r/2/scale) + x/scale
        # x = 1  + x/scale
        compare(x, f, g)
        # print(f'f({x:.3f}) == {np.log(iv(m/2 - 1, x))}')

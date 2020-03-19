import torch

import math
import numpy as np

"""
vMF distribution
p(e(w); µ, κ) = C_m(κ) * exp(κ * (µ @ e(w)))
C_m = κ ** (m/2 - 1) / ((2 * pi) ** (m/2) I_(m/2 - 1)(κ))

p(e_tgt; e_pred) = vMF(e_tgt); e_pred) = C_m(norm(e_pred)) * exp(e_pred @ e_tgt)
NLLvMF = - log(C_m(norm(e_pred))) - e_pred @ e_tgt
"""

bessel_coeffs = None
bessel_exps = None


def populate_bessel_consts(v, n_iters):
    global bessel_coeffs, bessel_exps
    bessel_coeffs = np.ndarray(n_iters)
    for i in range(n_iters):
        bessel_coeffs[i] = -math.log(math.factorial(i)) - math.lgamma(v + 1 + i)
    bessel_coeffs = torch.Tensor(bessel_coeffs)
    bessel_exps = torch.Tensor([2 * i + v for i in range(n_iters)])


def log_bessel(x):
    """Approximation of the log of modified Bessel function of the first kind

    `populate_bessel_consts()` must be called at least once before this in order to
    populate global variables conataining precomputed constants for this operation.
    """

    return torch.logsumexp(torch.log(x / 2) * bessel_exps + bessel_coeffs, 0)

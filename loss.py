from typing import Tuple
import math

from torch import Tensor
import numpy as np
import torch

"""
vMF distribution
p(e(w); µ, κ) = C_m(κ) * exp(κ * (µ @ e(w)))

p(e_tgt; e_pred) = vMF(e_tgt); e_pred) = C_m(norm(e_pred)) * exp(e_pred @ e_tgt)
NLLvMF = - log(C_m(norm(e_pred))) - e_pred @ e_tgt
"""

LOG_2_PI = Tensor([math.log(math.tau)])


def calculate_bessel_consts(v: float, n_iters: int) -> Tuple[Tensor, Tensor]:
    bessel_coeffs = np.ndarray(n_iters)
    for i in range(n_iters):
        bessel_coeffs[i] = -math.log(math.factorial(i)) - math.lgamma(v + 1 + i)
    bessel_coeffs = Tensor(bessel_coeffs)
    bessel_exps = Tensor([2 * i + v for i in range(n_iters)])
    return bessel_exps, bessel_coeffs


def log_bessel(x: Tensor, bessel_consts: Tuple[Tensor, Tensor]) -> Tensor:
    """Approximation of the log of modified Bessel function of the first kind

    `bessel_consts` should be retrieved from `calculate_bessel_consts()`

    """

    return torch.logsumexp(torch.log(x / 2) * bessel_consts[0] + bessel_consts[1], 0)


def log_vmf_normalizing_const(
    kappa: Tensor, m: Tensor, bessel_consts: Tuple[Tensor, Tensor]
):
    """Calculate the log normalizing constant C_m(kappa) for the vMF distribution

    C_m = κ ** (m/2 - 1) / ((2 * pi) ** (m/2) I_(m/2 - 1)(κ))
    where m is the dimension of κ

    `bessel_consts` should be retrieved from `calculate_bessel_consts()`
    """

    return (
        torch.log(kappa) * (m / 2 - 1)
        - LOG_2_PI * m / 2
        - log_bessel(kappa, bessel_consts)
    )


"""
vMF distribution
p(e(w); µ, κ) = C_m(κ) * exp(κ * (µ @ e(w)))
p(e_tgt; e_pred) = vMF(e_tgt); e_pred) = C_m(norm(e_pred)) * exp(e_pred @ e_tgt)
NLLvMF = - log(C_m(norm(e_pred))) - e_pred @ e_tgt
"""


def nll_vmf(
    e_pred: Tensor, e_tgt: Tensor, bessel_consts: Tuple[Tensor, Tensor]
) -> Tensor:
    return (
        -log_vmf_normalizing_const(e_pred.norm(), e_pred.shape[-1], bessel_consts)
        - e_tgt @ e_pred
    )

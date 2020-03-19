from typing import Tuple
import math

import numpy as np
import torch

"""
vMF distribution
p(e(w); µ, κ) = C_m(κ) * exp(κ * (µ @ e(w)))

p(e_tgt; e_pred) = vMF(e_tgt); e_pred) = C_m(norm(e_pred)) * exp(e_pred @ e_tgt)
NLLvMF = - log(C_m(norm(e_pred))) - e_pred @ e_tgt
"""

LOG_2_PI = torch.Tensor([math.log(math.tau)])


def calculate_bessel_consts(
    v: float, n_iters: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    bessel_coeffs = np.ndarray(n_iters)
    for i in range(n_iters):
        bessel_coeffs[i] = -math.log(math.factorial(i)) - math.lgamma(v + 1 + i)
    bessel_coeffs = torch.Tensor(bessel_coeffs)
    bessel_exps = torch.Tensor([2 * i + v for i in range(n_iters)])
    return bessel_exps, bessel_coeffs


def log_bessel(
    x: torch.Tensor, bessel_exps: torch.Tensor, bessel_coeffs: torch.Tensor
) -> torch.Tensor:
    """Approximation of the log of modified Bessel function of the first kind

    `bessel_exps` and `bessel_coeffs` should be retrieved from
    `calculate_bessel_consts()`

    """

    return torch.logsumexp(torch.log(x / 2) * bessel_exps + bessel_coeffs, 0)


def log_vmf_normalizing_const(
        kappa: torch.Tensor, m: torch.Tensor, bessel_exps: torch.Tensor, bessel_coeffs: torch.Tensor
):
    """Calculate the log normalizing constant C_m(kappa) for the vMF distribution

    C_m = κ ** (m/2 - 1) / ((2 * pi) ** (m/2) I_(m/2 - 1)(κ))
    where m is the dimension of κ

    `bessel_exps` and `bessel_coeffs` should be retrieved from
    `calculate_bessel_consts()`
    """

    return (
        torch.log(kappa) * (m / 2 - 1)
        - LOG_2_PI * m / 2
        - log_bessel(kappa, bessel_exps, bessel_coeffs)
    )

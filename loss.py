from typing import Tuple
import math

from torch import Tensor
import numpy as np
import torch


class VonMisesFisherLoss(torch.nn.modules.loss._Loss):
    """Loss according to a von Mises Fisher distribution

    PDF: p(e(w); µ, κ) = C_m(κ) * exp(κ * (µ @ e(w)))
        where C_m = κ ** (m/2 - 1) / ((2 * pi) ** (m/2) I_(m/2 - 1)(κ))
        where m is the dimension of κ
    """

    LOG_2_PI = Tensor([math.log(math.tau)])

    def __init__(self, input_dim: int, n_bessel_iters=10, reduction="mean") -> None:
        super(VonMisesFisherLoss, self).__init__(reduction=reduction)
        bessel_consts = VonMisesFisherLoss.calculate_bessel_consts(
            input_dim, n_bessel_iters
        )
        self.bessel_exps, self.bessel_coeffs = bessel_consts

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # TODO Add both types of regularization
        x = (
            -self.log_vmf_normalizing_const(input.norm(), input.shape[-1])
            - 1e-1*(target @ input.T)
        )
        if self.reduction == 'none':
            return x
        elif self.reduction == 'mean':
            return x.mean()
        elif self.reduction == 'mean':
            return x.sum()

    def calculate_bessel_consts(v: float, n_iters: int) -> Tuple[Tensor, Tensor]:
        bessel_coeffs = np.ndarray(n_iters)
        for i in range(n_iters):
            bessel_coeffs[i] = -math.log(math.factorial(i)) - math.lgamma(v + 1 + i)
        bessel_coeffs = Tensor(bessel_coeffs)
        bessel_exps = Tensor([2 * i + v for i in range(n_iters)])
        return bessel_exps, bessel_coeffs

    def log_bessel(self, x: Tensor) -> Tensor:
        """Approximation of the log of modified Bessel function of the first kind"""

        return torch.logsumexp(
            torch.log(x / 2) * self.bessel_exps + self.bessel_coeffs, 0
        )

    def log_vmf_normalizing_const(self, kappa: Tensor, m: Tensor) -> Tensor:
        """Calculate the log normalizing constant C_m(kappa) for the vMF distribution"""

        return (
            torch.log(kappa) * (m / 2 - 1)
            - VonMisesFisherLoss.LOG_2_PI * m / 2
            - self.log_bessel(kappa)
        )

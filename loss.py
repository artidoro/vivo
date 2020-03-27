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

    def __init__(
        self,
        input_dim: int,
        device: str = "cpu",
        n_bessel_iters=10,
        reduction="mean",
        ignore_index=-100,
    ) -> None:
        super(VonMisesFisherLoss, self).__init__(reduction=reduction)
        self.device = device
        bessel_consts = self.calculate_bessel_consts(input_dim / 2 - 1, n_bessel_iters)
        self.bessel_exps, self.bessel_coeffs = bessel_consts
        self.log_2_pi = Tensor([math.log(math.tau)]).to(self.device)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # TODO Add both types of regularization
        # Only the target and not the input vector must have unit norm
        unit_target = target / target.norm(dim=-1).reshape(-1, 1)
        # Second line is batch-wise dot product
        x = -self.log_vmf_normalizing_const(
            input.norm(dim=-1), input.shape[-1]
        ) - 1e-1 * (unit_target * input).sum(-1)
        if self.reduction == "none":
            return x
        elif self.reduction == "mean":
            return x.mean()
        elif self.reduction == "sum":
            return x.sum()

    def calculate_bessel_consts(self, v: float, n_iters: int) -> Tuple[Tensor, Tensor]:
        bessel_coeffs = np.ndarray(n_iters)
        for i in range(n_iters):
            bessel_coeffs[i] = -math.log(math.factorial(i)) - math.lgamma(v + 1 + i)
        bessel_coeffs = Tensor(bessel_coeffs).to(self.device)
        bessel_exps = Tensor([2 * i + v for i in range(n_iters)]).to(self.device)
        return bessel_exps.reshape(1, -1), bessel_coeffs.reshape(1, -1)

    def log_bessel(self, x: Tensor) -> Tensor:
        """Approximation of the log of modified Bessel function of the first kind"""

        # We need each value multipled with each coeff -> (|x|, 1) * (1, |coeffs|)
        x = x.reshape(-1, 1)
        return torch.logsumexp(
            torch.log(x / 2) * self.bessel_exps + self.bessel_coeffs, -1
        )

    def log_vmf_normalizing_const(self, kappa: Tensor, m: Tensor) -> Tensor:
        """Calculate the log normalizing constant C_m(kappa) for the vMF distribution"""

        return (
            torch.log(kappa) * (m / 2 - 1)
            - self.log_2_pi * m / 2
            - self.log_bessel(kappa)
        )

import math
from typing import Any

import torch
import torch.distributions as dist
from gpytorch.likelihoods.likelihood import _Likelihood
from gpytorch.likelihoods.noise_models import HomoskedasticNoise
from torch import Tensor

from .censored_normal import CensoredNormal


class CensoredGaussianLikelihood(_Likelihood):
    """Base class for Censored Gaussian Likelihoods."""

    def __init__(self, variance=None, low=None, high=None) -> None:
        super().__init__()

        variance = torch.tensor(1.) if variance is None else variance
        self.noise = HomoskedasticNoise()
        self.noise.initialize(noise=variance)
        self.low = low
        self.high = high

    @property
    def variance(self):
        return self.noise.noise

    def forward(self, function_samples: Tensor, *params: Any, **kwargs: Any):
        return CensoredNormal(
            function_samples, scale=self.variance.sqrt(), low=self.low, high=self.high,
            validate_args=False
        )


class CensoredGaussianLikelihoodAnalytic(CensoredGaussianLikelihood):
    standard_normal = dist.Normal(0, 1)

    def __init__(self, variance=None, low=None, high=None, jitter_=1e-16) -> None:
        super().__init__(variance, low, high)
        self.jitter_ = jitter_

    def upper_censor(self, x, s):
        return (self.high - x) / s

    def lower_censor(self, x, s):
        return (self.low - x) / s

    def expected_log_prob(
            self, target: Tensor, input: dist.MultivariateNormal, *params: Any, **kwargs: Any
    ) -> Tensor:
        mean, variance = input.mean, input.variance  # approximate posterior
        noise = self.variance  # likelihood
        sigma = noise.sqrt()
        std = variance.sqrt()

        has_high = ~torch.isinf(self.high)
        has_low = ~torch.isinf(self.low)
        upper_censor_pred = (self.high[has_high] - mean[has_high]) / std[has_high]
        lower_censor_pred = (self.low[has_low] - mean[has_low]) / std[has_low]

        upper_censor_obs = (self.high[has_high] - target[has_high]) / sigma
        lower_censor_obs = (self.low[has_low] - target[has_low]) / sigma

        # Gaussian term
        res = ((target - mean).square() + variance) / noise + noise.log() + math.log(2 * math.pi)
        res = res.mul(-0.5)
        up_cdf = torch.ones_like(res, dtype=torch.double)
        low_cdf = torch.zeros_like(res, dtype=torch.double)
        up_cdf[has_high] = self.standard_normal.cdf(upper_censor_pred)
        low_cdf[has_low] = self.standard_normal.cdf(lower_censor_pred)
        uncensored_cdf = up_cdf - low_cdf
        normal_part = res * uncensored_cdf  # maximized

        # upper term
        upper_censored_part = torch.zeros_like(normal_part, dtype=torch.double)
        x = 0.5 * (self.high[has_high] - 2 * target[has_high] + mean[has_high]) * std[has_high] * noise.reciprocal()
        upper_term_cdf = self.standard_normal.cdf(-upper_censor_obs)
        ln_cdf = torch.clamp_min(upper_term_cdf, self.jitter_).log()
        upper_term_pdf = self.standard_normal.log_prob(upper_censor_pred).exp()
        upper_censored_part[has_high] = ((ln_cdf + x) * upper_term_pdf)  # minimized

        # lower term
        lower_censored_part = torch.zeros_like(normal_part, dtype=torch.double)
        x = 0.5 * (self.low[has_low] - 2 * target[has_low] + mean[has_low]) * std[has_low] * noise.reciprocal()
        lower_term_cdf = self.standard_normal.cdf(lower_censor_obs)
        ln_cdf = torch.clamp_min(lower_term_cdf, self.jitter_).log()
        lower_term_pdf = self.standard_normal.log_prob(lower_censor_pred).exp()
        lower_censored_part[has_low] = ((ln_cdf - x) * lower_term_pdf)  # minimized

        return normal_part + upper_censored_part + lower_censored_part

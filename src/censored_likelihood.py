import math
import warnings
from copy import deepcopy
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributions as dist
from torch import Tensor
from .censored_normal import CensoredNormal
from gpytorch.likelihoods.likelihood import _Likelihood
from gpytorch.likelihoods.noise_models import HomoskedasticNoise


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
        return CensoredNormal(function_samples, scale=self.variance.sqrt(), low=self.low, high=self.high)


class CensoredGaussianLikelihoodMathematica(CensoredGaussianLikelihood):
    def __init__(self, variance=None, low=None, high=None) -> None:
        super().__init__(variance, low, high)

    def expected_log_prob(self, target: Tensor, input: dist.MultivariateNormal, *params: Any, **kwargs: Any) -> Tensor:
        noise = self.variance # likelihood
        mean, variance = input.mean, input.variance # approximate posterior

        jitter_=1e-16

        upper_censor = lambda x, s: (self.high - x) / s
        lower_censor = lambda x, s: (self.low - x) / s

        standard_normal = dist.Normal(0,1)
         # Gaussian term
        term1 = 2. + torch.log(noise) + math.log(2 * math.pi)
        term2 = ((target - mean).square() + variance) * noise.reciprocal()
        term3 = standard_normal.cdf(upper_censor(mean, variance.sqrt())) - standard_normal.cdf(lower_censor(mean, variance.sqrt()))
        normal_part_unscaled = (term1 - term2).mul(0.5)
        normal_part = normal_part_unscaled * term3

        # upper term
        x1 = (self.high - 2*target + mean) * variance.sqrt()
        term4 = torch.clamp_min(standard_normal.cdf(-upper_censor(target, variance.sqrt())), jitter_).log() + 0.5 * x1 * noise.reciprocal()
        upper_censored_part = term4 * standard_normal.log_prob(upper_censor(mean, variance.sqrt())).exp()

        # lower term
        x2 = (self.low - 2*target + mean) * variance.sqrt()
        term5 = torch.clamp_min(standard_normal.cdf(lower_censor(target, variance.sqrt())), jitter_).log() - 0.5 * x2 * noise.reciprocal()
        lower_censored_part = term5 * standard_normal.log_prob(lower_censor(mean, variance.sqrt())).exp()

        res = normal_part - lower_censored_part - upper_censored_part
        return res


class CensoredGaussianLikelihoodAnalytic(CensoredGaussianLikelihood):
    def __init__(self, variance=None, low=None, high=None, alpha=1., gamma=1., dzeta=1., jitter_=1e-16) -> None:
        super().__init__(variance, low, high)
        self.alpha = alpha
        self.gamma = gamma
        self.dzeta = dzeta
        self.jitter_ = jitter_

    def expected_log_prob(self, target: Tensor, input: dist.MultivariateNormal, *params: Any, **kwargs: Any) -> Tensor:
        noise = self.variance # likelihood
        mean, variance = input.mean, input.variance # approximate posterior

        upper_censor = lambda x, s: (self.high - x) / s
        lower_censor = lambda x, s: (self.low - x) / s

        standard_normal = dist.Normal(0,1)
         # Gaussian term
        term1 = torch.log(noise) + math.log(2 * math.pi)
        term2 = ((target - mean).square() + variance) * noise.reciprocal()
        term3 = standard_normal.cdf(upper_censor(mean, variance.sqrt())) - standard_normal.cdf(lower_censor(mean, variance.sqrt()))
        normal_part_unscaled = (term1 - term2).mul(0.5)
        normal_part = normal_part_unscaled * term3

        # upper term
        x1 = (self.high - 2*target + mean) * variance.sqrt()
        term4 = torch.clamp_min(standard_normal.cdf(-upper_censor(target, variance.sqrt())), self.jitter_).log() + 0.5 * x1 * noise.reciprocal()
        upper_censored_part = term4 * standard_normal.log_prob(upper_censor(mean, variance.sqrt())).exp()

        # lower term
        x2 = (self.low - 2*target + mean) * variance.sqrt()
        term5 = torch.clamp_min(standard_normal.cdf(lower_censor(target, variance.sqrt())), self.jitter_).log() - 0.5 * x2 * noise.reciprocal()
        lower_censored_part = term5 * standard_normal.log_prob(lower_censor(mean, variance.sqrt())).exp()

        res = self.alpha * normal_part - self.gamma * upper_censored_part - self.dzeta * lower_censored_part
        return res


class CensoredGaussianLikelihoodMathematica(CensoredGaussianLikelihoodAnalytic):
    def expected_log_prob(self, target: Tensor, input: dist.MultivariateNormal, *params: Any, **kwargs: Any) -> Tensor:
        noise = self.variance  # likelihood
        sigma = torch.sqrt(noise)
        sigma2 = noise

        mean, variance = input.mean, input.variance  # approximate posterior
        m = mean
        s = variance

        mu = target
        l = self.low
        u = self.high

        log = lambda x: torch.log(torch.maximum(x,torch.as_tensor(1e-8).to(x)))
        phi = lambda m, s, u: torch.exp(-((m - u) ** 2 / (2 * s ** 2)))
        Phi = lambda m, s, a: torch.special.erf((a - m) / (np.sqrt(2) * s))

        sqrt2pi = np.sqrt(2 * np.pi)
        log2 = np.log(2)
        logpi = np.log(np.pi)

        return (
                (log((1 / 2) * (1 - Phi(l, sigma, mu))) * phi(l, s, m)) / (sqrt2pi * s) +
                (log(1 + (1 / 2) * (-1 + Phi(u, sigma, mu))) * phi(u, s, m)) / (sqrt2pi * s) +
                (1 / (4 * sqrt2pi * sigma2)) * (
                        -2 * s * (l + m - 2 * mu) * phi(l, s, m) +
                        2 * s * (m + u - 2 * mu) * phi(m, s, u) -
                        sqrt2pi * Phi(m, s, l) * (
                                l ** 2 - m ** 2 - s ** 2 - 2 * l * mu + 2 * m * mu - sigma2 * log2 - sigma2 * logpi +
                                2 * sigma2 * log(phi(l, sigma, mu) / sigma)
                        ) +
                        sqrt2pi * Phi(u, s, m) * (
                                m ** 2 + s ** 2 - u ** 2 - 2 * m * mu + 2 * u * mu + sigma2 * log2 + sigma2 * logpi -
                                2 * sigma2 * log(phi(u, sigma, mu) / sigma)
                        )
                )
        )
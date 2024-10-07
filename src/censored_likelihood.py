import math
import warnings
from copy import deepcopy
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributions as dist
from torch import Tensor
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.likelihoods.noise_models import HomoskedasticNoise
from .censored_normal import CensoredNormal


class CensoredGaussianLikelihood(Likelihood):
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



class CensoredGaussianLikelihoodAnalytic(CensoredGaussianLikelihood):
    def __init__(self, variance=None, low=None, high=None, alpha=1., gamma=1., dzeta=1., jitter_=1e-32) -> None:
        super().__init__(variance, low, high)
        self.alpha = alpha
        self.gamma = gamma
        self.dzeta = dzeta
        self.jitter_ = jitter_

    def _expected_log_prob_terms(self, target: Tensor, input: dist.MultivariateNormal, *params: Any, **kwargs: Any) -> Tensor:

        noise = self.variance # likelihood
        sigma = noise.sqrt()
        mean, variance = input.mean, input.variance # approximate posterior
        std = variance.sqrt()

        upper_censor = lambda x, s: (self.high - x) / s
        lower_censor = lambda x, s: (self.low - x) / s

        standard_normal = dist.Normal(0., 1.)

        # continuous part --> this should be reasonably numerically stable
         # Gaussian term
        res = ((target - mean).square() + variance) / noise + noise.log() + math.log(2 * math.pi)
        res = res.mul(-0.5)
        uncensored_cdf = standard_normal.cdf(upper_censor(mean, std)) - standard_normal.cdf(lower_censor(mean, std))
        normal_part = res * uncensored_cdf  # maximized
        # upper boundary asymmetry contribution
        x_u = torch.where(torch.isinf(torch.Tensor([self.high])), 0.,
                          0.5 * (self.high - 2*target + mean) * std * noise.reciprocal())
        upper_term_pdf = standard_normal.log_prob(upper_censor(mean, std)).exp()
        upper_censored_part_cont = x_u * upper_term_pdf # minimized
        # lower boundery asymmetry contribution
        x_l = torch.where(torch.isinf(torch.Tensor([self.low])), 0.,
                          0.5 * (self.low - 2*target + mean) * std * noise.reciprocal())
        lower_term_pdf = standard_normal.log_prob(lower_censor(mean, std)).exp()
        lower_censored_part_cont =  x_l * lower_term_pdf # maximized

        upper_term_cdf = standard_normal.cdf(-upper_censor(target, sigma))
        ln_cdf = torch.log(torch.where(upper_term_cdf < self.jitter_, self.jitter_, upper_term_cdf))
        upper_censored_part_atom = ln_cdf * upper_term_pdf  # minimized

        lower_term_cdf = standard_normal.cdf(lower_censor(target, sigma))
        ln_cdf = torch.log(torch.where(lower_term_cdf < self.jitter_, self.jitter_, lower_term_cdf))
        lower_censored_part_atom = ln_cdf * lower_term_pdf  # minimized

        Phi = lambda m, s, a: torch.special.erf((a - m) / (np.sqrt(2) * s))

        return dict(normal_part_cont=normal_part,
                    normal_part_u = upper_censored_part_cont,
                    normal_part_l = lower_censored_part_cont,
                    normal_part_cdf_scaler=uncensored_cdf,
                    normal_part=normal_part - lower_censored_part_cont + upper_censored_part_cont,
                    upper_censored_part=upper_censored_part_atom, upper_term_pdf=upper_term_pdf,
                    lower_censored_part=lower_censored_part_atom, lower_term_pdf=lower_term_pdf,
                    lower_cdf_pred=Phi(self.low, std, mean), upper_cdf_pred=Phi(self.high, std, mean),
                    lower_cdf_target=Phi(self.low, sigma, target), upper_cdf_target=Phi(self.high, sigma, target)
                    )

    def expected_log_prob(self, target: Tensor, input: dist.MultivariateNormal, *params: Any, **kwargs: Any) -> Tensor:
      terms = self._expected_log_prob_terms(target, input, *params, **kwargs)
      return self.alpha * terms['normal_part'] + self.gamma * terms['upper_censored_part'] + self.dzeta * terms['lower_censored_part']

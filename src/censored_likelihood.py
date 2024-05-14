import math
import warnings
from copy import deepcopy
from typing import Any, Optional, Tuple, Union

import torch
from torch import Tensor
from .censored_normal import CensoredNormal
from gpytorch.likelihoods.likelihood import _Likelihood, _OneDimensionalLikelihood
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

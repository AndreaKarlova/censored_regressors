import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro.contrib.gp.likelihoods.likelihood import Likelihood
from pyro.nn.module import PyroParam, pyro_method
from .censored_normal import PyroCensoredNormal

class Gaussian(Likelihood):
    def __init__(self, variance=None):
        super().__init__()

        variance = torch.tensor(1.) if variance is None else variance
        self.variance = PyroParam(variance, constraints.positive)

    def forward(self, f_loc, f_var, y=None):
        y_dist = dist.Normal(f_loc + torch.randn(f_loc.dim(), device=f_loc.device)*f_var, self.variance.sqrt())
        self.y_dist = y_dist
        if y is not None:
            y_dist = y_dist.expand_by(y.shape[:-f_loc.dim()]).to_event(y.dim())
        return pyro.sample(self._pyro_get_fullname("y"), y_dist, obs=y)

class CensoredHomoscedGaussian(Likelihood):

    def __init__(self, variance=None, low=None, high=None):
        super().__init__()

        variance = torch.tensor(1.) if variance is None else variance
        self.variance = PyroParam(variance, constraints.positive)
        self.low = low
        self.high = high

    def forward(self, f_loc, f_var, y=None):
        y_dist = PyroCensoredNormal(loc=f_loc + torch.randn(f_loc.dim(), device=f_loc.device)*f_var, scale=self.variance.sqrt(), low=self.low, high=self.high)
        self.y_dist = y_dist
        if y is not None:
            y_dist = y_dist.expand_by(y.shape[:-f_loc.dim()]).to_event(y.dim())
        return pyro.sample("y", y_dist, obs=y)

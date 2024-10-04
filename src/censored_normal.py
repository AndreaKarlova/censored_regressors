import math
from numbers import Number, Real

import torch
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import _standard_normal, broadcast_all

__all__ = ["CensoredNormal"]


class CensoredNormal(ExponentialFamily):
    """
    Creates a censored normal (also called Gaussian) distribution parameterized by
    :attr:`loc` and :attr:`scale` and :attr:`low`  and :attr:`high`

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = CensoredNormal(
          torch.tensor([0.0]), torch.tensor([1.0]),
          torch.tensor([-0.5]), torch.tensor([0.5])
          )
        >>> m.sample(sample_shape=(3,1))
        tensor([[0.5000],
            [0.5000],
            [0.1836]])
        tensor([ 0.1046])

    Args:
        loc (float or Tensor): mean of the base normal distribution (often referred to as mu) to which the censoring is applied
        scale (float or Tensor): standard deviation of the base normal distribution to which the censoring is applied
        low (float or Tensor): lower censoring boundary
        high (float or Tensor): upper censoring boundary
    """
    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
        "low": constraints.dependent(is_discrete=False, event_dim=0),
        "high": constraints.dependent(is_discrete=False, event_dim=0)}
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        cdf_high = self._normal_cdf(self.high) # procedure rescales the given value
        pdf_high = self._normal_log_prob(self.high).exp()

        cdf_low = self._normal_cdf(self.low)
        pdf_low = self._normal_log_prob(self.low).exp()

        term1 = cdf_high - cdf_low
        term2 = pdf_high - pdf_low

        term3_l = torch.where(torch.isinf(self.low), 0., self.low * cdf_low)
        term3_u = torch.where(torch.isinf(self.high), 0., self.high * (1. - cdf_high))

        return self.loc * term1 - (self.scale**2) * term2 + term3_l + term3_u

    @property
    def stddev(self):
        return self.variance.sqrt()

    @property
    def variance(self):
        x_high = (self.high - self.loc)
        cdf_high = self._normal_cdf(self.high)
        pdf_high = math.exp(self._normal_log_prob(self.high))

        x_low = (self.low - self.loc)
        cdf_low = self._normal_cdf(self.low)
        pdf_low = math.exp(self._normal_log_prob(self.low))

        scaler = self.loc**2 + self.scale**2
        square_scale = self.scale**2
        square_low = self.low**2
        square_high = self.high**2

        term1 = cdf_high - cdf_low
        term2 = pdf_high - pdf_low

        term3_l = torch.where(torch.isinf(self.low), 0., x_low * pdf_low)
        term3_u = torch.where(torch.isinf(self.high), 0., x_high * pdf_high)
        term3 = term3_u - term3_l

        term4_l = torch.where(torch.isinf(self.low), 0., square_low * cdf_low)
        term4_u = torch.where(torch.isinf(self.high), 0., square_high * (1. - cdf_high))

        term4 = term4_l + term4_u
        return  scaler * term1 - 2 * self.loc * square_scale * term2  - square_scale * term3 + term4 - self.mean**2


    def __init__(self, loc, scale, low, high, validate_args=None):
        self.loc, self.scale, self.low, self.high = broadcast_all(loc, scale, low, high)
        if isinstance(loc, Number) and isinstance(scale, Number) and isinstance(low, Number) and isinstance(high, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        support = constraints.interval(low, high)
        self.jitter_ = 1e-16
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(CensoredNormal, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.low = self.low.expand(batch_shape)
        new.high = self.high.expand(batch_shape)
        super(CensoredNormal, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return constraints.interval(self.low, self.high)


    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            samples = torch.normal(self.loc.expand(shape), self.scale.expand(shape))
            return samples.clamp(min=self.low, max=self.high)


    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        rsamples =  self.loc + eps * self.scale
        return rsamples.clamp(min=self.low, max=self.high)

    def pdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        probs = torch.exp(self.log_prob(value))
        x_high = - (self.high - self.loc) / self.scale
        lower_cdf_mass = self._normal_cdf(self.low)
        upper_cdf_mass = self._normal_cdf_standardized(x_high)
        probs = torch.where(value <= self.low, lower_cdf_mass, probs)
        probs = torch.where(value >= self.high, upper_cdf_mass, probs)
        return probs

    def log_prob(self, value):
        """jitter: used to bounce off NormCDF from 0 before applying log """
        if self._validate_args:
            self._validate_sample(value)
        log_probs = self._normal_log_prob(value)
        x_high = - (self.high - self.loc) / self.scale
        lower_log_cdf_mass = math.log(self._normal_cdf(self.low) + self.jitter_) if isinstance(self._normal_cdf(self.low) + self.jitter_,
                                                                                                Number) else (self._normal_cdf(self.low) + self.jitter_).log()
        upper_log_cdf_mass = math.log(self._normal_cdf_standardized(x_high) + self.jitter_) if isinstance(self._normal_cdf_standardized(x_high) + self.jitter_,
                                                                                                    Number) else (self._normal_cdf_standardized(x_high) + self.jitter_).log()
        log_probs = torch.where(value <= self.low, lower_log_cdf_mass, log_probs)
        log_probs = torch.where(value >= self.high, upper_log_cdf_mass, log_probs)
        return log_probs

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        cdf_ = self._normal_cdf(value)
        cdf_ = torch.where(value < self.low, 0, cdf_)
        cdf_ = torch.where(value > self.high, 1, cdf_)
        return cdf_

    def icdf(self, value):
        result = self._normal_icdf(value)
        return result.clamp(min=self.low, max=self.high)

    def entropy(self):
        upper_censor = lambda x, s: (self.high - x) / s
        lower_censor = lambda x, s: (self.low - x) / s
        standard_normal = dist.Normal(0., 1.)

        cdf_high = standard_normal.cdf(upper_censor(self.loc, self.scale))
        cdf_high_neg = standard_normal.cdf(-upper_censor(self.loc, self.scale))
        pdf_high = standard_normal.log_prob(upper_censor(self.loc, self.scale)).exp()

        cdf_low = standard_normal.cdf(lower_censor(self.loc, self.scale))
        pdf_low = standard_normal.log_prob(lower_censor(self.loc, self.scale)).exp()

        if cdf_low <=self.jitter_:
          logcdf_x_low = math.log(cdf_low + self.jitter_) if isinstance(cdf_low + self.jitter_, Number) else (cdf_low + self.jitter_).log()
        else:
          logcdf_x_low = math.log(cdf_low) if isinstance(cdf_low, Number) else cdf_low.log()
        if cdf_high_neg <= self.jitter_:
          logcdf_x_high = math.log(1. - cdf_high + self.jitter_) if isinstance(cdf_high_neg + self.jitter_, Number) else (cdf_high_neg + self.jitter_).log()
        else:
          logcdf_x_high = math.log(cdf_high_neg) if isinstance(cdf_high_neg, Number) else cdf_high_neg.log()

        term1 = self._normal_entropy() * (cdf_high - cdf_low)

        term2_u = torch.where(torch.isinf(self.high), 0, upper_censor(self.loc, self.scale) * pdf_high)
        term2_l = torch.where(torch.isinf(self.low), 0, lower_censor(self.loc, self.scale) * pdf_low)
        term2 = 0.5 * (term2_u - term2_l)

        term3_u = torch.where(torch.isinf(self.high), 0, logcdf_x_high * cdf_high_neg)
        term3_u = torch.where(1. - cdf_high <= self.jitter_, 0, logcdf_x_high * cdf_high_neg)
        term3_l = torch.where(torch.isinf(self.low), 0, logcdf_x_low * cdf_low)
        term3_l = torch.where(cdf_low <= self.jitter_, 0, logcdf_x_low * cdf_low)
        term3 = term3_u + term3_l

        return term1 - term2 - term3

    def _normal_log_prob(self, value):
            # compute the variance
            var = self.scale**2
            log_scale = (
                math.log(self.scale) if isinstance(self.scale, Real) else self.scale.log()
            )
            return (
                -((value - self.loc) ** 2) / (2 * var)
                - log_scale
                - math.log(math.sqrt(2 * math.pi))
            )

    def _normal_cdf(self, value):
        return 0.5 * (
            1 + torch.erf((value - self.loc) * self.scale.reciprocal() / math.sqrt(2))
        )

    def _normal_cdf_standardized(self, value):
        return 0.5 * (
            1 + torch.erf(value / math.sqrt(2))
        )

    def _normal_icdf(self, value):
        return self.loc + self.scale * torch.erfinv(2 * value - 1) * math.sqrt(2)

    def _normal_icdf_standardized(self, value):
        return torch.erfinv(2 * value - 1) * math.sqrt(2)

    def _normal_entropy(self):
        # log(sqrt(2 pi e) * sigma) = 0.5 * log(2 pi e) + log(sigma) = 0.5 + 0.5 (log(2 pi)) + log(sigma)
        log_scale = (
                math.log(self.scale) if isinstance(self.scale, Real) else self.scale.log()
            )
        return 0.5 + 0.5 * math.log(2 * math.pi) + log_scale

    @property
    def _normal_natural_params(self):
        return (self.loc / self.scale.pow(2), -0.5 * self.scale.pow(2).reciprocal())

    def _log_normalizer(self, x, y):
        return -0.25 * x.pow(2) / y + 0.5 * torch.log(-math.pi / y)


    


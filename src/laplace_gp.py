import functools

import botorch
import gpytorch
import torch
from linear_operator import to_linear_operator


__all__ = ['HomemadeApproximateGP', 'train_laplace_approximation']

class HomemadeApproximateGP(gpytorch.Module):
    def __init__(
            self, train_x: torch.Tensor, train_y: torch.Tensor,
            covar_module: gpytorch.kernels.Kernel, mean_module: gpytorch.means.Mean,
            approximate_posterior: gpytorch.distributions.MultivariateNormal = None
    ):
        super().__init__()
        self.train_inputs = (train_x,)
        self.train_targets = train_y
        self.mean_module = mean_module
        self.covar_module = covar_module
        self._approximate_posterior = approximate_posterior
        assert self._approximate_posterior is None or self._approximate_posterior.shape()[0] == train_x.shape[0]

    @property
    def approximate_posterior(self):
        return self._approximate_posterior

    @approximate_posterior.setter
    def approximate_posterior(self, value):
        assert value is None or value.shape()[0] == self.train_inputs[0].shape[0]
        self._approximate_posterior = value

    def forward(self, *inputs, **kwargs):
        return self.prior_predictive(inputs[0])

    def prior_predictive(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def kz(self, x=None):
        return self.covar_module(self.train_inputs[0], x)

    def posterior_predictive(self, a):
        x = self.train_inputs[0]

        m, S = self.approximate_posterior.loc, self.approximate_posterior.lazy_covariance_matrix

        kz = to_linear_operator(self.kz())
        kzx = self.kz(x)
        kaa = self.covar_module(a)
        kza = self.kz(a) if a is not x else kzx
        kaz = torch.transpose(kza, 0, 1)
        kzi_kza = kz.inv_matmul(kza.to_dense())
        qaa = kaz @ kzi_kza

        mean = (kaz @ kz.inv_matmul((m - self.mean_module(x))[:, None]))[..., 0]
        covar = kaa - qaa + torch.transpose(kzi_kza, 0, 1) @ S @ kzi_kza

        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(a) + mean,
            to_linear_operator(covar)
        )

    def log_posterior(self, likelihood, f):
        return (
                self.prior_predictive(*self.train_inputs).log_prob(f) +
                likelihood(f).log_prob(self.train_targets).sum(-1)
        )


def train_laplace_approximation(
        model: HomemadeApproximateGP,
        likelihood: gpytorch.likelihoods.Likelihood,
        initial_map=None,
        callback=None, options=None
) -> botorch.optim.OptimizationResult:
    """
    Trains an approximate GP model with Laplace approximation
    :param model: Instance of HomemadeApproximateGP
    :param likelihood: A GPyTorch likelihood
    :param initial_map: Optional initial MAP estimate
    :return: Result of L-BFGS-B optimization
    """
    map_estimate = torch.zeros_like(model.train_targets, requires_grad=True)
    with torch.no_grad():
        if initial_map is not None:
            map_estimate.set_(initial_map.clone())

    parameters = dict(
        **botorch.optim.utils.get_parameters_and_bounds(model)[0],
        **botorch.optim.utils.get_parameters_and_bounds(likelihood)[0],
        map_estimate=map_estimate,
    )

    closure = botorch.optim.closures.core.ForwardBackwardClosure(
        forward=functools.partial(lambda m, l, x: -m.log_posterior(l, x), model, likelihood, map_estimate),
        parameters=parameters,
    )

    res = botorch.optim.core.scipy_minimize(
        closure=closure,
        parameters=parameters,
        bounds=None,
        method='L-BFGS-B',
        options=options,
        callback=callback,
    )

    hessian = torch.autograd.functional.hessian(functools.partial(model.log_posterior, likelihood), map_estimate)
    hessian = (hessian + hessian.T).div(2)

    laplace_dist = torch.distributions.MultivariateNormal(
        loc=map_estimate,
        precision_matrix=-hessian
    )
    model.approximate_posterior = gpytorch.distributions.MultivariateNormal(
        mean=laplace_dist.loc, covariance_matrix=laplace_dist.covariance_matrix
    )
    return res

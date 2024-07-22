import math

import gpytorch
import torch

from .censored_normal import CensoredNormal

__all__ = ['ExactGP', 'VariationalGP', 'LargeFeatureExtractor', 'CensoredMLL', 'initialize_gp']


class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self, data_dim, out_dim, latent_dim=10):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(data_dim, latent_dim))
        self.add_module('relu', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(latent_dim, out_dim))

    def reset_parameters(self):
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()


class BaseGPModel:
    def _initialize_parameters_(self, train_inputs, train_targets, categorical_size, kernel_type):
        d = train_inputs.shape[1]
        self.train_inputs = (train_inputs,)
        self.train_targets = train_targets
        self.mean_module = gpytorch.means.ConstantMean()
        self.feature_extractor = None
        self.kernel_type = kernel_type
        kernel_type, nu, *kernel_args = kernel_type
        if kernel_type == 'nn':
            q = kernel_args[0]
            self.feature_extractor = LargeFeatureExtractor(d, q).to(self.train_inputs[0])
            self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)
            if math.isinf(nu):
                base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=q)
            else:
                base_kernel = gpytorch.kernels.MaternKernel(nu=nu, ard_num_dims=q)
            self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
        elif kernel_type == 'prod':
            q = d
            if math.isinf(nu):
                base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=q)
            else:
                base_kernel = gpytorch.kernels.MaternKernel(nu=nu, ard_num_dims=q)
            self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
        elif kernel_type == 'xcat':
            self.continuous_columns = [i for i in range(d) if i not in categorical_size]
            self.covar_module = (
                    gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(
                        nu=nu,
                        ard_num_dims=len(self.continuous_columns),
                        active_dims=self.continuous_columns
                    )) *
                    gpytorch.kernels.AdditiveKernel(*(
                        gpytorch.kernels.IndexKernel(d, rank=d, active_dims=(i,))
                        for i, d in categorical_size.items()
                    ))
            )
        else:
            raise Exception(f'Missing kernel_type: {kernel_type!r} ({nu},{kernel_args})')

    def forward(self, data):
        if self.feature_extractor is not None:
            data = self.feature_extractor(data)
            data = self.scale_to_bounds(data)
        mean_x = self.mean_module(data)
        covar_x = self.covar_module(data)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class VariationalGP(BaseGPModel, gpytorch.models.ApproximateGP):
    def __init__(
            self, train_inputs, train_targets, categorical_size, likelihood=None,
            kernel_type=('prod', 5 / 2)
    ):
        inducing_points = train_inputs
        d = inducing_points.shape[1]
        # defines approximation
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=False
        )
        super().__init__(variational_strategy)
        super()._initialize_parameters_(train_inputs, train_targets, categorical_size, kernel_type)


class ExactGP(BaseGPModel, gpytorch.models.ExactGP):
    def __init__(
            self, train_inputs, train_targets, categorical_size, likelihood,
            kernel_type=('prod', 5 / 2)
    ):
        super().__init__(train_inputs, train_targets, likelihood)
        super()._initialize_parameters_(train_inputs, train_targets, categorical_size, kernel_type)
        self.variational_strategy = None


class CensoredMLL(gpytorch.mlls.MarginalLogLikelihood):
    def __init__(self, likelihood, model):
        super().__init__(likelihood, model)

    def forward(self, function_dist, target, *params, **kwargs):
        output = CensoredNormal(
            function_dist.loc, function_dist.stddev, self.likelihood.low, self.likelihood.high
        )
        res = output.log_prob(target)
        return res.mean()


def initialize_gp(mll: gpytorch.mlls.MarginalLogLikelihood, lmbd=4, kappa=10):
    """
    :param lmbd: λ∈[1,10], 1 -> very linear, 10 -> very non-linear
    :param kappa: κ∈[2,100], signal-to-noise ratio
    """

    model, lik = mll.model, mll.likelihood
    is_exact = isinstance(model, gpytorch.models.ExactGP)
    x, y = model.train_inputs[0], model.train_targets

    noise = lik if isinstance(lik, gpytorch.likelihoods.GaussianLikelihood) else lik.noise
    kernel_type, nu, *kernel_args = model.kernel_type
    if kernel_type == 'prod':
        model.covar_module.base_kernel.initialize(lengthscale=x.std(axis=0) / lmbd)
        model.covar_module.initialize(outputscale=y.var())
    elif kernel_type == 'nn':
        model.covar_module.base_kernel.initialize(lengthscale=lmbd)
        model.covar_module.initialize(outputscale=y.var())
        model.feature_extractor.reset_parameters()
    elif kernel_type == 'xcat':
        cont_kern = model.covar_module.kernels[0]
        cont_kern.base_kernel.initialize(lengthscale=x[:, model.continuous_columns].std(axis=0) / lmbd)
        cont_kern.initialize(outputscale=y.var())
        for index_kern in model.covar_module.kernels[1].kernels:
            index_kern.initialize(
                covar_factor=torch.randn(index_kern.covar_factor.shape),
                raw_var=torch.randn(index_kern.raw_var.shape)
            )
    model.mean_module.initialize(constant=y.mean())
    noise.initialize(noise=(y.std() / kappa).square())

    if not is_exact:
        model.variational_strategy._variational_distribution.initialize(
            variational_mean=torch.zeros(len(x)),
            chol_variational_covar=torch.eye(len(x), len(x))
        )

from copy import deepcopy

import botorch
import gpytorch
import numpy as np
import pandas as pd

from .models import CensoredMLL, initialize_gp


def set_trainable(train, *modules):
    for module in modules:
        if module is not None:
            module.requires_grad_(train)


def train_mll(mll, parameters=None):
    mll.train()
    try:
        trace = []
        res = botorch.optim.fit.fit_gpytorch_mll_scipy(
            mll, parameters=parameters,
            callback=lambda p, r: trace.append(r)
        )
        if len(trace) > 0:
            trace[-1] = res
        else:
            trace.append(res)
        return (res.status is not botorch.optim.core.OptimizationStatus.SUCCESS), trace
    except botorch.optim.utils.common.NotPSDError:
        return True, []


def ExactMLL(likelihood, model):
    if isinstance(likelihood, gpytorch.likelihoods.GaussianLikelihood):
        return gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    else:
        return CensoredMLL(likelihood, model)


def train_gp(
        likelihood, model,
        lmbdas=(4,), kappas=(10,),
        use_mll=False,
):
    if model.variational_strategy is None:
        elbo = ExactMLL(likelihood, model)
        use_mll = False
    else:
        elbo = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=model.train_targets.shape[0])

    results = []

    for i, (lmbda, kappa) in enumerate(zip(lmbdas, kappas)):
        initialize_gp(elbo, lmbda, kappa)
        set_trainable(True, likelihood, model)
        if use_mll:
            exact_mll = ExactMLL(likelihood, model)
            set_trainable(False, model.variational_strategy)
            failed, exact_trace = train_mll(exact_mll)
            if failed:
                continue
            set_trainable(False, likelihood, model)
            set_trainable(True, model.variational_strategy)
        else:
            exact_trace = []
        failed, trace = train_mll(elbo)
        if failed:
            continue
        results.append([
            trace, exact_trace,
            lmbda, kappa,
            deepcopy(model.state_dict()), deepcopy(likelihood.state_dict())
        ])
    results = pd.DataFrame(
        sorted(results, key=lambda x: x[0][-1].fval),
        columns=['trace', 'exact_trace', 'lambda', 'kappa', 'model', 'likelihood']
    )

    best_model, best_likelihood = results.loc[0, ['model', 'likelihood']].values
    model.load_state_dict(best_model)
    likelihood.load_state_dict(best_likelihood)
    return results

# %%
import os
import random
import sys
from pathlib import Path
from typing import Literal

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import torch
from sklearn.model_selection import train_test_split

import wandb
from config_parser import Optional, typecheck_schema, load_configs

sys.path.insert(0, '..')
from src.censored_likelihood import CensoredGaussianLikelihoodAnalytic
from src.models import ExactGP, VariationalGP
from src.training import train_gp

os.environ["WANDB_SILENT"] = "true"

# %% md [markdown]
# # Parameters
# %%
config_types = dict(
    dataset=str,
    censoring_relative_slack=Optional(int, None),
    include_categorical=Optional(bool, False),
    log_transform=Optional(bool, False),
    kernel_type=str,
    likelihood_type=Literal['gaussian', 'censored'],
    model_type=Literal['exact', 'variational'],
    use_mll=bool,
    optimizer=Optional(str, 'scipy'),
    n_starts=Optional(int, 10),
    data_random_seed=Optional(int, 11011996),
    init_random_seed=Optional(int, 42),
)
config = load_configs(sys.argv[1:])
# config = dict(
#     dataset='gbsg_cancer',
#     include_categorical=False,
#     kernel_type="('prod', 1/2)",
#     likelihood_type='censored',
#     model_type='variational',
#     use_mll=True,
#     optimizer='adam'
# )
config = typecheck_schema(config, config_types)
print(config)
print('=' * 24)
# %%
dataset = config['dataset']
censoring_relative_slack = config['censoring_relative_slack']
include_categorical = config['include_categorical']
log_transform = config['log_transform']
kernel_type = config['kernel_type'] = eval(config['kernel_type'])
assert isinstance(kernel_type, tuple)
likelihood_type = config['likelihood_type']
model_type = config['model_type']
use_mll = config['use_mll']
optimizer = config['optimizer']
n_starts = config['n_starts']
data_random_seed = config['data_random_seed']
init_random_seed = config['init_random_seed']
dump_dir = Path('dump')

run = wandb.init(
    project="censored-gps",
    config=config
)
# %% md [markdown]
# # Load dataset
# %%
random.seed(data_random_seed)
torch.manual_seed(data_random_seed)
np.random.seed(data_random_seed)

# %%
import data

(
    full_x, full_y, cat_vars, left_bound, right_bound, is_censored
) = getattr(data, dataset)(include_categorical=include_categorical)
cont_vars = full_x.columns.difference(cat_vars.index)

if log_transform:
    full_y = np.log(full_y)
    left_bound[~np.isinf(left_bound)] = np.log(left_bound[~np.isinf(left_bound)])
    right_bound[~np.isinf(right_bound)] = np.log(right_bound[~np.isinf(right_bound)])

train_idx, test_idx = train_test_split(range(len(full_x)), random_state=data_random_seed)  # , test_size = 0.8)

train_x_mean = full_x.loc[train_idx, cont_vars].mean()
train_x_std = full_x.loc[train_idx, cont_vars].std()
train_y_mean = full_y[train_idx].mean()
train_y_std = full_y[train_idx].std()

full_x[cont_vars] = (full_x[cont_vars] - train_x_mean) / train_x_std
full_y = (full_y - train_y_mean) / train_y_std

left_bound = (left_bound - train_y_mean) / train_y_std
right_bound = (right_bound - train_y_mean) / train_y_std

x = torch.as_tensor(full_x.loc[train_idx].values)
y = torch.as_tensor(full_y.loc[train_idx].values)
test_x = torch.as_tensor(full_x.loc[test_idx].values)
test_y = torch.as_tensor(full_y.loc[test_idx].values)

lik_right_bound = right_bound.copy()
lik_left_bound = left_bound.copy()
if censoring_relative_slack is not None:
    lik_right_bound[np.isinf(right_bound)] = (
            full_y[np.isinf(right_bound)] + censoring_relative_slack * np.abs(full_y[np.isinf(right_bound)])
    )
    lik_left_bound[np.isinf(left_bound)] = (
            full_y[np.isinf(left_bound)] - censoring_relative_slack * np.abs(full_y[np.isinf(left_bound)])
    )
lik_right_bound = torch.as_tensor(lik_right_bound.loc[train_idx].values) + 1e-6
lik_left_bound = torch.as_tensor(lik_left_bound.loc[train_idx].values) - 1e-6
# %% md [markdown]
# # Train and test gp
# %%
if likelihood_type == 'censored':
    likelihood = CensoredGaussianLikelihoodAnalytic(low=lik_left_bound, high=lik_right_bound)
elif likelihood_type == 'gaussian':
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
else:
    raise RuntimeError(f'Unknown likelihood type: {likelihood_type!r}')

if model_type == 'exact':
    model = ExactGP(x, y, cat_vars, likelihood, kernel_type)
elif model_type == 'variational':
    model = VariationalGP(x, y, cat_vars, likelihood, kernel_type)
else:
    raise RuntimeError(f'Unknown model type: {model_type!r}')

# %%
rng = np.random.default_rng(init_random_seed)
lmbdas = np.round(rng.uniform(1, 10, size=n_starts), 2)
kappas = np.round(np.exp(rng.uniform(np.log(2), np.log(100), size=n_starts)), 2)

if optimizer == 'scipy':
    optimizer = None
elif optimizer == 'adam':
    optimizer = torch.optim.Adam

results = train_gp(likelihood, model, lmbdas, kappas, use_mll=use_mll, optimizer=optimizer, step_limit=None)


# %%
def plot_errors(ax, y, pred, err_std, is_censored):
    with torch.no_grad():
        y_range = np.array([min(y), max(y)])
        ax.plot(y_range, y_range, '--k')
        ax.fill_betweenx(y_range, y_range - 2 * err_std, y_range + 2 * err_std, alpha=0.1, color='k')
        ferror = 2 * pred.stddev
        ax.errorbar(
            y[~is_censored], pred.loc[~is_censored],
            yerr=ferror[~is_censored],
            marker='.', color='C0', ls='None', capsize=3
        )
        ax.errorbar(
            y[is_censored], pred.loc[is_censored],
            yerr=ferror[is_censored],
            marker='.', color='C1', ls='None', capsize=3
        )
        xlim = ax.get_xlim()
        ax.plot(xlim, np.ones_like(xlim) * model.mean_module.constant.item(), '-.k', alpha=0.5)
        ax.set_xlim(xlim)


# %%
for _, v in results.iterrows():
    lmbd, kappa, m_state, l_state = v.loc[['lambda', 'kappa', 'model', 'likelihood']].values
    if v.name > 0:
        run = wandb.init(project="censored-gps", config=config)
    run.config.update({'kappa': v['kappa'], 'lambda': v['lambda']})
    final_step = 0
    for i in v['exact_trace']:
        run.log({'mll': i.fval}, step=i.step)
        final_step = i.step
    for i in v['trace']:
        pass
        run.log({'elbo': i.fval}, step=i.step + final_step)
    final_step = i.step + final_step

    model.load_state_dict(m_state)
    likelihood.load_state_dict(l_state)
    export_dir = dump_dir / run.name
    export_dir.mkdir(parents=True, exist_ok=True)

    torch.save(m_state, export_dir / 'model.pth')
    torch.save(l_state, export_dir / 'lik.pth')
    wandb.save(f'{export_dir.absolute()}/*.pth', base_path=str(export_dir.absolute()), policy='end')

    noise = (
        likelihood.noise
        if isinstance(likelihood, gpytorch.likelihoods.GaussianLikelihood)
        else likelihood.noise.noise
    )
    model.eval()
    likelihood.eval()

    # print(
    #     # f'ELBO: {elbo.item()}',
    #     'Signal to noise ratio: '
    #     # f'{100 * np.sqrt(censored_model.covar_module.outputscale.item() / noise.item()):0.5f}%',
    #     'Kernel kappa values:',
    #     # (full_x.std() / censored_model.covar_module.base_kernel.lengthscale.numpy(force=True)[0]).apply(lambda x: f'{x:0.2f}'),
    #     sep='\n'
    # )

    # Evaluate metrics
    eps_dist = gpytorch.distributions.MultivariateNormal(torch.zeros(1), torch.as_tensor([[noise.item()]]))
    err_std = noise.sqrt().item()
    is_censored_test = is_censored[test_idx].values
    is_censored_train = is_censored[train_idx].values
    with torch.no_grad():
        test_f_pred = model(test_x)
        f_pred = model(x)

        test_y_pred = (test_f_pred + eps_dist).to_data_independent_dist()
        y_pred = (f_pred + eps_dist).to_data_independent_dist()

        # p(y <= left or y >= right)
        censored_pred = (
                y_pred.cdf(torch.as_tensor(left_bound[train_idx].values)) -
                y_pred.cdf(torch.as_tensor(right_bound[train_idx].values)) + 1
        )
        test_censored_pred = (
                test_y_pred.cdf(torch.as_tensor(left_bound[test_idx].values)) -
                test_y_pred.cdf(torch.as_tensor(right_bound[test_idx].values)) + 1
        )

    if not log_transform:
        test_mae = sklearn.metrics.mean_absolute_percentage_error(
            test_y[~is_censored_test], test_y_pred.loc[~is_censored_test]
        )
        train_mae = sklearn.metrics.mean_absolute_percentage_error(
            y[~is_censored_train], y_pred.loc[~is_censored_train]
        )
    else:
        # Compute MAE in real space
        test_mae = sklearn.metrics.mean_absolute_percentage_error(
            np.exp(test_y[~is_censored_test]), np.exp(test_y_pred.loc[~is_censored_test])
        )
        train_mae = sklearn.metrics.mean_absolute_percentage_error(
            np.exp(y[~is_censored_train]), np.exp(y_pred.loc[~is_censored_train])
        )

    # TODO: use log-normal NLPD here
    test_nlpd = -test_y_pred.log_prob(test_y)[~is_censored_test].mean()
    train_nlpd = -y_pred.log_prob(y)[~is_censored_train].mean()

    wandb.log({
        'train_nlpd': train_nlpd,
        'train_mae': train_mae,
        'test_nlpd': test_nlpd,
        'test_mae': test_mae,
    }, step=final_step)

    train_confusion = sklearn.metrics.confusion_matrix(is_censored_train, censored_pred >= 0.5)
    test_confusion = sklearn.metrics.confusion_matrix(is_censored_test, test_censored_pred >= 0.5)
    wandb.log({
        'train_conf': wandb.plot.confusion_matrix(probs=None, y_true=is_censored_train,
                                                  preds=(censored_pred >= 0.5).numpy(),
                                                  class_names=['Uncensored', 'Censored']),
        'test_conf': wandb.plot.confusion_matrix(probs=None, y_true=is_censored_test,
                                                 preds=(test_censored_pred >= 0.5).numpy(),
                                                 class_names=['Uncensored', 'Censored']),
        'train_miss': train_confusion[1, 0] / train_confusion[1].sum(),
        'test_miss': test_confusion[1, 0] / test_confusion[1].sum(),
    }, step=final_step)

    f, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(5, 5))
    # ax = axs[1]
    plot_errors(ax=ax, y=test_y, pred=test_f_pred, err_std=err_std, is_censored=is_censored_test)
    ax.set_title(f'Test (MAE: {100 * test_mae:0.2f}% NLPD {test_nlpd:0.2f})')
    # plt.show(f)
    run.log({'test_error': wandb.Image(f)}, step=final_step)
    plt.close(f)

    f, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(5, 5))
    # ax = axs[0]
    plot_errors(ax=ax, y=y, pred=f_pred, err_std=err_std, is_censored=is_censored_train)
    ax.set_title(f'Train (MAE: {100 * train_mae:0.2f}% NLPD {train_nlpd:0.2f})')
    # plt.show(f)
    run.log({'train_error': wandb.Image(f)}, step=final_step)
    plt.close(f)

    # print('Train confusion', train_confusion[1])
    # print('Test confusion', test_confusion[1])
    # f, axs = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    # sklearn.metrics.ConfusionMatrixDisplay(train_confusion, display_labels=['Uncensored', 'Censored']).plot(ax=axs[0])
    # axs[0].set_title('Train')

    # sklearn.metrics.ConfusionMatrixDisplay(test_confusion, display_labels=['Uncensored', 'Censored']).plot(ax=axs[1])
    # axs[1].set_title('Test')
    # plt.show(f)
    # plt.close(f)

    run.finish()

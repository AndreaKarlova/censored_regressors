import torch
import torch.distributions as dist

# import numpy as np

def generate_rff(dim, M=100):
    """ Generate random Fourier feature function with dimensionality `dim` and `M` features """
    W = torch.randn(dim, M)
    b = torch.rand(M) * 2*torch.pi
    return lambda z, lengthscale: (2.0**0.5) * torch.cos(z @ W / lengthscale + b[None,:])

def fit_laplace_posterior(w_hat, log_joint_closure):
    """ Fit laplace posterior around mode `w_hat`.
    
        Requires `log_joint_closure`, a function of w_hat """
    return torch.autograd.functional.hessian(log_joint_closure, w_hat)

def laplace_predictive(phi, w_hat, H):
    """ Helper function
    
        Predict the marginal predictive variance for each input, given Gaussian posterior
    
        y_hat = phi @ w, w ~ N(w_hat, H) -> y_hat ~ N(phi @ w, phi @ H @ phi.T)
    """
    with torch.no_grad():
        ### TODO: np or torch for inv, here?
        return dist.Normal(phi @ w_hat, ((phi @ torch.linalg.inv(H)) * phi).sum(-1))

def bayesian_linear_model_loss(w, phi, y, noise):
    """ Loss function for training Bayesian linear model with Gaussian likelihood """
    scale = 1.0 # consider making optimizable
    prior = dist.Normal(0, scale).log_prob(w).sum(-1)
    y_hat = phi @ w
    likelihood = dist.Normal(y_hat, noise).log_prob(y).sum(-1)
    return -(prior + likelihood)

def bayesian_censored_model_loss(w, phi, y, noise, MAX_VALUE=0):
    """ Loss function for training Bayesian linear model with Tobit likelihood, 
        censored above at `MAX_VALUE`
    """
    scale = 1.0
    prior = dist.Normal(0, scale).log_prob(w).sum(-1)
    y_hat = phi @ w
    uncensored = (y < MAX_VALUE)
    standard_normal = dist.Normal(0,1)
    likelihood_a = standard_normal.log_prob((y[uncensored]-y_hat[uncensored])/noise)-noise.log()
    likelihood_b = torch.clamp_min(1 - standard_normal.cdf((MAX_VALUE - y_hat[~uncensored])/noise), 1e-8).log()
#     assert np.all(np.isfinite(likelihood_a.detach().numpy()))
#     assert np.all(np.isfinite(likelihood_b.detach().numpy()))
    return -(prior + likelihood_a.sum(-1) + likelihood_b.sum(-1))

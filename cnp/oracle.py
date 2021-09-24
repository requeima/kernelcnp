import torch
import numpy as np

from matrix import Diagonal
from stheno import *


# =============================================================================
# Custom kernels until we resolve issue with Stheno
# =============================================================================
    
def eq_cov(lengthscale, coefficient, noise=0.):
    
    def _eq_cov(x, x_, use_noise):
    
        diff = x[:, None, :] - x_[None, :, :]
        l2 = torch.sum((diff / lengthscale) ** 2, dim=2)
        cov = coefficient ** 2 * np.exp(-0.5 * l2)
        
        if use_noise:
            cov = cov + noise ** 2 * torch.eye(cov.shape[0])
            
        return cov
        
    return _eq_cov


def mat_cov(lengthscale, coefficient, noise=0.):
    
    def _mat_cov(x, x_, use_noise):
    
        diff = x[:, None, :] - x_[None, :, :]
        l2 = torch.sum((diff / lengthscale) ** 2, dim=2)
        cov = coefficient ** 2 * (1 + 5. ** 0.5 * l2**0.5 + 5 * l2 / 3)
        cov = cov * np.exp(- 5 ** 0.5 * l2**0.5)
        
        if use_noise:
            cov = cov + noise ** 2 * torch.eye(cov.shape[0])
            
        return cov
        
    return _mat_cov


def nm_cov(lengthscale1, lengthscale2, coefficient, noise=0.):
        
    eq_cov1 = eq_cov(lengthscale1, coefficient, noise)
    eq_cov2 = eq_cov(lengthscale2, coefficient, noise)
    
    def _nm_cov(x, x_, use_noise):
        
        cov1 = eq_cov1(x, x_, use_noise)
        cov2 = eq_cov2(x, x_, use_noise=False)
        
        return cov1 + cov2
        
    return _nm_cov


def wp_cov(period, lengthscale, coefficient, noise=0.):
    
    eq1 = eq_cov(lengthscale=lengthscale,
                 coefficient=coefficient,
                 noise=0.)
    
    eq2 = eq_cov(lengthscale=1.,
                 coefficient=1.,
                 noise=0.)
    
    def _wp_cov(x, x_, use_noise):
    
        trig = torch.cat([torch.sin(2 * np.pi * x / period),
                          torch.cos(2 * np.pi * x / period)],
                         axis=-1)
    
        trig_ = torch.cat([torch.sin(2 * np.pi * x_ / period),
                           torch.cos(2 * np.pi * x_ / period)],
                          axis=-1)
        
        cov = eq1(x=trig, x_=trig_, use_noise=False) * \
              eq2(x=x, x_=x_, use_noise=False)
        
        if use_noise:
            cov = cov + noise ** 2 * torch.eye(cov.shape[0])
            
        return cov
        
    return _wp_cov
    
    
def gp_sample(x, covariance):
    
    # Input shape is (N, D)
    assert len(x.shape) == 2
    
    # Compute prior covariance
    K = covariance(x, x, use_noise=True)
    Kchol = np.linalg.cholesky(K)
    
    # Draw sample from prior
    y = Kchol @ np.random.normal(shape=(x.shape[0],))
    
    return y

    
def gp_post_pred(xc, yc, xt, yt, covariance, use_target_noise):
    
    # Compute covariance matrix components
    Ktt = covariance(xt, xt, use_noise=use_target_noise)
    Kcc = covariance(xc, xc, use_noise=True)
    Kct = covariance(xc, xt, use_noise=False)
    
    # Compute mean of ground truth GP predictive
    mean = Kct.T @ np.linalg.solve(Kcc, yc)
    mean = mean[:, 0]

    # Compute covariance of ground truth GP predictive
    cov = Ktt - Kct.T @ np.linalg.solve(Kcc, Kct)
    
    return mean, cov


def gp_loglik(xc, yc, xt, yt, covariance):

    # Compute posterior predictive
    mean, cov = gp_post_pred(xc=xc,
                             yc=yc,
                             xt=xt,
                             yt=yt,
                             covariance=covariance,
                             use_target_noise=True)
    
    # Convert to torch tensors
    mean = torch.tensor(mean).double()
    cov = torch.tensor(cov).double()

    # Compute log probability of ground truth GP predictive
    dist = torch.distributions.MultivariateNormal(loc=mean,
                                                  covariance_matrix=cov)
    logprob = dist.log_prob(yt[:, 0].clone().detach().double())

    # Compute log probability of diagonal GP predictive
    diag_cov = torch.diag(cov)
    dist = torch.distributions.Normal(loc=mean, scale=diag_cov**0.5)

    diag_logprob = dist.log_prob(yt[:, 0].clone().detach().double())
    diag_logprob = torch.sum(diag_logprob)
    
    return logprob, diag_logprob


# =============================================================================
# Stheno log-likelihood calculation
# =============================================================================


def oracle_loglik(xc, yc, xt, yt, covariance, noise):
    
    # Create GP measure and condition on data
    p = GP(covariance)
    p_post = p | (p(xc, noise**2), yc)
    y_pred = p_post(xt, noise**2)
    
    loglik = y_pred.logpdf(yt)
    
    # Create GP measure and condition on data
    p = GP(covariance)
    p_post = p | (p(xc, noise**2), yc)
    y_pred = p_post(xt, noise**2)
    y_pred_diag = Normal(y_pred.mean, Diagonal(B.diag(y_pred.var)))
    
    loglik_diag = y_pred_diag.logpdf(yt)

    return loglik, loglik_diag
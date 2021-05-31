import torch
import numpy as np


# =============================================================================
# Custom kernels until we resolve issue with Stheno
# =============================================================================

def eq_cov(lengthscale, coefficient, noise):
    
    def _eq_cov(x, x_, use_noise):
    
        diff = x[:, None, :] - x_[None, :, :]
        l2 = torch.sum((diff / lengthscale) ** 2, dim=2)
        cov = coefficient ** 2 * np.exp(-0.5 * l2)
        
        if use_noise:
            cov = cov + noise ** 2 * torch.eye(cov.shape[0])
            
        return cov
        
    
    return _eq_cov


def mat_cov(lengthscale, coefficient, noise):
    
    def _mat_cov(x, x_, use_noise):
    
        diff = x[:, None, :] - x_[None, :, :]
        l1 = torch.sum(np.abs(diff / lengthscale), dim=2)
        cov = coefficient ** 2 * (1 + 5 ** 0.5 * l1 + 5 * l1 ** 2 / 3)
        cov = cov * np.exp(- 5 ** 0.5 * l1)
        
        if use_noise:
            cov = cov + noise ** 2 * torch.eye(cov.shape[0])
            
        return cov
        
    return _mat_cov


def wp_cov(period, lengthscale, coefficient, noise):
    
    def _wp_cov(x, x_, use_noise):
    
        diff = x[:, None, :] - x_[None, :, :]
        l1 = torch.sum(np.abs(diff / period), dim=2)
        l2 = torch.sum((diff / lengthscale) ** 2, dim=2)
        
        sin2 = (torch.sin(np.pi * l1) / lengthscale) ** 2
        
        cov = coefficient ** 2 * torch.exp(-2. * sin2)
        cov = cov * np.exp(-0.5 * l2)
        
        if use_noise:
            cov = cov + noise ** 2 * torch.eye(cov.shape[0])
            
        return cov
        
    return _wp_cov


def nm_cov(lengthscale1, lengthscale2, coefficient, noise):
        
    eq_cov1 = eq_cov(lengthscale1, coefficient, noise)
    eq_cov2 = eq_cov(lengthscale2, coefficient, noise)
    
    def _nm_cov(x, x_, use_noise):
        
        cov1 = eq_cov1(x, x_, use_noise)
        cov2 = eq_cov2(x, x_, use_noise=False)
        
        return cov1 + cov2
        
    return _nm_cov
    

def oracle_loglik(xc, yc, xt, yt, covariance):

    Ktt = covariance(xt, xt, use_noise=True)
    Kcc = covariance(xc, xc, use_noise=True)
    Kct = covariance(xc, xt, use_noise=False)
    
    # Compute mean and covariance of ground truth GP predictive
    mean = np.einsum('ij, ik -> jk', Kct, np.linalg.solve(Kcc, yc))
    mean = torch.tensor(mean[:, 0]).double()

    cov = Ktt - np.einsum('ij, ik -> jk', Kct, np.linalg.solve(Kcc, Kct))
    cov = cov.clone().detach().double()

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





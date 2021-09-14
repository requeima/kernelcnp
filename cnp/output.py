import numpy as np

import torch
import torch.nn as nn

from torch.distributions import MultivariateNormal

from abc import ABC, abstractmethod, abstractproperty


# =============================================================================
# General output layer
# =============================================================================

class OutputLayer(nn.Module):
    
    
    def __init__(self):
        
        super().__init__()

    
    @abstractmethod
    def loglik(self, tensor, y_target):
        """
        Implemented by child class.
        
        Computes the log-likelihood of *y_target* under the distribution which
        is specified by the parameters in *tensor*. The exact details of how
        *tensor* parameterises the predictive distribution are different for
        each child class and are specified there.
        
        Arguments:
            tensor   : torch.tensor, shape (B, T, C)
            y_target : torch.tensor, shape (B, T, 1)
        """
        pass

    
    @abstractmethod
    def sample(self, tensor, num_samples):
        """
        Implemented by child class.
        
        Draws **num_samples** function samples from the predictive distribution
        which is specified by the parameters in *tensor*. The exact details
        of how *tensor* parameterises the predictive distribution and how the
        samples are drawn are different for each child class and are specified
        there.
        
        Arguments:
            tensor      : torch.tensor, shape (B, T, C)
            num_samples : int
        """
        pass
    
    
# =============================================================================
# Meanfield output layer
# =============================================================================

class MeanFieldGaussianLayer(nn.Module):
    
    
    def __init__(self):
        
        super().__init__()
        
        self.noise_unconstrained = nn.Parameter(torch.tensor(0.))
        
        
    def mean_and_cov(self, tensor, double=False):
        """
        Computes mean and covariance of mean-field Gaussian layer, as
        specified by the parameters in *tensor*.
        
        Arguments:
            tensor : torch.tensor, (B, T, 2)
            double : optional bool, whether to use double precision
            
        Returns:
            mean   : torch.tensor, (B, T)
            cov    : torch.tensor, (B, T, T)
        """
        
        # Check tensor has three dimensions, and last dimension has size 2
        assert (len(tensor.shape) == 3) and (tensor.shape[2] == 2)
        
        # Mean is the first feature of the tensor
        mean = tensor[:, :, 0]
        
        # Covariance is a matrix with a 
        f_var = torch.nn.Softplus()(tensor[:, :, 1])
        y_var = f_var + torch.nn.Softplus()(self.noise_unconstrained)
        
        f_cov = torch.diag_embed(f_var)
        y_cov = torch.diag_embed(y_var)

        # Jitter to covariance for numerical stability
        jitter = 1e-6 * torch.eye(f_cov.shape[-1], device=y_cov.device)
        
        # Add jitter to both noiseless and noisy covariance
        f_cov = f_cov + jitter[None, :, :]
        y_cov = y_cov + jitter[None, :, :]
        
        if double:
            mean = mean.double()
            f_cov = f_cov.double()
            y_cov = y_cov.double()
        
        return mean, f_cov, y_cov
    
    
    def loglik(self, tensor, y_target):
        
        # Compute mean and covariance from feature vector
        y_mean, _, y_cov = self.mean_and_cov(tensor, double=True)
        
        # Initialise distribution to compute log probability
        dist = MultivariateNormal(loc=y_mean, covariance_matrix=y_cov)
        
        # Compute log likelihood and return
        loglik = dist.log_prob(y_target[:, :, 0]).float()
        loglik = torch.mean(loglik)
        
        return loglik
    
    
    def sample(self, tensor, num_samples):
        
        # Compute mean and covariance from feature vector
        y_mean, _, y_cov = self.mean_and_cov(tensor, double=True)
        
        # Initialise distribution to compute log probability
        dist = MultivariateNormal(loc=y_mean, covariance_matrix=y_cov)
        
        # Draw samples and return
        samples = dist.sample(sample_shape=[num_samples])
        
        return samples
    
    
    
# =============================================================================
# Innerprod output layer
# =============================================================================

class InnerprodGaussianLayer(nn.Module):
    
    
    def __init__(self, feature_dim, noise_type):
        
        super().__init__()
        
        self.feature_dim
        
        
    def mean_and_cov(self, tensor):
        """
        tensor : torch.tensor, (B, T, D)
        """
        

    def loglik(self, tensor, y_target):
        pass

    
    def sample(self, tensor, num_samples):
        pass

    
    def marginals(self, tensor):
        pass
    
    
    
# =============================================================================
# Kvv output layer
# =============================================================================

class KvvGaussianLayer(nn.Module):
    
    
    def __init__(self):
        
        super().__init__()

    
    def loglik(self, tensor, y_target):
        pass

    
    def sample(self, tensor, num_samples):
        pass

    
    def marginals(self, tensor):
        pass
    
    
    
# =============================================================================
# Gamma Gaussian output layer
# =============================================================================

class GammaGaussianCopulaLayer(nn.Module):
    
    
    def __init__(self):
        
        super().__init__()

    
    def loglik(self, tensor, y_target):
        pass

    
    def sample(self, tensor, num_samples):
        pass

    
    def marginals(self, tensor):
        pass
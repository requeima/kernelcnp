import numpy as np

import torch
import torch.nn as nn

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

    
    @abstractmethod
    def marginals(self, tensor):
        """
        Implemented by child class.
        
        Returns the marginals of the predictive distribution specified by the 
        parameters in *tensor*. The exact details of how *tensor* parameterises
        the predictive distribution (and therefore also the marginals) are 
        different for each child class and are specified there.
        
        Arguments:
            tensor      : torch.tensor, shape (B, T, C)
        """
        pass
    
    
# =============================================================================
# Meanfield output layer
# =============================================================================

class MeanFieldGaussianLayer(nn.Module):
    
    
    def __init__(self):
        
        super().__init__()
        
        self.noise_unconstrained = nn.Parameter(torch.tensor(0.))
        
        
    def mean_and_cov(self, tensor):
        """
        Computes mean and covariance of mean-field Gaussian layer, as
        specified by the parameters in *tensor*.
        
        Arguments:
            tensor : torch.tensor, (B, T, 2)
            
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
        
        return mean, f_cov, y_cov
        
    
    def loglik(self, tensor, y_target):
        
        # Compute mean and covariance from feature vector
        y_mean, _, y_cov = self.mean_and_cov(tensor)

        y_mean = mean.double()
        y_cov = cov.double()
        y_target = y_target.double()

        jitter = 1e-6 * torch.eye(y_cov.shape[-1], device=y_cov.device)
        jitter = jitter[None, :, :].double()
        
        cov = cov + jitter
        
        dist = MultivariateNormal(loc=mean[:, :, 0],
                                  covariance_matrix=cov)
        nll = - torch.mean(dist.log_prob(y_target[:, :, 0]))

        return nll.float()
    

    
    def sample(self, tensor, num_samples):
        pass

    
    def marginals(self, tensor):
        pass
    
    
    
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
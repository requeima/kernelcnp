import numpy as np

import torch
import torch.nn as nn

from abc import ABC, abstractmethod, abstractproperty


# =============================================================================
# General output layer
# =============================================================================

class OutputLayer(nn.Module):
    
    
    def __init__(self, noise_type):
        
        super().__init__()
        
        assert noise_type in ["homo", "hetero"]
        
        self.noise_type = noise_type

    
    @abstractmethod
    def loss(self, tensor, y_target):
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
        
        super().__init__(noise_type="hetero")
        
        
    def mean_and_cov(self, tensor):
        """
        Computes mean and covariance of mean-field Gaussian layer.
        
        tensor : torch.tensor, (B, T, 2)
        """
        
        # Check tensor has three dimensions, and last dimension has size 2
        assert (len(tensor.shape) == 3) and (tensor.shape[2] == 2)
        
        # Mean is the first feature of the tensor
        mean = tensor[:, :, :1]
        
        # Covariance is a matrix with a 
        cov = torch.nn.Softplus()(tensor[:, :, 1:])
        cov = 
        
        return 
        
    
    def loss(self, tensor, y_target):
        pass

    
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
        
        

    
    def loss(self, tensor, y_target):
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

    
    def loss(self, tensor, y_target):
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

    
    def loss(self, tensor, y_target):
        pass

    
    def sample(self, tensor, num_samples):
        pass

    
    def marginals(self, tensor):
        pass
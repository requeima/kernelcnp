import numpy as np

import torch
import torch.nn as nn

from torch.distributions import (
    Normal,
    MultivariateNormal,
    LowRankMultivariateNormal
)

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
            y_target : torch.tensor, shape (B, T)
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

class GaussianLayer(OutputLayer):
    
    
    def __init__(self, jitter=1e-6):
        super().__init__()
        
        self.jitter = jitter
    
    
    @abstractmethod
    def distribution(self, tensor, noiseless):
        pass
        
        
    @abstractmethod
    def _mean_and_cov(self, tensor):
        """
        Computes mean and covariance of mean-field Gaussian layer, as
        specified by the parameters in *tensor*. This method may give
        covariances which are close to singular, so the method mean_and_cov
        should be used instead.
        
        Arguments:
            tensor : torch.tensor, (B, T, 2)
            
        Returns:
            mean  : torch.tensor, (B, T)
            f_cov : torch.tensor, (B, T, T)
            y_cov : torch.tensor, (B, T, T)
        """
        pass
        
        
    def mean_and_cov(self, tensor, double=False):
        """
        Computes mean and covariance of mean-field Gaussian layer, as
        specified by the parameters in *tensor*. This method internally
        calls _mean_and_cov, and adds jitter to the covariances which this
        method produces.
        
        Arguments:
            tensor : torch.tensor, (B, T, 2)
            
        Returns:
            mean  : torch.tensor, (B, T)
            f_cov : torch.tensor, (B, T, T)
            y_cov : torch.tensor, (B, T, T)
        """
        
        # Compute mean and covariance
        mean, f_cov, y_cov = self._mean_and_cov(tensor)
        
        # Jitter to covariance for numerical stability
        jitter = self.jitter * torch.eye(f_cov.shape[-1], device=y_cov.device)
        
        # If specified, use double precision
        if double:
            mean = mean.double()
            f_cov = f_cov.double()
            y_cov = y_cov.double()
            jitter = jitter.double()
        
        # Add jitter to both noiseless and noisy covariance
        f_cov = f_cov + jitter[None, :, :]
        y_cov = y_cov + jitter[None, :, :]
        
        return mean, f_cov, y_cov
    
    
    def loglik(self, tensor, y_target):
        """
        Computes the log-likelihood of *y_target* under mean-field Gaussian
        layer, specified by the parameters in *tensor*.
        
        Arguments:
            tensor   : torch.tensor, (B, T, 2)
            y_target : torch.tensor, (B, T)
            
        Returns:
            loglik   : torch.tensor, ()
        """
        
        # Initialise distribution and compute log probability
        dist = self.distribution(tensor, noiseless=False)
        loglik = dist.log_prob(y_target).float()
        
        return loglik
    
    
    def sample(self, tensor, num_samples, noiseless):
        """
        Draws samples from the mean-field Gaussian
        layer, specified by the parameters in *tensor*.
        
        Arguments:
            tensor   : torch.tensor, (B, T, 2)
            y_target : torch.tensor, (B, T)
            
        Returns:
            loglik   : torch.tensor, ()
        """
        
        # Initialise distribution to compute log probability
        dist = self.distribution(tensor=tensor, noiseless=noiseless)
        
        # Draw samples and return
        samples = dist.sample(sample_shape=[num_samples])
        samples = samples.float()
        
        return samples
    
    
    
# =============================================================================
# Meanfield output layer
# =============================================================================


class MeanFieldGaussianLayer(GaussianLayer):
    
    
    def __init__(self, jitter=1e-6):
        
        super().__init__(jitter=jitter)
        
        self.noise_unconstrained = nn.Parameter(torch.tensor(0.))
        self.mean_dim = 1
        self.num_features = self.mean_dim + 1
        
        
    def _mean_and_cov(self, tensor):
        """
        Computes mean and covariance of mean-field Gaussian layer, as
        specified by the parameters in *tensor*. This method may give
        covariances which are close to singular, so the method mean_and_cov
        should be used instead.
        
        Arguments:
            tensor : torch.tensor, (B, T, 2)
            
        Returns:
            mean  : torch.tensor, (B, T)
            f_cov : torch.tensor, (B, T, T)
            y_cov : torch.tensor, (B, T, T)
        """
        
        # Check tensor has three dimensions, and last dimension has size 2
        assert (len(tensor.shape) == 3) and (tensor.shape[2] == 2)
        
        # Compute mean vector
        mean = tensor[:, :, 0]
        
        # Compute diagonal covariance matrix
        f_var = torch.nn.Softplus()(tensor[:, :, 1])
        y_var = f_var + torch.nn.Softplus()(self.noise_unconstrained)
        
        f_cov = torch.diag_embed(f_var)
        y_cov = torch.diag_embed(y_var)
        
        return mean, f_cov, y_cov
    
    
    def distribution(self, tensor, noiseless):
        
        # Get mean and covariances of distribution
        mean, f_cov, y_cov = self.mean_and_cov(tensor)
        
        # Set lower triangular scale equal to either noiseless or noisy scale
        sqrt_diag = lambda x : torch.diag_embed(torch.diag_part(x)**0.5)
        scale_tril = sqrt_diag(f_cov) if noiseless else sqrt_diag(y_cov)**0.5
        
        # Create distribution and return
        dist = MultivariateNormal(loc=mean, scale_tril=scale_tril)
        
        return dist
    
    
    
# =============================================================================
# Innerprod output layer
# =============================================================================

class InnerprodGaussianLayer(GaussianLayer):
    
    
    def __init__(self, num_embedding, noise_type, jitter=1e-6):
        
        super().__init__(jitter=jitter)
        
        # Noise type can be homoscedastic or heteroscedastic
        assert noise_type in ["homo", "hetero"]
        
        # Set noise type, initialise noise variable if necessary
        self.noise_type = noise_type
        
        if self.noise_type == "homo":
            self.noise_unconstrained = nn.Parameter(torch.tensor(0.))
        
        # Compute total number of features expected by layer
        self.mean_dim = 1
        self.extra_noise_dim = int(self.noise_type == "hetero")
        self.num_embedding = num_embedding
        
        self.num_features = self.mean_dim        + \
                            self.num_embedding   + \
                            self.extra_noise_dim
        
        
    def _mean_and_cov(self, tensor):
        """
        Computes mean and covariance of kvv Gaussian layer, as specified
        by the parameters in *tensor*. This method may give covariances
        which are close to singular, so the method mean_and_cov should be
        used instead.
        
        Arguments:
            tensor : torch.tensor, (B, T, 2)
            
        Returns:
            mean  : torch.tensor, (B, T)
            f_cov : torch.tensor, (B, T, T)
            y_cov : torch.tensor, (B, T, T)
        """
        
        # Check tensor has three dimensions, and last dimension has size 2
        assert (len(tensor.shape) == 3) and \
               (tensor.shape[2] == self.num_features)
        
        # Batch and datapoint dimensions
        B, T, C = tensor.shape
        
        # Compute mean vector
        mean = tensor[:, :, 0]
        
        # Slice out components of covariance - z and noise
        if self.noise_type == "homo":
            z = tensor[:, :, 1:] / C**0.5
            
            noise = torch.nn.Softplus()(self.noise_unconstrained)
            noise = noise[None, None].repeat(B, T)
            noise = torch.diag_embed(noise)
            
        else:
            z = tensor[:, :, 1:-1] / C**0.5
            
            noise = torch.nn.Softplus()(tensor[:, :, -1])
            noise = torch.diag_embed(noise)
        
        # Covariance is the product of the RBF and the v terms
        f_cov = torch.einsum("bnc, bmc -> bnm", z, z)
        y_cov = f_cov + noise
        
        return mean, f_cov, y_cov
    
    
    def distribution(self, tensor, noiseless):
        
        # Check tensor has three dimensions, and last dimension has size 2
        assert (len(tensor.shape) == 3) and \
               (tensor.shape[2] == self.num_features)
        
        B, T, C = tensor.shape
        
        # If num datapoints smaller than num embedding, return full-rank
        if tensor.shape[1] - 1 <= self.num_embedding:
            
            mean, f_cov, y_cov = self.mean_and_cov(tensor)
            cov = f_cov if noiseless else y_cov
            
            dist = MultivariateNormal(loc=mean, covariance_matrix=cov)
            
            return dist
        
        
        # Otherwise, return low-rank 
        else:
            
            # Split tensor into mean and embedding
            mean = tensor[:, :, 0]
            z = tensor[:, :, 1:-1] / C**0.5
            jitter = torch.tensor(1e-6).repeat(B, T)
        
            if noiseless:
                noise = jitter
                
            elif self.noise_type == "homo":
                noise = torch.nn.Softplus()(self.noise_unconstrained)
                noise = noise[None, None].repeat(B, T)
                noise = noise + jitter

            else:
                noise = torch.nn.Softplus()(tensor[:, :, -1])
                noise = noise + jitter
            
            dist = LowRankMultivariateNormal(loc=mean,
                                             cov_factor=z,
                                             cov_diag=noise)
            return dist
    
    
    
# =============================================================================
# Kvv output layer
# =============================================================================


class KvvGaussianLayer(GaussianLayer):
    
    
    def __init__(self, num_embedding, noise_type, jitter=1e-6):
        
        super().__init__(jitter=jitter)
        
        # Noise type can be homoscedastic or heteroscedastic
        assert noise_type in ["homo", "hetero"]
        
        # Set noise type, initialise noise variable if necessary
        self.noise_type = noise_type
        
        if self.noise_type == "homo":
            self.noise_unconstrained = nn.Parameter(torch.tensor(0.))
        
        # Compute total number of features expected by layer
        self.mean_dim = 1
        self.v_dim = 1
        self.extra_noise_dim = int(self.noise_type == "hetero")
        self.num_embedding = num_embedding
        
        self.num_features = self.mean_dim        + \
                            self.num_embedding   + \
                            self.v_dim           + \
                            self.extra_noise_dim 
        
        
    def _mean_and_cov(self, tensor):
        """
        Computes mean and covariance of kvv Gaussian layer, as specified
        by the parameters in *tensor*. This method may give covariances
        which are close to singular, so the method mean_and_cov should be
        used instead.
        
        Arguments:
            tensor : torch.tensor, (B, T, 2)
            
        Returns:
            mean  : torch.tensor, (B, T)
            f_cov : torch.tensor, (B, T, T)
            y_cov : torch.tensor, (B, T, T)
        """
        
        # Check tensor has three dimensions, and last dimension has size 2
        assert (len(tensor.shape) == 3) and \
               (tensor.shape[2] == self.num_features)
        
        # Batch and datapoint dimensions
        B, T, _ = tensor.shape
        
        # Compute mean vector
        mean = tensor[:, :, 0]
        
        # Slice out components of covariance
        z = tensor[:, :, 1:-2]
        v = tensor[:, :, -2]
        
        if self.noise_type == "homo":
            noise = torch.nn.Softplus()(self.noise_unconstrained)
            noise = noise[None, None].repeat(B, T)
            noise = torch.diag_embed(noise)
            
        else:
            noise = torch.nn.Softplus()(tensor[:, :, -1])
            noise = torch.diag_embed(noise)
            
        # Apply RBF function to embedding
        z = z / z.shape[-1]**0.5
        quad = -0.5 * (z[:, :, None, :] - z[:, None, :, :]) ** 2
        exp = torch.exp(torch.sum(quad, axis=-1))
        
        # Covariance is the product of the RBF and the v terms
        f_cov = exp * v[:, :, None] * v[:, None, :]
        y_cov = f_cov + noise
        
        return mean, f_cov, y_cov
    
    
    def distribution(self, tensor, noiseless):
        
        # Get mean and covariances of distribution
        mean, f_cov, y_cov = self.mean_and_cov(tensor, double=True)
        
        # Set covariance to either noiseless or noisy covariance
        cov = f_cov if noiseless else y_cov
        
        # Create distribution and return
        dist = MultivariateNormal(loc=mean, covariance_matrix=cov)
        
        return dist
    
    
    
# =============================================================================
# Log-logit copula output layer
# =============================================================================

class LogLogitCopulaLayer(OutputLayer):
    
    
    def __init__(self, gaussian_layer):
        
        super().__init__()
        
        # Initialise Gaussian layer
        self.gaussian_layer = gaussian_layer
        
        # Number of features equal to number of Gaussian layer features plus
        # two additional features for the Gamma - rate and concentration
        self.num_features = self.gaussian_layer.num_features + 2

    
    def loglik(self, tensor, y_target):
        """
        Arguments:
            tensor   : torch.tensor, (B, T, C)
            y_target : torch.tensor, (B, T)
            
        Returns:
            tensor : torch.tensor, (B, T)
        """
        
        # Unpack parameters and apply inverse transformation
        tensor, a, b = self.unpack_parameters(tensor=tensor)
        v_target = self.inverse_marginal_transformation(x=y_target,
                                                        a=a,
                                                        b=b)
        
        # Log-likelihood of transformed variables under Gaussian
        loglik = self.gaussian_layer.loglik(tensor=tensor, y_target=v_target)
        
        # Compute change-of-variables contribution (Jacobian is diagonal)
        grad = self.inverse_marginal_transformation(x=y_target,
                                                    a=a,
                                                    b=b,
                                                    grad=True)
        jacobian_term = torch.sum(torch.log(torch.abs(grad)), dim=-1)
        
        # Ensure shapes are compatible
        assert loglik.shape == jacobian_term.shape
        
        return loglik + jacobian_term

    
    def sample(self, tensor, num_samples, noiseless, double=False):
        """
        Arguments:
            tensor      : torch.tensor, (B, T, C)
            num_samples : int, number of samples to draw
            noiseless   : bool, whether to include the noise term
            
        Returns:
            tensor : torch.tensor, (B, T)
        """
        
        # Unpack parameters and apply inverse transformation
        tensor, a, b = self.unpack_parameters(tensor=tensor)
        
        # Draw samples from Gaussian and apply marginal transformation
        v_samples = self.gaussian_layer.sample(tensor=tensor,
                                               num_samples=num_samples,
                                               noiseless=noiseless,
                                               double=double)
        
        # Repeat a and b, (num_samples, B, T)
        a = a[None, :, :].repeat(num_samples, 1, 1)
        b = b[None, :, :].repeat(num_samples, 1, 1)
        
        # Apply marginal transformation to Gaussian samples
        samples = self.marginal_transformation(v_samples, a=a, b=b)
        
        return samples
        
        
    def unpack_parameters(self, tensor):
        """
        Arguments:
            tensor : torch.tensor, (B, T, C)
            
        Returns:
            tensor : torch.tensor, (B, T, C-2)
            a      : torch.tensor, (B, T)
            b      : torch.tensor, (B, T)
        """
        
        epsilon = 1e-3
        
        # Check tensor has correct number of features
        assert (len(tensor.shape) == 3) and \
               (tensor.shape[-1] == self.num_features)
        
        # Get rate and concentration from tensor
        a = 2e-2 * torch.nn.Softplus()(tensor[:, :, 0]) + 1e0 + epsilon
        b = 2e-2 * torch.nn.Softplus()(tensor[:, :, 1]) + 1e0 + epsilon
        
#         print(f'{a.min():.3f}, {a.max():.3f}, {b.min():.3f}, {b.max():.3f}')
        
        # Slice out rate and concentration
        tensor = tensor[:, :, 2:]
        
        return tensor, a, b
    
    
    def pdf(self, x, a, b):
        """
        Probability distribution function of the log-logistic distribution.
        
            PDF(x) = (b/a) * (x/a)^(b-1) / (1 + (x/a)^b)^2
        
        Arguments:
            x : torch.tensor, (B, T)
            a : torch.tensor, (B, T)
            b : torch.tensor, (B, T)
            
        Returns:
            tensor : torch.tensor, (B, T)
        """
        
        # Check shapes are compatible, all x values are positive
        assert x.shape == a.shape == b.shape
        assert torch.all(x > 0.)
        
        return (b/a) * (x/a)**(b-1) / (1+(x/a)**b)**2
    
    
    def cdf(self, x, a, b):
        """
        Cumulative distribution function of the log-logistic distribution.
        
            CDF(x) = 1 / (1 + (x/a)^-b)
        
        Arguments:
            x : torch.tensor, (B, T)
            a : torch.tensor, (B, T)
            b : torch.tensor, (B, T)
            
        Returns:
            tensor : torch.tensor, (B, T)
        """
        
        # Check shapes are compatible, all x values are positive
        assert x.shape == a.shape == b.shape
        assert torch.all(x > 0.)
        
        x = x.double()
        a = a.double()
        b = b.double()
        
        cdf = 1 / (1+(x/a)**-b)
        cdf = cdf.float()
        
        return cdf
    
    
    def icdf(self, x, a, b):
        """
        Inverse cumulative distribution function of the log-logistic
        distribution.
        
            CDF^-1(x) = a * (x^-1 - 1)^(-1/b)
        
        Arguments:
            x : torch.tensor, (B, T)
            a : torch.tensor, (B, T)
            b : torch.tensor, (B, T)
            
        Returns:
            tensor : torch.tensor, (B, T)
        """
        
        # Check shapes are compatible, all x values are positive
        assert x.shape == a.shape == b.shape
        assert torch.all(x > 0.)
        
        x = x.double()
        a = a.double()
        b = b.double()
        
        icdf = a * (x**-1 - 1) ** (-1/b)
        icdf = icdf.float()
        
        return icdf
    
    
    def marginal_transformation(self, x, a, b):
        """
        Arguments:
            x : torch.tensor, (B, T)
            a : torch.tensor, (B, T)
            b : torch.tensor, (B, T)
            
        Returns:
            tensor : torch.tensor, (B, T)
        """
        
        # Check shapes are compatible, all x values are positive
        assert x.shape == a.shape == b.shape
        
        zeros = torch.zeros(size=x.shape).double()
        ones = torch.ones(size=x.shape).double()
        
        gaussian = Normal(loc=zeros, scale=ones)
        
        x = gaussian.cdf(x)
        x = self.icdf(x, a, b)
        
        return x
        
        
    def inverse_marginal_transformation(self, x, a, b, grad=False):
        """
        Arguments:
            x : torch.tensor, (B, T)
            a : torch.tensor, (B, T)
            b : torch.tensor, (B, T)
            
        Returns:
            x : torch.tensor, (B, T)
        """
        
        # Check shapes are compatible, all x values are positive
        assert x.shape == a.shape == b.shape
        assert torch.all(x > 0.)
        
        zeros = torch.zeros(size=x.shape).double()
        ones = torch.ones(size=x.shape).double()
        
        gaussian = Normal(loc=zeros, scale=ones)
        
        if grad:
            x = self.pdf(x, a, b) / gaussian.icdf(self.cdf(x, a, b))
            x = x.float()
        
        else:
            x = self.cdf(x, a, b)
            x = gaussian.icdf(x)
            x = x.float()
        
        return x
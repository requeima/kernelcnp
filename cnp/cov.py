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
    def loglik(self, tensor, y_target, double):
        """
        Implemented by child class.
        
        Computes the log-likelihood of *y_target* under the distribution which
        is specified by the parameters in *tensor*. The exact details of how
        *tensor* parameterises the predictive distribution are different for
        each child class and are specified there.
        
        Arguments:
            tensor   : torch.tensor, shape (B, T, C)
            y_target : torch.tensor, shape (B, T)
            double   : bool, compute in double
        """
        pass

    
    @abstractmethod
    def sample(self, tensor, num_samples, double):
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
            double      : bool, compute in double
        """
        pass
    
    
    
# =============================================================================
# Gaussian output layer
# =============================================================================

class GaussianLayer(OutputLayer):
    
    
    def __init__(self, jitter=1e-6):
        
        super().__init__()
        
        self.jitter = jitter
    
    
    @abstractmethod
    def distribution(self, tensor, noiseless, double):
        pass
        
        
    @abstractmethod
    def _mean_and_cov(self, tensor):
        pass
        
        
    def mean_and_cov(self, tensor, double, **kwargs):
        
        # Compute mean and covariance
        mean, f_cov, y_cov = self._mean_and_cov(tensor, **kwargs)
        
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
    
    
    def loglik(self, tensor, y_target, double=True):
        
        # Initialise distribution and compute log probability
        dist = self.distribution(tensor, noiseless=False, double=double)
        loglik = dist.log_prob(y_target)
        
        return loglik
    
    
    def sample(self, tensor, num_samples, noiseless, double):
        
        # Initialise distribution to compute log probability
        dist = self.distribution(tensor=tensor,
                                 noiseless=noiseless,
                                 double=double)
        
        # Draw samples and return
        samples = dist.sample(sample_shape=[num_samples])
        
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
    
    
    def distribution(self, tensor, noiseless, double):
        
        # Get mean and covariances of distribution
        mean, f_cov, y_cov = self.mean_and_cov(tensor, double=double)
        
        # Set lower triangular scale equal to either noiseless or noisy scale
        sqrt_diag = lambda x : torch.diag_embed(torch.diagonal(x,
                                                               dim1=-2,
                                                               dim2=-1)**0.5)
        scale_tril = sqrt_diag(f_cov) if noiseless else sqrt_diag(y_cov)
        
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
    
    
    def distribution(self, tensor, noiseless, double):
        
        # Check tensor has three dimensions, and last dimension has size 2
        assert (len(tensor.shape) == 3) and \
               (tensor.shape[2] == self.num_features)
        
        B, T, C = tensor.shape
        
        # If num datapoints smaller than num embedding, return full-rank
        if tensor.shape[1] - 1 <= self.num_embedding:
            
            mean, f_cov, y_cov = self.mean_and_cov(tensor, double=double)
            cov = f_cov if noiseless else y_cov
            
            dist = MultivariateNormal(loc=mean, covariance_matrix=cov)
            
            return dist
        
        
        # Otherwise, return low-rank 
        else:
            
            # Convert tensor to double if required
            tensor = tensor.double() if double else tensor
            
            # Split tensor into mean and embedding
            mean = tensor[:, :, 0]
            z = tensor[:, :, 1:-1] / C**0.5
            
            jitter = torch.tensor(1e-6).repeat(B, T).to(z.device)
            jitter = jitter.double() if double else jitter
        
            if noiseless:
                noise = jitter
                
            elif self.noise_type == "homo":
                noise = torch.nn.Softplus()(self.noise_unconstrained)
                noise = noise[None, None].repeat(B, T)

            else:
                noise = torch.nn.Softplus()(tensor[:, :, -1])
                
            noise = noise.double() if double else noise
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
    
    
    def distribution(self, tensor, noiseless, double):
        
        # Get mean and covariances of distribution
        mean, f_cov, y_cov = self.mean_and_cov(tensor, double=double)
        
        # Set covariance to either noiseless or noisy covariance
        cov = f_cov if noiseless else y_cov
        
        # Create distribution and return
        dist = MultivariateNormal(loc=mean, covariance_matrix=cov)
        
        return dist
    
    
    
# =============================================================================
# General copula output layer
# =============================================================================

class CopulaLayer(OutputLayer):
    
    
    def __init__(self, gaussian_layer, device):
        
        super().__init__()
        
        # Initialise Gaussian layer
        self.gaussian_layer = gaussian_layer
        
        # Set device
        self.device = device

    
    def loglik(self, tensor, y_target):
        """
        Arguments:
            tensor   : torch.tensor, (B, T, C)
            y_target : torch.tensor, (B, T)
            
        Returns:
            tensor : torch.tensor, (B, T)
        """
        
        # Unpack parameters and apply inverse transformation
        tensor, marg_params = self.unpack_parameters(tensor=tensor)
        v_target = self.inverse_marginal_transformation(x=y_target,
                                                        marg_params=marg_params)
        
        # Log-likelihood of transformed variables under Gaussian
        loglik = self.gaussian_layer.loglik(tensor=tensor, y_target=v_target)
        
        # Compute change-of-variables contribution (Jacobian is diagonal)
        jacobian_term = self.jacobian_term(x=y_target,
                                           marg_params=marg_params)

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
        tensor, marg_params = self.unpack_parameters(tensor=tensor)
        
        # Draw samples from Gaussian and apply marginal transformation
        v_samples = self.gaussian_layer.sample(tensor=tensor,
                                               num_samples=num_samples,
                                               noiseless=noiseless,
                                               double=double)
        
        # Repeat a and b, (num_samples, B, T)
        marg_params = [marg_param[None, :, :].repeat(num_samples, 1, 1) \
                       for marg_param in marg_params]
        
        # Apply marginal transformation to Gaussian samples
        samples = self.marginal_transformation(v_samples,
                                               marg_params=marg_params)
        
        return samples
    
    
    def marginal_transformation(self, x, marg_params):
        """
        Arguments:
            x           : torch.tensor, (B, T)
            marg_params : List[torch.tensor], [(B, T), ..., (B, T)]
            
        Returns:
            x : torch.tensor, (B, T)
        """
        
        zeros = torch.zeros(size=x.shape).double().to(self.device)
        ones = torch.ones(size=x.shape).double().to(self.device)
        
        gaussian = Normal(loc=zeros, scale=ones)
        
        x = gaussian.cdf(x)
        x = self.icdf(x, marg_params)
        
        return x
        
        
    def inverse_marginal_transformation(self, x, marg_params):
        """
        Arguments:
            x           : torch.tensor, (B, T)
            marg_params : List[torch.tensor], [(B, T), ..., (B, T)]
            
        Returns:
            x : torch.tensor, (B, T)
        """
        
        zeros = torch.zeros(size=x.shape).double().to(self.device)
        ones = torch.ones(size=x.shape).double().to(self.device)
        
        gaussian = Normal(loc=zeros, scale=ones)
        
        x = self.cdf(x, marg_params)
        x = gaussian.icdf(x)
        
        return x
        
        
    def jacobian_term(self, x, marg_params):
        """
        Arguments:
            x           : torch.tensor, (B, T)
            marg_params : List[torch.tensor], [(B, T), ..., (B, T)]
            
        Returns:
            x : torch.tensor, (B, T)
        """
        
        zeros = torch.zeros(size=x.shape).double().to(self.device)
        ones = torch.ones(size=x.shape).double().to(self.device)
        
        gaussian = Normal(loc=zeros, scale=ones)
        
        jacobian_term = self.log_pdf(x, marg_params)
        jacobian_term = jacobian_term - \
                        gaussian.log_prob(self.cdf(x, marg_params))
        jacobian_term = torch.sum(jacobian_term, axis=-1)
        
        return jacobian_term
        
        
    @abstractmethod
    def unpack_parameters(self, tensor):
        pass
    
    
    @abstractmethod
    def log_pdf(self, x, marg_params):
        pass
    
    
    @abstractmethod
    def cdf(self, x, marg_params):
        pass
    
    
    @abstractmethod
    def icdf(self, x, marg_params):
        pass
    
    
# =============================================================================
# Exponential copula output layer
# =============================================================================


class ExponentialCopulaLayer(CopulaLayer):
    
    
    def __init__(self, gaussian_layer, device):
        
        super().__init__(gaussian_layer=gaussian_layer,
                         device=device)
        
        self.num_features = self.gaussian_layer.num_features + 1
        
        # Set sccale -- not currently used
        self.log_scale = torch.log(torch.tensor(1.))
        self.log_scale = torch.nn.Parameter(self.log_scale)
        
        
    def unpack_parameters(self, tensor):
        """
        Arguments:
            tensor : torch.tensor, (B, T, C)
            
        Returns:
            tensor      : torch.tensor, (B, T, C-2)
            marg_params : List[torch.tensor], [(B, T),]
        """
        
        # Check tensor has correct number of features
        assert (len(tensor.shape) == 3) and \
               (tensor.shape[-1] == self.num_features)
        
        # Get scale from tensor
        scale = torch.nn.Softplus()(tensor[:, :, 0]) + 1e0
    
        tensor = tensor[:, :, 1:]
        
        return tensor, [scale]
    
    
    def log_pdf(self, x, marg_params):
        """
        Arguments:
            x           : torch.tensor, (B, T)
            marg_params : List[torch.tensor], [(B, T)]
            
        Returns:
            tensor : torch.tensor, (B, T)
        """
        
        scale, = marg_params
        
        # Check shapes are compatible, all x values are positive
        assert x.shape == scale.shape
        assert torch.all(x >= 0.)
        
        return - torch.log(scale) - (x / scale)
    
    
    def cdf(self, x, marg_params):
        """
        Arguments:
            x           : torch.tensor, (B, T)
            marg_params : List[torch.tensor], [(B, T), (B, T)]
            
        Returns:
            tensor : torch.tensor, (B, T)
        """
        
        scale, = marg_params
        
        # Check shapes are compatible, all x values are positive
        assert x.shape == scale.shape
        assert torch.all(x >= 0.)
        
        x = x.double()
        scale = scale.double()
        
        cdf = 1 - torch.exp(-(x/scale))
        cdf = cdf.float()
        
        return cdf
    
    
    def icdf(self, x, marg_params):
        """
        Arguments:
            x           : torch.tensor, (B, T)
            marg_params : List[torch.tensor], [(B, T), (B, T)]
            
        Returns:
            tensor : torch.tensor, (B, T)
        """
        
        scale, = marg_params
        
        # Check shapes are compatible, all x values are positive
        assert x.shape == scale.shape
        assert torch.all(x >= 0.)
        
        x = x.double()
        scale = scale.double()
        
        icdf = -scale * torch.log(1. - x)
        icdf = icdf.float()
        
        return icdf
    
    
# =============================================================================
# Multioutput Gaussian layer
# =============================================================================

class MultiOutputGaussianLayer(GaussianLayer):
    
    
    def __init__(self, num_outputs, jitter=1e-6):
        
        super().__init__(jitter=jitter)
        
        self.num_outputs = num_outputs
    
    
    @abstractmethod
    def distribution(self, tensor, target_mask, noiseless, double):
        """
        Arguments:
            tensor      : torch.tensor, (B, D, T, F)
            target_mask : torch.tensor, (B, D, T)
            noiseless   : bool
            double      : bool
            
        Returns:
            dist        : torch.distribution, (B, M*T)
        """
        pass
    
    
    def loglik(self, tensor, y_target, target_mask, double=True):
        """
        Arguments:
            tensor      : torch.tensor, (B, D, T, F)
            y_target    : torch.tensor, (B, D, T)
            target_mask : torch.tensor, (B, D, T)
            double      : bool
            
        Returns:
            loglik      : torch.tensor, shape (B,)
        """
        
        assert y_target.shape == target_mask.shape
        
        # Create distribution - covariance shape (B, M*T, M*T)
        dist = self.distribution(tensor=tensor,
                                 target_mask=target_mask,
                                 noiseless=False,
                                 double=double)
        
        # Slice out masked channels
        mask_idx = torch.any(target_mask[0] == 1, dim=1)
        
        y_target = y_target[:, mask_idx, :]
        y_target = torch.reshape(y_target, (y_target.shape[0], -1))
        
        loglik = dist.log_prob(y_target)
        
        return loglik
    
    
    def sample(self, tensor, target_mask, num_samples, noiseless, double):
        """
        Arguments:
            tensor      : torch.tensor, (B, D, T, F)
            target_mask : torch.tensor, (B, D, T)
            num_samples : int
            noiseless   : bool
            double      : bool
            
        Returns:
            sample      : torch.tensor, (B, M, T)
        """
        
        mask_idx = torch.any(target_mask[0] == 1, dim=1)
        target_ones = torch.ones_like(target_mask).long().to(tensor.device)
        
        # Initialise distribution to compute log probability
        dist = self.distribution(tensor=tensor,
                                 target_mask=target_ones,
                                 noiseless=noiseless,
                                 double=double)
        
        # Draw samples and return
        samples = dist.sample(sample_shape=[num_samples])
        
        sample_shape = (samples.shape[0],
                        samples.shape[1],
                        self.num_outputs,
                        -1)
        
        samples = torch.reshape(samples, sample_shape)
        samples_masked = samples[:, :, mask_idx, :]
        
        return samples, samples_masked
    
    

# =============================================================================
# MultiOutput meanfield output layer
# =============================================================================


class MultiOutputMeanFieldGaussianLayer(MultiOutputGaussianLayer):
    
    
    def __init__(self, num_outputs, jitter=1e-6):
        
        super().__init__(num_outputs=num_outputs, jitter=jitter)
        
        self.noise_unconstrained = nn.Parameter(torch.tensor(0.))
        self.num_features = 2
        
        
    def _mean_and_cov(self, tensor, target_mask):
        """
        Arguments:
            tensor      : torch.tensor, (B, D, T, 2)
            target_mask : torch.tensor, (B, D, T)
            
        Returns:
            mean  : torch.tensor, (B, M*T)
            f_cov : torch.tensor, (B, M*T, M*T)
            y_cov : torch.tensor, (B, M*T, M*T)
        """
        
        # Check tensor shapes are correct
        assert (len(tensor.shape) == 4) and (tensor.shape[3] == 2)
        assert (len(target_mask.shape) == 3) and \
               (tensor.shape[:-1] == target_mask.shape)
        
        # Slice out masked channels - changes tensor shape to (B, M, T, 2)
        mask_idx = target_mask[0, :, 0] == 1
        tensor = tensor[:, mask_idx, :, :]
        
        # Compute mean vector
        mean = tensor[:, :, :, 0]
        mean = torch.reshape(mean, (mean.shape[0], -1))
        
        # Compute diagonal covariance matrix
        f_var = torch.nn.Softplus()(tensor[:, :, :, 1])
        f_var = torch.reshape(f_var, (f_var.shape[0], -1))
        y_var = f_var + torch.nn.Softplus()(self.noise_unconstrained)
        
        f_cov = torch.diag_embed(f_var)
        y_cov = torch.diag_embed(y_var)
        
        return mean, f_cov, y_cov
    
    
    def distribution(self, tensor, target_mask, noiseless, double):
        
        # Get mean and covariances of distribution
        mean, f_cov, y_cov = self.mean_and_cov(tensor=tensor,
                                               double=double,
                                               target_mask=target_mask)
        
        # Set lower triangular scale equal to either noiseless or noisy scale
        sqrt_diag = lambda x : torch.diag_embed(torch.diagonal(x,
                                                               dim1=-2,
                                                               dim2=-1)**0.5)
        scale_tril = sqrt_diag(f_cov) if noiseless else sqrt_diag(y_cov)
        
        # Create distribution and return
        dist = MultivariateNormal(loc=mean, scale_tril=scale_tril)
        
        return dist

    
# =============================================================================
# MultiOutput innerprod Gaussian layer
# =============================================================================

class MultiOutputInnerprodGaussianLayer(MultiOutputGaussianLayer):
    
    
    def __init__(self, num_outputs, num_embedding, noise_type, jitter=1e-6):
        
        super().__init__(num_outputs=num_outputs, jitter=jitter)
        
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
        self.num_outputs = num_outputs
        
        
    def _mean_and_cov(self, tensor, target_mask):
        """
        Arguments:
            tensor      : torch.tensor, (B, D, T, F)
            target_mask : torch.tensor, (B, D, T)
            
        Returns:
            mean  : torch.tensor, (B, M*T)
            f_cov : torch.tensor, (B, M*T, M*T)
            y_cov : torch.tensor, (B, M*T, M*T)
        """
        
        # Check tensors have correct shapes
        assert tensor.shape[:-1] == target_mask.shape
        
        # Slice out masked entries
        mask_idx = target_mask[0, :, 0] == 1
        tensor = tensor[:, mask_idx, :, :]
        
        # Unpack tensor dimensions
        B, M, T, F = tensor.shape
        
        # Reshape tensor to shape (B, M*T, F)
        tensor = torch.reshape(tensor, (B, M*T, F))
        
        # Compute mean vector
        mean = tensor[:, :, 0]
        
        # Slice out components of covariance - z and noise
        if self.noise_type == "homo":
            z = tensor[:, :, 1:] / F**0.5
            
            noise = torch.nn.Softplus()(self.noise_unconstrained)
            noise = noise[None, None].repeat(B, M*T)
            noise = torch.diag_embed(noise)
            
        else:
            z = tensor[:, :, 1:-1] / F**0.5

            noise = torch.nn.Softplus()(tensor[:, :, -1])
            noise = torch.diag_embed(noise)

        # Covariance is the product of the RBF and the v terms
        f_cov = torch.einsum("bnc, bmc -> bnm", z, z)
        y_cov = f_cov + noise
        
        return mean, f_cov, y_cov
    
    
    def distribution(self, tensor, target_mask, noiseless, double):
        """
        Arguments:
            tensor      : torch.tensor, (B, D, T, F)
            target_mask : torch.tensor, (B, D, T)
            
        Returns:
            mean  : torch.tensor, (B, M*T)
            f_cov : torch.tensor, (B, M*T, M*T)
            y_cov : torch.tensor, (B, M*T, M*T)
        """
        
        # Check tensors have correct shapes
        assert tensor.shape[:-1] == target_mask.shape
        assert tensor.shape[-1] == self.num_features
        
        # Slice out masked entries
        mask_idx = target_mask[0, :, 0] == 1
        M = torch.sum(mask_idx)
        
        # Unpack tensor dimension sizes
        B, D, T, F = tensor.shape
        
        # If num datapoints smaller than num embedding, return full-rank
        if M * T <= self.num_embedding:
            
            mean, f_cov, y_cov = self.mean_and_cov(tensor,
                                                   target_mask=target_mask,
                                                   double=double)
            cov = f_cov if noiseless else y_cov
            
            dist = MultivariateNormal(loc=mean, covariance_matrix=cov)
            
            return dist
        
        
        # Otherwise, return low-rank 
        else:
        
            # Slice out masked entries
            tensor = tensor[:, mask_idx, :, :]

            # Reshape tensor to shape (B, M*T, F)
            tensor = torch.reshape(tensor, (B, M*T, F))
            
            # Convert tensor to double if required
            tensor = tensor.double() if double else tensor
            
            # Split tensor into mean and embedding
            mean = tensor[:, :, 0]
            z = tensor[:, :, 1:-1] / F**0.5
            
            jitter = torch.tensor(1e-6).repeat(B, M*T).to(z.device)
            jitter = jitter.double() if double else jitter
        
            if noiseless:
                noise = jitter
                
            elif self.noise_type == "homo":
                noise = torch.nn.Softplus()(self.noise_unconstrained)
                noise = noise[None, None].repeat(B, M*T)

            else:
                noise = torch.nn.Softplus()(tensor[:, :, -1])
                
            noise = noise.double() if double else noise
            noise = noise + jitter
            
            dist = LowRankMultivariateNormal(loc=mean,
                                             cov_factor=z,
                                             cov_diag=noise)
            return dist
        
        

# =============================================================================
# MultiOutput kvv Gaussian layer
# =============================================================================


class MultiOutputKvvGaussianLayer(MultiOutputGaussianLayer):
    
    
    def __init__(self, num_outputs, num_embedding, noise_type, jitter=1e-6):
        
        super().__init__(num_outputs=num_outputs, jitter=jitter)
        
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
        self.num_outputs = num_outputs
        
        
    def _mean_and_cov(self, tensor, target_mask):
        """
        Arguments:
            tensor      : torch.tensor, (B, D, T, F)
            target_mask : torch.tensor, (B, D, T)
            
        Returns:
            mean  : torch.tensor, (B, M*T)
            f_cov : torch.tensor, (B, M*T, M*T)
            y_cov : torch.tensor, (B, M*T, M*T)
        """
        
        # Check tensor has three dimensions, and last dimension has size 2
        assert tensor.shape[:-1] == target_mask.shape
        
        # Slice out masked entries
        mask_idx = target_mask[0, :, 0] == 1
        tensor = tensor[:, mask_idx, :, :]
        
        # Unpack tensor dimensions
        B, M, T, F = tensor.shape
        
        # Reshape the tensor to shape (B, M*T, F)
        tensor = torch.reshape(tensor, (B, -1, F))
        
        mean = tensor[:, :, 0]
        
        if self.noise_type == "homo":
            
            z = tensor[:, :, 1:-1]
            v = tensor[:, :, -1]
        
            noise = torch.nn.Softplus()(self.noise_unconstrained)
            noise = noise[None, None].repeat(B, M*T)
            noise = torch.diag_embed(noise)
            
        else:
            
            z = tensor[:, :, 1:-2]
            v = tensor[:, :, -2]
            
            noise = torch.nn.Softplus()(tensor[:, :, -1])
            noise = torch.diag_embed(noise)
            
        # Apply RBF function to embedding
        z = z / z.shape[-1]**0.5
        quad = -0.5 * (z[:, :, None, :] - z[:, None, :, :])**2
        exp = torch.exp(torch.sum(quad, axis=-1))
        
        # Covariance is the product of the RBF and the v terms
        f_cov = exp * v[:, :, None] * v[:, None, :]
        y_cov = f_cov + noise
        
        return mean, f_cov, y_cov
    
    
    def distribution(self, tensor, target_mask, noiseless, double):
        
        # Get mean and covariances of distribution
        mean, f_cov, y_cov = self.mean_and_cov(tensor=tensor,
                                               double=double,
                                               target_mask=target_mask)
        
        cov = f_cov if noiseless else y_cov
        
        # Create distribution and return
        dist = MultivariateNormal(loc=mean, covariance_matrix=cov)
        
        return dist
    
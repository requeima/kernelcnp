import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from abc import ABC, abstractmethod, abstractproperty


def cdist(x):
    quad = (x[..., :, None, :] - x[..., None, :, :]) ** 2
    quad = torch.sum(quad, axis=-1)
    return quad

def rbf_kernel(x, scales):
    # Pariwise squared Euclidean distances between embedding vectors
    dists = cdist(x)

    # Compute the RBF kernel, broadcasting appropriately
    return torch.exp(-0.5 * dists / scales ** 2)


class Covariance(nn.Module):
    def __init__(self, num_basis_dim, extra_cov_dim):
        super().__init__()
        self.num_basis_dim = num_basis_dim
        self.extra_cov_dim = extra_cov_dim


class AddNoise(nn.Module):
    def __init__(self, extra_noise_dim):
        super().__init__()
        self.extra_noise_dim = extra_noise_dim


class InnerProdCov(Covariance):
    
    def __init__(self, num_basis_dim):
        # Extra dimension to add to the output
        extra_cov_dim = 0
        super().__init__(num_basis_dim, extra_cov_dim)
        
    def forward(self, embeddings):
        # Compute the covariance by taking inner products between embeddings
        basis_emb = embeddings[:, :, :self.num_basis_dim]
        cov = torch.einsum('bni, bmi -> bnm', basis_emb, basis_emb) / self.num_basis_dim
        
        return cov


class KvvCov(Covariance):
    def __init__(self, num_basis_dim):
        # Extra dimension to add to the output
        extra_cov_dim = 1
        super().__init__(num_basis_dim, extra_cov_dim)
        
        # Kernel Parameters
        init_length_scale = 0.5
        self.kernel_sigma = nn.Parameter(np.log(init_length_scale)* torch.ones(1), requires_grad=True)
        self.kernel_fn = torch.exp
    
    def forward(self, embeddings):
        # Extract the embeddings and v function
        basis_emb = embeddings[:, :, :self.num_basis_dim]
        v = embeddings[:, :, self.num_basis_dim: self.num_basis_dim + self.extra_cov_dim]

        #compute the covariance
        vv = torch.matmul(v, torch.transpose(v, dim0=-2, dim1=-1)) 
        scales = self.kernel_fn(self.kernel_sigma)
        cov = rbf_kernel(basis_emb, scales)
        cov = cov * vv
        return cov


class MeanFieldCov(Covariance):
    def __init__(self, num_basis_dim):
        assert num_basis_dim == 1, \
        'Mean Feald embedding must be one-dimensional.'
        # Extra dimension to add to the output
        extra_cov_dim = 0
        super().__init__(num_basis_dim, extra_cov_dim)
        
    def forward(self, embeddings):
        assert embeddings.shape[-1] == 1, \
        'Mean Feald embedding must be one-dimensional.'
        batch = embeddings.shape[0]
        dim = embeddings.shape[1]

        cov = torch.zeros(batch, dim, dim).to(embeddings.device)
        idx = np.arange(dim)
        cov[:, idx, idx] = torch.Softplus()(embeddings[:, :, 0])

        return cov


class AddHomoNoise(AddNoise):
    def __init__(self):
        # Extra dimension to add to the output
        extra_noise_dim = 0
        super().__init__(extra_noise_dim)

        # Noise Parameters
        self.noise_scale = nn.Parameter(torch.zeros(1), requires_grad=True)
    
    def forward(self, cov, embeddings):
        noise_var = torch.eye(cov.shape[1])[None, ...].to(cov.device)
        cov_plus_noise = cov + torch.exp(self.noise_scale) * noise_var
        
        return cov_plus_noise


class AddHeteroNoise(AddNoise):
    def __init__(self):
        # Extra dimension to add to the output
        extra_noise_dim = 1
        super().__init__(extra_noise_dim)

        # Noise Parameters
        self.noise_scale = nn.Parameter(np.log(1.0) * torch.ones(1), requires_grad=True)
        
    def forward(self,  cov, embeddings):
        # Extract the heteroskedastic noise function from the cov_layer_output
        hetero_noise_var = torch.exp(embeddings[:, :, -self.extra_noise_dim:])

        # Add the heteroskedastic noise to the covariance
        idx = np.arange(cov.shape[-1])
        cov_plus_noise = cov.clone()
        cov_plus_noise[:, idx, idx] = cov_plus_noise[:, idx, idx] + hetero_noise_var[:, :, 0]

        # Add homoskedastic noise to the covariance. This is for numerical stability of the initialization.
        noise_var =  torch.eye(cov.shape[1])[None, ...].to(cov.device)
        cov_plus_noise = cov_plus_noise + torch.exp(self.noise_scale) * noise_var
        return cov_plus_noise


class AddNoNoise(AddNoise):
    def __init__(self):
        # Extra dimension to add to the output
        extra_noise_dim = 0
        super().__init__(extra_noise_dim)
    
    def forward(self, cov, embeddings):
        return cov

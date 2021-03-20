import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import stheno.torch as stheno

import gnp.data
from gnp.experiment import report_loss, RunningAverage
from gnp.utils import gaussian_logpdf, init_sequential_weights, to_multiple, compute_dists
from gnp.architectures import SimpleConv, UNet
from abc import ABC, abstractmethod

from gnp.utils import device

def cdist(x):
    norms = torch.sum(x ** 2, axis=2)
    x_t = torch.transpose(x, dim0=1, dim1=2)
    inner_products = torch.matmul(x, x_t)
    return norms[:, :, None] + norms[:, None, :] - 2 * inner_products

def rbf_kernel(x, scales):
    # Pariwise squared Euclidean distances between embedding vectors
    dists = cdist(x)

    # Compute the RBF kernel, broadcasting appropriately
    return torch.exp(-0.5 * dists / scales ** 2)


class InnerProdCov(nn.Module):
    
    def __init__(self, num_basis_dim):
        super().__init__()
        # Extra dimension to add to the output
        self.extra_cov_dim = 0
        self.num_basis_dim
    
    def forward(self, embeddings):
        # Compute the covariance by taking innerproducts between embeddings
        basis_emb = embeddings[:, :, :self.num_basis_dim]
        cov = torch.matmul(basis_emb, torch.transpose(basis_emb, dim0=-2, dim1=-1)) / self.num_basis_dim
        return cov


class KvvCov(nn.Module):
    def __init__(self, num_basis_dim):
        super().__init__()
        # Extra dimension to add to the output
        self.extra_cov_dim = 1
        self.num_basis_dim = num_basis_dim
        
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
        scales = self.sigma_fn(self.kernel_sigma)
        cov = rbf_kernel(basis_emb, scales)
        cov = cov * vv
        return cov


class MeanFieldCov(nn.Module):
    def __init__(self, num_basis_dim):
        super().__init__()
        assert num_basis_dim == 1, \
        'Mean Feald embedding must be one-dimensional.'
        # Extra dimension to add to the output
        self.extra_cov_dim = 0
        self.num_basis_dim = num_basis_dim
        
    def forward(self, embeddings):
        assert embeddings.shape[-1] == 1, \
        'Mean Feald embedding must be one-dimensional.'
        batch = embeddings.shape[0]
        dim = embeddings.shape[1]

        cov = torch.zeros(batch, dim, dim)
        idx = np.arange(dim)
        cov[:, idx, idx] = torch.exp(embeddings[:, :, 0])

        return cov


class AddHomoNoise(nn.Module):
    def __init__(self):
        super().__init__()
        # Extra dimension to add to the output
        self.extra_noise_dim = 0
        # Noise Parameters
        self.noise_scale = nn.Parameter(np.log(1.0) * torch.ones(1), requires_grad=True)
    
    def forward(self, cov, embeddings):
        noise_var =  torch.eye(cov.shape[1])[None, ...].to(device)
        cov_plus_noise = cov + torch.exp(self.noise_scale) * noise_var
        
        return cov_plus_noise


class AddHeteroNoise(nn.Module):
    def __init__(self):
        super().__init__()
        # Extra dimension to add to the output
        self.extra_noise_dim = 1
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
        noise_var =  torch.eye(cov.shape[1])[None, ...].to(device)
        cov_plus_noise = cov_plus_noise + torch.exp(self.noise_scale) * noise_var
        return cov_plus_noise


class AddNoNoise(nn.Module):
    def __init__(self):
        super().__init__()
        # Extra dimension to add to the output
        self.extra_noise_dim = 0
        
    def forward(self, cov, embeddings):
        return cov
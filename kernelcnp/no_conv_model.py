import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import stheno.torch as stheno

import convcnp.data
from convcnp.experiment import report_loss, RunningAverage
from convcnp.utils import gaussian_logpdf, init_sequential_weights, to_multiple, compute_dists
from convcnp.architectures import SimpleConv, UNet
from convcnp.cnp import ConditionalNeuralProcess, StandardEncoder
from abc import ABC, abstractmethod

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NoConvKernelCNP(ConditionalNeuralProcess):
    def __init__(self,
                 latent_dim,
                 num_basis_dim,
                 num_output_dim,
                 use_attention=False):
    
        super().__init__(input_dim=1,
                         latent_dim=latent_dim,
                         num_channels=num_output_dim,
                         use_attention=use_attention)
        
        self.num_basis_dim = num_basis_dim

        # Overwrite the encoder with the correct input dimension 
        self.encoder = StandardEncoder(input_dim=2,
                                       latent_dim=self.latent_dim,
                                       use_attention=use_attention)

        # Noise Parameters
        self.noise_scale = nn.Parameter(np.log(1.0) * torch.ones(1), requires_grad=True)
    
        # Kernel Parameters
        init_length_scale = 0.5
        self.kernel_sigma = nn.Parameter(np.log(init_length_scale)* torch.ones(1), requires_grad=True)
        self.sigma_fn = torch.exp


    def forward(self, x, y, x_out, noiseless=False):
        cnp_out, _ = super().forward(x, y, x_out)

        # Produce mean
        mean = cnp_out[..., 0:1]
        
        # Produce cov
        cov_layer_output = cnp_out[..., 1:]
        cov, cov_plus_noise = self.cov(cov_layer_output)

        if noiseless:
            return mean, cov
        else:
            return mean, cov_plus_noise

    @abstractmethod
    def cov(self, cov_layer_output):
        pass

    def _cdist(self, x):
        norms = torch.sum(x ** 2, axis=2)
        x_t = torch.transpose(x, dim0=1, dim1=2)
        inner_products = torch.matmul(x, x_t)
        return norms[:, :, None] + norms[:, None, :] - 2 * inner_products

    def _rbf_kernel(self, basis_emb):
            
        # Pariwise squared Euclidean distances between embedding vectors
        dists = self._cdist(basis_emb)

        # Compute the RBF kernel, broadcasting appropriately
        scales = self.sigma_fn(self.kernel_sigma)

        return torch.exp(-0.5 * dists / scales ** 2)

    def _inner_prod_cov(self, cov_layer_output):
        # Compute the covariance by taking innerproducts between embeddings
        cov = torch.matmul(cov_layer_output, torch.transpose(cov_layer_output, dim0=-2, dim1=-1)) / cov_layer_output.shape[-1]
        
        return cov

    def _kvv_cov(self, cov_layer_output):
        # Extract the embeddings and v function
        basis_emb = cov_layer_output[:, :, :self.num_basis_dim]
        v = cov_layer_output[:, :, self.num_basis_dim: self.num_basis_dim + 1]

        #compute the covariance
        vv = torch.matmul(v, torch.transpose(v, dim0=-2, dim1=-1)) 
        cov = self._rbf_kernel(basis_emb)
        cov = cov * vv

        return cov
    
    def _add_homo_noise(self, cov, cov_layer_output):
        # Add homoskedastic noise to the covariance
        noise_var =  torch.eye(cov.shape[1])[None, ...].to(device)
        cov_plus_noise = cov + torch.exp(self.noise_scale) * noise_var
        
        return cov_plus_noise

    def _add_hetero_noise(self, cov, cov_layer_output):
        # Extract the heteroskedastic noise function from the cov_layer_output
        hetero_noise_var = torch.exp(cov_layer_output[:, :, -1:])

        # Add the heteroskedastic noise to the covariance
        idx = np.arange(cov.shape[-1])
        cov_plus_noise = cov.clone()
        cov_plus_noise[:, idx, idx] = cov_plus_noise[:, idx, idx] + hetero_noise_var[:, :, 0]

        # Add homoskedastic noise to the covariance. This is for numerical stability of the initialization.
        cov_plus_noise = self._add_homo_noise(cov_plus_noise, cov_layer_output)

        return cov_plus_noise

    @property
    def num_params(self):
        """Number of parameters in model."""
        return np.sum([torch.tensor(param.shape).prod()
                       for param in self.parameters()])


# CNP based models

class InnerProdHomoNoiseNoConvKernelCNP(NoConvKernelCNP):
    
    def __init__(self,
                 latent_dim,
                 num_basis_dim,
                 use_attention=False):

        super().__init__(latent_dim=latent_dim,
                         num_basis_dim=num_basis_dim,
                         num_output_dim=num_basis_dim + 1, 
                         use_attention=use_attention)


    def cov(self, cov_layer_output):
        cov = self._inner_prod_cov(cov_layer_output)
        cov_plus_noise = self._add_homo_noise(cov, cov_layer_output)
        return cov, cov_plus_noise



class InnerProdHeteroNoiseNoConvKernelCNP(NoConvKernelCNP):
    
    def __init__(self,
                 latent_dim,
                 num_basis_dim,
                 use_attention=False):


        super().__init__(latent_dim=latent_dim,
                         num_basis_dim=num_basis_dim,
                         num_output_dim=num_basis_dim + 2,
                         use_attention=use_attention)



    def cov(self, cov_layer_output):
        cov = self._inner_prod_cov(cov_layer_output)
        cov_plus_noise = self._add_hetero_noise( cov, cov_layer_output)
        return cov, cov_plus_noise


class KvvHomoNoiseNoConvKernelCNP(NoConvKernelCNP):
    
    def __init__(self,
                 latent_dim,
                 num_basis_dim,
                 use_attention=False):


        super().__init__(latent_dim=latent_dim,
                         num_basis_dim=num_basis_dim,
                         num_output_dim=num_basis_dim + 2,
                         use_attention=use_attention)


    def cov(self, cov_layer_output):
        cov = self._kvv_cov(cov_layer_output)
        cov_plus_noise = self._add_homo_noise(cov, cov_layer_output)

        return cov, cov_plus_noise


class KvvHeteroNoiseNoConvKernelCNP(NoConvKernelCNP):
    
    def __init__(self,
                 latent_dim,
                 num_basis_dim,
                 use_attention=False):

        super().__init__(latent_dim=latent_dim,
                         num_basis_dim=num_basis_dim,
                         num_output_dim=num_basis_dim + 3,
                         use_attention=use_attention)

    def cov(self, cov_layer_output):
        cov = self._kvv_cov(cov_layer_output)
        cov_plus_noise = self._add_hetero_noise(cov, cov_layer_output)
        return cov, cov_plus_noise


# ANP based models

class InnerProdHomoNoiseNoConvKernelANP(InnerProdHomoNoiseNoConvKernelCNP):
    
    def __init__(self,
                 latent_dim,
                 num_basis_dim):

        super().__init__(latent_dim=latent_dim,
                         num_basis_dim=num_basis_dim,
                         use_attention=True)


class InnerProdHeteroNoiseNoConvKernelANP(InnerProdHeteroNoiseNoConvKernelCNP):
    
    def __init__(self,
                 latent_dim,
                 num_basis_dim):


        super().__init__(latent_dim=latent_dim,
                         num_basis_dim=num_basis_dim,
                         use_attention=True)


class KvvHomoNoiseNoConvKernelANP(KvvHomoNoiseNoConvKernelCNP):
    
    def __init__(self,
                 latent_dim,
                 num_basis_dim):


        super().__init__(latent_dim=latent_dim,
                         num_basis_dim=num_basis_dim,
                         use_attention=True)


class KvvHeteroNoiseNoConvKernelANP(KvvHeteroNoiseNoConvKernelCNP):
    
    def __init__(self,
                 latent_dim,
                 num_basis_dim):

        super().__init__(latent_dim=latent_dim,
                         num_basis_dim=num_basis_dim,
                         use_attention=True)

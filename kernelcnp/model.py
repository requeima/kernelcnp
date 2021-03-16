import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import stheno.torch as stheno

import convcnp.data
from convcnp.experiment import report_loss, RunningAverage
from convcnp.utils import gaussian_logpdf, init_sequential_weights, to_multiple, compute_dists
from convcnp.architectures import SimpleConv, UNet
from abc import ABC, abstractmethod

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ConvDeepSet(nn.Module):
    """One-dimensional ConvDeepSet module. Uses an RBF kernel for psi(x, x').

    Args:
        out_channels (int): Number of output channels.
        init_length_scale (float): Initial value for the length scale.
    """

    def __init__(self, out_channels, init_length_scale):
        super(ConvDeepSet, self).__init__()
        self.out_channels = out_channels
        self.in_channels = 2
        self.g = self.build_weight_model()
        self.sigma = nn.Parameter(np.log(init_length_scale) *
                                  torch.ones(self.in_channels), requires_grad=True)
        self.sigma_fn = torch.exp

    def build_weight_model(self):
        """Returns a function point-wise function that transforms the
        (in_channels + 1)-dimensional representation to dimensionality
        out_channels.

        Returns:
            torch.nn.Module: Linear layer applied point-wise to channels.
        """
        model = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels),
        )
        init_sequential_weights(model)
        return model
    
    def rbf(self, dists):
        """Compute the RBF values for the distances using the correct length
        scales.

        Args:
            dists (tensor): Pair-wise distances between x and t.

        Returns:
            tensor: Evaluation of psi(x, t) with psi an RBF kernel.
        """
        # Compute the RBF kernel, broadcasting appropriately.
        scales = self.sigma_fn(self.sigma)[None, None, None, :]
        a, b, c = dists.shape
        return torch.exp(-0.5 * dists.view(a, b, c, -1) / scales ** 2)

    def forward(self, x, y, t):
        """Forward pass through the layer with evaluations at locations t.

        Args:
            x (tensor): Inputs of observations of shape (n, 1).
            y (tensor): Outputs of observations of shape (n, in_channels).
            t (tensor): Inputs to evaluate function at of shape (m, 1).

        Returns:
            tensor: Outputs of evaluated function at z of shape
                (m, out_channels).
        """
        # Compute shapes.
        batch_size = x.shape[0]
        n_in = x.shape[1]
        n_out = t.shape[1]

        # Compute the pairwise distances.
        # Shape: (batch, n_in, n_out).
        dists = compute_dists(x, t)

        # Compute the weights.
        # Shape: (batch, n_in, n_out, in_channels).
        wt = self.rbf(dists)

        # Compute the extra density channel.
        # Shape: (batch, n_in, 1).
        density = torch.ones(batch_size, n_in, 1).to(device)

        # Concatenate the channel.
        y_out = torch.cat([density, y], dim=2)

        # Perform the weighting.
        # Shape: (batch, n_in, n_out, in_channels + 1).
        y_out = y_out.view(batch_size, n_in, -1, self.in_channels) * wt

        # Sum over the inputs.
        # Shape: (batch, n_out, in_channels + 1).
        y_out = y_out.sum(1)

        # Use density channel to normalize convolution.
        density, conv = y_out[..., :1], y_out[..., 1:]
        normalized_conv = conv / (density + 1e-8)
        y_out = torch.cat((density, normalized_conv), dim=-1)

        # Apply the point-wise function.
        # Shape: (batch, n_out, out_channels).
        y_out = y_out.view(batch_size * n_out, self.in_channels)
        y_out = self.g(y_out)
        y_out = y_out.view(batch_size, n_out, self.out_channels)

        return y_out

class FinalLayer(nn.Module):
    """One-dimensional Set convolution layer. Uses an RBF kernel for psi(x, x').

    Args:
        in_channels (int): Number of inputs channels.
        init_length_scale (float): Initial value for the length scale.
    """

    def __init__(self, in_channels, init_length_scale, out_channels=1):
        super(FinalLayer, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.g = self.build_weight_model()
        self.sigma = nn.Parameter(np.log(init_length_scale) * torch.ones(self.in_channels), requires_grad=True)
        self.sigma_fn = torch.exp

    def build_weight_model(self):
        """Returns a function point-wise function that transforms the
        (in_channels + 1)-dimensional representation to dimensionality
        out_channels.

        Returns:
            torch.nn.Module: Linear layer applied point-wise to channels.
        """
        model = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels),
        )
        init_sequential_weights(model)
        return model
    
    def rbf(self, dists):
        """Compute the RBF values for the distances using the correct length
        scales.

        Args:
            dists (tensor): Pair-wise distances between x and t.

        Returns:
            tensor: Evaluation of psi(x, t) with psi an RBF kernel.
        """
        # Compute the RBF kernel, broadcasting appropriately.
        scales = self.sigma_fn(self.sigma)[None, None, None, :]
        a, b, c = dists.shape
        return torch.exp(-0.5 * dists.view(a, b, c, -1) / scales ** 2)

    def forward(self, x, y, t):
        """Forward pass through the layer with evaluations at locations t.

        Args:
            x (tensor): Inputs of observations of shape (n, 1).
            y (tensor): Outputs of observations of shape (n, in_channels).
            t (tensor): Inputs to evaluate function at of shape (m, 1).

        Returns:
            tensor: Outputs of evaluated function at z of shape
                (m, out_channels).
        """
        # Compute shapes.
        batch_size = x.shape[0]
        n_in = x.shape[1]
        n_out = t.shape[1]

        # Compute the pairwise distances.
        # Shape: (batch, n_in, n_out).
        dists = compute_dists(x, t)

        # Compute the weights.
        # Shape: (batch, n_in, n_out, in_channels).
        wt = self.rbf(dists)

        # Perform the weighting.
        # Shape: (batch, n_in, n_out, in_channels).
        y_out = y.view(batch_size, n_in, -1, self.in_channels) * wt

        # Sum over the inputs.
        # Shape: (batch, n_out, in_channels).
        y_out = y_out.sum(1)

        # Apply the point-wise function.
        # Shape: (batch, n_out, out_channels).
        y_out = y_out.view(batch_size * n_out, self.in_channels)
        y_out = self.g(y_out)
        y_out = y_out.view(batch_size, n_out, self.out_channels)

        return y_out

class KernelCNP(ABC, nn.Module):
    """One-dimensional KernelCNP model.

    Args:
        learn_length_scale (bool): Learn the length scale.
        points_per_unit (int): Number of points per unit interval on input.
            Used to discretize function.
    """

    def __init__(self, rho, points_per_unit, cov_layer_out_channels, num_basis_dim):
        super(KernelCNP, self).__init__()
        self.activation = nn.Sigmoid()
        self.sigma_fn = nn.Softplus()
        self.rho = rho
        self.multiplier = 2 ** self.rho.num_halving_layers
        self.num_basis_dim = num_basis_dim

        # Compute initialisation.
        self.points_per_unit = points_per_unit
        init_length_scale = 2.0 / self.points_per_unit
        init_weight_scale = 0.0
        num_noise_samples = 0
        init_noise_scale = 5.0

        
        # Instantiate encoder
        self.encoder = ConvDeepSet(out_channels=self.rho.in_channels,
                                   init_length_scale=init_length_scale)
        
        # Instantiate mean and standard deviation layers
        self.mean_layer = FinalLayer(in_channels=self.rho.out_channels,
                                     init_length_scale=init_length_scale,
                                     out_channels=1)
        self.cov_layer = FinalLayer(in_channels=self.rho.out_channels,
                                    init_length_scale=init_length_scale,
                                    out_channels=cov_layer_out_channels)

        # Kernel Parameters
        self.kernel_sigma = nn.Parameter(np.log(init_length_scale)* torch.ones(1), requires_grad=True)
        self.kernel_weight = nn.Parameter( init_weight_scale * torch.ones(1), requires_grad=True)
        self.kernel_fn = torch.exp

        # Noise Parameters
        self.noise_scale = nn.Parameter(np.log(1.0) * torch.ones(1), requires_grad=True)

    def forward(self, x, y, x_out, noiseless=False):
        """Run the model forward.

        Args:
            x (tensor): Observation locations of shape (batch, data, features).
            y (tensor): Observation values of shape (batch, data, outputs).
            x_out (tensor): Locations of outputs of shape (batch, data, features).
            
        Returns:
            tuple[tensor]: Means and standard deviations of shape (batch_out, channels_out).
        """
        n_out = x_out.shape[1]

        # Determine the grid on which to evaluate functional representation.
        x_min = min(torch.min(x).cpu().numpy(),
                    torch.min(x_out).cpu().numpy(), -2.) - 0.1
        x_max = max(torch.max(x).cpu().numpy(),
                    torch.max(x_out).cpu().numpy(), 2.) + 0.1
        num_points = int(to_multiple(self.points_per_unit * (x_max - x_min),
                                     self.multiplier))
        x_grid = torch.linspace(x_min, x_max, num_points).to(device)
        x_grid = x_grid[None, :, None].repeat(x.shape[0], 1, 1)

        # Apply first layer and conv net. Take care to put the axis ranging
        # over the data last.
        h = self.activation(self.encoder(x, y, x_grid))
        h = h.permute(0, 2, 1)
        h = h.reshape(h.shape[0], h.shape[1], num_points)
        h = self.rho(h)
        h = h.reshape(h.shape[0], h.shape[1], -1).permute(0, 2, 1)

        # Check that shape is still fine!
        if h.shape[1] != x_grid.shape[1]:
            raise RuntimeError('Shape changed.')

        # Produce mean
        mean = self.mean_layer(x_grid, h, x_out)

        # Produce cov
        cov_layer_output = self.cov_layer(x_grid, h, x_out)
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



class InnerProdHomoNoiseKernelCNP(KernelCNP):
    
    def __init__(self,
                 rho,
                 points_per_unit,
                 num_basis_dim):

        self.num_basis_dim = num_basis_dim

        super().__init__(rho=rho,
                         points_per_unit=points_per_unit, 
                         cov_layer_out_channels=num_basis_dim,
                         num_basis_dim=num_basis_dim)


    def cov(self, cov_layer_output):
        cov = self._inner_prod_cov(cov_layer_output)
        cov_plus_noise = self._add_homo_noise(cov, cov_layer_output)
        return cov, cov_plus_noise



class InnerProdHeteroNoiseKernelCNP(KernelCNP):
    
    def __init__(self,
                 rho,
                 points_per_unit,
                 num_basis_dim):

        self.num_basis_dim = num_basis_dim

        super().__init__(rho=rho,
                         points_per_unit=points_per_unit, 
                         cov_layer_out_channels=num_basis_dim + 1,
                         num_basis_dim=num_basis_dim)


    def cov(self, cov_layer_output):
        cov = self._inner_prod_cov(cov_layer_output)
        cov_plus_noise = self._add_hetero_noise( cov, cov_layer_output)
        return cov, cov_plus_noise


class KvvHomoNoiseKernelCNP(KernelCNP):
    
    def __init__(self,
                 rho,
                 points_per_unit,
                 num_basis_dim):
        
        self.num_basis_dim = num_basis_dim

        super().__init__(rho=rho,
                         points_per_unit=points_per_unit,  
                         cov_layer_out_channels=num_basis_dim + 1,
                         num_basis_dim=num_basis_dim)



    def cov(self, cov_layer_output):
        cov = self._kvv_cov(cov_layer_output)
        cov_plus_noise = self._add_homo_noise(cov, cov_layer_output)

        return cov, cov_plus_noise


class KvvHeteroNoiseKernelCNP(KernelCNP):
    
    def __init__(self,
                 rho,
                 points_per_unit,
                 num_basis_dim):

        self.num_basis_dim = num_basis_dim

        super().__init__(rho=rho,
                         points_per_unit=points_per_unit, 
                         cov_layer_out_channels=num_basis_dim + 2,
                         num_basis_dim=num_basis_dim)

    def cov(self, cov_layer_output):
        cov = self._kvv_cov(cov_layer_output)
        cov_plus_noise = self._add_hetero_noise(cov, cov_layer_output)
        return cov, cov_plus_noise

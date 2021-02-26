import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import stheno.torch as stheno

import convcnp.data
from convcnp.experiment import report_loss, RunningAverage
from convcnp.utils import gaussian_logpdf, init_sequential_weights, to_multiple
from convcnp.architectures import SimpleConv, UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def to_numpy(x):
    """Convert a PyTorch tensor to NumPy."""
    return x.squeeze().detach().cpu().numpy()

def compute_dists(x, y):
    """Fast computation of pair-wise distances for the 1d case.

    Args:
        x (tensor): Inputs of shape (batch, n, 1).
        y (tensor): Inputs of shape (batch, m, 1).

    Returns:
        tensor: Pair-wise distances of shape (batch, n, m).
    """
    return (x - y.permute(0, 2, 1)) ** 2


class ConvDeepSet(nn.Module):
    """One-dimensional ConvDeepSet module. Uses an RBF kernel for psi(x, x').

    Args:
        out_channels (int): Number of output channels.
        init_length_scale (float): Initial value for the length scale.
    """

    def __init__(self, out_channels, init_length_scale, init_noise_std=None, num_noise_samples=0):
        super(ConvDeepSet, self).__init__()
        self.out_channels = out_channels
        self.in_channels = 2 + num_noise_samples
        self.num_noise_samples = num_noise_samples
        self.g = self.build_weight_model()
        self.sigma = nn.Parameter(np.log(init_length_scale) *
                                  torch.ones(self.in_channels), requires_grad=True)
        self.sigma_fn = torch.exp
        if init_noise_std is not None:
            self.noise_std = nn.Parameter(init_noise_std * torch.ones(1), requires_grad=True)
        else:
            self.noise_std =  None
        self.noise_fn = torch.exp

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

    def sample_noise(self, batch_size, n_in):
        m = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0])) 
        samples = m.sample((batch_size, n_in, self.num_noise_samples)).to(device)
        samples = samples.view((batch_size, n_in, self.num_noise_samples))
        samples = samples * self.noise_fn(self.noise_std)
        return samples

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
        if self.num_noise_samples > 0:
            noise_samples = self.sample_noise(batch_size, n_in)
            y_out = torch.cat([density, y, noise_samples], dim=2)
        else:
            # Shape: (batch, n_in, in_channels + 1).
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

class KernelCNP(nn.Module):
    """One-dimensional KernelCNP model.

    Args:
        learn_length_scale (bool): Learn the length scale.
        points_per_unit (int): Number of points per unit interval on input.
            Used to discretize function.
    """

    def __init__(self, rho, points_per_unit, sigma_channels=1024, add_dists_in_kernel=False):
        super(KernelCNP, self).__init__()
        self.activation = nn.Sigmoid()
        self.sigma_fn = nn.Softplus()
        self.rho = rho
        self.multiplier = 2 ** self.rho.num_halving_layers
        self.add_dists_in_kernel = add_dists_in_kernel
        self.param_cov = "inner product" #options: "root", "kernel", "inner_product", "inner product with diag"
        self.correction_term = None

        # Compute initialisation.
        self.points_per_unit = points_per_unit
        init_length_scale = 2.0 / self.points_per_unit
        init_weight_scale = 0.0
        num_noise_samples = 0
        init_noise_scale = 5.0

        
        # Instantiate encoder
        self.encoder = ConvDeepSet(out_channels=self.rho.in_channels,
                                   init_length_scale=init_length_scale, 
                                   init_noise_std=init_noise_scale,
                                   num_noise_samples=num_noise_samples)
        
        # Instantiate mean and standard deviation layers
        self.mean_layer = FinalLayer(in_channels=self.rho.out_channels,
                                     init_length_scale=init_length_scale)
        self.sigma_layer = FinalLayer(in_channels=self.rho.out_channels,
                                      init_length_scale=init_length_scale,
                                      out_channels=sigma_channels)

        # Kernel Parameters
        self.kernel_sigma = nn.Parameter(np.log(init_length_scale)* torch.ones(1), requires_grad=True)
        self.kernel_weight = nn.Parameter( init_weight_scale * torch.ones(1), requires_grad=True)
        self.kernel_fn = torch.exp

        # Noise Parameters
        # self.noise_value = 1e-4
        self.noise_scale = nn.Parameter(np.log(1.0) * torch.ones(1), requires_grad=True)
    def rbf_kernel(self, basis_emb, x_out):
        if self.add_dists_in_kernel:
            x_dists = torch.cdist(x_out, x_out)
            emb_dists = torch.cdist(basis_emb, basis_emb) 
            dists = emb_dists + self.kernel_fn(self.kernel_weight) * x_dists
        else:
            dists = torch.cdist(basis_emb, basis_emb) 
        # Compute the RBF kernel, broadcasting appropriately.
        scales = self.sigma_fn(self.kernel_sigma)
        return torch.exp(-0.5 * dists / scales ** 2)
    
    def check_eig(self, cov):
        correction_term = 1e-6

        pos_eigs = False
        while not pos_eigs:
            eigs = torch.symeig(cov).eigenvalues
            if torch.sum(eigs < 0) > 0:
                cov = cov + correction_term * torch.eye(cov.shape[1])[None, ...].to(device)
                correction_term = correction_term * 10
            else:
                pos_eigs = True    
        return cov

    def check_det(self, cov):
        correction_term = 1e-10
        pos_det = False
        while not pos_det:
            if any(torch.det(cov)<= 1e-10):
                correction_term = correction_term * 10
                cov = cov + correction_term * torch.eye(cov.shape[1])[None, ...].to(device)
                # print(correction_term)
                # print(torch.det(cov))
            else:
                pos_det = True

        self.correction_term = correction_term
        return cov

    def forward(self, x, y, x_out):
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
        
        # Produce Covariance
        if self.param_cov == "root":
            basis_emb = self.sigma_layer(x_grid, h, torch.cat([x_out, x_grid], dim=1))
            root_cov = self.rbf_kernel(basis_emb, x_out)
            full_cov = torch.matmul(torch.transpose(root_cov, dim0=-2, dim1=-1), root_cov) 
            cov = full_cov[:, :n_out, :n_out]
            eps = self.noise_value * torch.eye(cov.shape[1])[None, ...].to(device)
        elif self.param_cov == "kernel":
            basis_emb = self.sigma_layer(x_grid, h, x_out)
            var_weights =  torch.exp(basis_emb[:, :, -1:])
            basis_emb = basis_emb[:, :, :-1]
            var_weights = torch.matmul(var_weights, torch.transpose(var_weights, dim0=-2, dim1=-1)) 
            cov = self.rbf_kernel(basis_emb, x_out)
            cov = cov * var_weights
        elif self.param_cov == "inner product":
            basis_emb = self.sigma_layer(x_grid, h, x_out)
            cov = torch.matmul(basis_emb, torch.transpose(basis_emb, dim0=-2, dim1=-1)) / basis_emb.shape[-1]
            eps = torch.exp(self.noise_scale) * torch.eye(cov.shape[1])[None, ...].to(device)
            # print(torch.exp(self.noise_scale))
            cov = cov + eps
            # cov = self.check_det(cov)
        elif self.param_cov == "inner product with diag":
            basis_emb = self.sigma_layer(x_grid, h, x_out)
            var_weights =  torch.exp(basis_emb[:, :, -1:])
            basis_emb = basis_emb[:, :, :-1]
            cov = torch.matmul(basis_emb, torch.transpose(basis_emb, dim0=-2, dim1=-1))
            idx = np.arange(cov.shape[-1])
            cov[:, idx, idx] = cov[:, idx, idx] + var_weights[:, :, 0]
            cov = self.check_det(cov)

        return mean, cov

    @property
    def num_params(self):
        """Number of parameters in model."""
        return np.sum([torch.tensor(param.shape).prod()
                       for param in self.parameters()])
    


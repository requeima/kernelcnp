import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

import numpy as np
import matplotlib.pyplot as plt
import os

__all__ = ['to_multiple',
           'BatchLinear',
           'init_layer_weights',
           'init_sequential_weights',
           'compute_dists',
           'pad_concat',
           'stacked_batch_mlp']


def to_multiple(x, multiple):
    """Convert `x` to the nearest above multiple.

    Args:
        x (number): Number to round up.
        multiple (int): Multiple to round up to.

    Returns:
        number: `x` rounded to the nearest above multiple of `multiple`.
    """
    if x % multiple == 0:
        return x
    else:
        return x + multiple - x % multiple


class BatchLinear(nn.Linear):
    """Helper class for linear layers on order-3 tensors.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): Use a bias. Defaults to `True`.
    """

    def __init__(self, in_features, out_features, bias=True):
        super(BatchLinear, self).__init__(in_features=in_features,
                                          out_features=out_features,
                                          bias=bias)
        nn.init.xavier_normal_(self.weight, gain=1)
        if bias:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, x):
        """Forward pass through layer. First unroll batch dimension, then pass
        through dense layer, and finally reshape back to a order-3 tensor.

        Args:
              x (tensor): Inputs of shape `(batch, n, in_features)`.

        Returns:
              tensor: Outputs of shape `(batch, n, out_features)`.
        """
        num_functions, num_inputs = x.shape[0], x.shape[1]
        x = x.view(num_functions * num_inputs, self.in_features)
        out = super(BatchLinear, self).forward(x)
        return out.view(num_functions, num_inputs, self.out_features)


def init_layer_weights(layer):
    """Initialize the weights of a :class:`nn.Layer` using Glorot
    initialization.

    Args:
        layer (:class:`nn.Module`): Single dense or convolutional layer from
            :mod:`torch.nn`.

    Returns:
        :class:`nn.Module`: Single dense or convolutional layer with
            initialized weights.
    """
    nn.init.xavier_normal_(layer.weight, gain=1)
    nn.init.constant_(layer.bias, 1e-3)


def init_sequential_weights(model, bias=0.0):
    """Initialize the weights of a nn.Sequential model with Glorot
    initialization.

    Args:
        model (:class:`nn.Sequential`): Container for model.
        bias (float, optional): Value for initializing bias terms. Defaults
            to `0.0`.

    Returns:
        (nn.Sequential): model with initialized weights
    """
    for layer in model:
        if hasattr(layer, 'weight'):
            nn.init.xavier_normal_(layer.weight, gain=1)
        if hasattr(layer, 'bias'):
            nn.init.constant_(layer.bias, bias)
    return model

def stacked_batch_mlp(input_features_dim, latent_features_dim, output_features_dim):
    """
    """
    mlp = nn.Sequential(BatchLinear(input_features_dim, latent_features_dim),
                        nn.ReLU(),
                        BatchLinear(latent_features_dim, latent_features_dim),
                        nn.ReLU(),
                        BatchLinear(latent_features_dim, output_features_dim))
    return mlp

def compute_dists(x, y):
    """Fast computation of pair-wise distances for the 1d case.

    Args:
        x (tensor): Inputs of shape `(batch, n, 1)`.
        y (tensor): Inputs of shape `(batch, m, 1)`.

    Returns:
        tensor: Pair-wise distances of shape `(batch, n, m)`.
    """
    assert x.shape[2] == 1 and y.shape[2] == 1, \
        'The inputs x and y must be 1-dimensional observations.'
    return (x - y.permute(0, 2, 1)) ** 2


def pad_concat(t1, t2):
    """Concat the activations of two layer channel-wise by padding the layer
    with fewer points with zeros.

    Args:
        t1 (tensor): Activations from first layers of shape `(batch, n1, c1)`.
        t2 (tensor): Activations from second layers of shape `(batch, n2, c2)`.

    Returns:
        tensor: Concatenated activations of both layers of shape
            `(batch, max(n1, n2), c1 + c2)`.
    """
    if t1.shape[2] > t2.shape[2]:
        padding = t1.shape[2] - t2.shape[2]
        if padding % 2 == 0:  # Even difference
            t2 = F.pad(t2, (int(padding / 2), int(padding / 2)), 'reflect')
        else:  # Odd difference
            t2 = F.pad(t2, (int((padding - 1) / 2), int((padding + 1) / 2)),
                       'reflect')
    elif t2.shape[2] > t1.shape[2]:
        padding = t2.shape[2] - t1.shape[2]
        if padding % 2 == 0:  # Even difference
            t1 = F.pad(t1, (int(padding / 2), int(padding / 2)), 'reflect')
        else:  # Odd difference
            t1 = F.pad(t1, (int((padding - 1) / 2), int((padding + 1) / 2)),
                       'reflect')

    return torch.cat([t1, t2], dim=1)


def compute_dists(x, y):
    """Fast computation of pair-wise distances for the 1d case.

    Args:
        x (tensor): Inputs of shape `(batch, n, 1)`.
        y (tensor): Inputs of shape `(batch, m, 1)`.

    Returns:
        tensor: Pair-wise distances of shape `(batch, n, m)`.
    """
    assert x.shape[2] == 1 and y.shape[2] == 1, \
        'The inputs x and y must be 1-dimensional observations.'
    return (x - y.permute(0, 2, 1)) ** 2


def build_grid(x_context, x_target, points_per_unit, grid_multiplier):
    
    n_out = x_target.shape[1]

    # Determine the grid on which to evaluate functional representation.
    x_min = min(torch.min(x_context).cpu().numpy(),
                torch.min(x_target).cpu().numpy(), -2.) - 0.1
    x_max = max(torch.max(x_context).cpu().numpy(),
                torch.max(x_target).cpu().numpy(), 2.) + 0.1
    num_points = int(to_multiple(points_per_unit * (x_max - x_min),
                                 grid_multiplier))
    x_grid = torch.linspace(x_min, x_max, num_points).to(x_context.device)
    x_grid = x_grid[None, :, None].repeat(x_context.shape[0], 1, 1)
    return x_grid, num_points



# =============================================================================
# Plotting util
# =============================================================================


def plot_samples_and_data(model,
                          gen_plot,
                          xmin,
                          xmax,
                          root,
                          epoch,
                          plot_marginals):

    # Sample datasets from generator
    data = list(gen_plot)[0]

    # Split context and target sets out
    ctx_in = data['x_context']
    ctx_out = data['y_context']

    trg_in = data['x_target']
    trg_out = data['y_target']

    # Locations to query predictions at
    xrange = xmax - xmin
    xmin = xmin - 0.5 * xrange
    xmax = xmax + 0.5 * xrange
    plot_inputs = torch.linspace(xmin, xmax, 100)[None, :, None]
    plot_inputs = plot_inputs.repeat(ctx_in.shape[0], 1, 1).to(ctx_in.device)

    # Make predictions 
    tensors = model(ctx_in, ctx_out, plot_inputs)
    mean, cov, cov_plus_noise = [tensor.detach().cpu() for tensor in tensors]

    plt.figure(figsize=(16, 3))

    for i in range(3):

        plt.subplot(1, 3, i + 1)

        # Plot samples from predictive distribution
        # Try samlping and plotting with jitter -- if error is raised, plot marginals
        try:
            
            if plot_marginals:
                plt.fill_between(plot_inputs[i, :, 0].cpu(),
                                 mean[i, :, 0] - 2 * torch.diag(cov[i, :, :]),
                                 mean[i, :, 0] + 2 * torch.diag(cov[i, :, :]),
                                 color='blue',
                                 alpha=0.3,
                                 zorder=1)
            
            else:
                cov_plus_jitter = cov[i, :, :].double() + \
                                  1e-4 * torch.eye(cov.shape[-1]).double()
                dist = torch.distributions.MultivariateNormal(loc=mean[i, :, 0].double(),
                                                              covariance_matrix=cov_plus_jitter)

                for j in range(100):
                    sample = dist.sample()
                    plt.plot(plot_inputs[i, :, 0].cpu(), sample, color='blue', alpha=0.05, zorder=2)
                
        except Exception as e:
            print(e)
            plt.fill_between(plot_inputs[i, :, 0].cpu(),
                             mean[i, :, 0] - 2 * torch.diag(cov[i, :, :]),
                             mean[i, :, 0] + 2 * torch.diag(cov[i, :, :]),
                             color='blue',
                             alpha=0.3,
                             zorder=1)


        plt.plot(plot_inputs[i, :, 0].cpu(),
                 mean[i, :, 0],
                 '--',
                 color='k')

        plt.scatter(ctx_in[i, :, 0].cpu(),
                    ctx_out[i, :, 0].cpu(),
                    s=100,
                    marker='+',
                    color='black',
                    label='Context',
                    zorder=3)

        plt.scatter(trg_in[i, :, 0].cpu(),
                    trg_out[i, :, 0].cpu(),
                    s=100,
                    marker='+',
                    color='red',
                    label='Target',
                    zorder=3)
        
        plt.xlim([xmin, xmax])

    plt.tight_layout()
    
    if not os.path.exists(f'{root}/plots'): os.mkdir(f'{root}/plots')
        
    plt.savefig(f'{root}/plots/{str(epoch).zfill(6)}.png')
    plt.close()
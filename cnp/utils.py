import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

try:
    import stheno
except ModuleNotFoundError:
    pass

import cnp
    

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


def build_grid(x_context, x_target, points_per_unit, grid_multiplier, grid_margin):
    
    x_mins = []
    x_maxs = []
    x_grids = []
    
    for d in range(x_context.shape[-1]):
        
        # Determine the grid on which to evaluate functional representation.
        x_min = min(torch.min(x_context[..., d]).cpu().numpy(),
                    torch.min(x_target[..., d]).cpu().numpy()) - grid_margin
        x_max = max(torch.max(x_context[..., d]).cpu().numpy(),
                    torch.max(x_target[..., d]).cpu().numpy()) + grid_margin
        # update the lists
        x_mins.append(x_min)
        x_maxs.append(x_max)
        
        n = int(to_multiple(points_per_unit * (x_max - x_min),
                                    grid_multiplier))
        
        # compute the x_grid
        x_grids.append(torch.linspace(x_min, x_max, n).to(x_context.device))

    x_grid = torch.stack(torch.meshgrid(x_grids), dim=-1)
    
    return x_grid


def move_channel_idx(x, to_last, num_dims):
    if to_last:
        perm_idx = [0] + [i + 2 for i in range(num_dims)] + [1]
    else:
        perm_idx = [0, num_dims + 1] + [i + 1 for i in range(num_dims)]
    
    return x.permute(perm_idx)



# =============================================================================
# Logger util class
# =============================================================================


class Logger(object):
    
    def __init__(self, log_directory, log_filename):
        self.terminal = sys.stdout
        self.log_directory = log_directory
        self.log_filename = log_filename

    def write(self, message):

        self.terminal.write(message)

        fhandle = open(self.log_directory.file(self.log_filename), "a")
        fhandle.write(message)
        fhandle.close()


    def flush(self):
        pass    

    

# =============================================================================
# Plotting util
# =============================================================================


def plot_samples_and_data(model,
                          valid_epoch,
                          x_plot_min,
                          x_plot_max,
                          root,
                          epoch,
                          latent_model,
                          plot_marginals,
                          device):

    # Get single iteration from validation epoch
    data = valid_epoch[0]

    # Split context and target sets out
    ctx_in = data['x_context'].to(device)[:3]
    ctx_out = data['y_context'].to(device)[:3]

    trg_in = data['x_target'].to(device)[:3]
    trg_out = data['y_target'].to(device)[:3]

    # Locations to query predictions at
    plot_inputs = torch.linspace(x_plot_min, x_plot_max, 200)[None, :, None]
    plot_inputs = plot_inputs.repeat(ctx_in.shape[0], 1, 1).to(ctx_in.device)
    num_samples = 10

    # Make predictions 
    if latent_model:
        tensors = model(ctx_in, ctx_out, plot_inputs, num_samples=num_samples)
        
        sample_means, noise_vars = tensors
        sample_means = sample_means.detach().cpu()
        
        idx = torch.arange(noise_vars.shape[2])
        noise_vars = noise_vars[:, :, idx, idx].detach().cpu()
        
        latent_marg_mean = torch.mean(sample_means[:, :, :, 0], dim=0)
        latent_marg_var = torch.mean(noise_vars, dim=0) + \
                          torch.var(sample_means[:, :, :, 0], dim=0)
    
    else:
        tensors = model(ctx_in, ctx_out, plot_inputs)
        mean, cov, cov_plus_noise = [tensor.detach().cpu() \
                                     for tensor in tensors]

    plt.figure(figsize=(16, 3))

    for i in range(3):

        plt.subplot(1, 3, i + 1)

        # Plot samples from predictive distribution
        # Try samlping with jitter - if error raised, plot marginals
        if latent_model:

            for j in range(num_samples):
                
                plt.plot(plot_inputs[i, :, 0].cpu(),
                         sample_means[j, i, :, 0],
                         color='blue',
                         alpha=0.5,
                         zorder=2)
                
            plt.fill_between(plot_inputs[i, :, 0].cpu(),
                             latent_marg_mean[i, :] - 2 * latent_marg_var[i, :] ** 0.5,
                             latent_marg_mean[i, :] + 2 * latent_marg_var[i, :] ** 0.5,
                             color='blue',
                             alpha=0.2,
                             zorder=1)
        
        else:
            try:

                if plot_marginals:
                    plt.fill_between(plot_inputs[i, :, 0].cpu(),
                                     mean[i, :, 0] - 2 * torch.diag(cov[i, :, :]),
                                     mean[i, :, 0] + 2 * torch.diag(cov[i, :, :]),
                                     color='blue',
                                     alpha=0.2,
                                     zorder=1)

                else:
                    cov_plus_jitter = cov[i, :, :].double() + \
                                      1e-4 * torch.eye(cov.shape[-1]).double()
                    dist = torch.distributions.MultivariateNormal(loc=mean[i, :, 0].double(),
                                                                  covariance_matrix=cov_plus_jitter)

                    for j in range(num_samples):
                        sample = dist.sample()
                        plt.plot(plot_inputs[i, :, 0].cpu(),
                                 sample,
                                 color='blue',
                                 alpha=0.5,
                                 zorder=2)

            except Exception as e:
                
                plt.fill_between(plot_inputs[i, :, 0].cpu(),
                                 mean[i, :, 0] - 2 * torch.diag(cov[i, :, :]),
                                 mean[i, :, 0] + 2 * torch.diag(cov[i, :, :]),
                                 color='blue',
                                 alpha=0.2,
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
        
        plt.xlim([x_plot_min, x_plot_max])

    plt.tight_layout()
    
    if not os.path.exists(f'{root}/plots'): os.mkdir(f'{root}/plots')
        
    plt.savefig(f'{root}/plots/{str(epoch).zfill(6)}.png')
    plt.close()


    
# =============================================================================
# Make Datagenerator Function
# =============================================================================

def make_generator(data_kind, gen_params, kernel_params):

    data_kind = data_kind[:-3] if data_kind[-3:] == '-lb' else data_kind

    if data_kind == 'sawtooth':
        gen = cnp.data.SawtoothGenerator(**gen_params)

    else:
        params = kernel_params[data_kind]
        if data_kind == 'eq':
            kernel = stheno.EQ().stretch(params[0])

        elif data_kind == 'matern':
            kernel = stheno.Matern52().stretch(params[0])

        elif data_kind in ['noisy-mixture', 'noisy-mixture-slow']:
            kernel = stheno.EQ().stretch(params[0]) + \
                        stheno.EQ().stretch(params[1])

        elif data_kind in ['weakly-periodic', 'weakly-periodic-slow']:
            kernel = stheno.EQ().stretch(params[0]) * \
                        stheno.EQ().periodic(period=params[1])

        else:
            raise ValueError(f'Unknown generator kind "{data_kind}".')

        gen = cnp.data.GPGenerator(kernel=kernel, **gen_params)

    return gen

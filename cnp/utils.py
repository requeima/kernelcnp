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

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()
    

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


# =============================================================================
# Plotting utility
# =============================================================================


def plot_pred_prey_fits(model,
                        valid_data,
                        holdout_data,
                        subsampled_data,
                        num_noisy_samples,
                        num_noiseless_samples,
                        device,
                        save_path):
    
    plt.figure(figsize=(16, 8))
    
    # Plot fit on validation with all context points
    plt.subplot(2, 2, 1)
    plot_single_1d_fit(model=model,
                       x_context=valid_data['x_context'][:1, :, None],
                       y_context=valid_data['y_context'][:1, 0, :, None] / 100 + 1e-2,
                       x_target=valid_data['x_target'][:1, :, None],
                       y_target=valid_data['y_target'][:1, 0, :, None] / 100 + 1e-2,
                       num_noisy_samples=num_noisy_samples,
                       num_noiseless_samples=num_noiseless_samples,
                       device=device,
                       xmin=0.)
    
    # Plot fit on validation with few context points
    plt.subplot(2, 2, 2)
    
    plot_single_1d_fit(model=model,
                       x_context=valid_data['x_context'][:1, ::5, None],
                       y_context=valid_data['y_context'][:1, 0, ::5, None] / 100 + 1e-2,
                       x_target=valid_data['x_target'][:1, ::5, None],
                       y_target=valid_data['y_target'][:1, 0, ::5, None] / 100 + 1e-2,
                       num_noisy_samples=num_noisy_samples,
                       num_noiseless_samples=num_noiseless_samples,
                       device=device,
                       xmin=0.)
    
    # Plot fit on holdout data
    plt.subplot(2, 2, 3)
    i = np.random.choice(np.arange(holdout_data[0]['x_context'].shape[0]))
    plot_single_1d_fit(model=model,
                       x_context=holdout_data[0]['x_context'][i:i+1],
                       y_context=holdout_data[0]['y_context'][i:i+1] / 100 + 1e-2,
                       x_target=holdout_data[0]['x_target'][i:i+1],
                       y_target=holdout_data[0]['y_target'][i:i+1] / 100 + 1e-2,
                       num_noisy_samples=num_noisy_samples,
                       num_noiseless_samples=num_noiseless_samples,
                       device=device,
                       xmin=0.)
    
    # Plot fit on subsampled data
    plt.subplot(2, 2, 4)
    i = np.random.choice(np.arange(subsampled_data[0]['x_context'].shape[0]))
    plot_single_1d_fit(model=model,
                       x_context=subsampled_data[0]['x_context'][i:i+1],
                       y_context=subsampled_data[0]['y_context'][i:i+1] / 100 + 1e-2,
                       x_target=subsampled_data[0]['x_target'][i:i+1],
                       y_target=subsampled_data[0]['y_target'][i:i+1] / 100 + 1e-2,
                       num_noisy_samples=num_noisy_samples,
                       num_noiseless_samples=num_noiseless_samples,
                       device=device,
                       xmin=0.)
    
    plt.tight_layout()
    plt.savefig(save_path)
    
    
def plot_single_1d_fit(model,
                       x_context,
                       y_context,
                       x_target,
                       y_target,
                       num_noisy_samples,
                       num_noiseless_samples,
                       device,
                       xmin=None,
                       xmax=None):
    
    assert x_context.shape[0] == 1
    
    # Draw noisy samples
    result = draw_1d_predictive_samples(model=model,
                                        x_context=x_context,
                                        y_context=y_context,
                                        x_target=x_target,
                                        num_samples=num_noisy_samples,
                                        noiseless=False,
                                        device=device)
    x_plot_noisy, samples_noisy, xmin, xmax = result
    
    x_plot_noisy = x_plot_noisy[0, :, 0]
    samples_noisy = torch.transpose(samples_noisy[:, 0, :], dim0=0, dim1=1)
    
    plt.plot(to_numpy(x_plot_noisy),
             to_numpy(samples_noisy),
             color='green',
             alpha=min(0.02, 1./num_noiseless_samples),
             zorder=1)
    
    # Draw noiseless samples
    result = draw_1d_predictive_samples(model=model,
                                        x_context=x_context,
                                        y_context=y_context,
                                        x_target=x_target,
                                        num_samples=num_noiseless_samples,
                                        noiseless=True,
                                        device=device)
    x_plot_noiseless, samples_noiseless, xmin, xmax = result
    
    x_plot_noiseless = x_plot_noiseless[0, :, 0]
    samples_noiseless = torch.transpose(samples_noiseless[:, 0, :], dim0=0, dim1=1)
    
    plt.plot(to_numpy(x_plot_noiseless),
             to_numpy(samples_noiseless),
             color='black',
             alpha=1.,
             zorder=2)
    
    # Slice context and target
    x_context = x_context[0, :, 0]
    y_context = y_context[0, :]
    x_target = x_target[0, :, 0]
    y_target = y_target[0, :]
    
    plt.scatter(to_numpy(x_context),
                to_numpy(y_context),
                marker='+',
                c='black',
                s=50,
                zorder=4)
    
    plt.scatter(to_numpy(x_target),
                to_numpy(y_target),
                marker='+',
                c='red',
                s=10,
                zorder=3)
                              
def draw_1d_predictive_samples(model,
                               x_context,
                               y_context,
                               x_target,
                               num_samples,
                               noiseless,
                               device,
                               xmin=None,
                               xmax=None):
    if xmin is None:
        xmin = min(torch.min(x_context), torch.min(x_target))
        
    if xmax is None:
        xmax = max(torch.max(x_context), torch.max(x_target))
    
    x_pad = (xmax - xmin) / 10.
    x_plot = torch.linspace(xmin-x_pad, xmax+x_pad, 200)[None, :, None]
    
    samples = model.sample(x_context=x_context.to(device),
                           y_context=y_context.to(device),
                           x_target=x_plot.to(device),
                           num_samples=num_samples,
                           noiseless=noiseless,
                           double=True)
    
    return x_plot, samples, xmin, xmax
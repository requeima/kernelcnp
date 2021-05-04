import numpy as np
import torch
import torch.nn as nn

from cnp.utils import (
    init_sequential_weights, 
    compute_dists, 
    build_grid
)
from cnp.aggregation import MeanPooling, FullyConnectedDeepSet
from cnp.architectures import FullyConnectedNetwork



# =============================================================================
# Standard fully connected decoder architecture
# =============================================================================


class StandardDecoder(nn.Module):
    """
    Decoder used for standard CNP model.

    Args:
        input_dim  (int): Dimension of the input
        latent_dim (int): Dimension of the hidden representation
        output_dim (int): Dimension of the output
    """

    def __init__(self, input_dim, latent_dim, output_dim):
        
        super().__init__()
        
        hidden_dims = [latent_dim]
        nonlinearity = 'Tanh'
        
        self.post_pooling_fn = FullyConnectedNetwork(input_dim=input_dim + hidden_dims,
                                                     output_dim=self.output_dim,
                                                     hidden_dims=hidden_dims,
                                                     nonlinearity=nonlinearity)

    def forward(self, r, x_context, y_context, *args, **kwargs):
        """
        Forward pass through standard decoder

        Args:
            r        (tensor): Hidden representation (B, 1, R)
            x_target (tensor): Target locations      (B, T, Din)

        Returns:
            z        (tensor): Output features       (B, T, Dout)
        """
        
        assert len(r.shape) ==           \
               len(x_context.shape) ==   \
               len(y_context.shape) == 3
        
        # Reshape inputs to model
        B = r.shape[0]
        T = x_target.shape[1]

        # If latent representation is global, repeat for every target point
        # to get tensor (B, T, R)
        if r.shape[1] == 1:
            r = r.repeat(1, T, 1)

        # Concatenate inputs with representation to get (B, T, R + Din)
        z = torch.cat([x_target, r], -1)
        z = self.post_pooling_fn(z)

        return z



# =============================================================================
# Standard convolutional decoder architecture
# =============================================================================


class ConvDecoder(nn.Module):
    """
    One-dimensional Set convolution layer. Uses an RBF kernel for psi(x, x').

    Args:
        in_channels (int): Number of inputs channels.
        init_length_scale (float): Initial value for the length scale.
    """

    def __init__(self,
                 input_dim, 
                 conv_architecture, 
                 conv_out_channels, 
                 out_channels, 
                 init_length_scale, 
                 points_per_unit, 
                 grid_multiplier,
                 grid_margin):

        super().__init__()
        
        self.conv = conv_architecture
        self.input_dim = input_dim
        self.conv_out_channels = conv_out_channels
        self.out_channels = out_channels
        self.grid_multiplier = grid_multiplier
        self.grid_margin = grid_margin
        self.points_per_unit = points_per_unit
        self.linear_model = self.build_weight_model()
        self.sigma = nn.Parameter(np.log(init_length_scale) * \
                                  torch.ones(self.input_dim),
                                  requires_grad=True)
        self.sigma_fn = torch.exp

    def build_weight_model(self):
        model = nn.Sequential(
            nn.Linear(self.conv_out_channels, self.out_channels),
        )
        init_sequential_weights(model)
        return model

    def forward(self, r, x_context, y_context, x_target):
        """Forward pass through the layer with evaluations at locations t.

        Args:
            x (tensor): Inputs of observations of shape (n, 1).
            y (tensor): Outputs of observations of shape (n, in_channels).
            t (tensor): Inputs to evaluate function at of shape (m, 1).

        Returns:
            tensor: Outputs of evaluated function at z of shape
                (m, out_channels).
        """
        
        # Put the channels in the last dimension
        # (batch, r_out, x_grid_1, ..., x_grid_d)
        r = self.conv(r)

        # Build grid
        x_grid = build_grid(x_context, 
                            x_target, 
                            self.points_per_unit, 
                            self.grid_multiplier,
                            self.grid_margin)
        
        # convert to (batch, n_target, x_grid_1, ..., x_grid_d, x_dims)
        x_grid = x_grid[None, None, ...]
        a, b, c = x_target.shape
        # convert to (batch, n_target, 1, ..., 1, x_dims)
        x_target = x_target.view(a, b, *([1] * c), c)

        # Shape: (1, 1, 1, ..., 1, x_dims)
        scales = torch.exp(self.sigma).view(1, 1, *([1] * c), c)

        # Compute RBF
        # Shape: (batch, n_target, x_grid_1, ..., x_grid_d)
        rbf = (x_grid - x_target) / scales
        rbf = torch.exp(-0.5 * (rbf ** 2).sum(dim=-1))

        # Perform the weighting.
        # Shape: (batch, n_target, r_out)
        z = torch.einsum('bt..., br... -> btr', rbf, r)

        # Apply the point-wise function
        # Shape: (batch, n_out, out_channels)
        z = self.linear_model(z)

        return z
        

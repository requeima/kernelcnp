import numpy as np
import torch
import torch.nn as nn

from .utils import (
    init_sequential_weights, 
    BatchLinear, 
    compute_dists, 
    stacked_batch_mlp,
    build_grid
)
from .aggregation import CrossAttention, MeanPooling, FullyConnectedDeepSet
from .architectures import FullyConnectedNetwork


class StandardDecoder(nn.Module):
    """Decoder used for standard CNP model.

    Args:
        input_dim (int): Dimensionality of the input.
        latent_dim (int): Dimensionality of the hidden representation.
        output_dim (int): Dimensionality of the output.
    """

    def __init__(self, input_dim, latent_dim, output_dim):
        super(StandardDecoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        post_pooling_fn = stacked_batch_mlp(self.input_dim, self.latent_dim, self.output_dim)
        self.post_pooling_fn = init_sequential_weights(post_pooling_fn)

    def forward(self, r, x_context, y_context, x_target):
        """Forward pass through the decoder.

        Args:
            r (torch.tensor): Hidden representation for each task of shape
                `(batch, None, latent_dim)`.
            x_target (tensor): Target locations of shape
                `(batch, num_targets, input_dim)`.

        Returns:
            tensor: Output values at each query point of shape
                `(batch, num_targets, output_dim)`
        """
        # Reshape inputs to model.
        num_functions, num_evaluations = r.shape[0], x_target.shape[1]

        # If latent representation is global, repeat once for each input.
        if r.shape[1] == 1:
            r = r.repeat(1, num_evaluations, 1)

        # Concatenate latents with inputs and pass through decoder.
        # Shape: (batch, num_targets, input_dim + latent_dim).
        z = torch.cat([x_target, r], -1)
        z = self.post_pooling_fn(z)

        # Separate mean and standard deviations and return.
        return z


class ConvDecoder(nn.Module):
    """One-dimensional Set convolution layer. Uses an RBF kernel for psi(x, x').

    Args:
        in_channels (int): Number of inputs channels.
        init_length_scale (float): Initial value for the length scale.
    """

    def __init__(self, 
                 conv_architecture, 
                 in_channels, 
                 out_channels, 
                 init_length_scale, 
                 points_per_unit, 
                 grid_multiplier):

        super().__init__()
        self.conv = conv_architecture
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.grid_multiplier = grid_multiplier
        self.points_per_unit = points_per_unit
        self.linear_model = self.build_weight_model()
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

        r = self.conv(r)
        r = r.reshape(r.shape[0], r.shape[1], -1).permute(0, 2, 1)

        x_grid, num_points = build_grid(x_context, 
                                x_target, 
                                self.points_per_unit, 
                                self.grid_multiplier)
        # Compute shapes.
        batch_size = x_grid.shape[0]
        n_in = x_grid.shape[1]
        n_out = x_target.shape[1]


        # Compute the pairwise distances.
        # Shape: (batch, n_in, n_out).
        dists = compute_dists(x_grid, x_target)

        # Compute the weights.
        # Shape: (batch, n_in, n_out, in_channels).
        wt = self.rbf(dists)

        # Perform the weighting.
        # Shape: (batch, n_in, n_out, in_channels).
        z = r.view(batch_size, n_in, -1, self.in_channels) * wt

        # Sum over the inputs.
        # Shape: (batch, n_out, in_channels).
        z = z.sum(1)

        # Apply the point-wise function.
        # Shape: (batch, n_out, out_channels).
        z = z.view(batch_size * n_out, self.in_channels)
        z = self.linear_model(z)
        z = z.view(batch_size, n_out, self.out_channels)

        return z
        
        
        
# =============================================================================
# Fully Connected Translation Equivariant Decoder
# =============================================================================


class FullyConnectedTEDecoder(nn.Module):
    
    def __init__(self):
        
        super().__init__()
    
    
    def forward(self, r, x_ctx, y_ctx, x_trg):
        
        """
        r     : (B, C, R)
        x_ctx : (B, C, Din)
        y_ctx : (B, C, Dout)
        x_ctx : (B, T, Din)
        """

        D = y_ctx.shape[-1]
        
        # Context and target inputs
        x_ctx = x_ctx[:, :, None, :]
        x_trg = x_trg[:, None, :, :]
        
        diff = x_ctx - x_trg
        
        y_ctx = y_ctx[:, :, None, :]
        y_ctx = y_ctx.repeat(1, 1, diff.shape[2], 1)
        
        r = r[:, :, None, :].repeat(1, 1, diff.shape[2], 1)
        
        ctx = torch.cat([diff, r, y_ctx], dim=-1)
        
        tensor = self.deepset(ctx)
        
        return tensor
        
        

# =============================================================================
# Standard Translation Equivariant Decoder
# =============================================================================


class StandardFullyConnectedTEDecoder(FullyConnectedTEDecoder):
    
    def __init__(self,
                 input_dim,
                 output_dim,
                 rep_dim,
                 embedding_dim):
        
        super().__init__()
        
        # Input dimension of encoder (Din + R)
        element_input_dim = input_dim + output_dim + rep_dim
        
        # Sizes of hidden layers and nonlinearity type
        # Used for both elementwise and aggregate networks
        hidden_dims = [128, 128]
        nonlinearity = 'ReLU'
        
        # Element network -- in (B, C, T, Din + R), out (B, C, T, R)
        element_network = FullyConnectedNetwork(input_dim=element_input_dim,
                                                output_dim=rep_dim,
                                                hidden_dims=hidden_dims,
                                                nonlinearity=nonlinearity)
        
        # Dimensions to mean over -- in (B, C, T, R), out (B, T, R)
        aggregation_dims = [1]
        
        # Aggregate network -- in (B, T, R), out (B, T, E)
        aggregate_network = FullyConnectedNetwork(input_dim=rep_dim,
                                                  output_dim=embedding_dim,
                                                  hidden_dims=hidden_dims,
                                                  nonlinearity=nonlinearity)
        
        # Deepset architecture
        deepset = FullyConnectedDeepSet(element_network,
                                        aggregation_dims,
                                        aggregate_network)
        
        self.deepset = deepset
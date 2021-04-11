import numpy as np
import torch
import torch.nn as nn

from cnp.aggregation import (
    CrossAttention,
    MeanPooling,
    FullyConnectedDeepSet
)

from cnp.architectures import FullyConnectedNetwork

from cnp.utils import (
    init_sequential_weights, 
    BatchLinear,
    compute_dists, 
    to_multiple, 
    stacked_batch_mlp,
    build_grid
)


class StandardEncoder(nn.Module):
    """Encoder used for standard CNP model.

    Args:
        input_dim (int): Dimensionality of the input.
        latent_dim (int): Dimensionality of the hidden representation.
        use_attention (bool, optional): Use attention. Defaults to `False`.
    """

    def __init__(self,
                 input_dim,
                 latent_dim,
                 use_attention=False):
        
        super(StandardEncoder, self).__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.use_attention = use_attention

        pre_pooling_fn = stacked_batch_mlp(self.input_dim, self.latent_dim, self.latent_dim)
        self.pre_pooling_fn = init_sequential_weights(pre_pooling_fn)
        
        if self.use_attention:
            self.pooling_fn = CrossAttention()
        else:
            self.pooling_fn = MeanPooling(pooling_dim=1)

    def forward(self, x_context, y_context, x_target):
        """Forward pass through the decoder.

        Args:
            x_context (tensor): Context locations of shape
                `(batch, num_context, input_dim_x)`.
            y_context (tensor): Context values of shape
                `(batch, num_context, input_dim_y)`.
            x_target (tensor, optional): Target locations of shape
                `(batch, num_target, input_dim_x)`.

        Returns:
            tensor: Latent representation of each context set of shape
                `(batch, 1, latent_dim)`.
        """
        assert len(x_context.shape) == 3, \
            'Incorrect shapes: ensure x_context is a rank-3 tensor.'
        assert len(y_context.shape) == 3, \
            'Incorrect shapes: ensure y_context is a rank-3 tensor.'

        decoder_input = torch.cat((x_context, y_context), dim=-1)
        
        h = self.pre_pooling_fn(decoder_input)
        return self.pooling_fn(h, x_context, x_target)

    

class ConvEncoder(nn.Module):
    """One-dimensional ConvDeepSet module. Uses an RBF kernel for psi(x, x').

    Args:
        out_channels (int): Number of output channels.
        init_length_scale (float): Initial value for the length scale.
    """

    def __init__(self, 
                 out_channels, 
                 init_length_scale, 
                 points_per_unit, 
                 grid_multiplier):
        super().__init__()
        self.activation = nn.Sigmoid()
        self.out_channels = out_channels
        self.in_channels = 2
        self.linear_model = self.build_weight_model()
        self.sigma = nn.Parameter(np.log(init_length_scale) *
                                  torch.ones(self.in_channels), requires_grad=True)
        self.sigma_fn = torch.exp
        self.grid_multiplier = grid_multiplier
        self.points_per_unit = points_per_unit

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

    def forward(self, x_context, y_context, x_target):
        """Forward pass through the layer with evaluations at locations t.

        Args:
            x (tensor): Inputs of observations of shape (n, 1).
            y (tensor): Outputs of observations of shape (n, in_channels).
            t (tensor): Inputs to evaluate function at of shape (m, 1).

        Returns:
            tensor: Outputs of evaluated function at z of shape
                (m, out_channels).
        """
        
        x_grid, num_points = build_grid(x_context, 
                                        x_target, 
                                        self.points_per_unit, 
                                        self.grid_multiplier)

        # Compute shapes.
        batch_size = x_context.shape[0]
        n_in = x_context.shape[1]
        n_out = x_grid.shape[1]

        # Compute the pairwise distances.
        # Shape: (batch, n_in, n_out).
        dists = compute_dists(x_context, x_grid)

        # Compute the weights.
        # Shape: (batch, n_in, n_out, in_channels).
        wt = self.rbf(dists)

        # Compute the extra density channel.
        # Shape: (batch, n_in, 1).
        density = torch.ones(batch_size, n_in, 1).to(x_context.device)

        # Concatenate the channel.
        y_out = torch.cat([density, y_context], dim=2)

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
        y_out = self.linear_model(y_out)
        y_out = y_out.view(batch_size, n_out, self.out_channels)

        # Apply the activation layer. Take care to put the axis ranging
        # over the data last.
        r = self.activation(y_out)
        r = r.permute(0, 2, 1)
        r = r.reshape(r.shape[0], r.shape[1], num_points)

        return r
        
        
        
# =============================================================================
# Fully Connected Translation Equivariant Encoder
# =============================================================================


class FullyConnectedTEEncoder(nn.Module):
    
    def __init__(self, deepset):
        
        super().__init__()
        
        self.deepset = deepset
    
    
    def forward(self, x_ctx, y_ctx, x_trg):
        
        assert len(x_ctx.shape) == 3
        assert len(y_ctx.shape) == 3
        assert len(x_trg.shape) == 3
        
        # Compute context input pairwise differences
        x_diff = x_ctx[:, None, :, :] - x_ctx[:, :, None, :]
        
        # Tile context outputs to concatenate with input differences
        y_ctx_tile1 = y_ctx[:, None, :, :].repeat(1, x_diff.shape[1], 1, 1)
        y_ctx_tile2 = y_ctx[:, :, None, :].repeat(1, 1, x_diff.shape[2], 1)
        
        # Concatenate input differences and outputs, to obtain complete context
        ctx = torch.cat([x_diff, y_ctx_tile1, y_ctx_tile2], dim=-1)
        
        # Latent representation of context set -- r has shape (B, C, R)
        r = self.deepset(ctx)
        
        return r
    

# =============================================================================
# Standard Translation Equivariant Encoder
# =============================================================================


class StandardFullyConnectedTEEncoder(FullyConnectedTEEncoder):
    
    def __init__(self,
                 input_dim,
                 output_dim,
                 rep_dim):
        
        # Input dimension of encoder (Din + 2 * Dout)
        element_input_dim = input_dim + 2 * output_dim
        
        # Sizes of hidden layers and nonlinearity type
        # Used for both elementwise and aggregate networks
        hidden_dims = [100]
        nonlinearity = 'Tanh'
        
        # Element network -- in (B, C, C, Din + 2 * Dout), out (B, C, C, R)
        element_network = FullyConnectedNetwork(input_dim=element_input_dim,
                                                output_dim=rep_dim,
                                                hidden_dims=hidden_dims,
                                                nonlinearity=nonlinearity)
        
        # Dimensions to mean over -- in (B, C, C, R), out (B, R)
        aggregation_dims = [2]
        
        # Aggregate network -- in (B, R), out (B, R)
        aggregate_network = FullyConnectedNetwork(input_dim=rep_dim,
                                                  output_dim=rep_dim,
                                                  hidden_dims=hidden_dims,
                                                  nonlinearity=nonlinearity)
        
        # Deepset architecture
        deepset = FullyConnectedDeepSet(element_network,
                                        aggregation_dims,
                                        aggregate_network)
        
        super().__init__(deepset=deepset)
        
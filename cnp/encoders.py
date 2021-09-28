import numpy as np
import torch
import torch.nn as nn

try:
    import lab.torch as B
except ModuleNotFoundError:
    pass

from cnp.aggregation import (
    FullyConnectedDeepSet,
    MultiHeadAttention
)

from cnp.architectures import FullyConnectedNetwork

from cnp.utils import (
    init_sequential_weights, 
    compute_dists, 
    to_multiple,
    build_grid,
    move_channel_idx,
    PositiveChannelwiseConv1D
)


# =============================================================================
# Standard Encoder for Fully Connected CNPs
# =============================================================================


class StandardEncoder(nn.Module):
    """
    Encoder used for standard CNP model

    Args:
        input_dim     (int) : Dimensionality of the input.
        latent_dim    (int) : Dimensionality of the hidden representation.
        use_attention (bool): Whether to use attention.
    """

    def __init__(self,
                 input_dim,
                 latent_dim,
                 num_layers,
                 use_attention):
        
        super(StandardEncoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.use_attention = use_attention
        self.output_dim = 1
        
        # Hidden dimensions and nonlinearity type for fully-connected network
        hidden_dims = num_layers * [latent_dim]
        nonlinearity = 'ReLU'
        
        # Number of attentive heads - used only if use_attention is True
        num_heads = 8
        
        self.pre_pooling = FullyConnectedNetwork(input_dim=input_dim+self.output_dim,
                                                 output_dim=latent_dim,
                                                 hidden_dims=hidden_dims,
                                                 nonlinearity=nonlinearity)
        
        if self.use_attention:
        
            self.key_query = FullyConnectedNetwork(input_dim=input_dim,
                                                   output_dim=latent_dim,
                                                   hidden_dims=hidden_dims,
                                                   nonlinearity=nonlinearity)
            
            self.pooling = MultiHeadAttention(key_input_dim=latent_dim,
                                              key_embedding_dim=latent_dim,
                                              value_input_dim=latent_dim,
                                              value_embedding_dim=latent_dim,
                                              output_embedding_dim=latent_dim,
                                              num_heads=num_heads)
            
        else:
            self.pooling = lambda _, __, tensor : torch.mean(tensor,
                                                             dim=1,
                                                             keepdim=True)
            

    def forward(self, x_context, y_context, x_target):
        """
        Forward pass through the decoder

        Args:
            x_context (tensor): Context inputs,              (B, C, Din)
            y_context (tensor): Context outputs,             (B, C, Dout)
            x_target  (tensor): Target inputs for attention, (B, C, Din)

        Returns:
            r         (tensor): Latent representation        (B, 1, R)
        """
        
        assert len(x_context.shape) ==   \
               len(y_context.shape) ==   \
               len(x_target.shape) == 3

        xy_context = torch.cat([x_context, y_context], dim=-1)
        
        tensor = self.pre_pooling(xy_context)
        
        if self.use_attention:
            
            x_context = self.key_query(x_context)
            x_target = self.key_query(x_target)
        
        r = self.pooling(x_context, x_target, tensor)
        
        return r



# =============================================================================
# Standard Encoder for Fully Connected ANPs
# =============================================================================


class StandardANPEncoder(nn.Module):

    def __init__(self, input_dim, latent_dim):
        
        super().__init__()
        
        nonlinearity = 'ReLU'
        
        self.latent_dim = latent_dim
        
        self.det_path = StandardEncoder(input_dim=input_dim,
                                        latent_dim=latent_dim,
                                        num_layers=6,
                                        use_attention=True)
        
        self.stoch_path = StandardEncoder(input_dim=input_dim,
                                          latent_dim=latent_dim,
                                          num_layers=2,
                                          use_attention=False)
        
        self.post_stoch_path = FullyConnectedNetwork(input_dim=latent_dim,
                                                     output_dim=2*latent_dim,
                                                     hidden_dims=2*[latent_dim],
                                                     nonlinearity=nonlinearity)


    def forward(self, x_context, y_context, x_target):
        """
        Forward pass through the decoder

        Args:
            x_context (tensor): Context inputs,              (B, C, Din)
            y_context (tensor): Context outputs,             (B, C, Dout)
            x_target  (tensor): Target inputs for attention, (B, C, Din)

        Returns:
            dist (torch.distribution): Latent distribution (B, T, 2*R)
        """
        
        assert len(x_context.shape) ==   \
               len(y_context.shape) ==   \
               len(x_target.shape) == 3
        
        # Deterministic path
        r_det = self.det_path(x_context, y_context, x_target)
        
        # Stochastic path
        r_stoch = self.stoch_path(x_context, y_context, x_target)
        r_stoch = self.post_stoch_path(r_stoch)
        
        r_stoch_mean = r_stoch[:, :, :self.latent_dim]
        r_stoch_scale = r_stoch[:, :, self.latent_dim:]
        r_stoch_scale = torch.nn.Sigmoid()(r_stoch_scale)
        
        return r_det, r_stoch_mean, r_stoch_scale
    
    
    def sample(self, forward_output):
        
        # Unpack outputs of forward
        r_det, r_stoch_mean, r_stoch_scale = forward_output
        
        # Create normal to sample from
        dist = torch.distributions.Normal(loc=r_stoch_mean,
                                          scale=r_stoch_scale)
        
        # Sample normal, repeat sample
        r_stoch = dist.rsample()
        r_stoch = r_stoch.repeat(1, r_det.shape[1], 1)
        
        # Concatenate with deterministic path
        r = torch.cat([r_det, r_stoch], dim=-1)
        
        return r



# =============================================================================
# Standard Convolutional Encoder for ConvCNPs
# =============================================================================
    

class ConvEncoder(nn.Module):

    def __init__(self,
                 input_dim, 
                 out_channels, 
                 init_length_scale, 
                 points_per_unit, 
                 grid_multiplier,
                 grid_margin,
                 density_normalize=True):
        
        super().__init__()
        
        self.activation = nn.ReLU()
        self.out_channels = out_channels
        self.input_dim = input_dim
        self.linear_model = self.build_weight_model()
        self.sigma = nn.Parameter(np.log(init_length_scale) *
                                  torch.ones(self.input_dim),
                                  requires_grad=True)
        self.grid_multiplier = grid_multiplier
        self.grid_margin = grid_margin
        self.points_per_unit = points_per_unit
        self.density_normalize = density_normalize

    def build_weight_model(self):
        """Returns a function point-wise function that transforms the
        (in_channels + 1)-dimensional representation to dimensionality
        out_channels.

        Returns:
            torch.nn.Module: Linear layer applied point-wise to channels.
        """
        model = nn.Sequential(
            nn.Linear(2, self.out_channels),
        )
        init_sequential_weights(model)
        return model


    def forward(self, x_context, y_context, x_target):
        """Forward pass through the layer with evaluations at locations t.

        Args:
            x (tensor): Inputs of observations of shape (n, d).
            y (tensor): Outputs of observations of shape (n, in_channels).
            t (tensor): Inputs to evaluate function at of shape (m, d).

        Returns:
            tensor: Outputs of evaluated function at z of shape
                (Batch, out_channels, x_grid_1,..., x_grid_n).
        """

        # Number of input dimensions, d
        # x_grid shape: (x_grid_1, ..., x_grid_d, d)
        x_grid = build_grid(x_context, 
                            x_target, 
                            self.points_per_unit, 
                            self.grid_multiplier,
                            self.grid_margin)


        # convert to (1, n_context, x_grid_1, ..., x_grid_d, x_dims)
        x_grid = x_grid[None, None, ...]

        # convert to (batch, n_context, 1, ..., 1, x_dims)
        a, b, c = x_context.shape
        x_context = x_context.view(a, b, *([1] * c), c)

        # Shape: (1, 1, 1, ..., 1, x_dims)
        scales = torch.exp(self.sigma).view(1, 1, *([1] * c), c)

        # Compute RBF
        # Shape: (batch, n_context, x_grid_1, ..., x_grid_d)
        rbf = (x_grid - x_context) / scales
        rbf = torch.exp(-0.5 * (rbf ** 2).sum(dim=-1))

        # Compute the extra density channel.
        # Shape: (batch, n_context, 1, ..., 1, 1)
        density = torch.ones(a, b, *([1] * c), 1).to(x_context.device)
        y_context = y_context.view(a, b, *([1] * c), 1)

        # Concatenate the channel.
        # Shape: (batch, n_context, 1, ..., 1, 2)
        y_out = torch.cat([density, y_context], dim=-1)

        # Perform the weighting.
        y_out = y_out * rbf[..., None]

        # Sum over the inputs.
        # Shape: (batch, x_grid_1, ..., x_grid_d, 2)
        y_out = y_out.sum(1)

        if self.density_normalize:
            # Use density channel to normalize convolution.
            density, conv = y_out[..., :1], y_out[..., 1:]
            normalized_conv = conv / (density + 1e-8)
            y_out = torch.cat((density, normalized_conv), dim=-1)

        # Apply linear function
        y_out = self.linear_model(y_out)

        # Apply the activation layer
        r = self.activation(y_out)

        # Move channels to second index 
        r = move_channel_idx(r, to_last=False, num_dims=c)

        return r
    
    


# =============================================================================
# Standard Convolutional Encoder for on-the-grid EEG data
# =============================================================================
    

class ConvEEGEncoder(nn.Module):

    def __init__(self,
                 num_channels_context,
                 conv_architecture):
        
        super().__init__()
        
        # Input dimension is 1 for EEG time series
        self.input_dim = 1
        
        # Dimension of channels to condition on
        self.num_channels_context = num_channels_context
        
        # Initialise positive convolution layer
        conv = PositiveChannelwiseConv1D(num_channels=2*num_channels_context,
                                         kernel_size=kernel_size,
                                         stride=stride)
        self.conv = conv
        
        # Initialise convolutional architecture
        self.cnn = conv_architecture
        

    def build_weight_model(self):
        
        model = nn.Sequential(
            nn.Linear(self.num_channels_context+1, self.out_channels),
        )
        init_sequential_weights(model)
        
        return model


    def forward(self, y_context, m_context):
        """
        Arguments:
            y_context : torch.tensor, (B, F, C)
            m_context : torch.tensor, (B, F, C)
        """
        
        assert y_context.shape == m_context.shape
        
        # Concatenate context and mask
        ym_context = torch.cat([m_context, y_context], axis=1)
        
        # Pass through positive-constrained convolution
        h = self.conv(ym_context)
        
        # Normalise using density channel
        h0 = h[:, :, :self.num_channels_context]
        h1 = h[:, :, self.num_channels_context:]
        h = h1 / (h0 + 1e-9)
        
        # Pass through CNN
        tensor = self.cnn(h)
        
        return tensor


class ConvPDEncoder(nn.Module):
    def __init__(
        self,
        out_channels=None,
        points_per_unit=20.0,
        grid_multiplier=4,
        grid_margin=1
    ):
        nn.Module.__init__(self)
        self.grid_multiplier = grid_multiplier
        self.grid_margin = grid_margin
        self.points_per_unit = points_per_unit
        self.log_scale = nn.Parameter(
            B.log(torch.tensor(2 / points_per_unit)),
            requires_grad=True,
        )

        if out_channels:
            # Build final linear layer.
            self.linear = nn.Sequential(nn.Linear(3, out_channels))
            init_sequential_weights(self.linear)

            def final_linear(x):
                # Put channels last, apply linear, and undo permutation.
                x = B.transpose(x, perm=(0, 2, 3, 1))
                x = self.linear(x)
                x = B.transpose(x, perm=(0, 3, 1, 2))
                return x

            self.final_linear = final_linear
        else:
            # Omit final linear layer.
            self.final_linear = lambda x: x

    def forward(self, xz, z, x):
        # Discretisation:
        x_grid = build_grid(
            xz,
            x,
            self.points_per_unit,
            self.grid_multiplier,
            self.grid_margin
        )

        with B.device(str(z.device)):
            # Construct density and identity channel.
            density_channel = B.ones(B.dtype(z), *B.shape(z)[:2], 1)
            identity_channel = B.eye(
                B.dtype(z),
                B.shape(z)[0],
                1,
                B.shape(x_grid)[0],
                B.shape(x_grid)[0],
            )

        # Prepend density channel.
        z = B.concat(density_channel, z, axis=2)

        # Put channel dimension second.
        z = B.transpose(z, perm=(0, 2, 1))[..., None]

        # Compute interpolation weights.
        dists2 = B.pw_dists2(xz, x_grid[None, :])
        weights = B.exp(-0.5 * dists2 / B.exp(2 * self.log_scale))
        weights = weights[:, None, :, :]  # Insert channel dimension.

        # Interpolate to grid.
        z = B.matmul(weights * z, weights, tr_a=True)

        # Normalise by density channel.
        z = B.concat(z[:, :1, ...], z[:, 1:, ...] / (z[:, :1, ...] + 1e-8), axis=1)

        # Prepend identity channel to complete the encoding.
        z = B.concat(identity_channel, z, axis=1)

        return x_grid, self.final_linear(z)


# =============================================================================
# Standard Latent Convolutional Encoder for ConvNPs
# =============================================================================
        
        
class StandardConvNPEncoder(ConvEncoder):

    def __init__(self,
                 input_dim,
                 conv_architecture,
                 init_length_scale, 
                 points_per_unit, 
                 grid_multiplier,
                 grid_margin):
        
        self.conv_input_channels = conv_architecture.in_channels
        self.conv_output_channels = conv_architecture.out_channels // 2
        
        super().__init__(input_dim=input_dim, 
                         out_channels=self.conv_input_channels, 
                         init_length_scale=init_length_scale, 
                         points_per_unit=points_per_unit, 
                         grid_multiplier=grid_multiplier,
                         grid_margin=grid_margin)
        
        self.conv_architecture = conv_architecture
        
        
    def forward(self, x_context, y_context, x_target):
        
        r = super().forward(x_context, y_context, x_target)
        r = self.conv_architecture(r)
        
        mean = r[:, ::2]
        scale = torch.nn.Sigmoid()(r[:, 1::2])
        scale = scale + 1e-3
        
        dist = torch.distributions.Normal(loc=mean, scale=scale)
        
        return dist
    
    
    def sample(self, forward_output):
        return forward_output.rsample()

import numpy as np
import torch
import torch.nn as nn

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
    move_channel_idx
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
                 use_attention):
        
        super(StandardEncoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.use_attention = use_attention
        
        # Hidden dimensions and nonlinearity type for fully-connected network
        hidden_dims = [latent_dim]
        nonlinearity = 'Tanh'
        
        # Number of attentive heads - used only if use_attention is True
        num_heads = 8
        
        self.pre_pooling_fn = FullyConnectedNetwork(input_dim=input_dim,
                                                    output_dim=latent_dim,
                                                    hidden_dims=hidden_dims,
                                                    nonlinearity=nonlinearity)
        
        if self.use_attention:
            
            self.pooling_fn = MultiHeadAttention(key_input_dim=input_dim,
                                                 key_embedding_dim=latent_dim,
                                                 value_input_dim=latent_dim,
                                                 value_embedding_dim=latent_dim,
                                                 output_embedding_dim=latent_dim,
                                                 num_heads=num_heads)
            
        else:
            self.pooling_fn = lambda tensor, y, x : torch.mean(tensor,
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
        
        tensor = self.pre_pooling_fn(xy_context)
        
        r = self.pooling_fn(x_context, x_target, tensor)
        
        return r



# =============================================================================
# Standard Encoder for Fully Connected ANPs
# =============================================================================


class StandardANPEncoder(nn.Module):

    def __init__(self, input_dim, latent_dim):
        
        super().__init__()
        
        assert latent_dim % 2 == 0

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.stoch_dim = 64
        self.det_dim = self.latent_dim - self.stoch_dim
        
        self.det_hidden_dims = [det_dim]
        self.stoch_hidden_dims = [stoch_dim]
        self.nonlinearity = 'Tanh'
        
        self.pre_pooling_fn_det = FullyConnectedNetwork(input_dim=self.input_dim,
                                                        output_dim=self.det_dim,
                                                        hidden_dims=self.det_hidden_dims,
                                                        nonlinearity=self.nonlinearity)
        
        self.pre_pooling_fn_stoch = FullyConnectedNetwork(input_dim=self.input_dim,
                                                          output_dim=2*self.stoch_dim,
                                                          hidden_dims=self.stoch_hidden_dims,
                                                          nonlinearity=self.nonlinearity)

        self.pooling_fn_det = MultiHeadAttention(key_input_dim=input_dim,
                                                 key_embedding_dim=det_dim,
                                                 value_input_dim=det_dim,
                                                 value_embedding_dim=det_dim,
                                                 output_embedding_dim=det_dim,
                                                 num_heads=num_heads)
        
        self.pooling_fn_stoch = lambda tensor, y, x : torch.mean(tensor,
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
            dist (torch.distributions.Normal): Latent distribution (B, T, 2*R)
        """
        
        assert len(x_context.shape) ==   \
               len(y_context.shape) ==   \
               len(x_target.shape) == 3

        tensor = torch.cat((x_context, y_context), dim=-1)
        
        # Deterministic path
        r_det = self.pre_pooling_fn_det(tensor)
        r_det = self.pooling_fn_det(r_det, x_context, x_target)
        
        r_det_mean = r_det
        r_det_scale = 1e-9 * torch.ones_like(r_det)
        
        # Stochastic path
        r_stoch = self.pre_pooling_fn_stoch(tensor)
        r_stoch = self.pooling_fn_stoch(r_stoch, x_context, x_target)
        r_stoch = r_stoch.repeat(1, x_target.shape[1], 1)
        
        r_stoch_mean = r_stoch[:, :, :self.stoch_dim]
        r_stoch_scale = r_stoch[:, :, self.stoch_dim:]
        r_stoch_scale = torch.nn.Sigmoid()(r_stoch_scale)
        
        # Create distribution
        mean = torch.cat([r_det_mean, r_stoch_mean], dim=-1)
        scale = torch.cat([r_det_scale, r_stoch_scale], dim=-1)
        
        dist = torch.distributions.Normal(loc=mean, scale=scale)
        
        return dist



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

        # Apply the activation layer. 
        r = self.activation(y_out)

        # Move channels to second index 
        r = move_channel_idx(r,to_last=False, num_dims=c)

        return r



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
        
        distribution = torch.distributions.Normal(loc=mean, scale=scale)
        
        return distribution
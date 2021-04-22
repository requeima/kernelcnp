import torch.nn as nn
import torch

from cnp.utils import (
    init_sequential_weights,
    init_layer_weights,
    pad_concat
)

__all__ = ['SimpleConv', 'UNet']


class DepthwiseSeparableConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, num_dims):
        
        if num_dims == 1:
            convf = nn.Conv1d
            
        elif num_dims == 2:
            convf = nn.Conv2d
            
        elif num_dims == 3:
            convf = nn.Conv3d
            
        else:
            raise ValueError('Number of dimensions > 3 not supported')

        assert kernel_size % 2 == 1
        
        padding = [kernel_size // 2] * num_dims
        kernel_size = [kernel_size] * num_dims

        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = convf(in_channels, 
                               in_channels, 
                               kernel_size=kernel_size, 
                               padding=padding, 
                               groups=in_channels)

        self.pointwise = convf(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class StandardDepthwiseSeparableCNN(nn.Module):
    
    def __init__(self, in_channels, out_channels, num_dims):
        
        super().__init__()
        
        latent_channels = 32
        kernel_size = 5
        num_layers = 12
        
        layers = [DepthwiseSeparableConv(in_channels,
                                         latent_channels,
                                         kernel_size,
                                         num_dims)]
        
        for i in range(num_layers - 2):
            
            layers.append(DepthwiseSeparableConv(latent_channels,
                                                 latent_channels,
                                                 kernel_size,
                                                 num_dims))
            
        layers.append(DepthwiseSeparableConv(latent_channels,
                                             out_channels,
                                             kernel_size,
                                             num_dims))
        
        self.layers = nn.ModuleList(layers)

        self.conv_net = nn.Sequential(*layers)
        init_sequential_weights(self.conv_net)

        
    def forward(self, tensor):
        
        relu = nn.ReLU()
        
        for conv in self.layers[:-1]:
            
            tensor = relu(conv(tensor)) # + tensor
            
        tensor = conv(tensor)
            
        return tensor
        
        
        
class SimpleConv(nn.Module):
    """Small convolutional architecture from 1d experiments in the paper.
    This is a 4-layer convolutional network with fixed stride and channels,
    using ReLU activations.

    Args:
        in_channels (int, optional): Number of channels on the input to the
            network. Defaults to 8.
        out_channels (int, optional): Number of channels on the output by the
            network. Defaults to 8.
    """

    def __init__(self, in_channels=8, out_channels=8):
        super(SimpleConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = nn.ReLU()
        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels, out_channels=16,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=16,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=self.out_channels,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
        )
        init_sequential_weights(self.conv_net)
        self.num_halving_layers = 0

    def forward(self, x):
        """Forward pass through the convolutional structure.

        Args:
            x (tensor): Inputs of shape `(batch, n_in, in_channels)`.

        Returns:
            tensor: Outputs of shape `(batch, n_out, out_channels)`.
        """
        return self.conv_net(x)


class UNet(nn.Module):
    """Large convolutional architecture from 1d experiments in the paper.
    This is a 12-layer residual network with skip connections implemented by
    concatenation.

    Args:
        in_channels (int, optional): Number of channels on the input to
            network. Defaults to 8.
    """

    def __init__(self, in_channels=8):
        super(UNet, self).__init__()
        self.activation = nn.ReLU()
        self.in_channels = in_channels
        self.num_halving_layers = 6
        # This cannot currently be changed
        self.out_channels = 16

        self.l1 = nn.Conv1d(in_channels=self.in_channels,
                            out_channels=self.in_channels,
                            kernel_size=5, stride=2, padding=2)
        self.l2 = nn.Conv1d(in_channels=self.in_channels,
                            out_channels=2 * self.in_channels,
                            kernel_size=5, stride=2, padding=2)
        self.l3 = nn.Conv1d(in_channels=2 * self.in_channels,
                            out_channels=2 * self.in_channels,
                            kernel_size=5, stride=2, padding=2)
        self.l4 = nn.Conv1d(in_channels=2 * self.in_channels,
                            out_channels=4 * self.in_channels,
                            kernel_size=5, stride=2, padding=2)
        self.l5 = nn.Conv1d(in_channels=4 * self.in_channels,
                            out_channels=4 * self.in_channels,
                            kernel_size=5, stride=2, padding=2)
        self.l6 = nn.Conv1d(in_channels=4 * self.in_channels,
                            out_channels=8 * self.in_channels,
                            kernel_size=5, stride=2, padding=2)

        for layer in [self.l1, self.l2, self.l3, self.l4, self.l5, self.l6]:
            init_layer_weights(layer)

        self.l7 = nn.ConvTranspose1d(in_channels=8 * self.in_channels,
                                     out_channels=4 * self.in_channels,
                                     kernel_size=5, stride=2, padding=2,
                                     output_padding=1)
        self.l8 = nn.ConvTranspose1d(in_channels=8 * self.in_channels,
                                     out_channels=4 * self.in_channels,
                                     kernel_size=5, stride=2, padding=2,
                                     output_padding=1)
        self.l9 = nn.ConvTranspose1d(in_channels=8 * self.in_channels,
                                     out_channels=2 * self.in_channels,
                                     kernel_size=5, stride=2, padding=2,
                                     output_padding=1)
        self.l10 = nn.ConvTranspose1d(in_channels=4 * self.in_channels,
                                      out_channels=2 * self.in_channels,
                                      kernel_size=5, stride=2, padding=2,
                                      output_padding=1)
        self.l11 = nn.ConvTranspose1d(in_channels=4 * self.in_channels,
                                      out_channels=self.in_channels,
                                      kernel_size=5, stride=2, padding=2,
                                      output_padding=1)
        self.l12 = nn.ConvTranspose1d(in_channels=2 * self.in_channels,
                                      out_channels=self.in_channels,
                                      kernel_size=5, stride=2, padding=2,
                                      output_padding=1)

        for layer in [self.l7, self.l8, self.l9, self.l10, self.l11, self.l12]:
            init_layer_weights(layer)

    def forward(self, x):
        """Forward pass through the convolutional structure.

        Args:
            x (tensor): Inputs of shape `(batch, n_in, in_channels)`.

        Returns:
            tensor: Outputs of shape `(batch, n_out, out_channels)`.
        """
        h1 = self.activation(self.l1(x))
        h2 = self.activation(self.l2(h1))
        h3 = self.activation(self.l3(h2))
        h4 = self.activation(self.l4(h3))
        h5 = self.activation(self.l5(h4))
        h6 = self.activation(self.l6(h5))
        h7 = self.activation(self.l7(h6))

        h7 = pad_concat(h5, h7)
        h8 = self.activation(self.l8(h7))
        h8 = pad_concat(h4, h8)
        h9 = self.activation(self.l9(h8))
        h9 = pad_concat(h3, h9)
        h10 = self.activation(self.l10(h9))
        h10 = pad_concat(h2, h10)
        h11 = self.activation(self.l11(h10))
        h11 = pad_concat(h1, h11)
        h12 = self.activation(self.l12(h11))

        return pad_concat(x, h12)


class BatchMLP(nn.Module):
    """Helper class for a simple MLP operating on order-3 tensors. Stacks
    several `BatchLinear` modules.

    Args:
        in_features (int): Dimensionality of inputs to MLP.
        out_features (int): Dimensionality of outputs of MLP.
    """

    def __init__(self, in_features, out_features):
        super(BatchMLP, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.net = nn.Sequential(
            nn.Linear(in_features=self.in_features,
                      out_features=self.out_features),
            nn.ReLU(),
            nn.Linear(in_features=self.out_features,
                      out_features=self.out_features)
        )

    def forward(self, x):
        """Forward pass through the network. Assumes a batch of tasks as input
        to the network.

        Args:
            x (tensor): Inputs of shape
                `(num_functions, num_points, input_dim)`.

        Returns:
            tensor: Representation of shape
                `(num_functions, num_points, output_dim)`.
        """
        num_functions, num_points = x.shape[0], x.shape[1]
        x = x.view(num_functions * num_points, -1)
        rep = self.net(x)
        return rep.view(num_functions, num_points, self.out_features)



# =============================================================================
# Fully Connected Neural Network
# =============================================================================


class FullyConnectedNetwork(nn.Module):
    
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dims,
                 nonlinearity):
        
        super().__init__()
        
        shapes = [input_dim] + hidden_dims + [output_dim]
        shapes = [(s1, s2) for s1, s2 in zip(shapes[:-1], shapes[1:])]
        
        self.W = []
        self.b = []
        self.num_linear = len(hidden_dims) + 1
        
        for shape in shapes:

            W = nn.Parameter(torch.randn(size=shape) / shape[0] ** 0.5)
            b = nn.Parameter(torch.randn(size=shape[1:]))

            self.W.append(W)
            self.b.append(b)
            
        self.W = torch.nn.ParameterList(self.W)
        self.b = torch.nn.ParameterList(self.b)
        
        self.nonlinearity = getattr(nn, nonlinearity)()
        
    
    def forward(self, tensor):
        
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            
            tensor = torch.einsum('...i, ij -> ...j', tensor, W)
                
            tensor = tensor + b[(None,) * (len(tensor.shape) - 1)]
            
            if i < self.num_linear - 1:
                tensor = self.nonlinearity(tensor)
        
        return tensor

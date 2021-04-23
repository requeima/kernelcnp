import torch.nn as nn
import torch

from cnp.utils import (
    init_sequential_weights,
    init_layer_weights,
    pad_concat
)

__all__ = ['StandardDepthwiseSeparableCNN', 'HalfUNet', 'UNet']



# =============================================================================
# Depthwise separable convolution layer
# =============================================================================


class DepthwiseSeparableConv(nn.Module):
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 num_dims,
                 stride,
                 transpose):
        
        if num_dims == 1:
            if transpose:
                convf = nn.ConvTranspose1d
            else:
                convf = nn.Conv1d
            
        elif num_dims == 2:
            if transpose:
                convf = nn.ConvTranspose2d
            else:
                convf = nn.Conv2d
            
        elif num_dims == 3:
            if transpose:
                convf = nn.ConvTranspose3d
            else:
                convf = nn.Conv3d
            
        else:
            raise ValueError('Number of dimensions > 3 not supported')

        assert kernel_size % 2 == 1
        
        padding = [kernel_size // 2] * num_dims
        kernel_size = [kernel_size] * num_dims

        super(DepthwiseSeparableConv, self).__init__()
        
        if transpose:

            self.pointwise = convf(in_channels, out_channels, kernel_size=1)
            
            self.depthwise = convf(out_channels, 
                                   out_channels, 
                                   kernel_size=kernel_size, 
                                   padding=padding, 
                                   groups=in_channels,
                                   stride=stride,
                                   output_padding=1)
            
        else:
            self.depthwise = convf(in_channels, 
                                   in_channels, 
                                   kernel_size=kernel_size, 
                                   padding=padding, 
                                   groups=in_channels,
                                   stride=stride)

            self.pointwise = convf(in_channels, out_channels, kernel_size=1)
        self.transpose = transpose

    def forward(self, x):
        
        if self.transpose:
            tensor = self.pointwise(x)
            tensor = self.depthwise(tensor)
            
        else:
            tensor = self.depthwise(x)
            tensor = self.pointwise(tensor)
        
        return tensor



# =============================================================================
# Standard depthwise separable CNN
# =============================================================================


class StandardDepthwiseSeparableCNN(nn.Module):
    
    def __init__(self, in_channels, out_channels, num_dims):
        
        super().__init__()
        
        latent_channels = 32
        kernel_size = 5
        stride = 2
        num_layers = 12
        
        layers = [DepthwiseSeparableConv(in_channels,
                                         latent_channels,
                                         kernel_size,
                                         num_dims,
                                         stride=stride,
                                         transpose=False)]
        
        for i in range(num_layers // 2 - 1):
            
            layers.append(DepthwiseSeparableConv(latent_channels,
                                                 latent_channels,
                                                 kernel_size,
                                                 num_dims,
                                                 stride=stride,
                                                 transpose=False))
        
        for i in range(num_layers // 2 - 1):
            
            layers.append(DepthwiseSeparableConv(latent_channels,
                                                 latent_channels,
                                                 kernel_size,
                                                 num_dims,
                                                 stride=stride,
                                                 transpose=True))
            
        layers.append(DepthwiseSeparableConv(latent_channels,
                                             out_channels,
                                             kernel_size,
                                             num_dims,
                                             stride=stride,
                                             transpose=True))
        
        self.layers = nn.ModuleList(layers)

        self.conv_net = nn.Sequential(*layers)
        init_sequential_weights(self.conv_net)

        
    def forward(self, tensor):
        
        relu = nn.ReLU()
        
        for conv in self.layers[:-1]:
            
            tensor = relu(conv(tensor)) # + tensor
            
        tensor = conv(tensor)
            
        return tensor

    

# =============================================================================
# UNet CNN architecture
# =============================================================================


class UNet(nn.Module):
    """Large convolutional architecture from 1d experiments in the paper.
    This is a 12-layer residual network with skip connections implemented by
    concatenation.

    Args:
        in_channels (int, optional): Number of channels on the input to
            network. Defaults to 8.
    """

    def __init__(self, input_dim, in_channels=8):
        
        super(UNet, self).__init__()
        
        
        conv = getattr(nn, f'Conv{input_dim}d')
        convt = getattr(nn, f'ConvTranspose{input_dim}d')
        
        self.activation = nn.ReLU()
        self.in_channels = in_channels
        self.num_halving_layers = 6

        self.l1 = conv(in_channels=self.in_channels,
                       out_channels=self.in_channels,
                       kernel_size=5,
                       stride=2,
                       padding=2)
        
        self.l2 = conv(in_channels=self.in_channels,
                       out_channels=2*self.in_channels,
                       kernel_size=5,
                       stride=2,
                       padding=2)
        
        self.l3 = conv(in_channels=2*self.in_channels,
                       out_channels=2*self.in_channels,
                       kernel_size=5,
                       stride=2,
                       padding=2)
        
        self.l4 = conv(in_channels=2*self.in_channels,
                       out_channels=4*self.in_channels,
                       kernel_size=5,
                       stride=2,
                       padding=2)
        
        self.l5 = conv(in_channels=4*self.in_channels,
                       out_channels=4*self.in_channels,
                       kernel_size=5,
                       stride=2,
                       padding=2)
        
        self.l6 = conv(in_channels=4*self.in_channels,
                       out_channels=8*self.in_channels,
                       kernel_size=5,
                       stride=2,
                       padding=2)

        for layer in [self.l1, self.l2, self.l3, self.l4, self.l5, self.l6]:
            init_layer_weights(layer)

        self.l7 = convt(in_channels=8*self.in_channels,
                        out_channels=4*self.in_channels,
                        kernel_size=5,
                        stride=2,
                        padding=2,
                        output_padding=1)
        
        self.l8 = convt(in_channels=8*self.in_channels,
                        out_channels=4*self.in_channels,
                        kernel_size=5,
                        stride=2,
                        padding=2,
                        output_padding=1)
        
        self.l9 = convt(in_channels=8*self.in_channels,
                        out_channels=2*self.in_channels,
                        kernel_size=5,
                        stride=2,
                        padding=2,
                        output_padding=1)
        
        self.l10 = convt(in_channels=4*self.in_channels,
                         out_channels=2*self.in_channels,
                         kernel_size=5,
                         stride=2,
                         padding=2,
                         output_padding=1)
        
        self.l11 = convt(in_channels=4*self.in_channels,
                         out_channels=self.in_channels,
                         kernel_size=5,
                         stride=2,
                         padding=2,
                         output_padding=1)
        
        self.l12 = convt(in_channels=2*self.in_channels,
                         out_channels=self.in_channels,
                         kernel_size=5,
                         stride=2,
                         padding=2,
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

        h7 = torch.cat([h5, h7], dim=1)
        h8 = self.activation(self.l8(h7))
        
        h8 = torch.cat([h4, h8], dim=1)
        h9 = self.activation(self.l9(h8))
        
        h9 = torch.cat([h3, h9], dim=1)
        h10 = self.activation(self.l10(h9))
        
        h10 = torch.cat([h2, h10], dim=1)
        h11 = self.activation(self.l11(h10))
        
        h11 = torch.cat([h1, h11], dim=1)
        h12 = self.activation(self.l12(h11))

        return torch.cat([x, h12], dim=1)
    
    

# =============================================================================
# HalfUNet CNN architecture
# =============================================================================
    
    
class HalfUNet(nn.Module):

    def __init__(self,
                 input_dim,
                 in_channels,
                 out_channels):
        
        super().__init__()
        
        conv = getattr(nn, f'Conv{input_dim}d')
        convt = getattr(nn, f'ConvTranspose{input_dim}d')
        
        self.activation = nn.ReLU()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_halving_layers = 6
        
        kernel_size = 5
        padding = kernel_size // 2

        self.l1 = conv(in_channels=self.in_channels,
                       out_channels=self.in_channels,
                       kernel_size=kernel_size,
                       stride=2,
                       padding=padding)
        
        self.l2 = conv(in_channels=self.in_channels,
                       out_channels=2*self.in_channels,
                       kernel_size=kernel_size,
                       stride=2,
                       padding=padding)
        
        self.l3 = conv(in_channels=2*self.in_channels,
                       out_channels=2*self.in_channels,
                       kernel_size=kernel_size,
                       stride=2,
                       padding=padding)
            
        self.l4 = convt(in_channels=2*self.in_channels,
                        out_channels=2*self.in_channels,
                        kernel_size=kernel_size,
                        stride=2,
                        padding=padding,
                        output_padding=1)
        
        self.l5 = convt(in_channels=4*self.in_channels,
                        out_channels=self.in_channels,
                        kernel_size=kernel_size,
                        stride=2,
                        padding=padding,
                        output_padding=1)
        
        self.l6 = convt(in_channels=2*self.in_channels,
                        out_channels=self.in_channels,
                        kernel_size=kernel_size,
                        stride=2,
                        padding=padding,
                        output_padding=1)


        self.last_layer_multiplier = conv(in_channels=2*self.in_channels,
                                          out_channels=self.out_channels,
                                          kernel_size=1,
                                          stride=1,
                                          padding=0)
            

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

        h4 = torch.cat([h4, h2], dim=1)
        h5 = self.activation(self.l5(h4))

        h5 = torch.cat([h5, h1], dim=1)
        h6 = self.activation(self.l6(h5))
        h6 = torch.cat([x, h6], dim=1)

        return self.last_layer_multiplier(h6)
    

# =============================================================================
# BatchMLP architecture
# =============================================================================


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

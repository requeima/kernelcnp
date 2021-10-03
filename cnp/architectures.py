import torch.nn as nn
import torch
import math

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

        super().__init__()
        
        if transpose:

            self.pointwise = convf(in_channels, out_channels, kernel_size=1)
            
            self.depthwise = convf(out_channels, 
                                   out_channels, 
                                   kernel_size=kernel_size, 
                                   padding=padding, 
                                   groups=out_channels,
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
    
    def __init__(self, input_dim, in_channels, out_channels, latent_channels=32):
        
        super().__init__()
        
        kernel_size = 21
        stride = 1
        num_layers = 12
        
        self.num_halving_layers = num_layers // 2
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.l1 = DepthwiseSeparableConv(in_channels,
                                         latent_channels,
                                         kernel_size,
                                         input_dim,
                                         stride=stride,
                                         transpose=False)
        
        self.l2 = DepthwiseSeparableConv(latent_channels,
                                         latent_channels,
                                         kernel_size,
                                         input_dim,
                                         stride=stride,
                                         transpose=False)
        
        self.l3 = DepthwiseSeparableConv(latent_channels,
                                         latent_channels,
                                         kernel_size,
                                         input_dim,
                                         stride=stride,
                                         transpose=False)
        
        self.l4 = DepthwiseSeparableConv(latent_channels,
                                         latent_channels,
                                         kernel_size,
                                         input_dim,
                                         stride=stride,
                                         transpose=False)
        
        self.l5 = DepthwiseSeparableConv(latent_channels,
                                         latent_channels,
                                         kernel_size,
                                         input_dim,
                                         stride=stride,
                                         transpose=False)
        
        self.l6 = DepthwiseSeparableConv(latent_channels,
                                         latent_channels,
                                         kernel_size,
                                         input_dim,
                                         stride=stride,
                                         transpose=False)
        
        self.l7 = DepthwiseSeparableConv(latent_channels,
                                         latent_channels,
                                         kernel_size,
                                         input_dim,
                                         stride=stride,
                                         transpose=False)
        
        self.l8 = DepthwiseSeparableConv(latent_channels,
                                         latent_channels,
                                         kernel_size,
                                         input_dim,
                                         stride=stride,
                                         transpose=False)
        
        self.l9 = DepthwiseSeparableConv(latent_channels,
                                         latent_channels,
                                         kernel_size,
                                         input_dim,
                                         stride=stride,
                                         transpose=False)
        
        self.l10 = DepthwiseSeparableConv(latent_channels,
                                          latent_channels,
                                          kernel_size,
                                          input_dim,
                                          stride=stride,
                                          transpose=False)
        
        self.l11 = DepthwiseSeparableConv(latent_channels,
                                          latent_channels,
                                          kernel_size,
                                          input_dim,
                                          stride=stride,
                                          transpose=False)
        
        self.l12 = DepthwiseSeparableConv(latent_channels,
                                          out_channels,
                                          kernel_size,
                                          input_dim,
                                          stride=stride,
                                          transpose=False)
        
        self.activation = nn.ReLU()

        
    def forward(self, tensor):
        
        tensor = self.l1(tensor)
        tensor = self.activation(tensor)
        
        tensor = self.l2(tensor)
        tensor = self.activation(tensor)
        
        tensor = self.l3(tensor)
        tensor = self.activation(tensor)
        
        tensor = self.l4(tensor)
        tensor = self.activation(tensor)
        
        tensor = self.l5(tensor)
        tensor = self.activation(tensor)
        
        tensor = self.l6(tensor)
        tensor = self.activation(tensor)
        
        tensor = self.l7(tensor)
        tensor = self.activation(tensor)
        
        tensor = self.l8(tensor)
        tensor = self.activation(tensor)
        
        tensor = self.l9(tensor)
        tensor = self.activation(tensor)
        
        tensor = self.l10(tensor)
        tensor = self.activation(tensor)
        
        tensor = self.l11(tensor)
        tensor = self.activation(tensor)
        
        tensor = self.l12(tensor)
            
        return tensor


def _compute_kernel_size(receptive_field, points_per_unit, num_layers):
    receptive_points = receptive_field * points_per_unit
    kernel_size = math.ceil(1 + (receptive_points - 1) / num_layers)
    # Ensure that the kernel size is odd.
    if kernel_size % 2 == 0:
        return kernel_size + 1
    else:
        return kernel_size


def _compute_padding(kernel_size):
    return math.floor(kernel_size / 2)


def build_dws_net(
    receptive_field,
    points_per_unit,
    dimensionality,
    num_in_channels,
    num_out_channels,
    num_layers=8,
    num_channels=64,
):

    kernel_size = _compute_kernel_size(receptive_field, points_per_unit, num_layers)
    padding = _compute_padding(kernel_size)

    if dimensionality == 1:
        conv = nn.Conv1d
    elif dimensionality == 2:
        conv = nn.Conv2d
    else:
        raise NotImplementedError(
            f"Cannot build a net of dimensionality {dimensionality}."
        )

    layers = [
        conv(
            in_channels=num_in_channels,
            out_channels=num_channels,
            kernel_size=1
        )
    ]
    for _ in range(num_layers):
        layers.append(conv(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=num_channels,
        ))
        layers.append(conv(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=1
        ))
        layers.append(nn.LeakyReLU())
    layers.append(conv(
        in_channels=num_channels,
        out_channels=num_out_channels,
        kernel_size=1
    ))

    conv_net = nn.Sequential(*layers)
    init_sequential_weights(conv_net)
    return conv_net


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

    def __init__(self, input_dim, in_channels, out_channels):
        
        super().__init__()
        
        
        conv = getattr(nn, f'Conv{input_dim}d')
        convt = getattr(nn, f'ConvTranspose{input_dim}d')
        
        self.activation = nn.ReLU()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = 8
        self.num_halving_layers = 6
        self.kernel_size = 5

        self.l1 = conv(in_channels=self.in_channels,
                       out_channels=self.latent_channels,
                       kernel_size=self.kernel_size,
                       stride=2,
                       padding=2)
        
        self.l2 = conv(in_channels=self.latent_channels,
                       out_channels=2*self.latent_channels,
                       kernel_size=self.kernel_size,
                       stride=2,
                       padding=2)
        
        self.l3 = conv(in_channels=2*self.latent_channels,
                       out_channels=2*self.latent_channels,
                       kernel_size=self.kernel_size,
                       stride=2,
                       padding=2)
        
        self.l4 = conv(in_channels=2*self.latent_channels,
                       out_channels=4*self.latent_channels,
                       kernel_size=self.kernel_size,
                       stride=2,
                       padding=2)
        
        self.l5 = conv(in_channels=4*self.latent_channels,
                       out_channels=4*self.latent_channels,
                       kernel_size=self.kernel_size,
                       stride=2,
                       padding=2)
        
        self.l6 = conv(in_channels=4*self.latent_channels,
                       out_channels=8*self.latent_channels,
                       kernel_size=self.kernel_size,
                       stride=2,
                       padding=2)

        for layer in [self.l1, self.l2, self.l3, self.l4, self.l5, self.l6]:
            init_layer_weights(layer)

        self.l7 = convt(in_channels=8*self.latent_channels,
                        out_channels=4*self.latent_channels,
                        kernel_size=self.kernel_size,
                        stride=2,
                        padding=2,
                        output_padding=1)
        
        self.l8 = convt(in_channels=8*self.latent_channels,
                        out_channels=4*self.latent_channels,
                        kernel_size=self.kernel_size,
                        stride=2,
                        padding=2,
                        output_padding=1)
        
        self.l9 = convt(in_channels=8*self.latent_channels,
                        out_channels=2*self.latent_channels,
                        kernel_size=self.kernel_size,
                        stride=2,
                        padding=2,
                        output_padding=1)
        
        self.l10 = convt(in_channels=4*self.latent_channels,
                         out_channels=2*self.latent_channels,
                         kernel_size=self.kernel_size,
                         stride=2,
                         padding=2,
                         output_padding=1)
        
        self.l11 = convt(in_channels=4*self.latent_channels,
                         out_channels=self.latent_channels,
                         kernel_size=self.kernel_size,
                         stride=2,
                         padding=2,
                         output_padding=1)
        
        self.l12 = convt(in_channels=2*self.latent_channels,
                         out_channels=self.in_channels,
                         kernel_size=self.kernel_size,
                         stride=2,
                         padding=2,
                         output_padding=1)

        for layer in [self.l7, self.l8, self.l9, self.l10, self.l11, self.l12]:
            init_layer_weights(layer)
            


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
        h12 = torch.cat([x, h12], dim=1)

        return self.last_layer_multiplier(h12)
    
    


# =============================================================================
# UNet CNN architecture
# =============================================================================


class EEGUNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        
        super().__init__()
        
        
        conv = nn.Conv1d
        convt = nn.ConvTranspose1d
        
        self.activation = nn.ReLU()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = 16
        self.num_halving_layers = 6
        self.kernel_size = 5

        self.l1 = conv(in_channels=self.in_channels,
                       out_channels=2*self.latent_channels,
                       kernel_size=self.kernel_size,
                       stride=2,
                       padding=2)
        
        self.l2 = conv(in_channels=2*self.latent_channels,
                       out_channels=4*self.latent_channels,
                       kernel_size=self.kernel_size,
                       stride=2,
                       padding=2)
        
        self.l3 = conv(in_channels=4*self.latent_channels,
                       out_channels=8*self.latent_channels,
                       kernel_size=self.kernel_size,
                       stride=2,
                       padding=2)
        
        self.l4 = conv(in_channels=8*self.latent_channels,
                       out_channels=16*self.latent_channels,
                       kernel_size=self.kernel_size,
                       stride=2,
                       padding=2)
        
        self.l5 = conv(in_channels=16*self.latent_channels,
                       out_channels=32*self.latent_channels,
                       kernel_size=self.kernel_size,
                       stride=2,
                       padding=2)
        
        self.l6 = conv(in_channels=32*self.latent_channels,
                       out_channels=64*self.latent_channels,
                       kernel_size=self.kernel_size,
                       stride=2,
                       padding=2)

        for layer in [self.l1, self.l2, self.l3, self.l4, self.l5, self.l6]:
            init_layer_weights(layer)

        self.l7 = convt(in_channels=64*self.latent_channels,
                        out_channels=32*self.latent_channels,
                        kernel_size=self.kernel_size,
                        stride=2,
                        padding=2,
                        output_padding=1)
        
        self.l8 = convt(in_channels=64*self.latent_channels,
                        out_channels=16*self.latent_channels,
                        kernel_size=self.kernel_size,
                        stride=2,
                        padding=2,
                        output_padding=1)
        
        self.l9 = convt(in_channels=32*self.latent_channels,
                        out_channels=8*self.latent_channels,
                        kernel_size=self.kernel_size,
                        stride=2,
                        padding=2,
                        output_padding=1)
        
        self.l10 = convt(in_channels=16*self.latent_channels,
                         out_channels=4*self.latent_channels,
                         kernel_size=self.kernel_size,
                         stride=2,
                         padding=2,
                         output_padding=1)
        
        self.l11 = convt(in_channels=8*self.latent_channels,
                         out_channels=2*self.latent_channels,
                         kernel_size=self.kernel_size,
                         stride=2,
                         padding=2,
                         output_padding=1)
        
        self.l12 = convt(in_channels=4*self.latent_channels,
                         out_channels=self.in_channels,
                         kernel_size=self.kernel_size,
                         stride=2,
                         padding=2,
                         output_padding=1)

        for layer in [self.l7, self.l8, self.l9, self.l10, self.l11, self.l12]:
            init_layer_weights(layer)
            


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
        h12 = torch.cat([x, h12], dim=1)

        return self.last_layer_multiplier(h12)
    
    

# =============================================================================
# HalfUNet CNN architecture
# =============================================================================
    
    
class HalfUNet(nn.Module):

    def __init__(self,
                 input_dim,
                 in_channels,
                 out_channels,
                 latent_channels):
        
        super().__init__()
        
        conv = getattr(nn, f'Conv{input_dim}d')
        convt = getattr(nn, f'ConvTranspose{input_dim}d')
        
        self.activation = nn.ReLU()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = 8
        self.num_halving_layers = 6
        
        kernel_size = 5
        padding = kernel_size // 2

        self.l1 = conv(in_channels=self.in_channels,
                       out_channels=self.latent_channels,
                       kernel_size=self.kernel_size,
                       stride=2,
                       padding=2)
        
        self.l2 = conv(in_channels=self.latent_channels,
                       out_channels=2*self.latent_channels,
                       kernel_size=self.kernel_size,
                       stride=2,
                       padding=2)
        
        self.l3 = conv(in_channels=2*self.latent_channels,
                       out_channels=2*self.latent_channels,
                       kernel_size=self.kernel_size,
                       stride=2,
                       padding=2)
            
        self.l4 = convt(in_channels=2*self.latent_channels,
                        out_channels=2*self.latent_channels,
                        kernel_size=kernel_size,
                        stride=2,
                        padding=padding,
                        output_padding=1)
        
        self.l5 = convt(in_channels=2*self.latent_channels,
                        out_channels=self.latent_channels,
                        kernel_size=kernel_size,
                        stride=2,
                        padding=padding,
                        output_padding=1)
        
        self.l6 = convt(in_channels=self.latent_channels,
                        out_channels=self.latent_channels,
                        kernel_size=kernel_size,
                        stride=2,
                        padding=padding,
                        output_padding=1)


        self.last_layer_multiplier = conv(in_channels=self.latent_channels,
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
            
            assert tensor.shape[-1] == W.shape[0]
            
            tensor = torch.einsum('...i, ij -> ...j', tensor, W)
            tensor = tensor + b[(None,) * (len(tensor.shape) - 1)]
            
            if i < self.num_linear - 1:
                tensor = self.nonlinearity(tensor)
        
        return tensor

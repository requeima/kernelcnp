import torch
import torch.nn as nn
from torch.distributions import Normal,MultivariateNormal
from torch.utils.data import Dataset, DataLoader
from utils import *

class ResidualBlock(nn.Module):
    """
    Residual block for decoder
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 upsample=False,
                 activation=nn.ReLU()):
        super().__init__()

        self.activation = activation
        self.upsample = upsample
        padding = kernel_size // 2

        self.conv = DepthSeparableConv2d(in_channels, 
                                         out_channels, 
                                         kernel_size=kernel_size, 
                                         padding=padding)

        self.depthwise = nn.Conv2d(in_channels,
                                   in_channels,
                                   kernel_size=kernel_size,
                                   padding=padding,
                                   groups=in_channels)

        self.pointwise = nn.Conv2d(in_channels,
                                   out_channels,
                                   kernel_size=1)
        
        self.upsample_layer = nn.Upsample(scale_factor=2, 
                                          mode="bilinear",
                                          align_corners=False)

    def forward(self, x):
        h = self.conv(self.activation(x))
        h = self.depthwise(self.activation(x))
        h = h + x
        h = self.pointwise(h)
        if self.upsample:
            h = self.upsample_layer(h)
        return h

class MLP(nn.Module):
    """
    Base MLP module with ReLU activation
    Parameters:
    -----------
    in_channels: Int
        Number of input channels
    out_channels: Int
        Number of output channels
    h_channels: Int
        Number of hidden channels
    h_layers: Int
        Number of hidden layers
    """

    def __init__(self, 
                in_channels, 
                out_channels, 
                h_channels=64,
                h_layers=4):

        super().__init__()

        def hidden_block(h_channels):
            h = nn.Sequential(
            nn.Linear(h_channels, h_channels),
            nn.ReLU())
            return h

        # Model
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, h_channels),
            nn.ReLU(),
            *[hidden_block(h_channels) for _ in range(h_layers)],
            nn.Linear(h_channels, out_channels) 
        )

    def forward(self, x):
        return self.mlp(x)

class DepthSeparableConv2d(nn.Module):
    """
    Depthwise separable version of pytorch conv2d
    Parameters:
    -----------
    in_channels: Int
    out_channels: Int
    kernel_size: Int
    bias: Boolean
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding,
                 bias=True):

        super().__init__()

        self.depthwise = nn.Conv2d(in_channels, 
                in_channels, 
                kernel_size=kernel_size, 
                padding=padding,
                groups=in_channels, 
                bias=bias)
        self.pointwise = nn.Conv2d(in_channels,
                out_channels,
                kernel_size=1,
                bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class Decoder(nn.Module):
    """
    Resnet CNN decoder. Channels transformed using MLP. 
    Parameters:
    -----------
    in_channels: Int
        input channels from context
    h channels: Int
        number of channels in residual blocks
    out_channels Int
        output channels from decoder
    """

    def __init__(self, 
                 in_channels,
                 out_channels,
                 h_channels=128,
                 kernel_size=3,
                 n_blocks=6,
                 upsample=False):

        super().__init__()
        self.out_channels = out_channels
        self.upsample=upsample

        padding = kernel_size//2

        self.in_conv = nn.Conv2d(
            in_channels, h_channels, kernel_size=kernel_size, padding=padding)

        self.resnet = nn.Sequential(
            *[ResidualBlock(h_channels, h_channels, kernel_size=kernel_size, upsample=self.upsample)
              for _ in range(n_blocks)])

        self.out_mlp = MLP(
            in_channels=h_channels,out_channels=out_channels)
              
        
    def forward(self, x):
        # Input [batch, ..., channels]
        x = self.in_conv(x)
        x = self.resnet(x)
        #x = channels_to_last_dim(x)
        x = self.out_mlp(channels_to_last_dim(x))
        return x

class SetConv(nn.Module):
    """
    Set conv module for translating gridded to ungridded using RBF kernel
    """

    def __init__(self,
                 init_ls):

        super().__init__()
        self.init_ls = torch.nn.Parameter(torch.tensor([init_ls]))
        self.init_ls.requires_grad = True

    def _rbf(self, dists):
        kernel = torch.exp(-0.5 * dists / self.init_ls ** 2)
        kernel = kernel.view(kernel.shape[0],-1)
        return kernel

    def forward(self, wt, dists):
        wt = wt.view(wt.shape[0],-1,wt.shape[-1])
        kernel = self._rbf(dists)

        out = torch.einsum('boc,po->bpc',wt,kernel)

        return out
        
        

import numpy as np
import torch
import torch.nn as nn

from cnp.encoders import StandardEncoder, ConvEncoder
from cnp.decoders import StandardDecoder, ConvDecoder
from cnp.architectures import UNet

class ConvditionalNeuralProcess(nn.Module):
    """Conditional Neural Process Module.

    Args:

    """
    def __init__(self, encoder, decoder, covariance, add_noise):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.covariance = covariance
        self.add_noise = add_noise

    
    def forward(self, x_context, y_context, x_target):
        r = self.encoder(x_context, y_context, x_target)
        z = self.decoder(r, x_context, y_context, x_target)
        # Produce mean
        mean = z[..., 0:1]
        
        # Produce cov
        embedding = z[..., 1:]
        cov = self.covariance(embedding)
        cov_plus_noise = self.add_noise(cov, embedding)

        return mean, cov, cov_plus_noise 



class StandardGNP(ConvditionalNeuralProcess):
    def __init__(self, covariance, add_noise, use_attention=False):
        # Hard-coded options
        input_dim = 1
        output_dim = 1
        latent_dim = 128
        num_out_channels = self.output_dim + covariance.num_basis_dim + covariance.extra_cov_dim + add_noise.extra_noise_dim

        # Attention

        # Construct the standard encoder and decoder
        encoder = \
            StandardEncoder(input_dim=self.input_dim + self.output_dim,
                            latent_dim=self.latent_dim,
                            use_attention=use_attention)
        decoder = StandardDecoder(input_dim=self.input_dim + self.latent_dim,
                                  latent_dim=self.latent_dim,
                                  output_dim=self.num_out_channels)

        super().__init__(encoder, decoder, covariance, add_noise)
        self.use_attention = use_attention
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.num_out_channels = num_out_channels


class StandardAGNP(StandardGNP):
    def __init__(self, covariance, add_noise):
        super().__init__(covariance, add_noise, use_attention=True)


class ConvGNP(ConvditionalNeuralProcess):
    def __init__(self, covariance, add_noise):
        # Hard-coded options
        input_dim = 1
        output_dim = 1
        conv_architecture = UNet()
        points_per_unit = 64

        # Construct the convolutional encoder
        grid_multiplyer =  2 ** conv_architecture.num_halving_layers
        init_length_scale = 2.0 / points_per_unit
        encoder = ConvEncoder(out_channels=conv_architecture.in_channels,
                              init_length_scale=init_length_scale,
                              points_per_unit=points_per_unit,
                              grid_multiplier=grid_multiplyer
                              )
        
        # Construct the convolutional decoder
        num_out_channels = output_dim + covariance.num_basis_dim + covariance.extra_cov_dim + add_noise.extra_noise_dim
        decoder = ConvDecoder(conv_architecture=conv_architecture,
                              in_channels=conv_architecture.out_channels,
                              out_channels=num_out_channels,
                              init_length_scale=init_length_scale,
                              points_per_unit=points_per_unit,
                              grid_multiplier=grid_multiplyer)

        super().__init__(encoder, decoder, covariance, add_noise)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv_architecture = conv_architecture

import numpy as np
import torch
import torch.nn as nn

from torch.distributions import MultivariateNormal

from cnp.encoders import (
    StandardEncoder,
    ConvEncoder,
    StandardFullyConnectedTEEncoder
)

from cnp.decoders import (
    StandardDecoder,
    ConvDecoder,
    StandardFullyConnectedTEDecoder
)

from cnp.architectures import UNet



# =============================================================================
# General Gaussian Neural Process
# =============================================================================


class GaussianNeuralProcess(nn.Module):
    
    
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
    
    
    @property
    def num_params(self):
        """Number of parameters."""
    
        return np.sum([torch.tensor(param.shape).prod() \
                       for param in self.parameters()])



# =============================================================================
# Standard Gaussian Neural Process
# =============================================================================


class StandardGNP(GaussianNeuralProcess):
    
    def __init__(self, covariance, add_noise, use_attention=False):
        
        # Standard input/output dimensions and latent representation dimension
        input_dim = 1
        output_dim = 1
        latent_dim = 128
        
        # Decoder output dimension
        decoder_output_dim = output_dim +               \
                             covariance.num_basis_dim + \
                             covariance.extra_cov_dim + \
                             add_noise.extra_noise_dim

        # Construct the standard encoder
        encoder = StandardEncoder(input_dim=input_dim + output_dim,
                                  latent_dim=latent_dim,
                                  use_attention=use_attention)
        
        # Construct the standard decoder
        decoder = StandardDecoder(input_dim=input_dim + latent_dim,
                                  latent_dim=latent_dim,
                                  output_dim=decoder_output_dim)

        super().__init__(encoder=encoder,
                         decoder=decoder,
                         covariance=covariance,
                         add_noise=add_noise)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.num_out_channels = decoder_output_dim
        self.use_attention = use_attention



# =============================================================================
# Standard Attentive Gaussian Neural Process
# =============================================================================


class StandardAGNP(StandardGNP):
    
    def __init__(self, covariance, add_noise):
        
        super().__init__(covariance=covariance,
                         add_noise=add_noise,
                         use_attention=True)



# =============================================================================
# Standard Fully Connected Translation Equivariant Gaussian Neural Process
# =============================================================================
        
        
class StandardFullyConnectedTEGNP(GaussianNeuralProcess):
    
    def __init__(self, covariance, add_noise):
        
        input_dim = 1
        output_dim = 1
        rep_dim = 100
        
        embedding_dim = output_dim +               \
                        covariance.num_basis_dim + \
                        covariance.extra_cov_dim + \
                        add_noise.extra_noise_dim
        
        encoder = StandardFullyConnectedTEEncoder(input_dim=input_dim,
                                                  output_dim=output_dim,
                                                  rep_dim=rep_dim)
        
        decoder = StandardFullyConnectedTEDecoder(input_dim=input_dim,
                                                  output_dim=output_dim,
                                                  rep_dim=rep_dim,
                                                  embedding_dim=embedding_dim)
        
        super().__init__(encoder=encoder, 
                         decoder=decoder,
                         covariance=covariance,
                         add_noise=add_noise)



# =============================================================================
# Standard Convolutional Translation Equivariant Gaussian Neural Process
# =============================================================================
        

class StandardConvGNP(GaussianNeuralProcess):
    
    def __init__(self, covariance, add_noise):
        
        # Standard input/output dimensions and discretisation density
        input_dim = 1
        output_dim = 1
        points_per_unit = 64
        
        # Standard convolutional architecture
        conv_architecture = UNet()

        # Construct the convolutional encoder
        grid_multiplyer =  2 ** conv_architecture.num_halving_layers
        init_length_scale = 2.0 / points_per_unit
        
        encoder = ConvEncoder(out_channels=conv_architecture.in_channels,
                              init_length_scale=init_length_scale,
                              points_per_unit=points_per_unit,
                              grid_multiplier=grid_multiplyer)
        
        # Construct the convolutional decoder
        num_out_channels = output_dim +               \
                           covariance.num_basis_dim + \
                           covariance.extra_cov_dim + \
                           add_noise.extra_noise_dim
        
        decoder = ConvDecoder(conv_architecture=conv_architecture,
                              in_channels=conv_architecture.out_channels,
                              out_channels=num_out_channels,
                              init_length_scale=init_length_scale,
                              points_per_unit=points_per_unit,
                              grid_multiplier=grid_multiplyer)

        super().__init__(encoder=encoder,
                         decoder=decoder,
                         covariance=covariance,
                         add_noise=add_noise)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv_architecture = conv_architecture

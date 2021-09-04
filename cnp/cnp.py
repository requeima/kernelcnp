import numpy as np
import torch
import torch.nn as nn

try:
    import lab.torch as B
except ModuleNotFoundError:
    pass

from torch.distributions import MultivariateNormal

from cnp.encoders import (
    StandardEncoder,
    ConvEncoder,
    ConvPDEncoder,
)

from cnp.decoders import (
    StandardDecoder,
    ConvDecoder,
    ConvPDDecoder,
)

from cnp.architectures import StandardDepthwiseSeparableCNN, UNet, build_dws_net



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

    
    def forward(self, x_context, y_context, x_target, **kwargs):
        
        r = self.encoder(x_context, y_context, x_target, **kwargs)
        z = self.decoder(r, x_context, y_context, x_target, **kwargs)
        
        # Produce mean
        mean = z[..., 0:1]
        
        # Produce cov
        embedding = z[..., 1:]
        cov = self.covariance(embedding)
        cov_plus_noise = self.add_noise(cov, embedding)
        
        return mean, cov, cov_plus_noise

    
    def loss(self, x_context, y_context, x_target, y_target):

        y_mean, _, y_cov = self.forward(x_context, y_context, x_target)

        y_mean = y_mean.double()
        y_cov = y_cov.double()
        y_target = y_target.double()

        jitter = 1e-3 * torch.eye(y_cov.shape[-1], device=y_cov.device).double()
        y_cov = y_cov + jitter[None, :, :]
        
        dist = MultivariateNormal(loc=y_mean[:, :, 0],
                                  covariance_matrix=y_cov)
        nll = - torch.mean(dist.log_prob(y_target[:, :, 0]))

        return nll.float()


    def mean_and_marginals(self, x_context, y_context, x_target):
        mean, cov, cov_plus_noise = self.forward(x_context, y_context, x_target)
        var = torch.diagonal(cov, dim1=-2, dim2=-2)
        var_plus_noise = torch.diagonal(cov_plus_noise, dim1=-2, dim2=-2)
        return mean, var, var_plus_noise


    @property
    def num_params(self):
        """Number of parameters."""
        return np.sum([torch.tensor(param.shape).prod() \
                       for param in self.parameters()])


# =============================================================================
# Standard Gaussian Neural Process
# =============================================================================


class StandardGNP(GaussianNeuralProcess):
    
    def __init__(self, input_dim, covariance, add_noise, use_attention=False):
        
        # Standard input/output dimensions and latent representation dimension
        output_dim = 1
        latent_dim = 128
        num_layers = 6
        
        # Decoder output dimension
        decoder_output_dim = output_dim +               \
                             covariance.num_basis_dim + \
                             covariance.extra_cov_dim + \
                             add_noise.extra_noise_dim

        # Construct the standard encoder
        encoder = StandardEncoder(input_dim=input_dim,
                                  latent_dim=latent_dim,
                                  use_attention=use_attention,
                                  num_layers=num_layers)
        
        # Construct the standard decoder
        decoder = StandardDecoder(input_dim=input_dim,
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
    
    def __init__(self, input_dim, covariance, add_noise):
        
        super().__init__(input_dim=input_dim,
                         covariance=covariance,
                         add_noise=add_noise,
                         use_attention=True)
   
         
# =============================================================================
# Standard UNet Convolutional Gaussian Neural Process with Noise Channel
# =============================================================================
        

class StandardConvGNP(GaussianNeuralProcess):
    
    def __init__(self, input_dim, covariance, add_noise):
        
        # Standard input/output dimensions and discretisation density
        output_dim = 1
        points_per_unit = 64 if input_dim == 1 else 32

        conv_channels = 8
        conv_in_channels = conv_channels
        conv_out_channels = 8
        
        # Standard convolutional architecture
        conv_architecture = UNet(input_dim=input_dim,
                                 in_channels=conv_in_channels,
                                 out_channels=conv_out_channels)

        # Construct the convolutional encoder
        grid_multiplyer =  2 ** conv_architecture.num_halving_layers
        init_length_scale = 2.0 / points_per_unit
        grid_margin = 0.2
        
        encoder = ConvEncoder(input_dim=input_dim,
                              out_channels=conv_channels,
                              init_length_scale=init_length_scale,
                              points_per_unit=points_per_unit,
                              grid_multiplier=grid_multiplyer,
                              grid_margin=grid_margin)
        
        # Construct the convolutional decoder
        decoder_out_channels = output_dim +               \
                               covariance.num_basis_dim + \
                               covariance.extra_cov_dim + \
                               add_noise.extra_noise_dim
        
        decoder = ConvDecoder(input_dim=input_dim,
                              conv_architecture=conv_architecture,
                              conv_out_channels=conv_architecture.out_channels,
                              out_channels=decoder_out_channels,
                              init_length_scale=init_length_scale,
                              points_per_unit=points_per_unit,
                              grid_multiplier=grid_multiplyer,
                              grid_margin=grid_margin)


        super().__init__(encoder=encoder,
                         decoder=decoder,
                         covariance=covariance,
                         add_noise=add_noise)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv_architecture = conv_architecture

    
# =============================================================================
# Standard Fully Convolutional GNP (AABI model)
# =============================================================================


class FullConvGNP(nn.Module):
    def __init__(
        self,
        points_per_unit=64,
        num_channels=64,
        unet=True,
        receptive_field=6,
    ):
        nn.Module.__init__(self)

        points_per_unit_mean = points_per_unit
        points_per_unit_kernel = 30

        num_channels_mean = num_channels
        num_channels_kernel = num_channels // 2

        self.log_sigma = nn.Parameter(B.log(torch.tensor(0.1)), requires_grad=True)

        input_dim = 1
        margin = 0.1

        # Build architectures:
        if unet:
            # Reduce number of channels.
            num_channels_mean = 8
            num_channels_kernel = 8

            self.mean_arch = UNet(
                input_dim=input_dim,
                in_channels=num_channels_mean,
                out_channels=num_channels_mean,
            )
            mean_conv_out_channels = self.mean_arch.out_channels
            mean_multiplier = 2 ** self.mean_arch.num_halving_layers
            self.kernel_arch = UNet(
                input_dim=2 * input_dim,
                in_channels=num_channels_kernel,
                out_channels=num_channels_kernel,
            )
            kernel_conv_out_channels = self.kernel_arch.out_channels
            kernel_multiplier = 2 ** self.kernel_arch.num_halving_layers
        else:
            self.mean_arch = build_dws_net(
                receptive_field=receptive_field,
                points_per_unit=points_per_unit_mean,
                num_in_channels=num_channels_mean,
                num_channels=num_channels_mean,
                num_out_channels=1,
                dimensionality=input_dim
            )
            mean_conv_out_channels = 1
            mean_multiplier = 1
            self.kernel_arch = build_dws_net(
                receptive_field=receptive_field,
                points_per_unit=points_per_unit_kernel,
                num_in_channels=num_channels_kernel,
                num_channels=num_channels_kernel,
                num_out_channels=1,
                dimensionality=2 * input_dim
            )
            kernel_conv_out_channels = 1
            kernel_multiplier = 1

        # Build encoders:
        self.mean_encoder = ConvEncoder(
            input_dim=input_dim,
            out_channels=num_channels_mean,
            init_length_scale=2 / points_per_unit_mean,
            points_per_unit=points_per_unit_mean,
            grid_multiplier=mean_multiplier,
            grid_margin=margin,
        )
        self.kernel_encoder = ConvPDEncoder(
            out_channels=num_channels_kernel,
            points_per_unit=points_per_unit_kernel,
            grid_multiplier=kernel_multiplier,
            grid_margin=margin
        )

        # Build decoders:
        self.mean_decoder = ConvDecoder(
            input_dim=input_dim,
            conv_architecture=self.mean_arch,
            conv_out_channels=mean_conv_out_channels,
            out_channels=1,
            init_length_scale=2 / points_per_unit_mean,
            points_per_unit=points_per_unit_mean,
            grid_multiplier=kernel_multiplier,
            grid_margin=margin,
        )
        self.kernel_decoder = ConvPDDecoder(
            points_per_unit=points_per_unit_kernel,
        )

    def forward(self, x_context, y_context, x_target):
        # Run mean.
        r = self.mean_encoder(x_context, y_context, x_target)
        mean = self.mean_decoder(r, x_context, y_context, x_target)

        # Run kernel.
        xz, z = self.kernel_encoder(x_context, y_context, x_target)
        z = self.kernel_arch(z)
        cov = self.kernel_decoder(xz, z, x_target)[1]
        cov = cov[:, 0, ...]  # Suppress the channels dimension.

        # Add noise to the kernel.
        with B.device(str(cov.device)):
            eye = B.eye(B.dtype(cov), B.shape(cov)[-1])
            cov_noisy = cov + B.exp(self.log_sigma) * eye[None, ...]

        return mean, cov, cov_noisy

    @property
    def num_params(self):
        return sum([int(np.prod(p.shape)) for p in self.parameters()])

    def loss(self, x_context, y_context, x_target, y_target):
        return GaussianNeuralProcess.loss(
            self,
            x_context,
            y_context,
            x_target,
            y_target
        )

    def mean_and_marginals(self, x_context, y_context, x_target):
        mean, cov, noisy_cov = self.forward(x_context, y_context, x_target)
        return mean, B.diag_extract(cov), B.diag_extract(noisy_cov)

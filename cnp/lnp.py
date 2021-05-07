import numpy as np
import torch
import torch.nn as nn

from torch.distributions import MultivariateNormal

from cnp.encoders import (
    StandardANPEncoder,
    StandardConvNPEncoder
)

from cnp.decoders import (
    StandardDecoder,
    ConvDecoder
)

from cnp.architectures import (
    UNet,
    HalfUNet,
    StandardDepthwiseSeparableCNN
)



# =============================================================================
# General Latent Neural Process
# =============================================================================


class LatentNeuralProcess(nn.Module):
    
    
    def __init__(self, encoder, decoder, add_noise, num_samples):
        
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.add_noise = add_noise
        self.num_samples = num_samples

    
    def forward(self, x_context, y_context, x_target, num_samples=None):
        
        num_samples = self.num_samples if num_samples is None else num_samples
        
        # Pass context set and target inputs through the encoder to obtain
        # the encoder output, as expected by encoder.sample
        encoder_forward_output = self.encoder(x_context, y_context, x_target)
        
        means = []
        noise_vars = []
        
        for i in range(num_samples):
            
            r = self.encoder.sample(encoder_forward_output)
            mean = self.decoder(r, x_context, y_context, x_target)
            
            zeros = torch.zeros(size=(mean.shape[0],
                                      mean.shape[1],
                                      mean.shape[1])).to(mean.device)
            
#             noise = torch.eye(size=(mean.shape[1],)).to(mean.device)
#             noise = 1e-2 * noise[None, :, :].repeat(mean.shape[0], 1, 1)
            
            means.append(mean)
#             noise_vars.append(noise)
            noise_vars.append(self.add_noise(zeros, None))
            
        means = torch.stack(means, dim=0)
        noise_vars = torch.stack(noise_vars, dim=0)
        
        return means, noise_vars
    
    
    def loss(self, x_context, y_context, x_target, y_target, num_samples=None):
        
        B = y_target.shape[0]
        
        num_samples = self.num_samples if num_samples is None else num_samples
        
        # Compute mean and variance tensors, each of shape (S, B, N, D)
        means, noise_vars = self.forward(x_context,
                                         y_context,
                                         x_target,
                                         num_samples=num_samples)
        
        means = means[:, :, :, 0]
        idx = torch.arange(noise_vars.shape[2])
        noise_vars = noise_vars[:, :, idx, idx]
        
        logprobs = []
        
        for mean, noise_var in zip(means, noise_vars):
            
            distribution = torch.distributions.Normal(loc=mean,
                                                      scale=noise_var ** 0.5)
            logprob = torch.sum(distribution.log_prob(y_target[:, :, 0]), axis=-1)
            
            logprobs.append(logprob)
            
        logprobs = torch.stack(logprobs, axis=1)
        logprob = 0
        
        for i, batch_logprobs in enumerate(logprobs):
            
            max_batch_logprob = torch.max(batch_logprobs)
        
            batch_logprobs = batch_logprobs - max_batch_logprob
            
            batch_mix_logprob = torch.log(torch.mean(torch.exp(batch_logprobs)))
            batch_mix_logprob = batch_mix_logprob + max_batch_logprob
            
            logprob = logprob + batch_mix_logprob
        
        return - logprob / B
    

    def mean_and_marginals(self, x_context, y_context, x_target):
        raise NotImplementedError


    @property
    def num_params(self):
        """Number of parameters."""
    
        return np.sum([torch.tensor(param.shape).prod() \
                       for param in self.parameters()])



# =============================================================================
# Attentive Latent Neural Process
# =============================================================================


class StandardANP(LatentNeuralProcess):
    
    
    def __init__(self, input_dim, add_noise, num_samples):
        
        # Standard input/output dim and latent representation dim
        # latent_dim is common to stochastic and deterministic paths, and
        # these are concatenated, producing a (2 * latent_dim) representation
        output_dim = 1
        latent_dim = 128
        
        # Decoder output dimension
        decoder_output_dim = output_dim + add_noise.extra_noise_dim

        # Construct the standard encoder
        encoder = StandardANPEncoder(input_dim=input_dim,
                                     latent_dim=latent_dim)
        
        # Construct the standard decoder
        decoder = StandardDecoder(input_dim=input_dim,
                                  latent_dim=latent_dim,
                                  output_dim=decoder_output_dim)

        super().__init__(encoder=encoder,
                         decoder=decoder,
                         add_noise=add_noise,
                         num_samples=num_samples)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        
        
        
        

# =============================================================================
# Convolutional Latent Neural Process
# =============================================================================
        
        
class StandardConvNP(LatentNeuralProcess):
    
    def __init__(self, input_dim, add_noise, num_samples):
        
        # Dimension of output is 1 for scalar outputs -- do not change
        output_dim = 1
        
        # Num channels of input passed to encoder CNN
        encoder_conv_input_channels = 16
        
        # Num channels of latent function
        # Outputted by encoder, expected by decoder
        latent_function_channels = 16
        
        # Num channels of output of decoder CNN
        decoder_conv_output_channels = 16
        
        # Num channels of output of decoder
        decoder_out_channels = 1
        
#         # Encoder convolutional architecture
#         encoder_conv = HalfUNet(input_dim=input_dim,
#                                 in_channels=encoder_conv_input_channels,
#                                 out_channels=2*latent_function_channels)
        
        # Standard convolutional architecture
        encoder_conv = UNet(input_dim=input_dim,
                            in_channels=encoder_conv_input_channels,
                            out_channels=2*latent_function_channels)
        
        
#         # Encoder convolutional architecture
#         decoder_conv = HalfUNet(input_dim=input_dim,
#                                 in_channels=latent_function_channels,
#                                 out_channels=decoder_conv_output_channels)
        
        # Standard convolutional architecture
        decoder_conv = UNet(input_dim=input_dim,
                            in_channels=latent_function_channels,
                            out_channels=decoder_conv_output_channels)

        # Construct the convolutional encoder
        grid_multiplier =  2 ** encoder_conv.num_halving_layers
        points_per_unit = 32
        init_length_scale = 8.0 / points_per_unit
        grid_margin = 0.2
        
        encoder = StandardConvNPEncoder(input_dim=input_dim,
                                        conv_architecture=encoder_conv,
                                        init_length_scale=init_length_scale, 
                                        points_per_unit=points_per_unit, 
                                        grid_multiplier=grid_multiplier,
                                        grid_margin=grid_margin)
        
        decoder = ConvDecoder(input_dim=input_dim,
                              conv_architecture=decoder_conv,
                              conv_out_channels=decoder_conv.out_channels,
                              out_channels=decoder_out_channels,
                              init_length_scale=init_length_scale,
                              points_per_unit=points_per_unit,
                              grid_multiplier=grid_multiplier,
                              grid_margin=grid_margin)


        super().__init__(encoder=encoder,
                         decoder=decoder,
                         add_noise=add_noise,
                         num_samples=num_samples)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
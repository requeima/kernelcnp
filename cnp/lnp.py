import numpy as np
import torch
import torch.nn as nn

from torch.distributions import MultivariateNormal

from cnp.encoders import (
    StandardANPEncoder
)

from cnp.decoders import (
    StandardDecoder
)

from cnp.architectures import UNet



# =============================================================================
# General Latent Neural Process
# =============================================================================


class LatentNeuralProcess(nn.Module):
    
    
    def __init__(self, encoder, decoder, add_noise):
        
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.add_noise = add_noise

    
    def forward(self, x_context, y_context, x_target, num_samples):
        
        # Pass context set and target inputs through the encoder to obtain
        # the distribution of the representation
        r_dist = self.encoder(x_context, y_context, x_target)
        
        means = []
        noise_vars = []
        
        for i in range(num_samples):
            
            r = r_dist.rsample()
            z = self.decoder(r, x_context, y_context, x_target)
            
            means.append(z[..., :1])
            noise_vars.append(self.add_noise(torch.zeros(size=(z.shape[0],
                                                               z.shape[1],
                                                               z.shape[1])),
                                             z[..., 1:]))
            
        means = torch.stack(means, dim=0)
        noise_vars = torch.stack(noise_vars, dim=0)
        
        return means, noise_vars

    
    def loss(self, x_context, y_context, x_target, y_target, num_samples):
        
        S = num_samples
        B = y_target.shape[0]
        N = y_target.shape[1]
        D = y_target.shape[2]
        
        # Compute mean and variance tensors, each of shape (S, B, N, D)
        means, noise_vars = self.forward(x_context,
                                         y_context,
                                         x_target,
                                         num_samples)
        
        means = means[:, :, :, 0]
        idx = torch.arange(noise_vars.shape[2])
        noise_vars = noise_vars[:, :, idx, idx]
        
        # Normal with batch shape (S, B, N, D), event shape ()
        normal = torch.distributions.Normal(loc=means, scale=noise_vars ** 0.5)
        
        # Normal with batch shape (S), event shape (B, N, D)
        normal = torch.distributions.Independent(normal, 2)
        
        # Categorical to use for mixing components
        logits = torch.ones(size=(num_samples,))
        categorical = torch.distributions.Categorical(logits=logits)
        
        mixture = torch.distributions.MixtureSameFamily(categorical, normal)
        
        logprob = mixture.log_prob(y_target[:, :, 0]) / (B * N * D)
        
        return - logprob
    

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
    
    
    def __init__(self, input_dim, add_noise):
        
        # Standard input/output dim and latent representation dim
        # latent_dim is common to stochastic and deterministic paths, and
        # these are concatenated, producing a (2 * latent_dim) representation
        output_dim = 1
        latent_dim = 128
        
        # Decoder output dimension
        decoder_output_dim = output_dim + add_noise.extra_noise_dim

        # Construct the standard encoder
        encoder = StandardANPEncoder(input_dim=input_dim + output_dim,
                                     latent_dim=latent_dim)
        
        # Construct the standard decoder
        decoder = StandardDecoder(input_dim=input_dim+latent_dim,
                                  latent_dim=latent_dim,
                                  output_dim=decoder_output_dim)

        super().__init__(encoder=encoder,
                         decoder=decoder,
                         add_noise=add_noise)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
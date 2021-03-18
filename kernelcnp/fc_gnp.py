import torch

from torch import nn
from torch.distributions import MultivariateNormal

import numpy as np


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
            tensor = tensor + b[None, None, :]
            
            if i < self.num_linear - 1:
                tensor = self.nonlinearity(tensor)
        
        return tensor


    
# =============================================================================
# Gaussian Neural Process
# =============================================================================
    
    
class GaussianNeuralProcess(nn.Module):
    
    def __init__(self,
                 encoder,
                 decoder,
                 log_noise):
        
        
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        
        self.log_noise = torch.tensor(log_noise)
        self.log_noise = nn.Parameter(self.log_noise)
        
        
    def forward(self,
                ctx_in,
                ctx_out,
                trg_in):
        
        D = ctx_out.shape[-1]
        
        ctx = torch.cat([ctx_in, ctx_out], dim=-1)
        
        theta = self.encoder(ctx)
        theta = torch.mean(theta, dim=1)[:, None, :]
        theta = theta.repeat(1, trg_in.shape[1], 1)
        
        tensor = torch.cat([trg_in, theta], dim=-1)
        tensor = self.decoder(tensor)
        
        mean = tensor[:, :, :1]
        log_noise = tensor[:, :, D:(2*D)]
        
        cov_root = tensor[:, :, (2*D):]
        cov = torch.einsum('bni, bmi -> bnm', cov_root, cov_root) / cov_root.shape[-1]
        
        cov_plus_noise = cov + torch.exp(self.log_noise) * torch.eye(cov.shape[1])[None, ...]
                
        return mean, cov, cov_plus_noise
    
    
    def _loss(self,
              ctx_in,
              ctx_out,
              trg_in,
              trg_out):
        
        mean, cov, cov_plus_noise = self.forward(ctx_in, ctx_out, trg_in)
        
        pred_dist = MultivariateNormal(loc=mean[:, :, 0],
                                       covariance_matrix=cov_plus_noise)
        
        log_prob = pred_dist.log_prob(trg_out[:, :, 0])
        log_prob = torch.mean(log_prob)
        
        return - log_prob
    
    
    def loss(self,
             inputs,
             outputs,
             num_samples):
        
        loss = 0
        
        for i in range(num_samples):
            
            N = np.random.choice(np.arange(1, inputs.shape[1]))
            
            ctx_in = inputs[:, :N]
            ctx_out = outputs[:, :N]
            trg_in = inputs[:, N:]
            trg_out = outputs[:, N:]
            
            loss = loss + self._loss(ctx_in,
                                     ctx_out,
                                     trg_in,
                                     trg_out)
        
        loss = loss / (num_samples * inputs.shape[1])
        
        return loss


    
# =============================================================================
# Translation Equivariant Gaussian Neural Process
# =============================================================================

    
class TranslationEquivariantGaussianNeuralProcess(GaussianNeuralProcess):
    
    def __init__(self,
                 encoder,
                 decoder,
                 log_noise):
        
        
        super().__init__(encoder=encoder,
                         decoder=decoder,
                         log_noise=log_noise)
        
        
    def forward(self,
                ctx_in,
                ctx_out,
                trg_in):
        
        D = ctx_out.shape[-1]
        
        # Context and target inputs
        ctx_in = ctx_in[:, None, :, :]
        trg_in = trg_in[:, :, None, :]
        
        diff = ctx_in - trg_in
        
        ctx_out = ctx_out[:, None, :, :]
        ctx_out = ctx_out.repeat(1, diff.shape[1], 1, 1)
        
        ctx = torch.cat([diff, ctx_out], dim=-1)
        
        theta = self.encoder(ctx)
        theta = torch.mean(theta, dim=2) # (B, T, R)
        
        tensor = torch.cat([trg_in[:, :, 0, :], theta], dim=-1)
        tensor = self.decoder(tensor)
        
        mean = tensor[:, :, :1]
        cov_root = tensor[:, :, 1:]
        cov = torch.einsum('bni, bmi -> bnm', cov_root, cov_root) / cov_root.shape[-1]
        
        diag_noise = torch.exp(self.log_noise) * torch.eye(cov.shape[1])[None, :, :]
        cov_plus_noise = cov + diag_noise
                
        return mean, cov, cov_plus_noise
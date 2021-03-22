from .gnp import GNP

import torch

__all__ = ['TEGNP']


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
# Translation Equivariant GNP
# =============================================================================
        
        
        
class TEGNP(GNP):

    def __init__(self,
                 latent_dim,
                 cov,
                 noise
                 use_attention):

        super().__init__()
        
        input_dim = 1
        output_dim = 1
        
        encoder_input_dim = input_dim + 2 * output_dim
        encoder_output_dim = latent_dim
        
        decoder_input_dim = input_dim + latent_dim
        decoder_output_dim = output_dim            + \
                             cov.num_basis_dim     + \
                             cov.extra_cov_dim     + \
                             noise.extra_noise_dim
        
        hidden_dims = [latent_dim, latent_dim, latent_dim]
        nonlinearity = 'ReLU'
        
        self.encoder = FullyConnectedNetwork(input_dim=encoder_input_dim,
                                             output_dim=encoder_output_dim,
                                             hidden_dims=hidden_dims,
                                             nonlinearity=nonlinearity)
        
        self.decoder = FullyConnectedNetwork(input_dim=decoder_input_dim,
                                             output_dim=decoder_output_dim,
                                             hidden_dims=hidden_dims,
                                             nonlinearity=nonlinearity)
        self.cov = cov
        self.noise = noise
        
        
    def encode(x_ctx, y_ctx):
        
        assert len(x_ctx.shape) == 3
        assert len(y_ctx.shape) == 2
        
        # Elementwise differences of context inputs
        x_diff = x_ctx[:, None, :, :] - x_ctx[:, :, None, :]
        
        # Tile context outputs to concatenate with input differences
        y_ctx_tile1 = y_ctx[:, None, :, :].repeat(1, x_diff.shape[1], 1, 1)
        y_ctx_tile2 = y_ctx[:, :, None, :].repeat(1, 1, x_diff.shape[1], 1)
        
        # Concatenate input differences and outputs, to obtain complete context
        ctx = torch.cat([x_ctx_diff, y_ctx_tile1, y_ctx_tile2], dim=-1)
        ctx = ctx.view(ctx.shape[0], ctx.shape[1] ** 2, ctx.shape[2])
        
        # Latent representation of context set -- resulting r has shape (B, R)
        r = self.encoder(ctx)
        r = torch.mean(r, dim=(1, 2))
        
        return r
    
    
    def decode(self, x_ctx, x_trg, r):
        
        # Elementwise differences of context and target inputs
        x_diff = x_ctx[:, :, None :] - x_trg[:, None, :, :]
        
        x_diff = x_diff.view(x_diff[0], -1, x_diff[3])
        
        
    def forward(self, x_ctx, y_ctx, x_trg, noiseless=False):
        
        r = self.encode(x_ctx, y_ctx)
        z = self.decode(x_ctx, x_ctx, r)
        
        # Produce mean and covariance
        mean = z[..., :1]
        
        cov = self.cov(z[..., 1:])
        cov_plus_noise = self.noise(cov, z[..., 1:])
        
        if noiseless:
            return mean, cov
        
        else:
            return mean, cov_plus_noise
        

    @property
    def num_params(self):
        return np.sum([torch.tensor(param.shape).prod()
                       for param in self.parameters()])
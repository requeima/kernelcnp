from .gnp import GNP
import torch


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
# Fully Connected DeepSet with mean aggregation
# =============================================================================
    
    
class FullyConnectedDeepSet(nn.Module):
    
    def __init__(self,
                 element_network,
                 aggregation_dims,
                 aggregate_network):
        
        assert type(aggregation_dims) == list
        
        self.element_network = element_network
        self.aggregation_dims = aggregation_dims
        self.aggregate_network = aggregate_network
        
    
    def forward(self, tensor):
        
        tensor = self.element_network(tensor)
        tensor = torch.mean(tensor, dim=aggregation_dims)
        tensor = self.aggregate_network(tensor)
        
        return tensor


    
# =============================================================================
# Fully Connected Translation Equivariant Encoder
# =============================================================================


class FullyConnectedTEEncoder(nn.Module):
    
    def __init__(self, deepset):
        
        super().__init__()
        
        self.deepset = deepset
    
    
    def forward(self, x_ctx, y_ctx, x_trg):
        
        assert len(x_ctx.shape) == 3
        assert len(y_ctx.shape) == 2
        assert len(x_trg.shape) == 3
        
        # Compute context input pairwise differences
        x_diff = x_ctx[:, None, :, :] - x_ctx[:, :, None, :]
        
        # Tile context outputs to concatenate with input differences
        y_ctx_tile1 = y_ctx[:, None, :, :].repeat(1, x_diff.shape[1], 1, 1)
        y_ctx_tile2 = y_ctx[:, :, None, :].repeat(1, 1, x_diff.shape[1], 1)
        
        # Concatenate input differences and outputs, to obtain complete context
        ctx = torch.cat([x_ctx_diff, y_ctx_tile1, y_ctx_tile2], dim=-1)
        
        # Latent representation of context set -- resulting r has shape (B, R)
        r = self.deepset(ctx)
        
        return r
     
        

# =============================================================================
# Fully Connected Translation Equivariant Decoder
# =============================================================================


class FullyConnectedTEDecoder(nn.Module):
    
    def __init__(self, deepset):
        
        super().__init__()
        
        self.deepset = deepset
    
    
    def forward(self, r, x_ctx, y_ctx, x_trg):
        
        # Compute context input pairwise differences
        x_diff = x_ctx[:, :, None, :] - x_ctx[:, None, :, :]
        
        # Tile representation vector r
        r = r[:, None, None, :].repeat(1, x_diff.shape[1], x_diff.shape[2], 1)
        
        # Concatenate input differences with tiled r's
        z = self.deepset(torch.cat([x_diff, r], dim=-1))
        
        return z
    

# =============================================================================
# Standard Translation Equivariant Encoder
# =============================================================================


class StandardFullyConnectedTEEncoder(FullyConnectedTEEncoder):
    
    def __init__(self,
                 input_dim,
                 output_dim,
                 rep_dim):
        
        # Input dimension of encoder (Din + 2 * Dout)
        encoder_input_dim = input_dim + 2 * output_dim
        
        # Sizes of hidden layers and nonlinearity type
        # Used for both elementwise and aggregate networks
        hidden_dims = [128, 128]
        nonlinearity = 'ReLU'
        
        # Element network -- in (B, C, C, Din + 2 * Dout), out (B, C, C, R)
        element_network = FullyConnectedNetwork(input_dim=encoder_input_dim,
                                                output_dim=rep_dim,
                                                hidden_dims=hidden_dims,
                                                nonlinearity=nonlinearity)
        
        # Dimensions to mean over -- in (B, C, C, R), out (B, R)
        aggregation_dims = [1, 2]
        
        # Aggregate network -- in (B, R), out (B, R)
        aggregate_network = FullyConnectedNetwork(input_dim=rep_dim,
                                                  output_dim=rep_dim,
                                                  hidden_dims=hidden_dims,
                                                  nonlinearity=nonlinearity)
        
        # Deepset architecture
        deepset = FullyConnectedDeepSet(element_network,
                                        aggregation_dims,
                                        aggregate_network)
        
        super().__init__(deepset=deepset)
        
        

# =============================================================================
# Standard Translation Equivariant Decoder
# =============================================================================


class StandardFullyConnectedTEDecoder(FullyConnectedTEEncoder):
    
    def __init__(self,
                 input_dim,
                 rep_dim,
                 embedding_dim):
        
        # Input dimension of encoder (Din + R)
        decoder_input_dim = input_dim + rep_dim
        
        # Sizes of hidden layers and nonlinearity type
        # Used for both elementwise and aggregate networks
        hidden_dims = [128, 128]
        nonlinearity = 'ReLU'
        
        # Element network -- in (B, C, T, Din + R), out (B, C, T, R)
        element_network = FullyConnectedNetwork(input_dim=decoder_input_dim,
                                                output_dim=rep_dim,
                                                hidden_dims=hidden_dims,
                                                nonlinearity=nonlinearity)
        
        # Dimensions to mean over -- in (B, C, T, R), out (B, T, R)
        aggregation_dims = [1]
        
        # Aggregate network -- in (B, T, R), out (B, T, E)
        aggregate_network = FullyConnectedNetwork(input_dim=rep_dim,
                                                  output_dim=embedding_dim,
                                                  hidden_dims=hidden_dims,
                                                  nonlinearity=nonlinearity)
        
        # Deepset architecture
        deepset = FullyConnectedDeepSet(element_network,
                                        aggregation_dims,
                                        aggregate_network)
        
        super().__init__(deepset=deepset)

        
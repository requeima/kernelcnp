import numpy as np
import torch
import torch.nn as nn

from cnp.utils import init_sequential_weights
from cnp.architectures import BatchMLP, FullyConnectedNetwork


class MeanPooling(nn.Module):
    """
    Helper class for performing mean pooling in CNPs.

    Args:
        pooling_dim (int, optional): Dimension to pool over.
    """

    def __init__(self, pooling_dim):
        
        super(MeanPooling, self).__init__()
        
        self.pooling_dim = pooling_dim
        

    def forward(self, h, *args, **kwargs):
        """
        Performs mean pooling.

        Args:
            h (tensor): Tensor to pool over.
        """
        return torch.mean(h, dim=self.pooling_dim, keepdim=True)


class DotProdAttention(nn.Module):
    """
    Args:
        embedding_dim (int):
        values_dim (int):
    """

    def __init__(self):
        
        super().__init__()
        

    def forward(self, keys, queries, values):
        """
        Forward pass of dot-product attention. Given keys K, queries Q and
        values V, this layer computes
        
            DotProdAttention(K, Q, V) = V * softmax(Q K.T / D1^0.5)
            
        where the following dimensions are assumed
        
            Q : (..., C, D1)
            K : (..., T, D1)
            V : (..., T, D2)

        Args:
            keys     (tensor): Keys K, shape (..., C, D1)
            queries  (tensor): Queries Q, shape (..., T, D1)
            values   (tensor): Queries V, shape (..., C, D2)

        Returns:
            attended (tensor): Attended values A, shape (B, T, D2)
        """
        
        D = keys.shape[-1]
        
        # Compute dot product between keys and queries, normalise by D^0.5
        dot = torch.einsum('...cd, ...td -> ...ct', keys, queries) / D ** 0.5
        
        # Apply softmax to get attention weights
        attn_weights = nn.functional.softmax(dot, dim=-1)
        
        # Weight values by attention, to get attended values
        attended = torch.einsum('...ct, ...cd -> ...td', attn_weights, values)
        
        return attended
    


class MultiHeadAttention(nn.Module):

    def __init__(self,
                 key_query_input_dim,
                 key_query_embedding_dim,
                 value_input_dim,
                 value_embedding_dim,
                 output_embedding_dim,
                 num_heads):
        
        super().__init__()
        
        self.key_query_input_dim = key_query_input_dim
        self.key_query_embedding_dim = key_query_embedding_dim
        
        self.value_input_dim = value_input_dim
        self.value_embedding_dim = value_embedding_dim
        
        self.output_embedding_dim = output_embedding_dim
        self.num_heads = num_heads
        
        # Initialise linear layers
        self.key_linear = nn.Linear(self.key_query_input_dim,
                                    self.key_query_embedding_dim * self.num_heads,
                                    bias=False)

        self.query_linear = nn.Linear(self.key_query_input_dim,
                                      self.key_query_embedding_dim * self.num_heads,
                                      bias=False)
        
        self.value_linear = nn.Linear(self.value_input_dim,
                                      self.value_embedding_dim * self.num_heads,
                                      bias=False)
        
        self.attention = DotProdAttention()
        
        self.head_mixer = nn.Linear(self.value_embedding_dim * self.num_heads,
                                    self.output_embedding_dim,
                                    bias=True)
        

    def forward(self, keys, queries, values):
        """
        Forward pass of dot-product attention. Given keys K, queries Q and
        values V, this layer computes
        
            MultiHeadAttention(K, Q, V)
            
        where the following dimensions are assumed
        
            Q : (B, C, D1)
            K : (B, T, D1)
            V : (B, T, D2)
        """
        
        B = queries.shape[0]
        C = queries.shape[1]
        T = values.shape[1]
        D2 = values.shape[2]
        
        key_embeddings = self.key_linear(keys)
        key_embeddings = torch.reshape(key_embeddings,
                                       (B, self.num_heads, self.key_query_embedding_dim, D1))
        
        query_embeddings = self.query_linear(queries)
        query_embeddings = torch.reshape(query_embeddings,
                                         (B, self.num_heads, self.key_query_embedding_dim, D1))
        
        value_embeddings = self.query_linear(values)
        value_embeddings = torch.reshape(value_embeddings,
                                         (B, self.num_heads, self.value_embedding_dim, D2))
        
        # (B, H, T, R)
        attended = self.attention(key_embeddings, query_embeddings, value_embeddings)
        attended = torch.reshape(attended, (B, T, -1))
        
        multi_head_attended = self.head_mixer(attended)
        
        return multi_head_attended


        
# =============================================================================
# Fully Connected DeepSet with mean aggregation
# =============================================================================
    
    
class FullyConnectedDeepSet(nn.Module):
    
    def __init__(self,
                 element_network,
                 aggregation_dims,
                 aggregate_network):

        super().__init__()
        
        assert type(aggregation_dims) == list
        
        self.element_network = element_network
        self.aggregation_dims = aggregation_dims
        self.aggregate_network = aggregate_network
        
    
    def forward(self, tensor):
        
        tensor = self.element_network(tensor)
        tensor = torch.mean(tensor, dim=self.aggregation_dims)
        tensor = self.aggregate_network(tensor)
        
        return tensor
    
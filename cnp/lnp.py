import numpy as np
import torch
import torch.nn as nn

from torch.distributions import MultivariateNormal

from cnp.encoders import (
    StandardEncoder,
    StandardMeanTEEncoder,
    ConvEncoder,
    StandardFullyConnectedTEEncoder
)

from cnp.decoders import (
    StandardDecoder,
    StandardMeanTEDecoder,
    ConvDecoder,
    StandardFullyConnectedTEDecoder
)

from cnp.architectures import UNet



# =============================================================================
# General Latent Neural Process
# =============================================================================


class LatentNeuralProcess(nn.Module):
    
    
    def __init__():
        super().__init__()
    
    def forward(self):
        pass
    
    def loss(self, x_context, y_context, x_target, y_target):
        pass

    def mean_and_marginals(self, x_context, y_context, x_target):
        pass


    @property
    def num_params(self):
        """Number of parameters."""
    
        return np.sum([torch.tensor(param.shape).prod() \
                       for param in self.parameters()])
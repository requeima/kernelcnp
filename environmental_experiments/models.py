import torch
import torch.nn as nn
from torch.distributions import Normal,MultivariateNormal
import numpy as np

from architectures import *
from utils import *

class convCNP(nn.Module):
    """
    Conditional neural process model 
    """
    
    def __init__(self, 
                 in_channels,
                 ls = 0.02,
                 elev_dim=3):
        
        super().__init__()

        self.decoder= Decoder(
            in_channels=in_channels,
            out_channels=2,
            n_blocks=6)

        self.elev_mlp = MLP(
            in_channels=5, out_channels=2
        )

        self.activation_function = torch.relu
        self.sc = SetConv(ls)
        self.fp = nn.Softplus()

    def forward(self, task):

        x = self.decoder(task["y_context"]) 

        # Transform to off-grid
        x = self.sc(x, task["dists"])
        # Do elevation
        x = torch.cat([x, task["elev"]], dim = -1)
        x = self.elev_mlp(x)

        # Force sigma > 0
        x[...,1] = force_positive(x[...,1])
        return x

    def loss(self, target_vals, v):
    
        # Deal with cases where data is missing for a station
        nans = torch.isnan(target_vals)
        v[nans, :] = 1
        target_vals[nans] = 1#target_vals[~torch.isnan(target_vals)]
        
        dist = Normal(loc=v[:,:,0], scale=v[:,:,1])
        logp = dist.log_prob(target_vals)
        logp[nans] = np.nan
        ll = torch.mean(torch.nansum(logp, dim=-1))

        return ll

    def preprocess_function(self, task):
        """
        Preprocess task
        """

        n_samples = task["y_target"].shape[-1] #np.random.randint(10,high=100)

        return task, n_samples


class convGNPKvv(nn.Module):
    """
    ConvGNP model kvv
    """
    
    def __init__(self, 
                 ls = 0.02,
                 elev_dim=3):
        
        super().__init__()

        self.decoder= Decoder(in_channels=25, out_channels=130, n_blocks=6)
        
        self.sc = SetConv(ls)

        self.elev_mlp = MLP(
            in_channels=130+elev_dim, 
            out_channels=130
        )

        init_length_scale = 0.5
        self.kernel_sigma = nn.Parameter(np.log(init_length_scale)* torch.ones(1), requires_grad=True)
        self.kernel_fn = torch.exp
 

    def forward(self, task):

        x = self.decoder(task["y_context"]) 

        x = self.sc(x, task["dists"])

        # Do elevation
        elev = task["elev"]
        x = torch.cat([x, elev], dim = -1)
        x = self.elev_mlp(x)

        return x


    def loss(self, yt, tensor):
        """
        Arguments:
            tensor : torch.tensor (B, N, C)
            yt     : torch.tensor (B, N)
        """

        tensor = tensor.double()
        yt = yt.double()

        mask = torch.isnan(torch.sum(yt, dim=0))
        yt = yt[:, ~mask]
        tensor = tensor[:, ~mask, :]

        mean = tensor[:, :, 0]
        noise = tensor[:, :, 1]
        noise = torch.exp(noise)[:, :, None]
        feat = tensor[:, :, 2:-1]
        v = tensor[:,:,-1:]

        #compute the covariance
        vv = torch.matmul(v, torch.transpose(v, dim0=-2, dim1=-1)) 
        scales = self.kernel_fn(self.kernel_sigma)
        cov = rbf_kernel(feat, scales)
        cov = cov * vv
        cov = cov + torch.eye(cov.shape[1]).double()[None, :, :].cuda() * (1e-6 + noise)

        dist = MultivariateNormal(loc=mean, covariance_matrix=cov)
        logprob = dist.log_prob(yt)
        loss = torch.mean(logprob) #/mean.shape[1]

        return loss

    def preprocess_function(self, task, n_samples=86):
        """
        Preprocess task
        """

        n_samples = 25 #np.random.randint(10,high=100)

        nan_locs = torch.isnan(task["y_target"].sum(dim=0))
        task["y_target"] = task["y_target"][:,~nan_locs]
        task["dists"] = task["dists"][~nan_locs,:,:]
        task["elev"] = task["elev"][:,~nan_locs,:]

        inc_inds = np.random.randint(0, high=task["y_target"].shape[1], size=n_samples)
        task["y_target"] = task["y_target"][:,inc_inds]
        task["dists"] = task["dists"][inc_inds,:,:]
        task["elev"] = task["elev"][:,inc_inds,:]

        return task, n_samples

class convGNPLinear(nn.Module):
    """
    ConvGNP Linear
    """
    
    def __init__(self, 
                 ls = 0.02,
                 elev_dim=3):
        
        super().__init__()

        self.decoder= Decoder(in_channels=25, out_channels=130, n_blocks=6)
        
        self.sc = SetConv(ls)

        self.elev_mlp = MLP(
            in_channels=130+elev_dim, 
            out_channels=130
        )
 

    def forward(self, task):
        x = self.decoder(task["y_context"]) 
        x = self.sc(x, task["dists"])

        # Do elevation
        elev = task["elev"]
        x = torch.cat([x, elev], dim = -1)
        x = self.elev_mlp(x)

        return x


    def loss(self, yt, tensor):
        """
        Arguments:
            tensor : torch.tensor (B, N, C)
            yt     : torch.tensor (B, N)
        """

        tensor = tensor.double()
        yt = yt.double()

        mask = torch.isnan(torch.sum(yt, dim=0))
        yt = yt[:, ~mask]
        tensor = tensor[:, ~mask, :]

        mean = tensor[:, :, 0]
        noise = tensor[:, :, 1]
        noise = torch.exp(noise)[:, :, None]
        feat = tensor[:, :, 2:]

        cov = torch.einsum('bnc, bmc -> bnm', feat, feat) / feat.shape[-1]
        cov = cov + torch.eye(cov.shape[1]).double()[None, :, :].cuda() * (1e-6 + noise)

        dist = MultivariateNormal(loc=mean, covariance_matrix=cov)
        logprob = dist.log_prob(yt)
        loss = torch.mean(logprob) #/mean.shape[1]

        return loss

    def preprocess_function(self, task, n_samples=86):
        """
        Preprocess task
        """

        n_samples = 25 #np.random.randint(10,high=100)

        nan_locs = torch.isnan(task["y_target"].sum(dim=0))
        task["y_target"] = task["y_target"][:,~nan_locs]
        task["dists"] = task["dists"][~nan_locs,:,:]
        task["elev"] = task["elev"][:,~nan_locs,:]

        inc_inds = np.random.randint(0, high=task["y_target"].shape[1], size=n_samples)
        task["y_target"] = task["y_target"][:,inc_inds]
        task["dists"] = task["dists"][inc_inds,:,:]
        task["elev"] = task["elev"][:,inc_inds,:]

        return task, n_samples

        
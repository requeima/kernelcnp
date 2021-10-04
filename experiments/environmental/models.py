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

    def loss(self, target_vals_in, v_in):

        target_vals = target_vals_in.clone()
        v = v_in.clone()
    
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


    def loss(self, yt_in, tensor_in):
        """
        Arguments:
            tensor : torch.tensor (B, N, C)
            yt     : torch.tensor (B, N)
        """

        print("In loss...")

        tensor = tensor_in.clone().double()
        yt = yt_in.clone().double()

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

        print("Out loss...")

        return loss

    def preprocess_function(self, task, n_samples=86):
        """
        Preprocess task
        """

        n_samples = 86 #np.random.randint(10,high=100)

        nan_locs = torch.isnan(task["y_target"].sum(dim=0))
        task["y_target"] = task["y_target"][:,~nan_locs]
        task["dists"] = task["dists"][~nan_locs,:,:]
        task["elev"] = task["elev"][:,~nan_locs,:]

        inc_inds = np.random.choice(task["y_target"].shape[1], n_samples, replace=False)
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


    def loss(self, yt_in, tensor_in):
        """
        Arguments:
            tensor : torch.tensor (B, N, C)
            yt     : torch.tensor (B, N)
        """

        tensor = tensor_in.clone().double()
        yt = yt_in.clone().double()

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

        inc_inds = np.random.choice(task["y_target"].shape[1], n_samples, replace=False)
        task["y_target"] = task["y_target"][:,inc_inds]
        task["dists"] = task["dists"][inc_inds,:,:]
        task["elev"] = task["elev"][:,inc_inds,:]

        return task, n_samples
    
class convNP(nn.Module):
    
    def __init__(self,
                 ls = 0.02,
                 n_samples=24,
                 n_latent_vars=128):

        super().__init__()

        self.n_samples = n_samples
        self.n_latent_vars = n_latent_vars
        self.decoder_1 = Decoder(in_channels=25, out_channels=n_latent_vars*2, n_blocks=3)
        self.decoder_2 = Decoder(in_channels=n_latent_vars, out_channels=2, n_blocks=3)
        self.sc = SetConv(ls)

        self.elev_mlp = MLP(
            in_channels=5, 
            out_channels=2
        )
        
    def _collapse_sample_dim(self, x):
        return x.view(-1,*x.shape[2:])
        
    def _sample_latent(self, x):
        # Sampling
        dist = Normal(
            loc = x[..., :self.n_latent_vars],
            scale=force_positive(x[..., self.n_latent_vars:]))
        x = dist.rsample(sample_shape=torch.Size([self.n_samples]))
        x = channels_to_2nd_dim(self._collapse_sample_dim(x))
        return x
            
    def forward(self, task):
        print("%%%%%%%%%%%%%")
        print(task["dists"].shape)
        batch = task["y_context"].shape[0]
        x = self.decoder_1(task["y_context"]) 
        x = self._sample_latent(x)
        x = self.decoder_2(x)
        print("34785629364516")
        print(task["dists"].shape)

        # Transform to off-grid
        x = self.sc(x, task["dists"])

        print(task["dists"].shape)
        print(task["y_target"].shape)

        # Do elevation
        elev = task["elev"][0:1,...].repeat(x.shape[0], 1, 1)
        print(elev.shape)
        print(x.shape)
        x = torch.cat([x, elev], dim = -1)
        x = self.elev_mlp(x)

        # Force sigma > 0
        x[...,1] = force_positive(x[...,1])
        return x.view(self.n_samples, batch, *x.shape[1:])
    
    def loss(self, targets, out):
        na = targets.sum(dim=0).isnan()

        out = out[:,:,~na,:]
        targets = targets[:,~na]
        
        n_samples = out.shape[0]
        targets = targets.unsqueeze(0).repeat(n_samples, 1, 1)
        dist = Normal(out[...,0], out[...,1])

        lps = dist.log_prob(targets)
        # First sum over target points (should be zeros where masked so 
        #won't contribute)
        lps = torch.sum(lps, axis = 2) # shape (samples, batch)

        # Do logsumexp of the sums
        x = torch.logsumexp(lps, dim = 0) - np.log(out.shape[0]) # shape (batch)
        
        # Return negative mean over bastch dimension
        return torch.mean(x)
        
    def preprocess_function(self, task):
        """
        Preprocess task
        """

        n_samples = task["y_target"].shape[-1] #np.random.randint(10,high=100)

        return task, n_samples

        
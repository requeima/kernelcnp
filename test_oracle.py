import argparse

import numpy as np
import stheno.torch as stheno
import torch
import matplotlib.pyplot as plt
import os
from datetime import datetime
import pickle

from torch._C import Value

# This is for an error that is now popping up when running on macos
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

import cnp.data

from copy import deepcopy

from cnp.experiment import (
    generate_root,
    WorkingDirectory,
    save_checkpoint,
    log_args
)

from cnp.utils import plot_samples_and_data, make_generator

from torch.distributions import MultivariateNormal


# =============================================================================
# Custom kernels until we resolve issue with Stheno
# =============================================================================

def eq_cov(lengthscale, coefficient, noise):
    
    def _eq_cov(x, x_, use_noise):
    
        diff = x[:, None, :] - x_[None, :, :]
        l2 = torch.sum((diff / lengthscale) ** 2, dim=2)
        cov = coefficient ** 2 * np.exp(-0.5 * l2)
        
        if use_noise:
            cov = cov + noise ** 2 * torch.eye(cov.shape[0])
            
        return cov
        
    
    return _eq_cov


def mat_cov(lengthscale, coefficient, noise):
    
    def _mat_cov(x, x_, use_noise):
    
        diff = x[:, None, :] - x_[None, :, :]
        l1 = torch.sum(np.abs(diff / lengthscale), dim=2)
        cov = coefficient ** 2 * (1 + 5 ** 0.5 * l1 + 5 * l1 ** 2 / 3)
        cov = cov * np.exp(- 5 ** 0.5 * l1)
        
        if use_noise:
            cov = cov + noise ** 2 * torch.eye(cov.shape[0])
            
        return cov
        
    return _mat_cov


def wp_cov(period, lengthscale, coefficient, noise):
    
    def _wp_cov(x, x_, use_noise):
    
        diff = x[:, None, :] - x_[None, :, :]
        l1 = torch.sum(np.abs(diff / period), dim=2)
        l2 = torch.sum((diff / lengthscale) ** 2, dim=2)
        
        sin2 = (torch.sin(np.pi * l1) / lengthscale) ** 2
        
        cov = coefficient ** 2 * torch.exp(-2. * sin2)
        cov = cov * np.exp(-0.5 * l2)
        
        if use_noise:
            cov = cov + noise ** 2 * torch.eye(cov.shape[0])
            
        return cov
        
    return _wp_cov


def nm_cov(lengthscale1, lengthscale2, coefficient, noise):
        
    eq_cov1 = eq_cov(lengthscale1, coefficient, noise)
    eq_cov2 = eq_cov(lengthscale2, coefficient, noise)
    
    def _nm_cov(x, x_, use_noise):
        
        cov1 = eq_cov1(x, x_, use_noise)
        cov2 = eq_cov2(x, x_, use_noise=False)
        
        return cov1 + cov2
        
    return _nm_cov
    

def oracle_loglik(xc, yc, xt, yt, covariance):

    Ktt = covariance(xt, xt, use_noise=True)
    Kcc = covariance(xc, xc, use_noise=True)
    Kct = covariance(xc, xt, use_noise=False)
    
    cov = Ktt - np.einsum('ij, ik -> jk', Kct, np.linalg.solve(Kcc, Kct))
    cov = torch.tensor(cov).double()
    
    mean = np.einsum('ij, ik -> jk', Kct, np.linalg.solve(Kcc, yc))
    mean = torch.tensor(mean[:, 0]).double()

    dist = torch.distributions.MultivariateNormal(loc=mean,
                                                  covariance_matrix=cov)
    logprob = dist.log_prob(torch.tensor(yt[:, 0]).double())

    return logprob



# =============================================================================
# Test oracle helper
# =============================================================================


def test_oracle(data, covariance):
    """ Compute the oracle test loss. """
    
    oracle_nll_list = []
    
    with torch.no_grad():
        for step, batch in enumerate(data):
            if step % 500 == 0:
                print(f'step {step}')
            print(step)
            oracle_nll = np.array(0.)
            for b in range(batch['x_context'].shape[0]):
                oracle_nll = -oracle_loglik(batch['x_context'][b],
                                            batch['y_context'][b],
                                            batch['x_target'][b],
                                            batch['y_target'][b],
                                            covariance=covariance)
                        
                # oracle_nll = oracle_nll + _oracle_nll
                oracle_nll_list.append(oracle_nll/50.)

        print(f"Oracle     neg. log-lik: "
            f"{np.mean(oracle_nll_list):.2f} +/- "
            f"{np.var(oracle_nll_list) ** 0.5:.2f}")
            
                
    mean_oracle = np.mean(oracle_nll_list)
    std_oracle = (np.var(oracle_nll_list) ** 0.5) / np.sqrt(step + 1)

    return mean_oracle, std_oracle


# Parse arguments given to the script.
parser = argparse.ArgumentParser()

# =============================================================================
# Data generation arguments
# =============================================================================

parser.add_argument('data',
                    choices=['sawtooth',
                            'eq',
                            'matern',
                            'noisy-mixture',
                            'noisy-mixture-slow',
                            'weakly-periodic',
                            'weakly-periodic-slow'],
                    help='Data set to train the CNP on. ')

parser.add_argument('--x_dim',
                    default=1,
                    choices=[1, 2, 3],
                    type=int,
                    help='Input dimension of data.')

parser.add_argument('--seed',
                    default=0,
                    type=int,
                    help='Random seed to use.')


# =============================================================================
# Experiment arguments
# =============================================================================


parser.add_argument('--root',
                    help='Experiment root, which is the directory from which '
                         'the experiment will run. If it is not given, '
                         'a directory will be automatically created.')

parser.add_argument('--gpu',
                    default=0,
                    type=int,
                    help='GPU to run experiment on. Defaults to 0.')


args = parser.parse_args()
    
# =============================================================================
# Set random seed, device
# =============================================================================

# Set seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Set device
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu)

device = torch.device('cpu') if not torch.cuda.is_available() and args.gpu == 0 \
                             else torch.device('cuda')

data_root = os.path.join('_experiments/toy-data', 
                         f'{args.data}',
                         'data',
                         f'seed-{args.seed}',
                         f'dim-{args.x_dim}')

experiment_name = os.path.join('_experiments/toy-results',
                                   f'{args.data}',
                                   f'models',
                                   'Oracle-GP',
                                   f'seed-{args.seed}',
                                   f'dim-{args.x_dim}')

working_directory = WorkingDirectory(root=experiment_name)
data_directory = WorkingDirectory(root=data_root)
    

# =============================================================================
# Load data and validation oracle generator
# =============================================================================
    
file = open(data_directory.file('test-data.pkl'), 'rb')
data_test = pickle.load(file)
file.close()

# Create the data generator for the oracle if gp data
if args.data == 'sawtooth' or args.data == 'random':
    raise ValueError('No oracle for data type')
    
elif args.data == 'eq':
    covariance = eq_cov(lengthscale=1.,
                        coefficient=1.,
                        noise=5e-2)

elif args.data == 'matern':
    covariance = mat_cov(lengthscale=1.,
                         coefficient=1.,
                         noise=5e-2)

elif args.data == 'noisy-mixture':
    covariance = nm_cov(lengthscale1=1.,
                        lengthscale2=0.25,
                        coefficient=1.,
                        noise=5e-2)

elif args.data == 'weakly-periodic':
    covariance = wp_cov(period=0.25,
                        lengthscale=1.,
                        coefficient=1.,
                        noise=5e-2)

elif args.data == 'noisy-mixture-slow':
    covariance = nm_cov(lengthscale1=1.,
                        lengthscale2=0.5,
                        coefficient=1.,
                        noise=5e-2)

elif args.data == 'weakly-periodic-slow':
    covariance = wp_cov(period=0.5,
                        lengthscale=1.,
                        coefficient=1.,
                        noise=5e-2)


# =============================================================================
# Test oracle
# =============================================================================

mean_oracle, std_oracle = test_oracle(data_test, covariance)
print('Oracle averages a log-likelihood of %s +- %s on unseen tasks.' % (mean_oracle, std_oracle))

with open(working_directory.file('test_log_likelihood.txt'), 'w') as f:
    f.write(str(mean_oracle))
    
with open(working_directory.file('test_log_likelihood_standard_error.txt'), 'w') as f:
    f.write(str(std_oracle))

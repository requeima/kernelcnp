import argparse

import os
import sys
import pickle
import time

from datetime import datetime
from copy import deepcopy

from cnp.experiment import (
    generate_root,
    WorkingDirectory,
    save_checkpoint,
    log_args
)

from cnp.cnp import StandardPredPreyConvGNP
from cnp.lnp import StandardPredPreyConvNP

from cnp.cov import (
    MeanFieldGaussianLayer,
    InnerprodGaussianLayer,
    KvvGaussianLayer,
    ExponentialCopulaLayer,
    LogLogitCopulaLayer
)

from cnp.utils import (
    Logger,
    plot_pred_prey_fits
)

import numpy as np
import matplotlib.pyplot as plt

import torch


# =============================================================================
# Testing helper
# =============================================================================


def test(data,
         model,
         args,
         device,
         latent_model):
    
    # Lists for logging model's training NLL and oracle NLL
    nll_list = []
    
    # If training a latent model, set the number of latent samples accordingly
    loss_kwargs = {'num_samples' : args.np_test_samples} if latent_model else {}
    
    with torch.no_grad():
        
        for step, batch in enumerate(data):

            nll = model.loss(batch['x_context'][:, :, None].to(device),
                             batch['y_context'][:, 0, :, None].to(device) / 100 + 1e-2,
                             batch['x_target'][:, :, None].to(device),
                             batch['y_target'][:, 0, :, None].to(device) / 100 + 1e-2)

            # Scale by the number of target points
            nll_list.append(nll / 100.)

            if step % 100 == 0:
                print(f"Validation neg. log-lik, {step+1}: "
                      f"{torch.mean(torch.tensor(nll_list)):.2f} +/- "
                      f"{torch.var(torch.tensor(nll_list))**0.5  / (step+1)**0.5:.2f}")

            
    mean_nll = torch.mean(torch.tensor(nll_list))
    std_nll = torch.var(torch.tensor(nll_list))**0.5 / (step + 1)**0.5
    
    return mean_nll, std_nll
        

# Parse arguments given to the script.
parser = argparse.ArgumentParser()


# =============================================================================
# Data generation arguments
# =============================================================================

parser.add_argument('data', help='Data set to train the CNP on.')

parser.add_argument('--seed',
                    default=0,
                    type=int,
                    help='Random seed to use.')

# =============================================================================
# Model arguments
# =============================================================================

parser.add_argument('model',
                    choices=['convGNP', 'convNP'],
                    help='Choice of model. ')

parser.add_argument('cov_type',
                    choices=['meanfield', 'innerprod',  'kvv'],
                    help='Choice of covariance method.')

parser.add_argument('noise_type',
                    choices=['homo', 'hetero'],
                    help='Choice of noise model.')

parser.add_argument('--marginal_type',
                    default='identity',
                    choices=['identity', 'exponential', 'loglogit'],
                    help='Choice of marginal transformation (optional).')

parser.add_argument('--np_loss_samples',
                    default=16,
                    type=int,
                    help='Number of latent samples for evaluating the loss, '
                         'used for ANP and ConvNP.')

parser.add_argument('--np_val_samples',
                    default=16,
                    type=int,
                    help='Number of latent samples for evaluating the loss, '
                         'when validating, used for ANP and ConvNP.')

parser.add_argument('--num_basis_dim',
                    default=32,
                    type=int,
                    help='Number of embedding basis dimensions.')

parser.add_argument('--jitter',
                    default=1e-4,
                    type=float,
                    help='Jitter.')

parser.add_argument('--gpu',
                    default=0,
                    type=int,
                    help='GPU to run experiment on. Defaults to 0.')

args = parser.parse_args()
    
# =============================================================================
# Set random seed and device
# =============================================================================

# Set seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Set device
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu)
    
use_cpu = not torch.cuda.is_available() and args.gpu == 0
device = torch.device('cpu') if use_cpu else torch.device('cuda')

root = 'experiments/predator-prey'

# Working directory for saving results
experiment_name = os.path.join(f'{root}',
                               f'results',
                               f'{args.data}',
                               f'models',
                               f'{args.model}',
                               f'{args.cov_type}',
                               f'{args.noise_type}',
                               f'{args.marginal_type}',
                               f'seed-{args.seed}')
working_directory = WorkingDirectory(root=experiment_name)

# Data directory for loading data
data_root = os.path.join(f'{root}',
                         f'simulated-data',
                         f'{args.data}')
data_directory = WorkingDirectory(root=data_root)

# Data directory for loading data
true_data_root = os.path.join(f'{root}', 'data')
true_data_root = WorkingDirectory(root=true_data_root)

log_path = f'{root}/logs'
log_filename = f'{args.data}-'          + \
               f'{args.model}-'         + \
               f'{args.cov_type}-'      + \
               f'{args.noise_type}-'    + \
               f'{args.marginal_type}'
                
log_directory = WorkingDirectory(root=log_path)
sys.stdout = Logger(log_directory=log_directory, log_filename=log_filename)
sys.stderr = Logger(log_directory=log_directory, log_filename=log_filename)
    
file = open(working_directory.file('data_location.txt'), 'w')
file.write(data_directory.root)
file.close()
    
# =============================================================================
# Create model
# =============================================================================

cov_types = {
    'meanfield' : MeanFieldGaussianLayer,
    'innerprod' : InnerprodGaussianLayer,
    'kvv'       : KvvGaussianLayer
}

if args.cov_type == 'meanfield':
    output_layer = MeanFieldGaussianLayer()
    
else:
    output_layer = cov_types[args.cov_type](num_embedding=args.num_basis_dim,
                                            noise_type=args.noise_type,
                                            jitter=args.jitter)

if args.marginal_type == 'exponential':
    print('Exponential marginals')
    output_layer = ExponentialCopulaLayer(gaussian_layer=output_layer,
                                          device=device)
    
elif args.marginal_type == 'loglogit':
    print('Log-logistic marginals')
    output_layer = LogLogitCopulaLayer(gaussian_layer=output_layer)
    
else:
    print('Gaussian marginals')
    
# Create model architecture
if args.model == 'convGNP':
    model = StandardPredPreyConvGNP(input_dim=1,
                                    output_layer=output_layer)
    
elif args.model == 'convNP':
    model = StandardPredPreyConvNP(input_dim=1,
                                   num_samples=args.np_loss_samples)
    
else:
    raise ValueError(f'Unknown model {args.model}.')


print(f'{args.data} '
      f'{args.model} '
      f'{args.cov_type} '
      f'{args.noise_type} '
      f'{args.marginal_type} '
      f'{args.num_basis_dim}: '
      f'{model.num_params}')

with open(working_directory.file('num_params.txt'), 'w') as f:
    f.write(f'{model.num_params}')
    
# Load model to appropriate device
model = model.to(device)

latent_model = args.model == 'convNP'

# Load model from saved state
load_dict = torch.load(working_directory.file('model_best.pth.tar', exists=True))
model.load_state_dict(load_dict['state_dict'])

# =============================================================================
# Load data
# =============================================================================

file = open(data_directory.file('test-data.pkl'), 'rb')
data_test = pickle.load(file)
file.close()

# =============================================================================
# Test model
# =============================================================================

# Compute negative log-likelihood on validation data
mean_nll, _ = test(data_test,
                   model,
                   args,
                   device,
                   latent_model)
mean_nll = float(mean_nll.item())

print(mean_nll)

with open(working_directory.file('test_log_likelihood.txt'), 'w') as file:
    file.write(str(mean_nll))

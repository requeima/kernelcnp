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

from cnp.cnp import StandardEEGConvGNP

from cnp.cov import (
    MultiOutputMeanFieldGaussianLayer,
    MultiOutputInnerprodGaussianLayer,
    MultiOutputKvvGaussianLayer
)

from cnp.data import EEGGenerator

from cnp.utils import Logger

import numpy as np
import matplotlib.pyplot as plt

import torch

# =============================================================================
# Test helper
# =============================================================================


def test(data_test, model, device, args):
    
    nlls = []
    
    with torch.no_grad():
        for batch in data_test:
        
            nll = model.loss(batch['x_context'].to(device),
                             batch['y_context'].to(device),
                             batch['m_context'].to(device),
                             batch['x_target'].to(device),
                             batch['y_target'].to(device),
                             batch['m_target'].to(device))
            
            nll = nll / (args.num_channels_target * args.target_length)
            
            nlls.append(nll.item())
            
    mean_nll = np.mean(nlls)

    # Print validation loss and oracle loss
    print(f"Test neg. log-lik: {mean_nll:.2f}")
    
    return mean_nll
        

# Parse arguments given to the script.
parser = argparse.ArgumentParser()


# =============================================================================
# Training data arguments
# =============================================================================

parser.add_argument('--epochs',
                    default=200,
                    type=int,
                    help='Number of epochs to train for.')

parser.add_argument('--batch_size',
                    default=16,
                    type=int,
                    help='Batch size.')

parser.add_argument('--batches_per_epoch',
                    default=256,
                    type=int,
                    help='Number of batches per epoch.')

parser.add_argument('--num_channels_total',
                    default=6,
                    type=int,
                    help='Number of EEG channels to use in experiment.')

parser.add_argument('--num_channels_target',
                    default=3,
                    type=int,
                    help='Number of EEG target channels to use in experiment.')

parser.add_argument('--target_length',
                    default=50,
                    type=int,
                    help='Size of time series to use in target set.')

parser.add_argument('--seed',
                    default=0,
                    type=int,
                    help='Random seed to use.')


# =============================================================================
# Model arguments
# =============================================================================

parser.add_argument('model',
                    choices=['convGNP'],
                    help='Choice of model. ')

parser.add_argument('cov_type',
                    choices=['meanfield', 'innerprod',  'kvv'],
                    help='Choice of covariance method.')

parser.add_argument('noise_type',
                    choices=['hetero'],
                    help='Choice of noise model.')

parser.add_argument('--init_length_scale',
                    default=1e-3,
                    type=float)

parser.add_argument('--num_basis_dim',
                    default=64,
                    type=int,
                    help='Number of embedding basis dimensions.')

parser.add_argument('--learning_rate',
                    default=2e-4,
                    type=float,
                    help='Learning rate.')

parser.add_argument('--jitter',
                    default=1e-3,
                    type=float,
                    help='The jitter level.')

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

root = 'experiments/eeg'

data_params = f'{args.epochs}-'              + \
              f'{args.batch_size}-'          + \
              f'{args.num_channels_total}-'  + \
              f'{args.num_channels_target}-' + \
              f'{args.target_length}-'       + \
              f'{args.seed}'

# Working directory for saving results
experiment_name = os.path.join(f'{root}',
                               f'results',
                               f'{data_params}',
                               f'{args.model}',
                               f'{args.cov_type}',
                               f'{args.noise_type}')
working_directory = WorkingDirectory(root=experiment_name)

log_path = f'{root}/logs'
log_filename = f'{data_params}-'        + \
               f'{args.model}-'         + \
               f'{args.cov_type}-'      + \
               f'{args.noise_type}'
                
log_directory = WorkingDirectory(root=log_path)
sys.stdout = Logger(log_directory=log_directory, log_filename=log_filename)
sys.stderr = Logger(log_directory=log_directory, log_filename=log_filename)

    
# =============================================================================
# Create model
# =============================================================================

cov_types = {
    'meanfield' : MultiOutputMeanFieldGaussianLayer,
    'innerprod' : MultiOutputInnerprodGaussianLayer,
    'kvv'       : MultiOutputKvvGaussianLayer
}

if args.cov_type == 'meanfield':
    output_layer = cov_types['meanfield'](num_outputs=args.num_channels_total)
    
else:
    output_layer = cov_types[args.cov_type](num_outputs=args.num_channels_total,
                                            num_embedding=args.num_basis_dim,
                                            noise_type=args.noise_type,
                                            jitter=args.jitter)
    
model = StandardEEGConvGNP(num_channels=args.num_channels_total,
                           output_layer=output_layer)

print(f'{data_params} '
      f'{args.model} '
      f'{args.cov_type} '
      f'{args.noise_type} '
      f'{args.num_basis_dim}: '
      f'{model.num_params}')

with open(working_directory.file('num_params.txt'), 'w') as f:
    f.write(f'{model.num_params}')
    
# Load model to appropriate device
model = model.to(device)


# =============================================================================
# Load data and validation oracle generator
# =============================================================================
    
data_test = EEGGenerator(split='test',
                         batch_size=args.batch_size,
                         batches_per_epoch=args.batches_per_epoch,
                         num_total_channels=args.num_channels_total,
                         num_target_channels=args.num_channels_target,
                         target_length=args.target_length,
                         device=device)

# =============================================================================
# Train or test model
# =============================================================================

# Number of epochs between validations
train_iteration = 0
log_every = 50
    
log_args(working_directory, args)

# Compute training negative log-likelihood
test_mean_nll = test(data_test, model, device, args)

file = open(working_directory.file('test_log_likelihood.txt'), 'w')
file.write(str(test_mean_nll))
file.close()
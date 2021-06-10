import argparse

import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import pickle
import time

# This is for an error that is now popping up when running on macos
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

from copy import deepcopy

from cnp.experiment import (
    generate_root,
    WorkingDirectory,
    save_checkpoint,
    log_args
)

from cnp.cnp import (
    StandardGNP,
    StandardAGNP,
    StandardConvGNP,
    FullConvGNP
)

from cnp.lnp import (
    StandardANP,
    StandardConvNP
)

from cnp.cov import (
    InnerProdCov,
    KvvCov,
    MeanFieldCov,
    AddHomoNoise,
    AddHeteroNoise,
    AddNoNoise
)

from cnp.utils import plot_samples_and_data, make_generator

import torch
from torch.distributions import MultivariateNormal
from torch.utils.tensorboard import SummaryWriter


# =============================================================================
# Validation helper
# =============================================================================


def test(data,
         model,
         args,
         device,
         latent_model):
    
    # Lists for logging model's training NLL and oracle NLL
    nll_list = []
    oracle_nll_list = []
    
    # If training a latent model, set the number of latent samples accordingly
    loss_kwargs = {'num_samples' : args.np_test_samples} if latent_model else \
                  {'double'      : True}
    
    with torch.no_grad():
        
        for step, batch in enumerate(data):

            nll = model.loss(batch['x_context'].to(device),
                             batch['y_context'].to(device),
                             batch['x_target'].to(device),
                             batch['y_target'].to(device),
                             **loss_kwargs)                        

            # Scale by the average number of target points
            nll_list.append(nll.item() / 50.)

            if step % 100 == 0:
                print(f"Validation neg. log-lik, {step+1}: "
                      f"{np.mean(nll_list):.2f} +/- "
                      f"{np.var(nll_list)**0.5  / (step+1)**0.5:.2f}")

            
    mean_nll = np.mean(nll_list)
    std_nll = np.var(nll_list)**0.5 / np.sqrt(step + 1)
    
    return mean_nll, std_nll
        
# Parse arguments given to the script.
parser = argparse.ArgumentParser()


# =============================================================================
# Data generation arguments
# =============================================================================

parser.add_argument('data',
                    choices=['eq',
                             'eq-lb',
                             'matern',
                             'matern-lb',
                             'noisy-mixture',
                             'noisy-mixture-lb',
                             'weakly-periodic',
                             'weakly-periodic-lb',
                             'noisy-mixture-slow',
                             'weakly-periodic-slow',
                             'noisy-mixture-slow-100',
                             'weakly-periodic-slow-100',
                             'sawtooth',
                             'sawtooth-lb'],
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

parser.add_argument('--validate_every',
                    default=10,
                    type=int,
                    help='Number of epochs between validations.')


# =============================================================================
# Model arguments
# =============================================================================

parser.add_argument('model',
                    choices=['GNP',
                             'AGNP',
                             'convGNP',
                             'FullConvGNP',
                             'ANP',
                             'convNP'],
                    help='Choice of model. ')

parser.add_argument('covtype',
                    choices=['innerprod-homo',
                             'innerprod-hetero', 
                             'kvv-homo',
                             'kvv-hetero',
                             'meanfield'],
                    help='Choice of covariance method.')

parser.add_argument('--np_loss_samples',
                    default=16,
                    type=int,
                    help='Number of latent samples for evaluating the loss, '
                         'used for ANP and ConvNP.')

parser.add_argument('--np_test_samples',
                    default=512,
                    type=int,
                    help='Number of latent samples for evaluating the loss, '
                         'when validating, used for ANP and ConvNP.')

parser.add_argument('--num_basis_dim',
                    default=512,
                    type=int,
                    help='Number of embedding basis dimensions.')

parser.add_argument('--learning_rate',
                    default=5e-4,
                    type=float,
                    help='Learning rate.')

parser.add_argument('--weight_decay',
                    default=0.,
                    type=float,
                    help='Weight decay.')


# =============================================================================
# Experiment arguments
# =============================================================================


parser.add_argument('--root',
                    type=str,
                    default='_experiments',
                    help='Experiment root, which is the directory from which '
                         'the experiment will run. If it is not given, '
                         'a directory will be automatically created.')

parser.add_argument('--num_params',
                    action='store_true',
                    help='Print the total number of parameters in the moodel '
                         'and exit.')

parser.add_argument('--gpu',
                    default=0,
                    type=int,
                    help='GPU to run experiment on. Defaults to 0.')


args = parser.parse_args()

    
# =============================================================================
# Set random seed, device and tensorboard writer
# =============================================================================

# Set seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Set device
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu)
    
use_cpu = not torch.cuda.is_available() and args.gpu == 0
device = torch.device('cpu') if use_cpu else torch.device('cuda')

data_root = os.path.join(f'toy-data',
                         f'{args.data}',
                         f'data',
                         f'seed-{args.seed}',
                         f'dim-{args.x_dim}')
    
experiment_name = os.path.join(f'toy-results',
                               f'{args.data}',
                               f'models',
                               f'{args.model}',
                               f'{args.covtype}',
                               f'seed-{args.seed}',
                               f'dim-{args.x_dim}')

working_directory = WorkingDirectory(root=experiment_name)
data_directory = WorkingDirectory(root=data_root)

writer = SummaryWriter(f'{experiment_name}/log')
    
# =============================================================================
# Create model
# =============================================================================

# Create covariance method
if args.covtype == 'innerprod-homo':
    cov = InnerProdCov(args.num_basis_dim)
    noise = AddHomoNoise()
    
elif args.covtype == 'innerprod-hetero':
    cov = InnerProdCov(args.num_basis_dim)
    noise = AddHeteroNoise()
    
elif args.covtype == 'kvv-homo':
    cov = KvvCov(args.num_basis_dim)
    noise = AddHomoNoise()
    
elif args.covtype == 'kvv-hetero':
    cov = KvvCov(args.num_basis_dim)
    noise = AddHomoNoise()
    
elif args.covtype == 'meanfield':
    cov = MeanFieldCov(num_basis_dim=1)
    noise = AddNoNoise()
    
else:
    raise ValueError(f'Unknown covariance method {args.covtype}.')
    
# Create model architecture
if args.model == 'GNP':
    model = StandardGNP(input_dim=args.x_dim,
                        covariance=cov,
                        add_noise=noise)
    
elif args.model == 'AGNP':
    model = StandardAGNP(input_dim=args.x_dim,
                         covariance=cov,
                         add_noise=noise)
    
elif args.model == 'convGNP':
    model = StandardConvGNP(input_dim=args.x_dim,
                            covariance=cov,
                            add_noise=noise)

elif args.model == 'FullConvGNP':
    model = FullConvGNP()
    

elif args.model == 'ANP':
    
    noise = AddHomoNoise()
    model = StandardANP(input_dim=args.x_dim,
                        add_noise=noise,
                        num_samples=args.np_loss_samples)
    
elif args.model == 'convNP':
    
    noise = AddHomoNoise()
    model = StandardConvNP(input_dim=args.x_dim,
                           add_noise=noise,
                           num_samples=args.np_loss_samples)
    
else:
    raise ValueError(f'Unknown model {args.model}.')


print(f'{args.model} '
      f'{args.covtype} '
      f'{args.num_basis_dim}: '
      f'{model.num_params}')

with open(working_directory.file('num_params.txt'), 'w') as f:
    f.write(f'{model.num_params}')
        
if args.num_params:
    exit()
    
    
# Load model to appropriate device
model = model.to(device)

latent_model = args.model in ['ANP', 'convNP']

# Load model from saved state
load_dict = torch.load(working_directory.file('model_best.pth.tar', exists=True))
model.load_state_dict(load_dict['state_dict'])

# =============================================================================
# Load data and validation oracle generator
# =============================================================================
    
file = open(data_directory.file('test-data.pkl'), 'rb')
data_test = pickle.load(file)
file.close()


# =============================================================================
# Train or test model
# =============================================================================
print("Starting testing...")

start_time = time.time()
test_mean_nll, test_std_nll = test(data_test,
                                   model,
                                   args,
                                   device,
                                   latent_model)
stop_time = time.time()
elapsed_time = stop_time - start_time

print("finished testing.")

file = open(working_directory.file('test_log_likelihood.txt'), 'w')
file.write(str(test_mean_nll))
file.close()

file = open(working_directory.file('test_log_likelihood_standard_error.txt'), 'w')
file.write(str(test_std_nll))
file.close()

file = open(working_directory.file('test_time.txt'), 'w')
file.write(str(elapsed_time))
file.close()

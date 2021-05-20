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


def test_oracle(data, data_generator):
    """ Compute the oracle test loss. """
    
    oracle_nll_list = []
    
    with torch.no_grad():
        for step, batch in enumerate(data):
            print(step)
            oracle_nll = np.array(0.)
            if (type(data_generator) == cnp.data.GPGenerator):
                for b in range(batch['x_context'].shape[0]):
                    oracle_nll =  - data_generator.log_like(batch['x_context'][b],
                                                            batch['y_context'][b],
                                                            batch['x_target'][b],
                                                            batch['y_target'][b])
                    # oracle_nll = oracle_nll + _oracle_nll
                    oracle_nll_list.append(oracle_nll/50.)

        print(f"Oracle     neg. log-lik: "
            f"{np.mean(oracle_nll_list):.2f} +/- "
            f"{np.var(oracle_nll_list) ** 0.5:.2f}")
                
    mean_oracle = np.mean(oracle_nll_list)
    std_oracle = np.var(oracle_nll_list) ** 0.5

    return mean_oracle, std_oracle


# Parse arguments given to the script.
parser = argparse.ArgumentParser()

# =============================================================================
# Data generation arguments
# =============================================================================

parser.add_argument('data',
                    choices=['eq',
                             'matern',
                             'noisy-mixture',
                             'weakly-periodic',
                             'sawtooth',
                             'random'],
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

experiment_name = os.path.join('_experiments/toy-data',
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
    
else:
    file = open(data_directory.file('gen-test-dict.pkl'), 'rb')
    gen_valid_gp_params = pickle.load(file)
    file.close()

    file = open(data_directory.file('kernel-params.pkl'), 'rb')
    kernel_params = pickle.load(file)
    file.close()
    
    gen_test = make_generator(args.data, gen_valid_gp_params, kernel_params)


# =============================================================================
# Test oracle
# =============================================================================

mean_oracle, std_oracle = test_oracle(data_test, gen_test)
print('Oracle averages a log-likelihood of %s +- %s on unseen tasks.' % (mean_oracle, std_oracle))

with open(working_directory.file('test_log_likelihood.txt'), 'w') as f:
    f.write(str(mean_oracle))
    
with open(working_directory.file('test_log_likelihood_standard_error.txt'), 'w') as f:
    f.write(str(std_oracle))
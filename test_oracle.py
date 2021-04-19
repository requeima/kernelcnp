import argparse

import numpy as np
import stheno.torch as stheno
import torch
import matplotlib.pyplot as plt
import os
from datetime import datetime
import pickle

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

from cnp.utils import plot_samples_and_data

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
                    _oracle_nll =  - data_generator.log_like(batch['x_context'][b],
                                                            batch['y_context'][b],
                                                            batch['x_target'][b],
                                                            batch['y_target'][b])
                    oracle_nll = oracle_nll + _oracle_nll
                        
            oracle_nll_list.append(oracle_nll)

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
                             'sawtooth'],
                    help='Data set to train the CNP on. ')

parser.add_argument('--seed',
                    default=0,
                    type=int,
                    help='Random seed to use.')

parser.add_argument('--std_noise',
                    default=1e-1,
                    type=float,
                    help='Standard dev. of noise added to GP-generated data.')

parser.add_argument('--batch_size',
                    default=128,
                    type=int,
                    help='Number of tasks per batch sampled.')

parser.add_argument('--max_num_context',
                    default=32,
                    type=int,
                    help='Maximum number of context points.')

parser.add_argument('--max_num_target',
                    default=32,
                    type=int,
                    help='Maximum number of target points.')


parser.add_argument('--num_test_iters',
                    default=2048,
                    type=int,
                    help='Iterations (# batches sampled) for testing.')


parser.add_argument('--eq_params',
                    default=[1.],
                    nargs='+',
                    type=float,
                    help='.')

parser.add_argument('--m52_params',
                    default=[1.],
                    nargs='+',
                    type=float,
                    help='.')

parser.add_argument('--mixture_params',
                    default=[1., 0.5],
                    nargs='+',
                    type=float,
                    help='.')

parser.add_argument('--wp_params',
                    default=[1., 0.5],
                    nargs='+',
                    type=float,
                    help='.')

parser.add_argument('--x_range',
                    default=[-3., 3.],
                    nargs='+',
                    type=float,
                    help='Range of input x for sampled data.')

parser.add_argument('--freq_range',
                    default=[3., 5.],
                    nargs='+',
                    type=float,
                    help='Range of frequencies for sawtooth data.')

parser.add_argument('--shift_range',
                    default=[-5., 5.],
                    nargs='+',
                    type=float,
                    help='Range of frequency shifts for sawtooth data.')

parser.add_argument('--trunc_range',
                    default=[10., 20.],
                    nargs='+',
                    type=float,
                    help='Range of truncations for sawtooth data.')


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
# Set random seed, device and tensorboard writer
# =============================================================================

# Set seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Set device
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu)

device = torch.device('cpu') if not torch.cuda.is_available() and args.gpu == 0 \
                             else torch.device('cuda')

data_root = os.path.join('_experiments',
                         f'{args.data}',
                         'data',
                         f'{args.seed}')

experiment_name = os.path.join('_experiments',
                                   f'{args.data}',
                                   f'models',
                                   'Oracle-GP')
working_directory = WorkingDirectory(root=experiment_name)
data_directory = WorkingDirectory(root=data_root)

    

file = open(working_directory.file('data_location.txt'), 'w')
file.write(data_directory.root)
file.close()
    

# =============================================================================
# Create data generators
# =============================================================================


# Training data generator parameters -- used for both Sawtooth and GP
gen_params = {
    'batch_size'                : args.batch_size,
    'x_range'                   : args.x_range,
    'max_num_context'           : args.max_num_context,
    'max_num_target'            : args.max_num_target,
    'include_context_in_target' : False,
    'device'                    : device
}

# Plotting data generator parameters -- used for both Sawtooth and GP
gen_plot_params = deepcopy(gen_params)
gen_plot_params['iterations_per_epoch'] = 1
gen_plot_params['batch_size'] = 3
gen_plot_params['max_num_context'] = 16

# Training data generator parameters -- specific to Sawtooth
gen_train_sawtooth_params = {
    'freq_range'  : args.freq_range,
    'shift_range' : args.shift_range,
    'trunc_range' : args.trunc_range
}

                    
if args.data == 'sawtooth':
    gen_test = cnp.data.SawtoothGenerator(args.num_test_iters,
                                          **gen_train_sawtooth_params,
                                          **gen_params)
    
else:
    
    if args.data == 'eq':
        kernel = stheno.EQ().stretch(args.eq_params[0])
        
    elif args.data == 'matern':
        kernel = stheno.Matern52().stretch(args.m52_params[0])
        
    elif args.data == 'noisy-mixture':
        kernel = stheno.EQ().stretch(args.mixture_params[0]) + \
                 stheno.EQ().stretch(args.mixture_params[1])
        
    elif args.data == 'weakly-periodic':
        kernel = stheno.EQ().stretch(args.wp_params[0]) * \
                 stheno.EQ().periodic(period=args.wp_params[1])
        
    else:
        raise ValueError(f'Unknown generator kind "{args.data}".')
        
    gen_test = cnp.data.GPGenerator(iterations_per_epoch=args.num_test_iters,
                                    kernel=kernel,
                                    std_noise=args.std_noise,
                                    **gen_params)
    
# =============================================================================
# Load data
# =============================================================================

file = open(data_directory.file('test-data.pkl'), 'rb')
data_test = pickle.load(file)
file.close()

# =============================================================================
# Test oracle
# =============================================================================

mean_oracle, std_oracle = test_oracle(data_test, gen_test)
print('Oracle averages a log-likelihood of %s +- %s on unseen tasks.' % (mean_oracle, std_oracle))

with open(working_directory.file('test_log_likelihood.txt'), 'w') as f:
    f.write(str(mean_oracle))
    
with open(working_directory.file('test_log_likelihood_standard_error.txt'), 'w') as f:
    f.write(str(std_oracle))
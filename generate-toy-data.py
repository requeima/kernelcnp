import argparse

import numpy as np
import stheno.torch as stheno
import torch
import matplotlib.pyplot as plt
import os
from datetime import datetime
from tqdm import trange
import pickle

# This is for an error that is now popping up when running on macos
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

import cnp.data

from itertools import product
from copy import deepcopy

from cnp.experiment import WorkingDirectory, log_args


# Parse arguments given to the script.
parser = argparse.ArgumentParser()


# =============================================================================
# Data generation arguments
# =============================================================================

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

parser.add_argument('--num_train_iters',
                    default=1,
                    type=int,
                    help='Iterations (# batches sampled) per training epoch.')

parser.add_argument('--num_valid_iters',
                    default=10,
                    type=int,
                    help='Iterations (# batches sampled) for validation.')

parser.add_argument('--num_test_iters',
                    default=2048,
                    type=int,
                    help='Iterations (# batches sampled) for testing.')

parser.add_argument('--validate_every',
                    default=1000,
                    type=int,
                    help='.')

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

parser.add_argument('--epochs',
                    default=10000,
                    type=int,
                    help='Number of epochs to train for.')


parser.add_argument('--root',
                    help='Experiment root, which is the directory from which '
                         'the experiment will run. If it is not given, '
                         'a directory will be automatically created.')

args = parser.parse_args()
    

data_kinds = ['eq',
              'matern',
              'noisy-mixture',
              'weakly-periodic',
              'sawtooth']


seeds = list(range(0, 2))

for seed in seeds:
    for data_kind in data_kinds:

        # =============================================================================
        # Set random seed and device
        # =============================================================================

        # Set seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        device = torch.device('cpu')

        path = os.path.join('_experiments', f'{data_kind}', 'data', f'{seed}')


        # =============================================================================
        # Create data generators
        # =============================================================================


        # Training data generator parameters -- used for both Sawtooth and GP
        gen_params = {
            'x_range'                   : args.x_range,
            'max_num_context'           : args.max_num_context,
            'max_num_target'            : args.max_num_target,
            'include_context_in_target' : False,
            'device'                    : device
        }

        # Training data generator parameters -- specific to Sawtooth
        gen_train_sawtooth_params = {
            'freq_range'  : args.freq_range,
            'shift_range' : args.shift_range,
            'trunc_range' : args.trunc_range
        }


        if data_kind == 'sawtooth':

            gen_train = cnp.data.SawtoothGenerator(args.num_train_iters,
                                                   batch_size=args.batch_size,
                                                   **gen_train_sawtooth_params,
                                                   **gen_params)

            gen_valid = cnp.data.SawtoothGenerator(args.num_valid_iters,
                                                   batch_size=args.batch_size,
                                                   **gen_train_sawtooth_params,
                                                   **gen_params)

        else:

            if data_kind == 'eq':
                kernel = stheno.EQ().stretch(args.eq_params[0])

            elif data_kind == 'matern':
                kernel = stheno.Matern52().stretch(args.m52_params[0])

            elif data_kind == 'noisy-mixture':
                kernel = stheno.EQ().stretch(args.mixture_params[0]) + \
                         stheno.EQ().stretch(args.mixture_params[1])

            elif data_kind == 'weakly-periodic':
                kernel = stheno.EQ().stretch(args.wp_params[0]) * \
                         stheno.EQ().periodic(period=args.wp_params[1])

            else:
                raise ValueError(f'Unknown generator kind "{data_kind}".')

            gen_train = cnp.data.GPGenerator(iterations_per_epoch=args.num_train_iters,
                                             batch_size=args.batch_size,
                                             kernel=kernel,
                                             std_noise=args.std_noise,
                                             **gen_params)

            gen_valid = cnp.data.GPGenerator(iterations_per_epoch=args.num_valid_iters,
                                             batch_size=args.batch_size,
                                             kernel=kernel,
                                             std_noise=args.std_noise,
                                             **gen_params)


        train_data = [[batch for batch in gen_train] for epoch in trange(args.epochs + 1)]
        valid_data = [[batch for batch in gen_valid] for epoch in trange(args.epochs // args.validate_every + 1)]

        wd = WorkingDirectory(root=path)
        log_args(wd, args)

        with open(wd.file('train-data.pkl'), 'wb') as file:
            pickle.dump(train_data, file)
            file.close()

        with open(wd.file('valid-data.pkl'), 'wb') as file:
            pickle.dump(valid_data, file)
            file.close()
        
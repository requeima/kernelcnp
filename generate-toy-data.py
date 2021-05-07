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

parser.add_argument('--test',
                    action='store_true',
                    help='Test the model and record the values in the'
                         'experimental root.')

parser.add_argument('--x_dims',
                    default=[1, 2],
                    nargs='+',
                    type=int,
                    help='Dimensions of x to loop over.')

parser.add_argument('--x_context_range',
                    default=[-2., 2.],
                    nargs='+',
                    type=float,
                    help='Range of inputs for sampled data.')

parser.add_argument('--x_target_range',
                    default=None,
                    nargs='+',
                    type=float,
                    help='Range of inputs for sampled data.')

parser.add_argument('--std_noise',
                    default=5e-2,
                    type=float,
                    help='Standard dev. of noise added to GP-generated data.')

parser.add_argument('--batch_size',
                    default=16,
                    type=int,
                    help='Number of tasks per batch sampled.')

parser.add_argument('--min_num_context',
                    default=50,
                    type=int,
                    help='Minimum number of context points.')

parser.add_argument('--max_num_context',
                    default=50,
                    type=int,
                    help='Maximum number of context points.')

parser.add_argument('--max_num_target',
                    default=50,
                    type=int,
                    help='Maximum number of target points.')

parser.add_argument('--num_train_iters',
                    default=1024,
                    type=int,
                    help='Iterations (# batches sampled) per training epoch.')

parser.add_argument('--num_valid_iters',
                    default=64,
                    type=int,
                    help='Iterations (# batches sampled) for validation.'
                         'Only used if generate_data_at_traintime is set to True.')

parser.add_argument('--num_test_iters',
                    default=2048,
                    type=int,
                    help='Iterations (# batches sampled) for testing.')

parser.add_argument('--validate_every',
                    default=5,
                    type=int,
                    help='.')

parser.add_argument('--epochs',
                    default=10,
                    type=int,
                    help='Number of epochs to train for.')

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
                    default=[1., 0.25],
                    nargs='+',
                    type=float,
                    help='.')

parser.add_argument('--wp_params',
                    default=[1., 0.25],
                    nargs='+',
                    type=float,
                    help='.')

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


parser.add_argument('--root',
                    help='Experiment root, which is the directory from which '
                         'the experiment will run. If it is not given, '
                         'a directory will be automatically created.')

args = parser.parse_args()


# data_kinds = ['eq',
#               'matern',
#               'noisy-mixture',
#               'weakly-periodic',
#               'sawtooth']

data_kinds = ['sawtooth']

seeds = list(range(0, 2))

for seed in seeds:
    for x_dim in args.x_dims:
        for data_kind in data_kinds:
            
            x_context_ranges = x_dim * [args.x_context_range]
            
            x_target_ranges = x_dim * [args.x_context_range] \
                                    if args.x_target_range is None else \
                              x_dim * [args.x_target_range]

            # =================================================================
            # Set random seed and device
            # =================================================================

            # Set seed
            np.random.seed(seed)
            torch.manual_seed(seed)

            device = torch.device('cpu')

            path = os.path.join('_experiments',
                                f'{data_kind}',
                                'data',
                                f'seed-{seed}',
                                f'dim-{x_dim}')


            # =================================================================
            # Create data generators
            # =================================================================


            # Training data generator parameters -- used for both Sawtooth and GP
            gen_params = {
                'min_num_context'           : args.min_num_context,
                'max_num_context'           : args.max_num_context,
                'max_num_target'            : args.max_num_target,
                'device'                    : device
            }

            # Training data generator parameters -- specific to Sawtooth
            gen_train_sawtooth_params = {
                'freq_range'  : args.freq_range,
                'shift_range' : args.shift_range,
                'trunc_range' : args.trunc_range
            }


            if data_kind == 'sawtooth':
                
                if x_dim > 1: continue

                else:

                    gen_train = cnp.data.SawtoothGenerator(args.num_train_iters,
                                                           batch_size=args.batch_size,
                                                           x_context_ranges=x_context_ranges,
                                                           x_target_ranges=x_target_ranges,
                                                           **gen_train_sawtooth_params,
                                                           **gen_params)

                    gen_valid = cnp.data.SawtoothGenerator(args.num_valid_iters,
                                                           batch_size=args.batch_size,
                                                           x_context_ranges=x_context_ranges,
                                                           x_target_ranges=x_target_ranges,
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
                                                 x_context_ranges=x_context_ranges,
                                                 x_target_ranges=x_target_ranges,
                                                 **gen_params)

                gen_valid = cnp.data.GPGenerator(iterations_per_epoch=args.num_valid_iters,
                                                 batch_size=args.batch_size,
                                                 kernel=kernel,
                                                 std_noise=args.std_noise,
                                                 x_context_ranges=x_context_ranges,
                                                 x_target_ranges=x_target_ranges,
                                                 **gen_params)


            wd = WorkingDirectory(root=path)
            
            train_data = [[batch for batch in gen_train] for epoch in trange(args.epochs + 1)]
            valid_data = [[batch for batch in gen_valid] for epoch in trange(args.epochs // args.validate_every + 1)]
            
            
            import os
            import sys
            import matplotlib.pyplot as plt
            
            xctx = train_data[0][0]['x_context'][0, :, 0]
            yctx = train_data[0][0]['y_context'][0, :, 0]
            
            plt.scatter(xctx, yctx)
            plt.savefig('./tmp/fig.pdf')
            plt.close()
            raise Exception

            with open(wd.file('train-data.pkl'), 'wb') as file:
                pickle.dump(train_data, file)
                file.close()

            with open(wd.file('valid-data.pkl'), 'wb') as file:
                pickle.dump(valid_data, file)
                file.close()

            with open(wd.file('gen-train.pkl'), 'wb') as file:
                pickle.dump(gen_train, file)
                file.close()

            with open(wd.file('gen-valid.pkl'), 'wb') as file:
                pickle.dump(gen_valid, file)
                file.close()

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
from cnp.utils import make_generator

from itertools import product
from copy import deepcopy

from cnp.experiment import WorkingDirectory, log_args


# Parse arguments given to the script.
parser = argparse.ArgumentParser()


# =============================================================================
# Make Random DataGenerator Function
# =============================================================================

def make_random_generator(gen_train_gp_params,
                          gen_train_sawtooth_params,
                          kernel_params):
    
    gp_data_kinds = ['eq',
                     'matern',
                     'noisy-mixture',
                     'weakly-periodic',]

    gen_list = []
    
    # Generate sawtooth seperately
    gen  = make_generator('sawtooth', gen_train_sawtooth_params, None)
    gen_list.append(gen)

    
    for dk in gp_data_kinds:
        gen  = make_generator(dk, gen_train_gp_params, kernel_params)
        gen_list.append(gen)
    
    return gen_list

# =============================================================================
# Data generation arguments
# =============================================================================

parser.add_argument('--test',
                    action='store_true',
                    help='Test the model and record the values in the'
                         'experimental root.')

parser.add_argument('--x_dims',
                    default=[1],
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
                    default=1,
                    type=int,
                    help='Number of tasks per batch sampled.')

parser.add_argument('--max_num_context',
                    default=10,
                    type=int,
                    help='Maximum number of context points.')

parser.add_argument('--min_num_target',
                    default=50,
                    type=int,
                    help='Maximum number of target points.')

parser.add_argument('--max_num_target',
                    default=50,
                    type=int,
                    help='Maximum number of target points.')

parser.add_argument('--num_test_iters',
                    default=5000,
                    type=int,
                    help='Iterations (# batches sampled) per training epoch.')

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
#               'sawtooth',
#               'random']

data_kinds = ['eq',
              'matern',
              'noisy-mixture',
              'weakly-periodic',
              'sawtooth']

seeds = list(range(0, 2))

for x_dim in args.x_dims:
    for seed in seeds:
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

            path = os.path.join('_experiments/toy-data',
                                f'{data_kind}',
                                'data',
                                f'seed-{seed}',
                                f'dim-{x_dim}')


            # =================================================================
            # Create data generators
            # =================================================================
        

            # Training data generator parameters -- used for both Sawtooth and GP
            gen_params = {
                'batch_size'                : args.batch_size,
                'x_context_ranges'          : x_context_ranges,
                'x_target_ranges'           : x_target_ranges,
                'max_num_context'           : args.max_num_context,
                'min_num_target'            : args.min_num_target,
                'max_num_target'            : args.max_num_target,
                'device'                    : device,
                'iterations_per_epoch'      : args.num_test_iters
            }

            # Training data generator parameters -- specific to Sawtooth
            gen_test_sawtooth_params = gen_params.copy()
            gen_test_sawtooth_params.update({
                'freq_range'  : args.freq_range,
                'shift_range' : args.shift_range,
                'trunc_range' : args.trunc_range
            })

            # Training data generator parameters -- specific to GPs
            gen_test_gp_params = gen_params.copy()
            gen_test_gp_params.update({
                'std_noise' : args.std_noise
            })

            wd = WorkingDirectory(root=path)

            kernel_params = {'eq'              : args.eq_params,
                             'matern'          : args.m52_params,
                             'noisy-mixture'   : args.mixture_params,
                             'weakly-periodic' : args.wp_params
            }

            if data_kind == 'sawtooth':
                if x_dim > 1: continue

                gen_test = make_generator(data_kind, gen_test_sawtooth_params, None)            

                test_data = [batch for batch in gen_test]

            elif data_kind == 'random':
                if x_dim > 1: continue

                gen_test = make_random_generator(gen_test_gp_params,
                                                  gen_test_sawtooth_params, 
                                                  kernel_params
                )

                test_idx = np.random.randint(5, size=(args.num_test_iters + 1))


                test_data = []
                for i in range(args.batch_size):
                    # The batch size is set to 1 in the generator
                    test_data.append([b for b in gen_test[test_idx[i]]][0])

            else:
                gen_test = make_generator(data_kind, gen_test_gp_params, kernel_params)
                test_data = gen_test.pregen_epoch()

                # Save the generating parameters
                with open(wd.file('gen-test-dict.pkl'), 'wb') as file:
                    pickle.dump(gen_test_gp_params, file)
                    file.close()

                # Save the kernel parameters
                temp = {data_kind: kernel_params[data_kind]}
                with open(wd.file('kernel-params.pkl'), 'wb') as file:
                    pickle.dump(temp, file)
                    file.close()

            with open(wd.file('test-data.pkl'), 'wb') as file:
                pickle.dump(test_data, file)
                file.close()
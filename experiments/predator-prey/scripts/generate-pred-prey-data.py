import argparse

import numpy as np
import stheno.torch as stheno
import torch
import matplotlib.pyplot as plt
import os
from datetime import datetime
from tqdm import trange
import pickle
import itertools

import cnp.data
from cnp.utils import make_generator

from itertools import product
from copy import deepcopy

from cnp.experiment import WorkingDirectory, log_args


# Parse arguments given to the script.
parser = argparse.ArgumentParser()

# =============================================================================
# Data generation arguments
# =============================================================================

parser.add_argument('--batch_size',
                    default=16,
                    type=int,
                    help='Number of tasks per batch sampled.')

parser.add_argument('--max_num_context',
                    default=50,
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

parser.add_argument('--num_train_iters',
                    default=1024,
                    type=int,
                    help='Iterations (# batches sampled) per training epoch.')

parser.add_argument('--num_valid_iters',
                    default=32,
                    type=int,
                    help='Iterations (# batches sampled) for validation.')

parser.add_argument('--num_test_iters',
                    default=1024,
                    type=int,
                    help='Iterations (# batches sampled).')

parser.add_argument('--validate_every',
                    default=10,
                    type=int,
                    help='.')

parser.add_argument('--epochs',
                    default=100,
                    type=int,
                    help='Number of epochs to train for.')

parser.add_argument('--root',
                    help='Experiment root, which is the directory from which '
                         'the experiment will run. If it is not given, '
                         'a directory will be automatically created.')

parser.add_argument('--test',
                    action='store_true',
                    default=False,
                    help='Whether to generate test data (default train).')

args = parser.parse_args()

seeds = list(range(0, 1))

input(f'About to do batch size {args.batch_size}. '
      f'Enter to continue, Ctrl+C to cancel generation.')

for seed in seeds:

    # =================================================================
    # Set random seed and device
    # =================================================================

    # Set seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    if args.test:
        seed_ = np.random.randint(int(1e6), int(2e6))

        np.random.seed(seed_)
        torch.manual_seed(seed_)


    device = torch.device('cpu')

    root = 'experiments/synthetic/'
    data_name = f'sim-pred-prey-'          + \
                f'{args.batch_size}-'      + \
                f'{args.max_num_context}-' + \
                f'{args.min_num_target}-'  + \
                f'{args.max_num_target}-'  + \
                f'{seed}'

    path = os.path.join(root, 'toy-data', data_name)

    # =========================================================================
    # Create data generators
    # =========================================================================


    # Training data generator parameters, used for both Sawtooth and GP
    gen_params = {
        'batch_size'       : args.batch_size,
        'x_context_ranges' : [0, 100],
        'max_num_context'  : args.max_num_context,
        'min_num_target'   : args.min_num_target,
        'max_num_target'   : args.max_num_target,
        'device'           : device,
        'x_target_ranges'  : None
    }

    if args.test:
        
        gen_params.update({'iterations_per_epoch' : args.num_test_iters})

        gen_test = PredatorPreyGenerator(**gen_params)
        test_data = gen_test.pregen_epoch()

    else:
        
        gen_train_params.update(
            {'iterations_per_epoch' : args.num_train_iters}
        )
        
        gen_valid_params.update(
            {'iterations_per_epoch' : args.num_valid_iters}
        )

        gen_train = PredatorPreyGenerator(**gen_train_params)
        gen_valid = PredatorPreyGenerator(**gen_valid_params)  

        train_data = [gen_train.pregen_epoch() \
                      for i, epoch in enumerate(trange(args.epochs + 1))]
        
        valid_data = [gen_valid.pregen_epoch() \
                      for i, epoch in enumerate(trange(args.epochs // args.validate_every + 1))]

if args.test:
    with open(wd.file('test-data.pkl'), 'wb') as file:
        pickle.dump(test_data, file)
        file.close()

else:
    with open(wd.file('train-data.pkl'), 'wb') as file:
        pickle.dump(train_data, file)
        file.close()

    with open(wd.file('valid-data.pkl'), 'wb') as file:
        pickle.dump(valid_data, file)
        file.close()

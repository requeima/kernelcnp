import argparse
from matrix import diagonal

import numpy as np
# import stheno.torch as stheno
import torch
import matplotlib.pyplot as plt
import os
from datetime import datetime
import pickle

from torch._C import Value
import time

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

from cnp.oracle import (
    eq_cov,
    mat_cov,
    nm_cov,
    wp_cov,
    oracle_loglik
)

from stheno import (
    EQ,
    Matern52
)

from cnp.utils import make_generator

from torch.distributions import MultivariateNormal

torch.set_default_dtype(torch.double)


# =============================================================================
# Test oracle helper
# =============================================================================


def test_oracle(data, covariance, noise):
    """ Compute the oracle test loss. """
    
    oracle_ll_list = []
    diag_oracle_ll_list = []
    
    with torch.no_grad():
        for step, batch in enumerate(data):

            for b in range(batch['x_context'].shape[0]):
                logliks = oracle_loglik(batch['x_context'][b].cpu().detach().numpy(),
                                        batch['y_context'][b].cpu().detach().numpy(),
                                        batch['x_target'][b].cpu().detach().numpy(),
                                        batch['y_target'][b].cpu().detach().numpy(),
                                        covariance=covariance,
                                        noise=noise)
                logprob, diag_logprob = logliks

                oracle_ll_list.append(logprob / 100.)
                diag_oracle_ll_list.append(diag_logprob / 100.)

            if step % 100 == 0 or step == len(data) - 1:

                n = len(oracle_ll_list)
                
                print(f"{args.data} step {step} \n"
                      f"Oracle full-cov log-lik: "
                      f"{np.mean(oracle_ll_list):.2f} +/- "
                      f"{np.var(oracle_ll_list) ** 0.5 / n**0.5:.2f}, diag:"
                      f"{np.mean(diag_oracle_ll_list):.2f} +/- "
                      f"{np.var(diag_oracle_ll_list) ** 0.5 / n**0.5:.2f}")

    mean_oracle = np.mean(oracle_ll_list)
    std_oracle = (np.var(oracle_ll_list) ** 0.5) / np.sqrt(step + 1)

    mean_diag_oracle = np.mean(diag_oracle_ll_list)
    std_diag_oracle = (np.var(diag_oracle_ll_list) ** 0.5) / np.sqrt(step + 1)

    return mean_oracle, std_oracle, mean_diag_oracle, std_diag_oracle


# Parse arguments given to the script.
parser = argparse.ArgumentParser()

# =============================================================================
# Data generation arguments
# =============================================================================

parser.add_argument('data',
                    type=str,
                    help='Data set to train the CNP on. ')

parser.add_argument('--x_dim',
                    default=1,
                    choices=[1, 2],
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

root = 'experiments/synthetic'

data_root = os.path.join(f'{root}',
                         f'toy-data',
                         f'{args.data}')

experiment_name = os.path.join(f'{root}',
                               f'results',
                               f'{args.data}',
                               f'models',
                               f'oracle',
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

noise = 5e-2

if 'eq' in args.data:
    oracle_cov = EQ().stretch(1.)

elif 'matern' in args.data:
    oracle_cov = Matern52().stretch(1.)

elif 'noisy-mixture-slow' in args.data:
    oracle_cov = EQ().stretch(1.) + \
                 EQ().stretch(0.5)

elif 'weakly-periodic-slow' in args.data:
    oracle_cov = EQ().stretch(1.) * \
                 EQ().periodic(period=0.5)
        
elif 'noisy-mixture' in args.data:
    oracle_cov = EQ().stretch(1.) + \
                 EQ().stretch(0.25)

elif 'weakly-periodic' in args.data:
    oracle_cov = EQ().stretch(1.) * \
                 EQ().periodic(period=0.25)


# =============================================================================
# Test oracle
# =============================================================================

start_time = time.time()
test_result = test_oracle(data_test, oracle_cov, noise)
end_time = time.time()
elapsed_time = end_time - start_time
# Record experiment time\
with open(working_directory.file('test_time.txt'), 'w') as f:
    f.write(str(elapsed_time))

mean_oracle, std_oracle, mean_diag_oracle, std_diag_oracle = test_result
print(f'Oracle averages a log-likelihood of {mean_oracle:.4f} '
      f'+- {std_oracle:.4f} on unseen tasks.')

with open(working_directory.file('test_nll_mean.txt'), 'w') as f:
    f.write(str(mean_oracle))
    
with open(working_directory.file('test_nll_error.txt'), 'w') as f:
    f.write(str(std_oracle))

with open(working_directory.file('test_diag_nll_mean.txt'), 'w') as f:
    f.write(str(mean_diag_oracle))
    
with open(working_directory.file('test_diag_nll_std.txt'), 'w') as f:
    f.write(str(std_diag_oracle))

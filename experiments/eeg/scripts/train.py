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
from cnp.lnp import StandardEEGConvNP

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
from torch.utils.tensorboard import SummaryWriter

# =============================================================================
# Training epoch helper
# =============================================================================

def train(data_train,
          model,
          optimiser,
          log_every,
          device,
          writer,
          iteration,
          args):
    
    for batch in data_train:
        
        nll = model.loss(batch['x_context'].to(device),
                         batch['y_context'].to(device),
                         batch['m_context'].to(device),
                         batch['x_target'].to(device),
                         batch['y_target'].to(device),
                         batch['m_target'].to(device))
            
        nll = nll / (args.num_channels_target * args.target_length)

        if iteration % log_every == 0:
            print(f"Training   neg. log-lik: {nll:.2f}")
        
        nll.backward()
        optimiser.step()
        optimiser.zero_grad()

        # Write to tensorboard
        writer.add_scalar('Train log-lik.', -nll, iteration)
        
        iteration = iteration + 1
        
    return iteration


# =============================================================================
# Validation helper
# =============================================================================


def validate(data_valid,
             model,
             device,
             writer,
             args):
    
    nlls = []
    
    with torch.no_grad():
        for batch in data_valid:
        
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
    print(f"Validation neg. log-lik: {mean_nll:.2f}")
    
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

parser.add_argument('--np_samples',
                    default=32,
                    type=int,
                    help='Number of samples used for ConvNP.')

parser.add_argument('--validate_every',
                    default=1,
                    type=int,
                    help='Number of epochs between validations.')


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
                    choices=['hetero'],
                    help='Choice of noise model.')

parser.add_argument('--init_length_scale',
                    default=1e-3,
                    type=float)

parser.add_argument('--num_basis_dim',
                    default=512,
                    type=int,
                    help='Number of embedding basis dimensions.')

parser.add_argument('--learning_rate',
                    default=5e-4,
                    type=float,
                    help='Learning rate.')

parser.add_argument('--jitter',
                    default=1e-3,
                    type=float,
                    help='The jitter level.')

parser.add_argument('--weight_decay',
                    default=0.,
                    type=float,
                    help='Weight decay.')

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

# Tensorboard writer
writer = SummaryWriter(f'{experiment_name}/log')
    
# =============================================================================
# Create model
# =============================================================================

cov_types = {
    'meanfield' : MultiOutputMeanFieldGaussianLayer,
    'innerprod' : MultiOutputInnerprodGaussianLayer,
    'kvv'       : MultiOutputKvvGaussianLayer
}

if args.model == 'convGNP':

	if args.cov_type == 'meanfield':
		output_layer = cov_types['meanfield'](num_outputs=args.num_channels_total)
		
	else:
		output_layer = cov_types[args.cov_type](num_outputs=args.num_channels_total,
												num_embedding=args.num_basis_dim,
												noise_type=args.noise_type,
												jitter=args.jitter)
    
	model = StandardEEGConvGNP(num_channels=args.num_channels_total,
							   output_layer=output_layer)

else:

	output_layer = cov_types['meanfield'](num_outputs=args.num_channels_total)

	model = StandardEEGConvNP(num_channels=args.num_channels_total,
							  output_layer=output_layer,
							  num_samples=args.np_samples)


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
    
data_train = EEGGenerator(split='train',
                          batch_size=args.batch_size,
                          batches_per_epoch=args.batches_per_epoch,
                          num_total_channels=args.num_channels_total,
                          num_target_channels=args.num_channels_target,
                          target_length=args.target_length,
                          device=device)
    
data_valid = EEGGenerator(split='valid',
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

# Create optimiser
optimiser = torch.optim.Adam(model.parameters(),
                             args.learning_rate,
                             weight_decay=args.weight_decay)

# Run the training loop, maintaining the best objective value
best_nll = np.inf

start_time = time.time()
for epoch in range(args.epochs):

    print('\nEpoch: {}/{}'.format(epoch + 1, args.epochs))

    if epoch % args.validate_every == 0:

        # Compute negative log-likelihood on validation data
        val_nll = validate(data_valid,
                           model,
                           device,
                           writer,
                           args)

        # Log information to tensorboard
        writer.add_scalar('Valid log-lik.', -val_nll, epoch)

        # Update the best objective value and checkpoint the model
        is_best, best_obj = (True, val_nll) if val_nll < best_nll else \
                            (False, best_nll)

        save_checkpoint(working_directory,
                        {'epoch'         : epoch + 1,
                         'state_dict'    : model.state_dict(),
                         'best_acc_top1' : best_obj,
                         'optimizer'     : optimiser.state_dict()},
                        is_best=is_best,
                        epoch=epoch)

    # Compute training negative log-likelihood
    train_iteration = train(data_train,
                            model,
                            optimiser,
                            log_every,
                            device,
                            writer,
                            train_iteration,
                            args)

end_time = time.time()
elapsed_time = end_time - start_time

# Record experiment time\
with open(working_directory.file('train_time.txt'), 'w') as f:
    f.write(str(elapsed_time))

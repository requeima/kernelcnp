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

from cnp.cnp import StandardPredPreyConvGNP # StandardConvGNP
from cnp.lnp import StandardConvNP

from cnp.cov import (
    MeanFieldGaussianLayer,
    InnerprodGaussianLayer,
    KvvGaussianLayer,
    ExponentialCopulaLayer,
    LogLogitCopulaLayer
)

from cnp.utils import (
    Logger,
    plot_pred_prey_fits
)

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.tensorboard import SummaryWriter

# =============================================================================
# Training epoch helper
# =============================================================================

def train(data,
          model,
          optimiser,
          log_every,
          device,
          writer,
          iteration):
    
    for step, batch in enumerate(data):

        nll = model.loss(batch['x_context'][:, :, None].to(device),
                         batch['y_context'][:, 0, :, None].to(device) / 100 + 1e-2,
                         batch['x_target'][:, :, None].to(device),
                         batch['y_target'][:, 0, :, None].to(device) / 100 + 1e-2)

        # Log information to tensorboard
        writer.add_scalar('Training data log-lik.', -nll, epoch)
        
        encoder_scale = torch.exp(model.encoder.sigma).detach().cpu().numpy().squeeze()
        decoder_scale = torch.exp(model.decoder.sigma).detach().cpu().numpy().squeeze()

        if step % log_every == 0:
            print(f"Training   neg. log-lik: {nll:.2f}, "
                  f"Encoder/decoder scales {encoder_scale:.3f} "
                  f"{decoder_scale:.3f}")

        # Compute gradients and apply them
        nll.backward()
        optimiser.step()
        optimiser.zero_grad()
        
        iteration = iteration + 1
        
    return iteration


# =============================================================================
# Validation helper
# =============================================================================


def validate(data,
             data_holdout,
             data_subsampled,
             model,
             device,
             writer,
             latent_model,
             figure_path):
    
    # Lists for logging model's training NLL and oracle NLL
    nll_list = []
    nll_holdout_list = []
    nll_subsampled_list = []
    oracle_nll_list = []
    
    # If training a latent model, set the number of latent samples accordingly
    loss_kwargs = {'num_samples' : args_np_val_samples} \
                  if latent_model else {}
    
    with torch.no_grad():
        for step, batch in enumerate(data):
            
            nll = model.loss(batch['x_context'][:, :, None].to(device),
                             batch['y_context'][:, 0, :, None].to(device) / 100 + 1e-2,
                             batch['x_target'][:, :, None].to(device),
                             batch['y_target'][:, 0, :, None].to(device) / 100 + 1e-2,
                             **loss_kwargs)
            
            nll_list.append(nll.item())
            
        for step, batch in enumerate(data_holdout):
            
            nll_holdout = model.loss(batch['x_context'].to(device),
                                     batch['y_context'].to(device) / 100 + 1e-2,
                                     batch['x_target'].to(device),
                                     batch['y_target'].to(device) / 100 + 1e-2,
                                  **loss_kwargs)
            
            nll_holdout_list.append(nll_holdout.item())
            
        for step, batch in enumerate(data_subsampled):
            
            nll_subsampled = model.loss(batch['x_context'].to(device),
                                        batch['y_context'].to(device) / 100 + 1e-2,
                                        batch['x_target'].to(device),
                                        batch['y_target'].to(device) / 100 + 1e-2,
                                        **loss_kwargs)
            
            nll_subsampled_list.append(nll_subsampled.item())

    mean_nll = np.mean(nll_list)
    mean_holdout_nll = np.mean(nll_holdout_list)
    mean_subsampled_nll = np.mean(nll_subsampled_list)

    # Print validation loss and oracle loss
    print(f"Validation data neg. log-lik: "
          f"{mean_nll:.2f}")
    
    print(f"Holdout    data neg. log-lik: "
          f"{mean_holdout_nll:.2f}")
    
    print(f"Subsampled data neg. log-lik: "
          f"{mean_subsampled_nll:.2f}")
    
    plot_pred_prey_fits(model=model,
                        valid_data=data[0],
                        holdout_data=data_holdout,
                        subsampled_data=data_subsampled,
                        num_noisy_samples=200,
                        num_noiseless_samples=3,
                        device=device,
                        save_path=figure_path)

    return mean_nll, mean_holdout_nll, mean_subsampled_nll
        

# Parse arguments given to the script.
parser = argparse.ArgumentParser()


# =============================================================================
# Data generation arguments
# =============================================================================

parser.add_argument('data', help='Data set to train the CNP on.')

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
                    choices=['convGNP', 'convNP'],
                    help='Choice of model. ')

parser.add_argument('cov_type',
                    choices=['meanfield', 'innerprod',  'kvv'],
                    help='Choice of covariance method.')

parser.add_argument('noise_type',
                    choices=['homo', 'hetero'],
                    help='Choice of noise model.')

parser.add_argument('--marginal_type',
                    default='identity',
                    choices=['identity', 'exponential', 'loglogit'],
                    help='Choice of marginal transformation (optional).')

parser.add_argument('--np_loss_samples',
                    default=16,
                    type=int,
                    help='Number of latent samples for evaluating the loss, '
                         'used for ANP and ConvNP.')

parser.add_argument('--exponential_scale',
                    default=4.,
                    type=float,
                    help='Exponential decay parameter for exponential copula.')

parser.add_argument('--points_per_unit',
                    default=32,
                    type=int)

parser.add_argument('--init_length_scale',
                    default=1e-1,
                    type=float)

parser.add_argument('--np_val_samples',
                    default=16,
                    type=int,
                    help='Number of latent samples for evaluating the loss, '
                         'when validating, used for ANP and ConvNP.')

parser.add_argument('--num_basis_dim',
                    default=32,
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

root = 'experiments/predator-prey'

# Working directory for saving results
experiment_name = os.path.join(f'{root}',
                               f'results',
                               f'{args.data}',
                               f'models',
                               f'{args.model}',
                               f'{args.cov_type}',
                               f'{args.noise_type}',
                               f'{args.marginal_type}',
                               f'seed-{args.seed}')
working_directory = WorkingDirectory(root=experiment_name)

# Data directory for loading data
data_root = os.path.join(f'{root}',
                         f'simulated-data',
                         f'{args.data}')
data_directory = WorkingDirectory(root=data_root)

# Data directory for loading data
true_data_root = os.path.join(f'{root}', 'data')
true_data_root = WorkingDirectory(root=true_data_root)

log_path = f'{root}/logs'
log_filename = f'{args.data}-'          + \
               f'{args.model}-'         + \
               f'{args.cov_type}-'      + \
               f'{args.noise_type}-'    + \
               f'{args.marginal_type}'
                
log_directory = WorkingDirectory(root=log_path)
sys.stdout = Logger(log_directory=log_directory, log_filename=log_filename)
sys.stderr = Logger(log_directory=log_directory, log_filename=log_filename)

# Tensorboard writer
writer = SummaryWriter(f'{experiment_name}/log')
    
file = open(working_directory.file('data_location.txt'), 'w')
file.write(data_directory.root)
file.close()
    
# =============================================================================
# Create model
# =============================================================================

cov_types = {
    'meanfield' : MeanFieldGaussianLayer,
    'innerprod' : InnerprodGaussianLayer,
    'kvv'       : KvvGaussianLayer
}

if args.cov_type == 'meanfield':
    output_layer = MeanFieldGaussianLayer()
    
else:
    output_layer = cov_types[args.cov_type](num_embedding=args.num_basis_dim,
                                            noise_type=args.noise_type)

if args.marginal_type == 'exponential':
    print('Exponential marginals')
    output_layer = ExponentialCopulaLayer(gaussian_layer=output_layer,
                                          scale=args.exponential_scale,
                                          device=device)
    
elif args.marginal_type == 'loglogit':
    print('Log-logistic marginals')
    output_layer = LogLogitCopulaLayer(gaussian_layer=output_layer)
    
else:
    print('Gaussian marginals')
    
# Create model architecture
if args.model == 'convGNP':
    model = StandardPredPreyConvGNP(input_dim=1,
                                    output_layer=output_layer)
    
elif args.model == 'convNP':
    model = StandardConvNP(input_dim=1,
                           num_samples=args.np_loss_samples)
    
else:
    raise ValueError(f'Unknown model {args.model}.')


print(f'{args.data} '
      f'{args.model} '
      f'{args.cov_type} '
      f'{args.noise_type} '
      f'{args.marginal_type} '
      f'{args.num_basis_dim}: '
      f'{model.num_params}')

with open(working_directory.file('num_params.txt'), 'w') as f:
    f.write(f'{model.num_params}')
        
if args.num_params:
    exit()
    
# Load model to appropriate device
model = model.to(device)

latent_model = args.model == 'convNP'


# =============================================================================
# Load data and validation oracle generator
# =============================================================================
    
file = open(data_directory.file('train-data.pkl'), 'rb')
data_train = pickle.load(file)
file.close()

file = open(data_directory.file('valid-data.pkl'), 'rb')
data_val = pickle.load(file)
file.close()

file = open(true_data_root.file('subsampled_tasks.pkl'), 'rb')
data_subsampled = pickle.load(file)
file.close()

file = open(true_data_root.file('holdout_tasks.pkl'), 'rb')
data_holdout = pickle.load(file)
file.close()

# =============================================================================
# Train or test model
# =============================================================================

# Number of epochs between validations
train_iteration = 0
log_every = 100
    
log_args(working_directory, args)

# Create optimiser
optimiser = torch.optim.Adam(model.parameters(),
                         args.learning_rate,
                         weight_decay=args.weight_decay)

# Run the training loop, maintaining the best objective value
best_nll = np.inf

epochs = len(data_train)

start_time = time.time()
for epoch in range(epochs):

    print('\nEpoch: {}/{}'.format(epoch + 1, epochs))

    if epoch % args.validate_every == 0:

        valid_epoch = data_val[epoch // args.validate_every]
        
        figure_path = f'{experiment_name}/figures/{epoch:04d}.pdf'
        if not os.path.exists(f'{experiment_name}/figures'):
            os.mkdir(f'{experiment_name}/figures')

        # Compute negative log-likelihood on validation data
        result = validate(valid_epoch,
                          data_holdout,
                          data_subsampled,
                          model,
                          device,
                          None,
                          latent_model,
                          figure_path)
        
        val_nll, val_holdout_nll, val_subsampled_nll = result

        # Log information to tensorboard
        writer.add_scalar('Holdout data log-lik.',
                          -val_holdout_nll,
                          epoch)
        
        # Log information to tensorboard
        writer.add_scalar('Subsampled data log-lik.',
                          -val_subsampled_nll,
                          epoch)


        # Log information to tensorboard
        writer.add_scalar('Validation log-lik.',
                          -val_nll,
                          epoch)

        # Update the best objective value and checkpoint the model
        is_best, best_obj = (True, val_nll) if val_nll < best_nll else \
                            (False, best_nll)

    train_epoch = data_train[epoch]

    # Compute training negative log-likelihood
    train_iteration = train(train_epoch,
                            model,
                            optimiser,
                            log_every,
                            device,
                            writer,
                            train_iteration)

    save_checkpoint(working_directory,
                    {'epoch'         : epoch + 1,
                     'state_dict'    : model.state_dict(),
                     'best_acc_top1' : best_obj,
                     'optimizer'     : optimiser.state_dict()},
                    is_best=is_best,
                    epoch=epoch)

end_time = time.time()
elapsed_time = end_time - start_time

# Record experiment time\
with open(working_directory.file('train_time.txt'), 'w') as f:
    f.write(str(elapsed_time))

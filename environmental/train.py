import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

from datetime import datetime
import pickle

import torch
from torch.utils.tensorboard import SummaryWriter

from cnp.experiment import (
    generate_root,
    WorkingDirectory,
    save_checkpoint,
    log_args
)

from cnp.cnp import StandardConvGNP

from cnp.lnp import StandardConvNP

from cnp.cov import (
    InnerProdCov,
    KvvCov,
    AddHomoNoise,
    AddHeteroNoise
)

from cnp.data import EnvironmentalDataloader


# =============================================================================
# Training epoch helper
# =============================================================================


def train(data, model, optimiser, log_every, device, writer, iteration):
    
    for step, batch in enumerate(data):

        nll = model.loss(batch['x_context'].to(device),
                         batch['y_context'].to(device),
                         batch['x_target'].to(device),
                         batch['y_target'].to(device))

        if step % log_every == 0:
            print(f"Training   neg. log-lik: {nll:.2f}")

        # Compute gradients and apply them
        nll.backward()
        optimiser.step()
        optimiser.zero_grad()

        # Write to tensorboard
        writer.add_scalar('Train log-lik.', - nll, iteration)
        iteration = iteration + 1
        
    return iteration


# =============================================================================
# Validation helper
# =============================================================================


def validate(data, model, args, device, writer, latent_model):
    
    # Lists for logging model's training NLL and oracle NLL
    nll_list = []
    
    # If training a latent model, set the number of latent samples accordingly
    loss_kwargs = {'num_samples' : args.np_val_samples} if latent_model else {}
    
    with torch.no_grad():
        for step, batch in enumerate(data):
            
            nll = model.loss(batch['x_context'].to(device),
                             batch['y_context'].to(device),
                             batch['x_target'].to(device),
                             batch['y_target'].to(device),
                             **loss_kwargs)
            
            # Scale by the average number of target points
            nll_list.append(nll.item())
            
    mean_nll = np.mean(nll_list)
    std_nll = np.var(nll_list)**0.5
    
    # Print validation loss and oracle loss
    print(f"Validation neg. log-lik: "
          f"{mean_nll:.2f} +/- "
          f"{std_nll:.2f}")

    return mean_nll, std_nll
        

# =============================================================================
# Model arguments
# =============================================================================

parser.add_argument('model',
                    choices=['convGNP', 'convNP'],
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

parser.add_argument('--np_val_samples',
                    default=1024,
                    type=int,
                    help='Number of latent samples for evaluating the loss, '
                         'when validating, used for ANP and ConvNP.')

parser.add_argument('--num_basis_dim',
                    default=512,
                    type=int,
                    help='Number of embedding basis dimensions.')


# =============================================================================
# Training arguments
# =============================================================================

parser.add_argument('--learning_rate',
                    default=5e-4,
                    type=float,
                    help='Learning rate.')

parser.add_argument('--weight_decay',
                    default=0.,
                    type=float,
                    help='Weight decay.')

parser.add_argument('--validate_every',
                    default=10,
                    type=int,
                    help='Number of epochs between validations.')

parser.add_argument('--epochs',
                    default=100,
                    type=int,
                    help='Number of epochs to train for.')

parser.add_argument('--iterations_per_epoch',
                    default=1024,
                    type=int,
                    help='Number of iterations in each epoch.')

parser.add_argument('--batch_size',
                    default=16,
                    type=int,
                    help='Number of datasets in each batch.')


# =============================================================================
# Experiment arguments
# =============================================================================

parser.add_argument('--root',
                    help='Experiment root, which is the directory from which '
                         'the experiment will run. If it is not given, '
                         'a directory will be automatically created.')

parser.add_argument('--seed',
                    default=0,
                    type=int,
                    help='Random seed to use.')

parser.add_argument('--num_params',
                    action='store_true',
                    help='Print the total number of parameters in the moodel '
                         'and exit.')

parser.add_argument('--gpu',
                    default=0,
                    type=int,
                    help='GPU to run experiment on. Defaults to 0.')

parser = argparse.ArgumentParser()


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

data_root = os.path.join('environmental', 'data')
data_directory = WorkingDirectory(root=data_root)

experiment_name = os.path.join(f'environmental',
                               f'results',
                               f'models',
                               f'{args.model}',
                               f'{args.covtype}',
                               f'seed-{args.seed}')
working_directory = WorkingDirectory(root=experiment_name)

writer = SummaryWriter(f'{experiment_name}/log')

# Log all arguments passed to the experiment script
log_args(working_directory, args)
    

# =============================================================================
# Load training and validation data
# =============================================================================

# Training set name, subsample even-index times for train, odd for validation
train_dataset = 'era5land_daily_central_eu_complete.nc'
train_subsampler = lambda t : t % 2 == 0
valid_subsampler = lambda t : t % 2 == 1

# Test dataset locations, including one for validation
test_datasets = [train_dataset] + \
                ['era5land_daily_east_eu_test.nc',
                 'era5land_daily_north_eu_test.nc',
                 'era5land_daily_west_eu_test.nc']
test_subsamplers = [valid_subsampler] + \
                   [lambda t : True for _ in test_datasets]

# Create training dataloader
train_dataloader = EnvironmentalDataloader(train_dataset,
                                           args.iterations_per_epoch,
                                           args.min_num_context,
                                           args.max_num_context,
                                           args.min_num_target,
                                           args.max_num_target,
                                           args.batch_size,
                                           train_subsampler,
                                           scale_inputs_by=None)
scale_inputs_by = train_dataloader.scale_by

# Create validation/testing dataloaders
test_dataloaders = [EnvironmentalDataloader(dataset,
                                            args.iterations_per_epoch,
                                            args.min_num_context,
                                            args.max_num_context,
                                            args.min_num_target,
                                            args.max_num_target,
                                            args.batch_size,
                                            subsampler,
                                            scale_inputs_by=scale_inputs_by)
                     for dataset, subsampler in zip(]


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
if args.model == 'convGNP':
    model = StandardConvGNP(input_dim=2,
                            covariance=cov,
                            add_noise=noise)
   
elif args.model == 'convNP':
    
    noise = AddHomoNoise()
    model = StandardConvNP(input_dim=2,
                           add_noise=noise,
                           num_samples=args.np_loss_samples)
    
else:
    raise ValueError(f'Unknown model {args.model}.')


# Print model to the log
print(f'{args.model} '
      f'{args.covtype}: '
      f'{model.num_params}')

with open(working_directory.file('num_params.txt'), 'w') as f:
    f.write(f'{model.num_params}')
        
if args.num_params:
    exit()
    
# Load model to appropriate device
model = model.to(device)


# =============================================================================
# Train or test model
# =============================================================================

# Number of epochs between validations
train_iteration = 0
log_every = 1
    
# Create optimiser
optimiser = torch.optim.Adam(model.parameters(),
                         args.learning_rate,
                         weight_decay=args.weight_decay)

# Run the training loop, maintaining the best objective value
best_nll = np.inf

for epoch in range(args.epochs):

    if train_iteration % log_every == 0:
        print('\nEpoch: {}/{}'.format(epoch + 1, epochs))

    if epoch % args.validate_every == 0:

        # Compute validation negative log-likelihood
        val_nll, _ = validate(valid_dataloader,
                              model,
                              args,
                              device,
                              writer)

        # Log information to tensorboard
        writer.add_scalar('Valid log-lik.',
                          -val_nll,
                          epoch)

        # Update the best objective value and checkpoint the model
        is_best, best_obj = (True, val_nll) if val_nll < best_nll else \
                            (False, best_nll)

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

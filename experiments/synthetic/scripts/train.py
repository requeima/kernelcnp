import argparse

import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import pickle
import time
import sys

from stheno import (
    EQ,
    Matern52
)

# This is for an error that is now popping up when running on macos
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

from copy import deepcopy

from cnp.experiment import (
    generate_root,
    WorkingDirectory,
    save_checkpoint,
    log_args
)

from cnp.cnp import (
    StandardGNP,
    StandardAGNP,
    StandardConvGNP,
    FullConvGNP
)

from cnp.lnp import (
    StandardANP,
    StandardConvNP
)

from cnp.cov import (
    MeanFieldGaussianLayer,
    InnerprodGaussianLayer,
    KvvGaussianLayer,
    LogLogitCopulaLayer
)

from cnp.oracle import oracle_loglik

from cnp.utils import (
    plot_samples_and_data,
    make_generator,
    Logger
)

import torch
from torch.distributions import MultivariateNormal
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


def validate(data,
             oracle_cov,
             noise,
             model,
             args,
             device,
             writer,
             latent_model):
    
    # Lists for logging model's training NLL and oracle NLL
    nll_list = []
    mae_list = []
    oracle_nll_list = []
    
    # If training a latent model, set the number of latent samples accordingly
    loss_kwargs = {'num_samples' : args.np_val_samples} \
                  if latent_model else {}
    
    with torch.no_grad():
        
        for step, batch in enumerate(data):
            
            nll = model.loss(batch['x_context'].to(device),
                             batch['y_context'].to(device),
                             batch['x_target'].to(device),
                             batch['y_target'].to(device),
                             **loss_kwargs)
            
            if latent_model:
                mean, _ = model.forward(batch['x_context'].to(device),
                                        batch['y_context'].to(device),
                                        batch['x_target'].to(device),
                                        **loss_kwargs)
                
                diff = batch['y_target'][None, :, :, 0].to(device) - mean[:, :, :, 0]
                mae = torch.sum(torch.abs(diff), axis=-1)
                mae = torch.mean(mae)
                
            else:
                mean, _, _ = model.mean_and_marginals(batch['x_context'].to(device),
                                                      batch['y_context'].to(device),
                                                      batch['x_target'].to(device))
                
                diff = batch['y_target'][:, :, 0].to(device) - mean
                mae = torch.sum(torch.abs(diff), axis=1)
                mae = torch.mean(mae, axis=0)
                
            
            oracle_nll = torch.tensor(0.)

            # Oracle loss exists only for GP-generated data, not sawtooth
            if oracle_cov is not None:
                for b in range(batch['x_context'].shape[0]):
                    
                    x_context = batch['x_context'][b].clone().detach().numpy()
                    y_context = batch['y_context'][b].clone().detach().numpy()
                    x_target = batch['x_target'][b].clone().detach().numpy()
                    y_target = batch['y_target'][b].clone().detach().numpy()
                    
                    oracle_nll = oracle_nll - oracle_loglik(x_context,
                                                            y_context,
                                                            x_target,
                                                            y_target,
                                                            oracle_cov,
                                                            noise)

            # Scale by the average number of target points
            nll_list.append(nll.item())
            mae_list.append(mae.item())
            oracle_nll_list.append(oracle_nll.item() / \
                                   batch['x_context'].shape[0])

    mean_nll = np.mean(nll_list)
    std_nll = np.var(nll_list)**0.5
    
    mean_oracle_nll = np.mean(oracle_nll_list)
    std_oracle_nll = np.var(oracle_nll_list)**0.5
    
    mean_mae = np.mean(mae_list)
    std_mae = np.var(mae_list)**0.5

    # Print validation loss and oracle loss
    print(f"Validation neg. log-lik: "
          f"{mean_nll:.2f}")

    print(f"Oracle     neg. log-lik: "
          f"{mean_oracle_nll:.2f}")
    
    print(f"Validation          MAE: "
          f"{mean_mae:.2f}")

    return mean_nll, std_nll, mean_oracle_nll, std_oracle_nll, mean_mae, std_mae


# Parse arguments given to the script.
parser = argparse.ArgumentParser()


# =============================================================================
# Data generation arguments
# =============================================================================

parser.add_argument('data', help='Data set to train the CNP on.')

parser.add_argument('--x_dim',
                    default=1,
                    choices=[1, 2],
                    type=int,
                    help='Input dimension of data.')

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
                    choices=['GNP',
                             'AGNP',
                             'convGNP',
                             'FullConvGNP',
                             'ANP',
                             'convNP'],
                    help='Choice of model. ')

parser.add_argument('cov_type',
                    choices=['meanfield',
                             'innerprod', 
                             'kvv'],
                    help='Choice of covariance method.')

parser.add_argument('noise_type',
                    choices=['homo', 'hetero'],
                    help='Choice of noise model.')

parser.add_argument('--marginal_type',
                    default='identity',
                    choices=['loglogit'],
                    help='Choice of marginal transformation (optional).')

parser.add_argument('--np_loss_samples',
                    default=10,
                    type=int,
                    help='Number of latent samples for evaluating the loss, '
                         'used for ANP and ConvNP.')

parser.add_argument('--np_val_samples',
                    default=5,
                    type=int,
                    help='Number of latent samples for evaluating the loss, '
                         'when validating, used for ANP and ConvNP.')

parser.add_argument('--num_basis_dim',
                    default=512,
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

root = 'experiments/synthetic'

# Working directory for saving results
experiment_name = os.path.join(f'{root}',
                               f'results',
                               f'{args.data}',
                               f'models',
                               f'{args.model}',
                               f'{args.cov_type}',
                               f'{args.noise_type}',
                               f'{args.marginal_type}',
                               f'seed-{args.seed}',
                               f'dim-{args.x_dim}')
working_directory = WorkingDirectory(root=experiment_name)

# Data directory for loading data
data_root = os.path.join(f'{root}',
                         f'toy-data',
                         f'{args.data}')
data_directory = WorkingDirectory(root=data_root)

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

if args.marginal_type == 'loglogit':
    output_layer = LogLogitCopulaLayer(gaussian_layer=output_layer)
    
# Create model architecture
if args.model == 'GNP':
    model = StandardGNP(input_dim=args.x_dim, output_layer=output_layer)
    
elif args.model == 'AGNP':
    model = StandardAGNP(input_dim=args.x_dim, output_layer=output_layer)
    
elif args.model == 'convGNP':
    model = StandardConvGNP(input_dim=args.x_dim, output_layer=output_layer)

elif args.model == 'FullConvGNP':
    model = FullConvGNP()

elif args.model == 'ANP':
    model = StandardANP(input_dim=args.x_dim,
                        num_samples=args.np_loss_samples)
    
elif args.model == 'convNP':
    model = StandardConvNP(input_dim=args.x_dim,
                           num_samples=args.np_loss_samples)
    
else:
    raise ValueError(f'Unknown model {args.model}.')


print(f'{args.model} '
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

latent_model = args.model in ['ANP', 'convNP']


# =============================================================================
# Load data and validation oracle generator
# =============================================================================
    
file = open(data_directory.file('train-data.pkl'), 'rb')
data_train = pickle.load(file)
file.close()

file = open(data_directory.file('valid-data.pkl'), 'rb')
data_val = pickle.load(file)
file.close()

oracle_cov = None
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
# Train or test model
# =============================================================================

# Number of epochs between validations
train_iteration = 0
log_every = 500
    
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

        # Compute validation negative log-likelihood
        val_nll, _, val_oracle, _, val_mae, _ = validate(valid_epoch,
                                                         oracle_cov,
                                                         noise,
                                                         model,
                                                         args,
                                                         device,
                                                         writer,
                                                         latent_model)

        # Log information to tensorboard
        writer.add_scalar('Valid log-lik.',
                          -val_nll,
                          epoch)
        
        writer.add_scalar('Valid MAE',
                          val_mae,
                          epoch)

        writer.add_scalar('Valid oracle log-lik.',
                          -val_oracle,
                          epoch)

        writer.add_scalar('Oracle minus valid log-lik.',
                          -val_oracle + val_nll,
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

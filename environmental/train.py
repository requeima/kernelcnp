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

from cnp.utils import make_generator


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
             model,
             args,
             device,
             writer,
             latent_model):
    
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

    return mean_nll, std_nll, mean_oracle_nll, std_oracle_nll
        

# Parse arguments given to the script.
parser = argparse.ArgumentParser()

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
    

# =============================================================================
# Load training and validation data
# =============================================================================

train_dataset = 'era5land_daily_central_eu_complete.nc'

valid_datasets = ['era5land_daily_east_eu_test.nc',
                  'era5land_daily_north_eu_test.nc',
                  'era5land_daily_west_eu_test.nc']


train_dataset = Dataset(f"{data_root}/{train_dataset}", "r", format="NETCDF4")
valid_datasets = [Dataset(f"{data_root}/{dataset}", "r", format="NETCDF4") \
                  for dataset in valid_datasets]

class LambdaIterator:

    def __init__(self, generator, num_elements):
        self.generator = generator
        self.num_elements = num_elements
        self.index = 0

    def __next__(self):
        self.index += 1
        if self.index <= self.num_elements:
            return self.generator()
        else:
            raise StopIteration()

    def __iter__(self):
        return self


class EnvironmentalDataloader:

    def __init__(self,
                 dataset,
                 iterations_per_epoch,
                 min_num_context,
                 max_num_context,
                 min_num_target,
                 max_num_target,
                 num_datasets,
                 scale_inputs_by):

        self.dataset = dataset
        self.iterations_per_epoch = iterations_per_epoch

        self.min_num_context = min_num_context
        self.max_num_context = max_num_context
        self.min_num_context = min_num_context
        self.max_num_target = max_num_target
        self.num_datasets = num_datasets

        lat = np.array(dataset.variables['latitude'])
        lon = np.array(dataset.variables['longitude'])

        lat, lon, scale_by = self.lat_lon_scale(lat=lat,
                                                lon=lon,
                                                scale_by=scale_inputs_by)
        self.lat = lat
        self.lon = lon
        self.scale_by = scale_by
        self.latlon_idx = np.meshgrid(np.arange(lat.shape[0]),
                                      np.arange(lon.shape[0]))
        self.latlon_idx = np.reshape(np.stack(self.latlon_idx, axis=-1),
                                     (-1, 2))

        # Predict total precipitation
        self.variables = ["tp"]


    def lat_lon_scale(self, lat, lon, scale_by=None):
        """
        Computes the translation and scaling amounts which convert the given
        longitude and latitude arrays to both be in the range lat_lon_range.
        """

        if scale_by is None:
        
            lat_min = lat.min()
            lat_max = lat.max()

            lon_min = lon.min()
            lon_max = lon.max()

            lat_trans = (lat_max + lat_min) / 2
            lat_scale = 0.5 * (lat_max - lat_min)

            lon_trans = (lon_max + lat_min) / 2
            lon_scale = 0.5 * (lon_max - lon_min)

            self.scale_by = (lat_trans, lat_scale, lon_trans, lon_scale)

        else:
            lat_trans, lat_scale, lon_trans, lon_scale = scale_by

        lat = (lat - lat_trans) / lat_scale
        lon = (lon - lon_trans) / lon_scale

        return lat, lon, (lat_trans, lat_scale, lon_trans, lon_scale)


    def __iter__(self):
        return LambdaIterator(lambda : self.generate_task(), self.iterations_per_epoch)


    def generate_task(self):

        # Latitude and logitude resolutions
        num_lat = self.lat.shape[0]
        num_lon = self.lon.shape[0]

        # Dict to store sampled batch
        batch = {
            'x'         : [],
            'y'         : [],
            'x_context' : [],
            'y_context' : [],
            'x_target'  : [],
            'y_target'  : []
        }

        # Sample number of context and target points
        num_context = np.random.randint(1, self.max_num_context+1)
        num_target = np.random.randint(1, self.max_num_target+1)
        num_data = num_context + num_target

        idx = np.arange(self.latlon_idx.shape[0])

        for i in range(self.num_datasets):
            
            # Sample indices for current batch (C + T, 2)
            _idx = np.random.choice(idx, size=(num_data,), replace=False)
            _idx = self.latlon_idx[_idx]

            # Slice out latitude and longitude values, stack to (C + T, 2)
            # These latitude and longitude values are already rescaled
            x = np.stack([self.lat[_idx[:, 0]], self.lon[_idx[:, 1]]], axis=-1)

            # Slice out output values to be predicted
            t = np.random.randint(0, self.dataset.variables['time'].shape[0])
            y = [self.dataset.variables[variable][t][_idx[:, 0], _idx[:, 1]] \
                 for variable in self.variables]
            y = np.stack(y, axis=-1)

            # Append results to lists in batch dict
            batch['x'].append(x)
            batch['y'].append(y)

            batch['x_context'].append(x[:num_context])
            batch['y_context'].append(y[:num_context])

            batch['x_target'].append(x[num_context:])
            batch['y_target'].append(y[num_context:])

        # Stack arrays and convert to tensors
        batch = {name : torch.tensor(np.stack(tensors, axis=0)) \
                 for name, tensors in batch.items()}

        return batch


iterations_per_epoch = 1000
min_num_context = 50
max_num_context = 50
min_num_target = 10
max_num_target = 50
num_datasets = 16
epochs = 100

dataloader = EnvironmentalDataloader(train_dataset,
                                     iterations_per_epoch,
                                     min_num_context,
                                     max_num_context,
                                     min_num_target,
                                     max_num_target,
                                     num_datasets,
                                     scale_inputs_by=None)


for epoch in range(epochs):
    for batch in dataloader:
        
        print(batch['x'].shape)
        print(batch['y'].shape)

        print(batch['x'][0])

## =============================================================================
## Create model
## =============================================================================
#
## Create covariance method
#if args.covtype == 'innerprod-homo':
#    cov = InnerProdCov(args.num_basis_dim)
#    noise = AddHomoNoise()
#    
#elif args.covtype == 'innerprod-hetero':
#    cov = InnerProdCov(args.num_basis_dim)
#    noise = AddHeteroNoise()
#    
#elif args.covtype == 'kvv-homo':
#    cov = KvvCov(args.num_basis_dim)
#    noise = AddHomoNoise()
#    
#elif args.covtype == 'kvv-hetero':
#    cov = KvvCov(args.num_basis_dim)
#    noise = AddHomoNoise()
#    
#elif args.covtype == 'meanfield':
#    cov = MeanFieldCov(num_basis_dim=1)
#    noise = AddNoNoise()
#    
#else:
#    raise ValueError(f'Unknown covariance method {args.covtype}.')
#    
## Create model architecture
#if args.model == 'GNP':
#    model = StandardGNP(input_dim=args.x_dim,
#                        covariance=cov,
#                        add_noise=noise)
#    
#elif args.model == 'AGNP':
#    model = StandardAGNP(input_dim=args.x_dim,
#                         covariance=cov,
#                         add_noise=noise)
#    
#elif args.model == 'convGNP':
#    model = StandardConvGNP(input_dim=args.x_dim,
#                            covariance=cov,
#                            add_noise=noise)
#
#elif args.model == 'FullConvGNP':
#    model = FullConvGNP()
#    
#
#elif args.model == 'ANP':
#    
#    noise = AddHomoNoise()
#    model = StandardANP(input_dim=args.x_dim,
#                        add_noise=noise,
#                        num_samples=args.np_loss_samples)
#    
#elif args.model == 'convNP':
#    
#    noise = AddHomoNoise()
#    model = StandardConvNP(input_dim=args.x_dim,
#                           add_noise=noise,
#                           num_samples=args.np_loss_samples)
#    
#else:
#    raise ValueError(f'Unknown model {args.model}.')
#
#
#print(f'{args.model} '
#      f'{args.covtype} '
#      f'{args.num_basis_dim}: '
#      f'{model.num_params}')
#
#with open(working_directory.file('num_params.txt'), 'w') as f:
#    f.write(f'{model.num_params}')
#        
#if args.num_params:
#    exit()
#    
#    
## Load model to appropriate device
#model = model.to(device)
#
#latent_model = args.model in ['ANP', 'convNP']
#
#
#
## =============================================================================
## Load data and validation oracle generator
## =============================================================================
#    
#file = open(data_directory.file('train-data.pkl'), 'rb')
#data_train = pickle.load(file)
#file.close()
#
#file = open(data_directory.file('valid-data.pkl'), 'rb')
#data_val = pickle.load(file)
#file.close()
#
## Create the data generator for the oracle if gp data
#if args.data == 'sawtooth' or args.data == 'random':
#    gen_val = None
#    
#else:
#    file = open(data_directory.file('gen-valid-dict.pkl'), 'rb')
#    gen_valid_gp_params = pickle.load(file)
#    file.close()
#
#    file = open(data_directory.file('kernel-params.pkl'), 'rb')
#    kernel_params = pickle.load(file)
#    file.close()
#    
#    gen_val = make_generator(args.data, gen_valid_gp_params, kernel_params)
#
#        
## =============================================================================
## Train or test model
## =============================================================================
#
## Number of epochs between validations
#train_iteration = 0
#log_every = 1
#    
#log_args(working_directory, args)
#
## Create optimiser
#optimiser = torch.optim.Adam(model.parameters(),
#                         args.learning_rate,
#                         weight_decay=args.weight_decay)
#
## Run the training loop, maintaining the best objective value
#best_nll = np.inf
#
#epochs = len(data_train)
#
#for epoch in range(epochs):
#
#    if train_iteration % log_every == 0:
#        print('\nEpoch: {}/{}'.format(epoch + 1, epochs))
#
#    if epoch % args.validate_every == 0:
#
#        valid_epoch = data_val[epoch // args.validate_every]
#
#        # Compute validation negative log-likelihood
#        val_nll, _, val_oracle, _ = validate(valid_epoch,
#                                             gen_val,
#                                             model,
#                                             args,
#                                             device,
#                                             writer,
#                                             latent_model)
#
#        # Log information to tensorboard
#        writer.add_scalar('Valid log-lik.',
#                          -val_nll,
#                          epoch)
#
#        writer.add_scalar('Valid oracle log-lik.',
#                          -val_oracle,
#                          epoch)
#
#        writer.add_scalar('Oracle minus valid log-lik.',
#                          -val_oracle + val_nll,
#                          epoch)
#
#        # Update the best objective value and checkpoint the model
#        is_best, best_obj = (True, val_nll) if val_nll < best_nll else \
#                            (False, best_nll)
#
#        plot_marginals = args.covtype == 'meanfield'
#
#        if args.x_dim == 1 and \
#           not (args.data == 'sawtooth' or args.data == 'random'):
#
#            plot_samples_and_data(model=model,
#                                  gen_plot=gen_val,
#                                  xmin=-2,
#                                  xmax=2,
#                                  root=working_directory.root,
#                                  epoch=epoch,
#                                  latent_model=latent_model,
#                                  plot_marginals=plot_marginals,
#                                  device=device)
#
#
#    train_epoch = data_train[epoch]
#
#    # Compute training negative log-likelihood
#    train_iteration = train(train_epoch,
#                            model,
#                            optimiser,
#                            log_every,
#                            device,
#                            writer,
#                            train_iteration)
#
#    save_checkpoint(working_directory,
#                    {'epoch'         : epoch + 1,
#                     'state_dict'    : model.state_dict(),
#                     'best_acc_top1' : best_obj,
#                     'optimizer'     : optimiser.state_dict()},
#                    is_best=is_best,
#                    epoch=epoch)

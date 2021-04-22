import argparse

import numpy as np
import stheno.torch as stheno
import torch
import matplotlib.pyplot as plt
import os
from datetime import datetime
import pickle

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

from cnp.cnp import (
    StandardGNP,
    StandardAGNP,
    StandardConvGNP
)

from cnp.cov import (
    InnerProdCov,
    KvvCov,
    MeanFieldCov,
    AddHomoNoise,
    AddHeteroNoise,
    AddNoNoise
)

from cnp.utils import plot_samples_and_data

from torch.distributions import MultivariateNormal

from torch.utils.tensorboard import SummaryWriter


def validate(data, data_generator, model, args, device, oracle=True):
    """ Compute the validation loss. """
    
    nll_list = []
    oracle_nll_list = []
    
    with torch.no_grad():
        for step, batch in enumerate(data):
            nll = model.loss(batch['x_context'].to(device),
                             batch['y_context'].to(device),
                             batch['x_target'].to(device),
                             batch['y_target'].to(device))
            
            oracle_nll = np.array(0.)
            if oracle:
                if (type(data_generator) == cnp.data.GPGenerator):
                    for b in range(batch['x_context'].shape[0]):
                        oracle_nll =  - data_generator.log_like(batch['x_context'][b],
                                                                batch['y_context'][b],
                                                                batch['x_target'][b],
                                                                batch['y_target'][b])
                        

            # Scale by the maximum number of target points            
            nll_list.append(nll.item()/args.max_num_target)
            oracle_nll_list.append(oracle_nll.item()/args.max_num_target)
                
    mean_nll = np.mean(nll_list)
    std_nll = (np.var(nll_list) ** 0.5) / np.sqrt(len(nll_list))
    mean_oracle = np.mean(oracle_nll_list)
    std_oracle = (np.var(oracle_nll_list) ** 0.5) / np.sqrt(len(oracle_nll_list)) 


    print(f"Validation neg. log-lik: "
        f"{mean_nll:.2f} +/- "
        f"{std_nll:.2f}")

    print(f"Oracle     neg. log-lik: "
        f"{mean_oracle:.2f} +/- "
        f"{std_oracle:.2f}")

    return mean_nll, std_nll, mean_oracle, std_oracle


def train(data, model, optimiser, log, device):
    """ Perform a training epoch. """

    nll = 0.
    
    for step, batch in enumerate(data):

        nll = nll + model.loss(batch['x_context'].to(device),
                                 batch['y_context'].to(device),
                                 batch['x_target'].to(device),
                                 batch['y_target'].to(device))
        
    # Scale objective by number of iterations
    nll = nll / (step + 1)
    
    if log:
        print(f"Training   neg. log-lik: {nll:.2f}")

    # Compute gradients and apply them
    nll.backward()
    optimiser.step()
    optimiser.zero_grad()

    return nll


# Parse arguments given to the script.
parser = argparse.ArgumentParser()


# =============================================================================
# Data generation arguments
# =============================================================================

parser.add_argument('data',
                    choices=['eq',
                             'matern',
                             'noisy-mixture',
                             'weakly-periodic',
                             'sawtooth'],
                    help='Data set to train the CNP on. ')

parser.add_argument('--x_dim',
                    default=1,
                    choices=[1, 2, 3],
                    type=int,
                    help='Input dimension of data.')

parser.add_argument('--seed',
                    default=0,
                    type=int,
                    help='Random seed to use.')

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
                    default=25,
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

parser.add_argument('--x_context_range',
                    default=[-3., 3.],
                    nargs='+',
                    type=float,
                    help='Range of input x for sampled data.')

parser.add_argument('--x_target_range',
                    default=None,
                    nargs='+',
                    type=float,
                    help='Range of inputs for sampled data.')

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


# =============================================================================
# Model arguments
# =============================================================================

parser.add_argument('model',
                    choices=['GNP',
                             'AGNP',
                             'convGNP',
                             'StandardNDConvGNP'],
                    help='Choice of model. ')

parser.add_argument('covtype',
                    choices=['innerprod-homo',
                             'innerprod-hetero', 
                             'kvv-homo',
                             'kvv-hetero',
                             'meanfield'],
                    help='Choice of covariance method.')

parser.add_argument('--num_basis_dim',
                    default=512,
                    type=int,
                    help='Number of embedding basis dimensions.')

parser.add_argument('--learning_rate',
                    default=1e-3,
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

parser.add_argument('--train',
                    action='store_true',
                    help='Perform training. If this is not specified, '
                         'the model will be attempted to be loaded from the '
                         'experiment root.')

parser.add_argument('--test',
                    action='store_true',
                    help='Test the model and record the values in the'
                         'experimental root.')

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

device = torch.device('cpu') if not torch.cuda.is_available() and args.gpu == 0 \
                             else torch.device('cuda')

data_root = os.path.join('_experiments',
                         f'{args.data}',
                         'data',
                         f'seed-{args.seed}',
                         f'dim-{args.x_dim}')

# Load working directory
if args.root:
    working_directory = WorkingDirectory(root=args.root)
    data_directory = WorkingDirectory(root=data_root)
    
    writer = SummaryWriter(f'{args.root}/log')
    
else:
    experiment_name = os.path.join('_experiments',
                                   f'{args.data}',
                                   f'models',
                                   f'{args.model}',
                                   f'{args.covtype}',
                                   f'seed-{args.seed}',
                                   f'dim-{args.x_dim}')
    working_directory = WorkingDirectory(root=experiment_name)
    data_directory = WorkingDirectory(root=data_root)
    
    writer = SummaryWriter(f'{experiment_name}/log')
    

file = open(working_directory.file('data_location.txt'), 'w')
file.write(data_directory.root)
file.close()
    

# =============================================================================
# Create data generators
# =============================================================================


# Training data generator parameters -- used for both Sawtooth and GP
gen_params = {
    'batch_size'                : args.batch_size,
    'x_context_ranges'          : args.x_context_range,
    'max_num_context'           : args.max_num_context,
    'max_num_target'            : args.max_num_target,
    'device'                    : device
}

# Plotting data generator parameters -- used for both Sawtooth and GP
gen_plot_params = deepcopy(gen_params)
gen_plot_params['iterations_per_epoch'] = 1
gen_plot_params['batch_size'] = 3
gen_plot_params['max_num_context'] = 16

# Training data generator parameters -- specific to Sawtooth
gen_train_sawtooth_params = {
    'freq_range'  : args.freq_range,
    'shift_range' : args.shift_range,
    'trunc_range' : args.trunc_range
}

                    
if args.data == 'sawtooth':
    
    gen_train = cnp.data.SawtoothGenerator(args.num_train_iters,
                                           **gen_train_sawtooth_params,
                                           **gen_params)
    
    gen_val = cnp.data.SawtoothGenerator(args.num_valid_iters,
                                         **gen_train_sawtooth_params,
                                         **gen_params)
    
    gen_test = cnp.data.SawtoothGenerator(args.num_test_iters,
                                          **gen_train_sawtooth_params,
                                          **gen_params)
    
    gen_plot = cnp.data.SawtoothGenerator(**gen_train_sawtooth_params,
                                          **gen_plot_params)
    
else:
    
    if args.data == 'eq':
        kernel = stheno.EQ().stretch(args.eq_params[0])
        
    elif args.data == 'matern':
        kernel = stheno.Matern52().stretch(args.m52_params[0])
        
    elif args.data == 'noisy-mixture':
        kernel = stheno.EQ().stretch(args.mixture_params[0]) + \
                 stheno.EQ().stretch(args.mixture_params[1])
        
    elif args.data == 'weakly-periodic':
        kernel = stheno.EQ().stretch(args.wp_params[0]) * \
                 stheno.EQ().periodic(period=args.wp_params[1])
        
    else:
        raise ValueError(f'Unknown generator kind "{args.data}".')
        
    gen_train = cnp.data.GPGenerator(iterations_per_epoch=args.num_train_iters,
                                     kernel=kernel,
                                     std_noise=args.std_noise,
                                     **gen_params)
        
    gen_val = cnp.data.GPGenerator(iterations_per_epoch=args.num_valid_iters,
                                   kernel=kernel,
                                   std_noise=args.std_noise,
                                   **gen_params)
        
    gen_test = cnp.data.GPGenerator(iterations_per_epoch=args.num_test_iters,
                                    kernel=kernel,
                                    std_noise=args.std_noise,
                                    **gen_params)
        
    gen_plot = cnp.data.GPGenerator(kernel=kernel,
                                    std_noise=args.std_noise,
                                    **gen_plot_params)
    


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
if args.model == 'GNP':
    model = StandardGNP(input_dim=input_dim,
                        covariance=cov,
                        add_noise=noise)
    
elif args.model == 'AGNP':
    model = StandardAGNP(input_dim=input_dim,
                         covariance=cov,
                         add_noise=noise)
    
elif args.model == 'convGNP':
    model = StandardConvGNP(input_dim=input_dim,
                            covariance=cov,
                            add_noise=noise)
    
else:
    raise ValueError(f'Unknown model {args.model}.')


print(f'{args.model} '
      f'{args.covtype} '
      f'{args.num_basis_dim}: '
      f'{model.num_params}')

with open(working_directory.file('num_params.txt'), 'w') as f:
    f.write(f'{model.num_params}')
        
if args.num_params:
    exit()
    
    
# Load model to appropriate device
model = model.to(device)


# =============================================================================
# Load data
# =============================================================================

if args.train:
    file = open(data_directory.file('train-data.pkl'), 'rb')
    data_train = pickle.load(file)
    file.close()

    file = open(data_directory.file('valid-data.pkl'), 'rb')
    data_val = pickle.load(file)
    file.close()
if args.test:
    file = open(data_directory.file('test-data.pkl'), 'rb')
    data_test = pickle.load(file)
    file.close()

# =============================================================================
# Train or test model
# =============================================================================

# Number of epochs between validations
LOG_EVERY = 1

if args.train:
    
    log_args(working_directory, args)

    # Create optimiser
    optimiser = torch.optim.Adam(model.parameters(),
                                 args.learning_rate,
                                 weight_decay=args.weight_decay)
    
    # Run the training loop, maintaining the best objective value
    best_nll = np.inf
    
    for epoch in range(args.epochs + 1):
        
        log = epoch % LOG_EVERY == 0
        
        if log:
            print('\nEpoch: {}/{}'.format(epoch + 1, args.epochs))

        # Compute training negative log-likelihood
        train_nll = train(data_train[epoch],
                          model,
                          optimiser,
                          log,
                          device)

        writer.add_scalar('Train log-lik.', - train_nll, epoch)

        if epoch % args.validate_every == 0:
            
            # Compute validation negative log-likelihood
            val_nll, _, val_oracle, _ = validate(data_val[epoch // args.validate_every],
                                           gen_val,
                                           model,
                                           args=args,
                                           device=device)
            
            writer.add_scalar('Valid log-lik.', - val_nll, epoch)
            writer.add_scalar('Valid oracle log-lik.', - val_oracle, epoch)
            writer.add_scalar('Oracle minus valid log-lik.', - val_oracle + val_nll, epoch)

            # Update the best objective value and checkpoint the model
            is_best, best_obj = (True, val_nll) if val_nll < best_nll else \
                                (False, best_nll)
            
            plot_marginals = args.covtype == 'meanfield'
            
            # plot_samples_and_data(model=model,
            #                       gen_plot=gen_plot,
            #                       xmin=args.x_context_range[0],
            #                       xmax=args.x_context_range[1],
            #                       root=working_directory.root,
            #                       epoch=epoch,
            #                       plot_marginals=plot_marginals)
            
        save_checkpoint(working_directory,
                        {'epoch'         : epoch + 1,
                         'state_dict'    : model.state_dict(),
                         'best_acc_top1' : best_obj,
                         'optimizer'     : optimiser.state_dict()},
                        is_best=is_best,
                        epoch=epoch)
        
        

elif args.test:

    print('Testing...')
    
    # Load model on appropriate device
    if device.type == 'cpu':
        load_dict = torch.load(working_directory.file('model_best.pth.tar',
                                                      exists=True),
                               map_location=torch.device('cpu'))
    else:
        load_dict = torch.load(working_directory.file('model_best.pth.tar',
                                                      exists=True))
        
    model.load_state_dict(load_dict['state_dict'])
    
    # Test model on ~2000 tasks.
    test_obj, test_obj_std_error, _, _ = validate(data_test, 
                                            gen_test, 
                                            model, 
                                            args, 
                                            device,
                                            oracle=False)
    
    print('Model averages a log-likelihood of %s +- %s on unseen tasks.' % (test_obj, test_obj_std_error))
    
    with open(working_directory.file('test_log_likelihood.txt'), 'w') as f:
        f.write(str(test_obj))
        
    with open(working_directory.file('test_log_likelihood_standard_error.txt'), 'w') as f:
        f.write(str(test_obj_std_error))

import argparse

import numpy as np
import stheno.torch as stheno
import torch
import matplotlib.pyplot as plt
import os

import cnp.data

from copy import deepcopy

from cnp.experiment import (
    generate_root,
    WorkingDirectory,
    save_checkpoint
)

from cnp.cnp import (
    StandardGNP,
    StandardAGNP,
    StandardConvGNP,
    StandardFullyConnectedTEGNP
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


def validate(data, model, report_freq, std_error=False):
    """ Compute the validation loss. """
    
    nll_list = []
    
    with torch.no_grad():
        for step, batch in enumerate(data):

            y_mean, _, y_cov = model(batch['x_context'],
                                     batch['y_context'],
                                     batch['x_target'])

            dist = MultivariateNormal(loc=y_mean[:, :, 0],
                                      covariance_matrix=y_cov)
            
            nll = - dist.log_prob(batch['y_target'][:, :, 0]).sum()
            
            nll_list.append(nll.item())

            if (step + 1) % report_freq == 0:
                print(f"Validation neg. log-lik: "
                      f"{np.mean(nll_list):.2f} +/- "
                      f"{np.var(nll_list) ** 0.5:.2f}")
                
    mean_nll = np.mean(nll_list)
    
    return mean_nll


def train(data, model, optimiser, log):
    """ Perform a training epoch. """

    nll = 0.
    
    for step, batch in enumerate(data):

        y_mean, _, y_cov = model(batch['x_context'],
                                 batch['y_context'],
                                 batch['x_target'])
        

        dist = MultivariateNormal(loc=y_mean[:, :, 0],
                                  covariance_matrix=y_cov)
        nll = nll - dist.log_prob(batch['y_target'][:, :, 0]).sum()
        
    # Scale objective by number of iterations
    nll = nll / (step + 1)
    
    if log:
        print(f"Training   neg. log-lik: {nll:.2f}")

    # Compute gradients and apply them
    nll.backward()
    optimiser.step()
    optimiser.zero_grad()

    return nll

def log_args(wd, args):
    args_file = wd.file('args_file.txt')

    args_str = ""

    for arg in vars(args):
        args_str += f"{arg}: {getattr(args, arg)}"

        args_str += "\n"

    with open(args_file, 'w') as args_file_file_write:
        args_file_file_write.write(args_str)

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

parser.add_argument('--batch_size',
                    default=16,
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
                    default=100,
                    type=int,
                    help='Iterations (# batches sampled) for validation.')

parser.add_argument('--num_test_iters',
                    default=2048,
                    type=int,
                    help='Iterations (# batches sampled) for testing.')

parser.add_argument('--x_range',
                    default=[-3., 3.],
                    nargs='+',
                    type=float,
                    help='Range of input x for sampled data.')

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
                             'TEGNP'],
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

parser.add_argument('--gpu',
                    default=0,
                    type=int,
                    help='GPU to run experiment on. Defaults to 0.')


args = parser.parse_args()

device = torch.device('cpu') if not torch.cuda.is_available() and args.gpu == 0 \
                             else torch.device(f'cuda:{args.gpu}')


# Load working directory
if args.root:
    working_directory = WorkingDirectory(root=args.root)
    
else:
    experiment_name = os.path.join('_experiments',
                                   f'{args.data}',
                                   f'{args.model}',
                                   f'{args.covtype}-{args.num_basis_dim}basisdims')
    working_directory = WorkingDirectory(root=experiment_name)
    


# =============================================================================
# Create data generators
# =============================================================================

EQ_PARAMS = [1.]
M52_PARAMS = [1.]
MIXTURE_PARAMS = [1., 0.25]
WP_PARAMS = [1., 0.25]


# Training data generator parameters -- used for both Sawtooth and GP
gen_params = {
    'batch_size'                : args.batch_size,
    'x_range'                   : args.x_range,
    'max_num_context'           : args.max_num_context,
    'max_num_target'            : args.max_num_target,
    'include_context_in_target' : False,
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
        kernel = stheno.EQ().stretch(EQ_PARAMS[0])
        
    elif args.data == 'matern':
        kernel = stheno.Matern52().stretch(M52_PARAMS[0])
        
    elif args.data == 'noisy-mixture':
        kernel = stheno.EQ().stretch(MIXTURE_PARAMS[0]) + \
                 stheno.EQ().stretch(MIXTURE_PARAMS[1])
        
    elif args.data == 'weakly-periodic':
        kernel = stheno.EQ().stretch(WP_PARAMS[0]) * \
                 stheno.EQ().periodic(period=WP_PARAMS[1])
        
    else:
        raise ValueError(f'Unknown generator kind "{args.data}".')
        
    kernel = kernel + 1e-2 * stheno.Delta()
        
    gen_train = cnp.data.GPGenerator(iterations_per_epoch=args.num_train_iters,
                                     kernel=kernel,
                                     **gen_params)
        
    gen_val = cnp.data.GPGenerator(iterations_per_epoch=args.num_valid_iters,
                                   kernel=kernel,
                                   **gen_params)
        
    gen_test = cnp.data.GPGenerator(iterations_per_epoch=args.num_test_iters,
                                    kernel=kernel,
                                    **gen_params)
        
    gen_plot = cnp.data.GPGenerator(kernel=kernel,
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
    model = StandardGNP(covariance=cov,
                        add_noise=noise)
    
elif args.model == 'AGNP':
    model = StandardAGNP(covariance=cov,
                         add_noise=noise)
    
elif args.model == 'convGNP':
    model = StandardConvGNP(covariance=cov,
                            add_noise=noise)
    
elif args.model == 'TEGNP':
    model = StandardFullyConnectedTEGNP(covariance=cov,
                                        add_noise=noise)
    
else:
    raise ValueError(f'Unknown model {args.model}.')
    
    
# Load model to appropriate device
model = model.to(device)


# =============================================================================
# Train or test model
# =============================================================================

# Number of epochs between validations
LOG_EVERY = 10
VALIDATE_EVERY = 100

if args.train:

    log_args(working_directory, args)

    # Create optimiser
    optimiser = torch.optim.Adam(model.parameters(),
                                 args.learning_rate,
                                 weight_decay=args.weight_decay)
    
    # Run the training loop, maintaining the best objective value
    best_nll = np.inf
    
    for epoch in range(args.epochs):
        
        log = epoch % LOG_EVERY == 0
        
        if log:
            print('\nEpoch: {}/{}'.format(epoch + 1, args.epochs))

        # Compute training negative log-likelihood
        train_nll = train(gen_train,
                          model,
                          optimiser,
                          log=log)

        if epoch % VALIDATE_EVERY == 0:
            
            # Compute validation negative log-likelihood
            val_nll = validate(gen_val,
                               model,
                               report_freq=args.num_valid_iters)

            # Update the best objective value and checkpoint the model
            is_best, best_obj = (True, val_nll) if val_nll < best_nll else \
                                (False, best_nll)
            
            plot_marginals = args.covtype == 'meanfield'
            
            print(plot_marginals)
            
            plot_samples_and_data(model=model,
                                  gen_plot=gen_plot,
                                  xmin=args.x_range[0],
                                  xmax=args.x_range[1],
                                  root=working_directory.root,
                                  epoch=epoch,
                                  plot_marginals=plot_marginals)
            
        save_checkpoint(working_directory,
                        {'epoch'         : epoch + 1,
                         'state_dict'    : model.state_dict(),
                         'best_acc_top1' : best_obj,
                         'optimizer'     : optimiser.state_dict()},
                        is_best=is_best,
                        epoch=epoch)
        
        

elif args.test:
    
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
    test_obj, test_obj_std_error = validate(gen_test, model, std_error=True)
    
    print('Model averages a log-likelihood of %s +- %s on unseen tasks.' % (test_obj, test_obj_std_error))
    
    with open(working_directory.file('test_log_likelihood.txt'), 'w') as f:
        f.write(str(test_obj))
        
    with open(working_directory.file('test_log_likelihood_standard_error.txt'), 'w') as f:
        f.write(str(test_obj_std_error))

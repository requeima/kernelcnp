import argparse

import numpy as np
import stheno.torch as stheno
import torch
import matplotlib.pyplot as plt
import os

import cnp.data

from cnp.experiment import (
    report_loss,
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

from torch.distributions import MultivariateNormal

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def validate(data, model, report_freq=None, std_error=False):
    """Compute the validation loss."""
    model.eval()
    likelihoods = []
    with torch.no_grad():
        for step, task in enumerate(data):

            y_mean, _, y_cov = model(task['x_context'],
                                     task['y_context'],
                                     task['x_target'])

            dist = MultivariateNormal(loc=y_mean[:, :, 0],
                                      covariance_matrix=y_cov)
            obj = - dist.log_prob(task['y_target'][:, :, 0]).sum()

            likelihoods.append(obj.item())

            if report_freq:

                avg_ll = np.array(likelihoods).mean()
                report_loss('Validation', avg_ll, step, report_freq)

    likelihoods = np.array(likelihoods)
    avg_ll = likelihoods.mean()
    if std_error:
        std_error = likelihoods.std()/np.sqrt(len(likelihoods))
        return avg_ll, std_error
    else:
        return avg_ll


def train(data, model, optimiser, report_freq):
    """Perform a training epoch."""
    model.train()
    losses = []
    for step, task in enumerate(data):

        y_mean, _, y_cov = model(task['x_context'],
                                 task['y_context'],
                                 task['x_target'])
        

        dist = MultivariateNormal(loc=y_mean[:, :, 0],
                                  covariance_matrix=y_cov)
        obj = - dist.log_prob(task['y_target'][:, :, 0]).sum()

        # Optimization
        obj.backward()
        optimiser.step()
        optimiser.zero_grad()

        # Track training progress
        losses.append(obj.item())
        avg_loss = np.array(losses).mean()
        report_loss('Training', avg_loss, step, report_freq)

    return avg_loss


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

parser.add_argument('--num_train_iterations',
                    default=1,
                    type=int,
                    help='Iterations (# batches sampled) per training epoch.')

parser.add_argument('--num_valid_iterations',
                    default=32,
                    type=int,
                    help='Iterations (# batches sampled) for validation.')

parser.add_argument('--num_test_iterations',
                    default=2048,
                    type=int,
                    help='Iterations (# batches sampled) for testing.')

parser.add_argument('--xrange',
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
                    default=100,
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
                    default=1024,
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


args = parser.parse_args()


# Load working directory
if args.root:
    working_directory = WorkingDirectory(root=args.root)
    
else:
    experiment_name = os.path.join('_experiments',
                                   f'{args.data}',
                                   f'{args.model}',
                                   f'{args.covtype}')
    working_directory = WorkingDirectory(root=experiment_name)
    


# =============================================================================
# Create data generators
# =============================================================================

EQ_PARAMS = [1.]
M52_PARAMS = [1.]
MIXTURE_PARAMS = [1., 0.25]
WP_PARAMS = [1., 0.25]


# Generator parameters -- used for both Sawtooth and GP generators
generator_parameters = {
    'batch_size'                : args.batch_size,
    'x_range'                   : args.x_range,
    'max_num_context'           : args.max_num_context,
    'max_num_target'            : args.max_num_target,
    'include_context_in_target' : False
}

# Generator parameters -- specific to sawtooth
train_sawtooth_parameters = {
    'freq_range'  : args.freq_range,
    'shift_range' : args.shift_range,
    'trunc_range' : args.trunc_range
}

                    
if args.data == 'sawtooth':
    
    gen_train = cnp.data.SawtoothGenerator(args.num_train_iterations,
                                           **sawtooth_parameters,
                                           **generator_parameters)
    
    gen_val = cnp.data.SawtoothGenerator(args.num_valid_iterations,
                                         **sawtooth_parameters,
                                         **generator_parameters)
    
    gen_test = cnp.data.SawtoothGenerator(args.num_test_iterations,
                                          **sawtooth_parameters,
                                          **generator_parameters)
    
else:
    
    if args.data == 'eq':
        kernel = stheno.EQ().stretch(EQ_PARAMS[0])
        
    elif args.data == 'matern':
        kernel = stheno.Matern52().stretch(M52_PARAMS[0])
        
    elif args.data == 'noisy-mixture':
        kernel = stheno.EQ().stretch(MIXTURE_PARAMS[0]) + \
                 stheno.EQ().stretch(MIXTURE_PARAMS[1]) + \
                 1e-3 * stheno.Delta()
        
    elif args.data == 'weakly-periodic':
        kernel = stheno.EQ().stretch(WP_PARAMS[0]) * \
                 stheno.EQ().periodic(period=WP_PARAMS[1])
        
    else:
        raise ValueError(f'Unknown generator kind "{args.data}".')
        
    gen_train = cnp.data.GPGenerator(iterations_per_epoch=args.num_train_iterations,
                                     kernel=kernel,
                                     **generator_parameters)
        
    gen_valid = cnp.data.GPGenerator(iterations_per_epoch=args.num_valid_iterations,
                                     kernel=kernel,
                                     **generator_parameters)
        
    gen_test = cnp.data.GPGenerator(iterations_per_epoch=args.num_test_iterations,
                                    kernel=kernel,
                                    **generator_parameters)
    


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
model.to(device)
    


# =============================================================================
# Train or test model
# =============================================================================


if args.train:

    # Create optimiser
    optimiser = torch.optim.Adam(model.parameters(),
                                 args.learning_rate,
                                 weight_decay=args.weight_decay)
    
    # Run the training loop, maintaining the best objective value.
    best_obj = -np.inf
    for epoch in range(args.epochs):
        print('\nEpoch: {}/{}'.format(epoch + 1, args.epochs))

        # Compute training objective.
        train_obj = train(gen, model, optimiser, report_freq=50)
        report_loss('Training', train_obj, 'epoch')

        # Compute validation objective.
        val_obj = validate(gen_val, model, report_freq=20)
        report_loss('Validation', val_obj, 'epoch')

        # Update the best objective value and checkpoint the model.
        is_best = False
        if val_obj > best_obj:
            best_obj = val_obj
            is_best = True
            
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

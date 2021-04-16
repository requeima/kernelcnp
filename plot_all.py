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
    StandardMeanTEGNP,
    StandardMeanTEAGNP,
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

# Move data to device
to_numpy = lambda x: x.squeeze().cpu().numpy()

def plot_tasks(data_gen, model, wd, args, i):
    num_points = 200
    x_all = torch.linspace(data_gen.x_range[0], data_gen.x_range[1], num_points)
    fig = plt.figure(figsize=(24, 8))
    fig.suptitle(f'{args.model}.  {args.covtype}.  {args.data}.', fontsize=14)
       

    for step, batch in enumerate(data_gen):
        plt.subplot(1, 3, step + 1)
        x_context = batch['x_context'].to(device)
        y_context = batch['y_context'].to(device)

        # Make predictions with model
        with torch.no_grad():

            y_mean, _,  y_std = model(x_context, y_context, x_all[None, :, None].to(device))

            # Get the marginals if we are predicting the full covariance
            if y_std.shape[-1] > 1:
                dist = torch.distributions.multivariate_normal.MultivariateNormal(loc=y_mean[0,:, 0], covariance_matrix=y_std[0, :, :])
                y_std = torch.diagonal(y_std, 0, dim1=-2, dim2=-1)[:, :, None]

            # Plot context set
            plt.scatter(to_numpy(x_context), to_numpy(y_context), label='Context Set', color='black')

            # Plot model predictions
            plt.plot(to_numpy(x_all), to_numpy(y_mean), label='Model Output', color='blue')
            plt.fill_between(to_numpy(x_all),
                            to_numpy(y_mean +  1.96 * y_std),
                            to_numpy(y_mean -  1.96 * y_std),
                            color='tab:blue', alpha=0.2)

            
            # Plot Samples
            # sample1, sample2 = to_numpy(dist.sample()), to_numpy(dist.sample())        
            # plt.plot(to_numpy(x_all), sample1, label='Sample', color='green', alpha=0.5)
            # plt.plot(to_numpy(x_all), sample2, label='Sample', color='orange', alpha=0.5)

            # Make predictions with oracle GP (if GP dataset)
            if args.data != 'sawtooth':
                gp = data_gen.gp
                post = gp.measure | (gp(to_numpy(x_context).astype(np.float64)), to_numpy(y_context).astype(np.float64))
                gp_mean, gp_lower, gp_upper = post(gp(to_numpy(x_all))).marginals()

                plt.plot(to_numpy(x_all), gp_mean, color='black', label='Oracle GP')
                # plt.plot(to_numpy(x_all), gp_lower, color='black', alpha=0.8)
                # plt.plot(to_numpy(x_all), gp_upper, color='black', alpha=0.8)
                plt.fill_between(to_numpy(x_all),
                                gp_lower,
                                gp_upper,
                                color='black', alpha=0.2)

            plt.ylim(-3., 3)
            # plt.axis('off')
            plt.legend(prop={'size': 16})
    plt.savefig(wd.file(f'plot-{i + 1}.png'), bbox_inches='tight')
    plt.close()
            


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

parser.add_argument('--seed',
                    default=0,
                    type=int,
                    help='Random seed to use.')

parser.add_argument('--std_noise',
                    default=1e-1,
                    type=float,
                    help='Standard dev. of noise added to GP-generated data.')

parser.add_argument('--batch_size',
                    default=1,
                    type=int,
                    help='Number of tasks per batch sampled.')

parser.add_argument('--max_num_context',
                    default=32,
                    type=int,
                    help='Maximum number of context points.')

parser.add_argument('--max_num_target',
                    default=100,
                    type=int,
                    help='Maximum number of target points.')

parser.add_argument('--num_plots',
                    default=3,
                    type=int,
                    help='Iterations (# batches sampled) for testing.')

parser.add_argument('--iterations',
                    default=5,
                    type=int,
                    help='Iterations (# batches sampled) for testing.')

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


# =============================================================================
# Model arguments
# =============================================================================

parser.add_argument('model',
                    choices=['GNP',
                             'AGNP',
                             'MeanTEGNP',
                             'MeanTEAGNP',
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

# Load working directory
if args.root:
    working_directory = WorkingDirectory(root=args.root)
        
else:
    experiment_name = os.path.join('_experiments',
                                   f'{args.data}',
                                   f'models',
                                   f'{args.model}',
                                   f'{args.covtype}',
                                   f'{args.seed}')
    working_directory = WorkingDirectory(root=experiment_name)

    

# =============================================================================
# Create data generators
# =============================================================================


# Training data generator parameters -- used for both Sawtooth and GP
gen_params = {
    'batch_size'                : args.batch_size,
    'x_range'                   : args.x_range,
    'max_num_context'           : args.max_num_context,
    'max_num_target'            : args.max_num_target,
    'include_context_in_target' : False,
    'device'                    : device
}

# Training data generator parameters -- specific to Sawtooth
gen_train_sawtooth_params = {
    'freq_range'  : args.freq_range,
    'shift_range' : args.shift_range,
    'trunc_range' : args.trunc_range
}

                    
if args.data == 'sawtooth':
    gen_plot = cnp.data.SawtoothGenerator(args.num_plots,
                                          **gen_train_sawtooth_params,
                                          **gen_params)
    
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
        
    gen_plot = cnp.data.GPGenerator(iterations_per_epoch=args.num_plots,
                                    kernel=kernel,
                                    std_noise=args.std_noise,
                                    **gen_params)

    


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
    
elif args.model == 'MeanTEGNP':
    model = StandardMeanTEGNP(covariance=cov,
                              add_noise=noise)
    
elif args.model == 'MeanTEAGNP':
    model = StandardMeanTEAGNP(covariance=cov,
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
# Plot
# =============================================================================
if device.type == 'cpu':
    load_dict = torch.load(working_directory.file('model_best.pth.tar',
                                                    exists=True),
                            map_location=torch.device('cpu'))
else:
    load_dict = torch.load(working_directory.file('model_best.pth.tar',
                                                    exists=True))
    
    model.load_state_dict(load_dict['state_dict'])


plot_dir = os.path.join('_experiments',
                        f'{args.data}',
                        f'models',
                        f'{args.model}',
                        f'{args.covtype}',
                        f'{args.seed}',
                        'TrainedModelPlots')
plot_dir = WorkingDirectory(root=plot_dir)
for i in range(args.iterations):
    plot_tasks(gen_plot, model, plot_dir, args, i)

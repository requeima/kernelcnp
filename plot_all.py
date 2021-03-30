import argparse

import numpy as np
import stheno.torch as stheno
import torch
import matplotlib.pyplot as plt
import os

import gnp.data
from gnp.architectures import SimpleConv, UNet
from gnp.experiment import (
    report_loss,
    generate_root,
    WorkingDirectory,
    save_checkpoint
)
from gnp.gnp import GNP, AGNP
from gnp.convgnp import ConvGNP
from gnp.cov import (
    InnerProdCov,
    KvvCov,
    MeanFieldCov,
    AddHomoNoise,
    AddHeteroNoise,
    AddNoNoise
)

from gnp.set_conv import ConvCNP
from gnp.utils import device, gaussian_logpdf

# Move data to device
to_numpy = lambda x: x.squeeze().cpu().numpy()

def plot_task(task, model):
    num_points = 200
    x_all = torch.linspace(-2., 2., num_points)
    
    x_context = task['x_context'].to(device)
    y_context = task['y_context'].to(device)

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
    plt.axis('off')
    plt.legend(prop={'size': 16})

# Parse arguments given to the script.
parser = argparse.ArgumentParser()
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

parser.add_argument('--x_range',
                    default=[-3., 3.],
                    nargs='+',
                    type=float,
                    help='Range of input x for sampled data.')

parser.add_argument('--max_num_context',
                    default=32,
                    type=int,
                    help='Maximum number of context points.')

parser.add_argument('--max_num_target',
                    default=32,
                    type=int,
                    help='Maximum number of target points.')

parser.add_argument('--gpu',
                    default=0,
                    type=int,
                    help='GPU to run experiment on. Defaults to 0.')
                    
args = parser.parse_args()

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
    


# Model list
models = ["GNP", "AGNP", "convGNP", "TEGNP"]
covs = ["innerprod-homo-4basisdims", "innerprod-hetero-4basisdims", "kvv-homo-4basisdims", "kvv-hetero-4basisdims",
        "innerprod-homo-512basisdims", "innerprod-hetero-512basisdims", "kvv-homo-512basisdims", "kvv-hetero-512basisdims",
        "meanfield"]


for task_num, task in enumerate(gen_plot):
    for m in models:
        for c in covs:
            experiment_name = os.path.join('_experiments', 
                                            f'{args.data}', 
                                            f'{m}', 
                                            f'{c}')
            wd = WorkingDirectory(root=experiment_name)

            # Covariance method
            if c == 'innerprod-homo':
                cov = InnerProdCov(args.num_basis_dim)
                noise = AddHomoNoise()
            elif c == 'innerprod-hetero':
                cov = InnerProdCov(args.num_basis_dim)
                noise = AddHeteroNoise()
            elif c == 'kvv-homo':
                cov = KvvCov(args.num_basis_dim)
                noise = AddHomoNoise()
            elif c == 'kvv-hetero':
                cov = KvvCov(args.num_basis_dim)
                noise = AddHomoNoise()
            elif c == 'meanfield':
                cov = MeanFieldCov(num_basis_dim=1)
                noise = AddNoNoise()
            
            # Load model.
            if m == 'GNP':
                model = GNP(latent_dim=128,
                            cov=cov,
                            noise=noise)
            elif m == 'AGNP':
                model = AGNP(latent_dim=128,
                            cov=cov,
                            noise=noise)
            elif m == 'convGNP':
                model = ConvGNP(rho=UNet(), 
                                points_per_unit=64,
                                cov=cov,
                                noise=noise)
            
            model.to(device)
            
            # Load saved model.
            if device.type == 'cpu':
                load_dict = torch.load(wd.file('model_best.pth.tar', exists=True), map_location=torch.device('cpu'))
            else:
                load_dict = torch.load(wd.file('model_best.pth.tar', exists=True))
            model.load_state_dict(load_dict['state_dict'])

            
            fig = plt.figure(figsize=(24, 8))       
            plot_task(task, model)
            plt.savefig(wd.file('tmp_plot_%s' % task_num), bbox_inches='tight')
            plt.close()
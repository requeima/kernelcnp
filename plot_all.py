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
parser.add_argument('--num_basis_dim',
                    default=1024,
                    type=int,
                    help='Maximum number of context points for test set.')                
args = parser.parse_args()

# Load data generator.
if args.data == 'sawtooth':
    gen = gnp.data.SawtoothGenerator()
    gen_val = gnp.data.SawtoothGenerator(num_tasks=60)
    gen_test = gnp.data.SawtoothGenerator(num_tasks=2048)
    gen_plot = gnp.data.SawtoothGenerator(num_tasks=16, batch_size=1, max_train_points=20)
else:
    if args.data == 'eq':
        kernel = stheno.EQ().stretch(0.25)
    elif args.data == 'matern':
        kernel = stheno.Matern52().stretch(0.25)
    elif args.data == 'noisy-mixture':
        kernel = stheno.EQ().stretch(1.) + \
                 stheno.EQ().stretch(.25) + \
                 0.001 * stheno.Delta()
    elif args.data == 'weakly-periodic':
        kernel = stheno.EQ().stretch(0.5) * stheno.EQ().periodic(period=0.25)
    else:
        raise ValueError(f'Unknown data "{args.data}".')
    gp = stheno.GP(kernel)
    gen = gnp.data.GPGenerator(kernel=kernel)
    gen_val = gnp.data.GPGenerator(kernel=kernel, num_tasks=60)
    gen_test = gnp.data.GPGenerator(kernel=kernel, num_tasks=2048)
    gen_plot = gnp.data.GPGenerator(kernel=kernel, max_train_points=20, num_tasks=16, batch_size=1)

# Model list
models = ["GNP", "AGNP", "convGNP"]
covs = ["innerprod-homo", "innerprod-hetero", "kvv-homo", "kvv-hetero", "meanfield"]


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
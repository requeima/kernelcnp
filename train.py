import argparse

import numpy as np
import stheno.torch as stheno
import torch
import matplotlib.pyplot as plt

import convcnp.data
from convcnp.architectures import SimpleConv, UNet
from convcnp.cnp import RegressionANP as ANP
from convcnp.cnp import RegressionCNP as CNP
from convcnp.experiment import (
    report_loss,
    generate_root,
    WorkingDirectory,
    save_checkpoint
)
from kernelcnp.model import (
    InnerProdHomoNoiseKernelCNP, 
    InnerProdHeteroNoiseKernelCNP, 
    KvvHomoNoiseKernelCNP, 
    KvvHeteroNoiseKernelCNP
)

from convcnp.set_conv import ConvCNP
from convcnp.utils import device, gaussian_logpdf


def validate(data, model, report_freq=None):
    """Compute the validation loss."""
    model.eval()
    likelihoods = []
    with torch.no_grad():
        for step, task in enumerate(data):
            num_target = task['y_target'].shape[1]
            y_mean, y_std = \
                model(task['x_context'], task['y_context'], task['x_target'])
            obj = \
                 gaussian_logpdf(task['y_target'], y_mean, y_std,
                                 'batched_mean')
            likelihoods.append(obj.item() / num_target)
            if report_freq:
                avg_ll = np.array(likelihoods).mean()
                report_loss('Validation', avg_ll, step, report_freq)
    avg_ll = np.array(likelihoods).mean()
    return avg_ll


def train(data, model, opt, report_freq):
    """Perform a training epoch."""
    model.train()
    losses = []
    for step, task in enumerate(data):
        y_mean, y_std = model(task['x_context'], task['y_context'],
                              task['x_target'])
        
        
        obj = -gaussian_logpdf(task['y_target'], y_mean, y_std, 'batched_mean')

        # Optimization
        obj.backward()
        opt.step()
        opt.zero_grad()

        # Track training progress
        losses.append(obj.item())
        avg_loss = np.array(losses).mean()
        report_loss('Training', avg_loss, step, report_freq)
    return avg_loss

# Move data to device
to_numpy = lambda x: x.squeeze().cpu().numpy()

def plot_task(task, model):
    num_points = 200
    x_all = torch.linspace(-2., 2., num_points)
    
    x_context = task['x_context'].to(device)
    y_context = task['y_context'].to(device)

    # Make predictions with model
    with torch.no_grad():
        y_mean, y_std = model(x_context, y_context, x_all[None, :, None].to(device))

        # Get the marginals if we are predicting the full covariance
        if y_std.shape[-1] > 1:
            dist = torch.distributions.multivariate_normal.MultivariateNormal(loc=y_mean[0,:, 0], covariance_matrix=y_std[0, :, :])
            y_std = torch.diagonal(y_std, 0, dim1=-2, dim2=-1)[:, :, None]

        # Plot context set
        plt.scatter(to_numpy(x_context), to_numpy(y_context), label='Context Set', color='black')

        # Plot model predictions
        plt.plot(to_numpy(x_all), to_numpy(y_mean), label='Model Output', color='blue')
        plt.fill_between(to_numpy(x_all),
                         to_numpy(y_mean +  y_std),
                         to_numpy(y_mean -  y_std),
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
parser.add_argument('model',
                    choices=['convcnp', 
                             'convcnpxl', 
                             'cnp', 
                             'anp',
                             'InnerProdHomoNoiseKernelCNP', 
                             'InnerProdHeteroNoiseKernelCNP', 
                             'KvvHomoNoiseKernelCNP', 
                             'KvvHeteroNoiseKernelCNP'],
                    help='Choice of model. ')
parser.add_argument('--root',
                    help='Experiment root, which is the directory from which '
                         'the experiment will run. If it is not given, '
                         'a directory will be automatically created.')
parser.add_argument('--train',
                    action='store_true',
                    help='Perform training. If this is not specified, '
                         'the model will be attempted to be loaded from the '
                         'experiment root.')
parser.add_argument('--epochs',
                    default=100,
                    type=int,
                    help='Number of epochs to train for.')
parser.add_argument('--learning_rate',
                    default=1e-3,
                    type=float,
                    help='Learning rate.')
parser.add_argument('--weight_decay',
                    default=1e-5,
                    type=float,
                    help='Weight decay.')
args = parser.parse_args()

# Load working directory.
if args.root:
    wd = WorkingDirectory(root=args.root)
else:
    experiment_name = f'{args.model}-{args.data}'
    wd = WorkingDirectory(root=generate_root(experiment_name))

# Load data generator.
if args.data == 'sawtooth':
    gen = convcnp.data.SawtoothGenerator()
    gen_val = convcnp.data.SawtoothGenerator(num_tasks=60)
    gen_test = convcnp.data.SawtoothGenerator(num_tasks=2048)
    gen_plot = convcnp.data.SawtoothGenerator(num_tasks=16, batch_size=1, max_train_points=20)
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
    gen = convcnp.data.GPGenerator(kernel=kernel)
    gen_val = convcnp.data.GPGenerator(kernel=kernel, num_tasks=60)
    gen_test = convcnp.data.GPGenerator(kernel=kernel, num_tasks=2048)
    gen_plot = convcnp.data.GPGenerator(kernel=kernel, max_train_points=20, num_tasks=16, batch_size=1)

# Load model.
if args.model == 'convcnp':
    model = ConvCNP(learn_length_scale=True,
                    points_per_unit=64,
                    architecture=SimpleConv())
elif args.model == 'convcnpxl':
    model = ConvCNP(learn_length_scale=True,
                    points_per_unit=64,
                    architecture=UNet())
elif args.model == 'cnp':
    model = CNP(latent_dim=128)
elif args.model == 'anp':
    model = ANP(latent_dim=128)
elif args.model == 'InnerProdHomoNoiseKernelCNP':
    model = InnerProdHomoNoiseKernelCNP(rho=UNet(), points_per_unit=64, num_basis_dim=1024)
elif args.model == 'InnerProdHeteroNoiseKernelCNP':
    model = InnerProdHeteroNoiseKernelCNP(rho=UNet(), points_per_unit=64, num_basis_dim=1024)
elif args.model == 'KvvHomoNoiseKernelCNP':
    model = KvvHomoNoiseKernelCNP(rho=UNet(), points_per_unit=64, num_basis_dim=1024)
elif args.model == 'KvvHeteroNoiseKernelCNP':
    model = KvvHeteroNoiseKernelCNP(rho=UNet(), points_per_unit=64, num_basis_dim=1024)
else:
    raise ValueError(f'Unknown model {args.model}.')

model.to(device)

# Perform training.
opt = torch.optim.Adam(model.parameters(),
                       args.learning_rate,
                       weight_decay=args.weight_decay)
if args.train:
    # Run the training loop, maintaining the best objective value.
    best_obj = -np.inf
    for epoch in range(args.epochs):
        print('\nEpoch: {}/{}'.format(epoch + 1, args.epochs))

        # Compute training objective.
        train_obj = train(gen, model, opt, report_freq=50)
        report_loss('Training', train_obj, 'epoch')

        # Compute validation objective.
        val_obj = validate(gen_val, model, report_freq=20)
        report_loss('Validation', val_obj, 'epoch')

        # Update the best objective value and checkpoint the model.
        is_best = False
        if val_obj > best_obj:
            best_obj = val_obj
            is_best = True
        save_checkpoint(wd,
                        {'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'best_acc_top1': best_obj,
                         'optimizer': opt.state_dict()},
                        is_best=is_best)

else:
    # Load saved model.
    if device.type == 'cpu':
        load_dict = torch.load(wd.file('model_best.pth.tar', exists=True), map_location=torch.device('cpu'))
    else:
        load_dict = torch.load(wd.file('model_best.pth.tar', exists=True))
    model.load_state_dict(load_dict['state_dict'])

# Test model on ~2000 tasks.
test_obj = validate(gen_test, model)
print('Model averages a log-likelihood of %s on unseen tasks.' % test_obj)
with open(wd.file('test_log_likelihood.txt'), 'w') as f:
    f.write(str(test_obj))

# Plot the models
for task_num, task in enumerate(gen_plot):
    fig = plt.figure(figsize=(24, 8))
    plot_task(task, model)
    plt.savefig(wd.file('tmp_plot_%s' % task_num), bbox_inches='tight')
    plt.close()
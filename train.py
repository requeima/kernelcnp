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

from cnp.cnp import StandardGNP, StandardAGNP, StandardConvGNP, StandardFullyConnectedTEGNP
from cnp.cov import (
    InnerProdCov,
    KvvCov,
    MeanFieldCov,
    AddHomoNoise,
    AddHeteroNoise,
    AddNoNoise
)

from cnp.utils import device
from torch.distributions import MultivariateNormal


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


def train(data, model, opt, report_freq):
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
        opt.step()
        opt.zero_grad()

        # Track training progress
        losses.append(obj.item())
        avg_loss = np.array(losses).mean()
        report_loss('Training', avg_loss, step, report_freq)

    return avg_loss


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
                    help='Test the model and record the values in the experimental root.')
parser.add_argument('--test_context_num',
                    default=2048,
                    type=int,
                    help='Maximum number of context points for test set.')
parser.add_argument('--num_basis_dim',
                    default=1024,
                    type=int,
                    help='Maximum number of context points for test set.')
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
    experiment_name = os.path.join('_experiments', f'{args.data}', f'{args.model}', f'{args.covtype}')
    wd = WorkingDirectory(root=experiment_name)

# Load data generator.
if args.data == 'sawtooth':
    gen = cnp.data.SawtoothGenerator()
    gen_val = cnp.data.SawtoothGenerator(num_tasks=60)
    gen_test = cnp.data.SawtoothGenerator(num_tasks=args.test_context_num)
    gen_plot = cnp.data.SawtoothGenerator(num_tasks=16, batch_size=1, max_train_points=20)
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
    gen = cnp.data.GPGenerator(kernel=kernel)
    gen_val = cnp.data.GPGenerator(kernel=kernel, num_tasks=60)
    gen_test = cnp.data.GPGenerator(kernel=kernel, num_tasks=args.test_context_num)
    gen_plot = cnp.data.GPGenerator(kernel=kernel, max_train_points=20, num_tasks=16, batch_size=1)

# Covariance method
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
# Load model.
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
                        is_best=is_best,
                        epoch=epoch)

else:
    # Load saved model.
    if device.type == 'cpu':
        load_dict = torch.load(wd.file('model_best.pth.tar', exists=True), map_location=torch.device('cpu'))
    else:
        load_dict = torch.load(wd.file('model_best.pth.tar', exists=True))
    model.load_state_dict(load_dict['state_dict'])

if args.test:
    # Test model on ~2000 tasks.
    test_obj, test_obj_std_error = validate(gen_test, model, std_error=True)
    print('Model averages a log-likelihood of %s +- %s on unseen tasks.' % (test_obj, test_obj_std_error))
    with open(wd.file('test_log_likelihood.txt'), 'w') as f:
        f.write(str(test_obj))
    with open(wd.file('test_log_likelihood_standard_error.txt'), 'w') as f:
        f.write(str(test_obj_std_error))

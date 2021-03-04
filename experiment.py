import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import stheno.torch as stheno

import convcnp.data
from convcnp.experiment import report_loss, RunningAverage
from convcnp.utils import gaussian_logpdf, init_sequential_weights, to_multiple
from convcnp.architectures import SimpleConv, UNet
from kernelcnp.model import InnerProductHomoscedasticKernelCNP
import time
#import lab as B

#B.epsilon = 1e-6

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def to_numpy(x):
    """Convert a PyTorch tensor to NumPy."""
    return x.squeeze().detach().cpu().numpy() 

def plot_task(task, idx, legend):
    x_context, y_context = to_numpy(task['x_context'][idx]), to_numpy(task['y_context'][idx])
    x_target, y_target = to_numpy(task['x_target'][idx]), to_numpy(task['y_target'][idx])
    
    # Plot context and target sets.
    plt.scatter(x_context, y_context, label='Context Set', color='black')
    
    # Infer GP posterior.
    post = gp.measure  | (gp(x_context), y_context)
    
    # Make and plot predictions on desired range.
    gp_mean, gp_lower, gp_upper = post(gp(x_test)).marginals()
    plt.plot(x_test, gp_mean, color='tab:green', label='Oracle GP')
    plt.fill_between(x_test, gp_lower, gp_upper, color='tab:green', alpha=0.1)
    if legend:
        plt.legend()

def train(data, model, opt):
    """Perform a training epoch."""
    ravg = RunningAverage()
    model.train()
    for step, task in enumerate(data):
        y_mean, y_cov = model(task['x_context'], task['y_context'], task['x_target'])
        dist = torch.distributions.multivariate_normal.MultivariateNormal(loc=y_mean[:, :, 0], covariance_matrix=y_cov)
        obj = torch.mean(-dist.log_prob(task['y_target'][:, :, 0]))
        obj.backward()
        opt.step()
        opt.zero_grad()
        ravg.update(obj.item() / data.batch_size, data.batch_size)
    return ravg.avg

def plot_model_task(model, task, idx, legend):
    num_functions = task['x_context'].shape[0]
    
    # Make predictions with the model.
    model.eval()
    with torch.no_grad():
        y_mean, y_cov = model(task['x_context'], task['y_context'], x_test.repeat(num_functions, 1, 1))
    
    y_std = torch.diagonal(y_cov, 0, dim1=-2, dim2=-1)[:, :, None]
    # Plot the task and the model predictions.
    x_context, y_context = to_numpy(task['x_context'][idx]), to_numpy(task['y_context'][idx])
    x_target, y_target = to_numpy(task['x_target'][idx]), to_numpy(task['y_target'][idx])
    dist = torch.distributions.multivariate_normal.MultivariateNormal(loc=y_mean[idx, :, 0], covariance_matrix=y_cov[idx, :, :])
    sample1, sample2 = to_numpy(dist.sample()), to_numpy(dist.sample())
    y_mean, y_std = to_numpy(y_mean[idx]), to_numpy(y_std[idx])

    # Plot Samples
    plt.plot(to_numpy(x_test[0]), sample1, label='Sample', color='green', alpha=0.5)
    plt.plot(to_numpy(x_test[0]), sample2, label='Sample', color='orange', alpha=0.5)

    # Plot context and target sets.
    plt.scatter(x_context, y_context, label='Context Set', color='black')
    plt.scatter(x_target, y_target, label='Target Set', color='red')
    
    # Plot model predictions.
    plt.plot(to_numpy(x_test[0]), y_mean, label='Model Output', color='blue')

    plt.fill_between(to_numpy(x_test[0]),
                     y_mean + 2 * y_std,
                     y_mean - 2 * y_std,
                     color='tab:blue', alpha=0.2)
    if legend:
        plt.legend()

model = InnerProductHomoscedasticKernelCNP(rho=UNet(), points_per_unit=64, num_basis_dim=1024)
model.to(device)

# Some training hyper-parameters:
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100
PLOT_FREQ = 10

# Initialize optimizer
opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

kernel = stheno.Matern52().stretch(0.25)
gen = convcnp.data.GPGenerator(kernel=kernel, batch_size=8, include_context_in_target=False)

# x_test = np.linspace(-2, 2, 300)
# gp = stheno.GP(kernel)

# task = gen.generate_task()
# fig = plt.figure(figsize=(24, 5))
# for i in range(3):
#     plt.subplot(1, 3, i + 1)
#     plot_task(task, i, legend=i==2)
# plt.show()

# Create a fixed set of outputs to predict at when plotting.
x_test = torch.linspace(-2., 2., 200)[None, :, None].to(device)

t = time.time()
# Run the training loop.
for epoch in range(NUM_EPOCHS):

    # Compute training objective.
    train_obj = train(gen, model, opt)

    # Plot model behaviour every now and again.
    if epoch % PLOT_FREQ == 0:
        print('Epoch %s: NLL %.3f' % (epoch, train_obj))
        task = gen.generate_task()
        fig = plt.figure(figsize=(24, 5))
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plot_model_task(model, task, idx=i, legend=i==2)
        plt.savefig('model epoch: ' + str(epoch))
        print('Elapsed: ' + str(time.time() - t))

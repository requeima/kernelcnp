import numpy as np
import matplotlib.pyplot as plt

import torch


def eq_covariance(inputs1,
                  inputs2,
                  scale,
                  cov_coeff,
                  noise_coeff):
    
    diff = inputs1[:, :, None, :] - inputs2[:, None, :, :]
    
    quad = np.sum((diff / (2 * scale ** 2)) ** 2, axis=-1)
    
    cov = cov_coeff ** 2 * np.exp(- quad)
    
    return cov


def sample_datasets_from_gps(low,
                             high,
                             num_batches,
                             batch_size,
                             scale,
                             cov_coeff,
                             noise_coeff,
                             as_tensor):
    
    x = np.random.uniform(low=low,
                          high=high,
                          size=(num_batches, batch_size))[:, :, None]

    cov = eq_covariance(x, x, scale, cov_coeff, noise_coeff)
    cov = cov + noise_coeff ** 2 * np.eye(cov.shape[-1])[None, :, :]
    
    zeros = np.zeros((cov.shape[1],))
    ones = np.diag(np.ones_like(zeros))
    
    noise = np.random.multivariate_normal(zeros,
                                          ones,
                                          size=(num_batches,))
    chol = np.linalg.cholesky(cov)
    
    y = np.einsum('bij, bj -> bi', chol, noise)
    
    x = torch.tensor(x).float()
    y = torch.tensor(y).float()[:, :, None]
    
    return x, y


def gp_post_pred(train_inputs,
                 train_outputs,
                 pred_inputs,
                 scale,
                 cov_coeff,
                 noise_coeff):
    
    K = eq_covariance(train_inputs,
                      train_inputs,
                      scale,
                      cov_coeff,
                      noise_coeff)[0]
    
    K = K + noise_coeff ** 2 * np.eye(K.shape[-1])
    
    k_star = eq_covariance(pred_inputs,
                           train_inputs,
                           scale,
                           cov_coeff,
                           noise_coeff)[0]
    
    means = np.dot(k_star, np.linalg.solve(K, train_outputs[0, :, 0]))
    
    K_inv_k_star = np.linalg.solve(K, k_star.T)
    stds = (cov_coeff - np.diag(np.einsum('ij, jk -> ik', k_star, K_inv_k_star))) ** 0.5
    
    return means, stds


def plot_samples_and_predictions(gnp,
                                 xmin,
                                 xmax,
                                 batch_size,
                                 scale,
                                 cov_coeff,
                                 noise_coeff,
                                 step):
    
    num_batches = 3
    
    plt.figure(figsize=(16, 3))

    # Sample three datasets
    inputs, outputs = sample_datasets_from_gps(xmin,
                                               xmax,
                                               num_batches,
                                               batch_size,
                                               scale,
                                               cov_coeff,
                                               noise_coeff,
                                               True)
    
    
    for i in range(3):
        
        plt.subplot(1, 3, i + 1)
            
        N = np.random.choice(np.arange(1, inputs.shape[1]-1))
        
        ctx_in = inputs[i:i+1, :N, :]
        ctx_out = outputs[i:i+1, :N, :]
        
        trg_in = inputs[i:i+1, N:, :]
        trg_out = outputs[i:i+1, N:, :]
        
        plot_inputs = torch.linspace(xmin, xmax, 100)[None, :, None]

        tensors = gnp(ctx_in, ctx_out, plot_inputs)
        
        mean, cov, cov_plus_noise = [tensor.detach() for tensor in tensors]

        gp_means, gp_stds = gp_post_pred(ctx_in.numpy(),
                                         ctx_out.detach().numpy(),
                                         plot_inputs.numpy(),
                                         scale,
                                         cov_coeff,
                                         noise_coeff)
        
        try:
            cov_plus_jitter = cov[0, :, :] + 1e-6 * torch.eye(cov.shape[-1])
            dist = torch.distributions.MultivariateNormal(loc=mean[0, :, 0],
                                                          covariance_matrix=cov_plus_jitter)
            
            for j in range(100):
                sample = dist.sample()
                plt.plot(plot_inputs[0, :, 0], sample, color='blue', alpha=0.05, zorder=2)
                
        except:
            plt.fill_between(plot_inputs[0, :, 0],
                             mean[0, :, 0] - torch.diag(cov[0, :, :]),
                             mean[0, :, 0] + torch.diag(cov[0, :, :]),
                             color='blue',
                             alpha=0.3,
                             zorder=1)
            

        plt.plot(plot_inputs[0, :, 0],
                 mean[0, :, 0],
                 '--',
                 color='k')
        
        plt.fill_between(plot_inputs[0, :, 0],
                         gp_means - gp_stds,
                         gp_means + gp_stds,
                         color='gray',
                         alpha=0.3,
                         zorder=1)

        plt.scatter(ctx_in[0, :, 0],
                    ctx_out[0, :, 0],
                    s=100,
                    marker='+',
                    color='black',
                    label='Context',
                    zorder=3)

        plt.scatter(trg_in[0, :, 0],
                    trg_out[0, :, 0],
                    s=100,
                    marker='+',
                    color='red',
                    label='Target',
                    zorder=3)

    plt.show()
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
                          size=(num_batches, batch_size, 1))

    cov = eq_covariance(x, x, scale, cov_coeff, noise_coeff)
    cov = cov + noise_coeff ** 2 * np.eye(cov.shape[-1])[None, :, :]

    zeros = np.zeros((cov.shape[1],))
    ones = np.diag(np.ones_like(zeros))

    noise = np.random.normal(size=(num_batches, batch_size, 1))
    
    chol = np.linalg.cholesky(cov)

    y = np.einsum('bij, bjd -> bid', chol, noise)

    x = torch.tensor(x).float()
    y = torch.tensor(y).float()

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
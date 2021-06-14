import subprocess
from itertools import product
import time
import torch
import nvsmi
import os

# Use all GPUs by default, and memory % above which no experiments are sent
GPUS_TO_USE = [str(i) for i in range(torch.cuda.device_count())]
GPU_MEMORY_PERCENTAGE = 30.

# Model and data generator configurations
data_generators = ['eq',
                   'matern',
                   'noisy-mixture',
                   'noisy-mixture-slow-100',
                   'noisy-mixture-slow',
                   'weakly-periodic',
                   'weakly-periodic-slow-100',
                   'weakly-periodic-slow',
                   'sawtooth']

data_generators = ['eq',
                   'matern',
                   'noisy-mixture',
                   'weakly-periodic',
                   'sawtooth']

cond_models = ['GNP', 'AGNP', 'convGNP']
latent_models = ['ANP', 'convNP']
fcgnp_models = ["FullConvGNP"]

covs = ['innerprod-homo', 'kvv-homo', 'meanfield']

x_dims = ['1']

seeds = [str(i) for i in range(1)]

# Configs for conditional models
cond_configs = list(product(seeds, x_dims, data_generators, cond_models, covs))
#latent_configs = list(product(seeds, x_dims, data_generators, latent_models, ["meanfield"]))
fcgnp_configs = list(product(seeds, x_dims, data_generators, fcgnp_models, ["meanfield"]))

configs = cond_configs + fcgnp_configs

FNULL = open(os.devnull, 'w')

if __name__ == '__main__':

    while len(configs) > 0:

        for gpu_id in GPUS_TO_USE:
            
            percent_memory_used = list(nvsmi.get_gpus())[int(gpu_id)].mem_util

            if percent_memory_used < GPU_MEMORY_PERCENTAGE:

                seed, x_dim, gen, model, cov = configs[0]
                
                

                command = ['python',
                            '-W',
                            'ignore',
                           'test.py',
                           gen,
                           model,
                           cov,
                           '--x_dim',
                           x_dim,
                           '--seed',
                           seed,
                           '--gpu',
                           gpu_id]
                
                print(f'Starting experiment, memory: {percent_memory_used:.1f}% '
                      f'(max. allowed {GPU_MEMORY_PERCENTAGE}%)\n{command}')
                
                process = subprocess.call(command)

                configs = configs[1:]

                time.sleep(5e0)
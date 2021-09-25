import subprocess
from itertools import product
import time
import torch
import nvsmi
import os
from cnp.experiment import WorkingDirectory

# Use all GPUs by default, and memory % above which no experiments are sent
GPUS_TO_USE = [str(i) for i in range(torch.cuda.device_count())]
GPU_MEMORY_PERCENTAGE = 50.

# Model and data generator configurations
data_generators = ['weakly-periodic']
models = ['convGNP']
covs = ['sum-kvv-homo']
x_dims = ['1']
# num_basis_dim = ["2", "8", "64", "128", "512", "4096"]
# num_sum_elements = ["1", "2", "8", "16", "64"]

num_basis_dim = ["256",]
num_sum_elements = ["4", "6", "12","20", "28", "34", "36", "42", "48", "52", "58", "72"]

seeds = [str(i) for i in range(1)]

configs = list(product(seeds, x_dims, data_generators, models, covs, num_basis_dim, num_sum_elements))
working_directory = WorkingDirectory(root='experiments/synthetic/scripts')

FNULL = open(os.devnull, 'w')

if __name__ == '__main__':

    while len(configs) > 0:

        for gpu_id in GPUS_TO_USE:
            
            percent_memory_used = list(nvsmi.get_gpus())[int(gpu_id)].mem_util

            if percent_memory_used < GPU_MEMORY_PERCENTAGE:

                
                

                seed, x_dim, gen, model, cov, num_bas, num_sum = configs[0]
                
                command = ['python',
                           working_directory.file('test.py'),
                           gen,
                           model,
                           cov,
                           '--x_dim', x_dim,
                           '--seed', seed,
                           '--gpu', gpu_id,
                           '--num_basis_dim', num_bas,
                           '--num_sum_elements', num_sum
                           ]
                
                print(f'Starting experiment, memory: {percent_memory_used:.1f}% '
                      f'(max. allowed {GPU_MEMORY_PERCENTAGE}%)\n{command}')
                
                process = subprocess.call(command)

                configs = configs[1:]

                time.sleep(5e0)

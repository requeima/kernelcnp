import subprocess
from itertools import product
import time
import torch
import nvsmi
import os

# Use all GPUs by default, and memory % above which no experiments are sent
GPUS_TO_USE = [str(i) for i in range(torch.cuda.device_count())]
GPU_MEMORY_PERCENTAGE = 5.

# Model and data generator configurations
data_generators = ['eq',
                   'matern',
                   'noisy-mixture',
                   'weakly-periodic',
                   'sawtooth']

models = ['GNP',
          'AGNP',
          'convGNP',
          'TEGNP']

covs = ['innerprod-homo',
        'kvv-homo',
        'meanfield']

seeds = [str(i) for i in range(1, 3)]

configs = list(product(seeds, data_generators, models, covs))

# Other experiment parameters
optional_params = {
    '--epochs'          : 8000,
    '--num_train_iters' : 1,
    '--learning_rate'   : 1e-3
}

optional_params = [(k, str(v)) for k, v in optional_params.items()]
optional_params = [param for tup in optional_params for param in tup]

FNULL = open(os.devnull, 'w')

if __name__ == '__main__':

    while len(configs) > 0:

        for gpu_id in GPUS_TO_USE:
            
            percent_memory_used = list(nvsmi.get_gpus())[int(gpu_id)].mem_util

            if percent_memory_used < GPU_MEMORY_PERCENTAGE:

                seed, gen, moodel, cov = configs[0]
                
                command = ['python', 'train.py', gen, moodel, cov, '--train', '--seed', seed, '--gpu', gpu_id]
                command = command + optional_params
                
                print(f'Starting experiment, memory: {percent_memory_used:.1f}% '
                      f'(max. allowed {GPU_MEMORY_PERCENTAGE}%)\n{command}')
                
                process = subprocess.Popen(command)

                configs = configs[1:]

                time.sleep(5e0)

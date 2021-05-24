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
                   'noisy-mixture-slow',
                   'weakly-periodic',
                   'weakly-periodic-slow']

x_dims = ['1']

seeds = [str(i) for i in range(1)]

configs = list(product(seeds, x_dims, data_generators))

FNULL = open(os.devnull, 'w')

if __name__ == '__main__':

    while len(configs) > 0:

        for gpu_id in GPUS_TO_USE:
            
            percent_memory_used = list(nvsmi.get_gpus())[int(gpu_id)].mem_util

            if percent_memory_used < GPU_MEMORY_PERCENTAGE:

                seed, x_dim, gen = configs[0]
                
                command = ['python',
                            '-W',
                            'ignore',
                           'test_oracle.py',
                           gen,
                           '--x_dim',
                           x_dim,
                           '--seed',
                           seed]
                
                print(f'Starting experiment, memory: {percent_memory_used:.1f}% '
                      f'(max. allowed {GPU_MEMORY_PERCENTAGE}%)\n{command}')
                
                process = subprocess.Popen(command)

                configs = configs[1:]

                time.sleep(5e0)

import subprocess
from itertools import product
import time
import torch
import nvsmi



# Maximum number of experiments per GPU at any one time
MAX_EXPERIMENTS_PER_GPU = 3

# Use all GPUs by default, and memory % above which no experiments are sent
GPUS_TO_USE = list(range(torch.cuda.device_count()))
GPU_MEMORY_PERCENTAGE = 70

# Model and data generator configurations
data_generators = ['eq',
                   'matern',
                   'noisy-mixture',
                   'weakly-periodic',
                   'sawtooth']

models = ['GNP',
          'AGNP',
          'convGNP',
          'TEGNP',
          'TEGNP']

covs = ['innerprod-homo',
        'innerprod-hetero',
        'kvv-homo',
        'kvv-hetero',
        'meanfield']

configs = list(product(data_generators, models, covs))

# Other experiment parameters
optional_params = {
    '--epochs'          : 10000,
    '--num_train_iters' : 5,
    '--learning_rate'   : 1e-3
}

optional_params = [(k, str(v)) for k, v in optional_params.items()]
optional_params = [param for tup in optional_params for param in tup]

if __name__ == '__main__':

    while len(configs) > 0:

        time.sleep(1.)

        for gpu_id in GPUS_TO_USE:
            
            percent_memory_used = list(nvsmi.get_gpus())[gpu_id].gpu_util

            if percent_memory_used < GPU_MEMORY_PERCENTAGE:

                gen, moodel, cov = configs[0]
                
                command = ['python', 'train.py', gen, moodel, cov, '--train']
                command = command + optional_params
                
                process = subprocess.Popen(command)

                configs = configs[1:]
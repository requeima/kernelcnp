from itertools import product

# Model and data generator configurations
data_generators = ['random',
                   'sawtooth',
                   'eq',
                   'matern',
                   'noisy-mixture',
                   'weakly-periodic']

# Conditional models -- without FullConvGNP
cond_models = ['GNP', 'AGNP', 'convGNP']

# Latent models
latent_models = ['ANP', 'convNP']

# Covariances for conditional models
covs = ['innerprod-homo', 'kvv-homo', 'meanfield']

# Seeds to try
seeds = [str(i) for i in range(1)]

# Configs for conditional models
cond_configs = list(product(seeds, data_generators, cond_models, covs))

# Configs for FullConvGNP
fcgnp_configs = list(product(seeds, data_generators, ['FullConvGNP'], ['meanfield']))

# Configs for latent models
latent_configs = list(product(seeds, data_generators, latent_models, ['meanfield']))


# Configs for all experiments
configs = cond_configs + fcgnp_configs + latent_configs

commands = [['python',
             'train.py',
             gen,
             model,
             cov,
             '--x_dim',
             '1',
             '--seed',
             seed,
             '--gpu',
             '0']
            for seed, gen, model, cov in configs]


for command in commands:

    print(command)

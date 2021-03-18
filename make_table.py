import pandas as pd
import numpy as np
from convcnp.experiment import WorkingDirectory, generate_root

models = ['convcnp', 
          'convcnpxl', 
          'cnp', 
          'anp',
          'InnerProdHomoNoiseKernelCNP', 
          'InnerProdHeteroNoiseKernelCNP',
          'KvvHomoNoiseKernelCNP',
          'KvvHeteroNoiseKernelCNP']

experiments = ["eq",
               "matern",
               "noisy-mixture",
               "weakly-periodic",
               "sawtooth"]

experiments_with_error = ["eq", "eq-error", 
               "matern", "matern-errror", 
               "noisy-mixture", "noisy-mixture-error", 
               "weakly-periodic", "weakly-periodic-error", 
               "sawtooth", "sawtooth-error"]

# Create an empty 
df = pd.DataFrame(index=models, columns=experiments_with_error)

# Fill the dataframe
for m in models:
    for e, err in zip(experiments, experiments_with_error):
        root = '_experiments/%s-%s' % (m,e)
        wd = WorkingDirectory(root=root)
        mean = np.loadtxt(wd.file('test_log_likelihood.txt', exists=True))
        error = np.loadtxt(wd.file('test_log_likelihood_standard_error.txt', exists=True))
        df.at[m, e] = mean
        df.at[m, err] = error


df.to_csv('_experiments/full_results.csv', float_format='%.3f')



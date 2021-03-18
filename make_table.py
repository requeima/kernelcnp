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

experiments = ["eq", "weakly-periodic", "sawtooth"]

experiments_with_error = []
for e in experiments:
    experiments_with_error.append(e)
    experiments_with_error.append(e + "-error")    

# Create an empty 
df = pd.DataFrame(index=models, columns=experiments_with_error)

# Fill the dataframe
for m in models:
    for e in experiments:
        err = e + "-error"
        root = '_experiments/%s-%s' % (m,e)
        wd = WorkingDirectory(root=root)
        mean = np.loadtxt(wd.file('test_log_likelihood.txt', exists=True))
        error = np.loadtxt(wd.file('test_log_likelihood_standard_error.txt', exists=True))
        df.at[m, e] = mean
        df.at[m, err] = error


df.to_csv('_experiments/full_results.csv', float_format='%.3f')



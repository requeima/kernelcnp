import pandas as pd
import numpy as np
from gnp.experiment import WorkingDirectory, generate_root
import os

models = ["GNP", "AGNP", "convGNP"]
covs = ["innerprod-homo", "innerprod-hetero", "kvv-homo", "kvv-hetero", "meanfield"]

fullmodels = []

for c in covs:
    for m in models:
        fullmodels.append(" ".join([c, m]))

experiments = ["eq", "matern", "noisy-mixture", "weakly-periodic", "sawtooth"]

experiments_with_error = []
for e in experiments:
    experiments_with_error.append(e)
    experiments_with_error.append(e + "-error")    

# Create an empty dataframe
df = pd.DataFrame(index=fullmodels, columns=experiments_with_error)

# Fill the dataframe
for c in covs:
    for m in models:
        for e in experiments:            
            if (c == 'meanfield') and (n == 'homo' or n == 'hetero'):
                pass
            else:
                experiment_name = os.path.join('_experiments', 
                                                f'{e}', 
                                                f'{m}', 
                                                f'{c}')
                wd = WorkingDirectory(root=experiment_name)
                err = e + "-error"
                mean = np.loadtxt(wd.file('test_log_likelihood.txt', 
                                            exists=True))
                error = np.loadtxt(wd.file('test_log_likelihood_standard_error.txt',
                                            exists=True))
                
                idx = " ".join([c, m])
                df.at[idx, e] = mean
                df.at[idx, err] = error


df.to_csv('_experiments/full_results.csv', float_format='%.3f')



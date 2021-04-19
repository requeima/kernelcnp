import pandas as pd
import numpy as np
from gnp.experiment import WorkingDirectory, generate_root
import os

models = ["GNP", "AGNP", "convGNP", "TEGNP"]
# covs = ["innerprod-homo-4basisdims", "innerprod-hetero-4basisdims", "kvv-homo-4basisdims", "kvv-hetero-4basisdims",
#         "innerprod-homo-512basisdims", "innerprod-hetero-512basisdims", "kvv-homo-512basisdims", "kvv-hetero-512basisdims",
#         "meanfield"]

fullmodels = []

for c in covs:
    for m in models:
        fullmodels.append(" ".join([c, m]))

datas = ["eq", "matern", "noisy-mixture", "Sawtooth"]

experiments_with_error = []
for e in experiments:
    experiments_with_error.append(e)
    experiments_with_error.append(e + "-error")    

# Create an empty dataframe
df = pd.DataFrame(index=fullmodels, columns=experiments_with_error)

# Fill the dataframe

for m in models:
    for d in datas:            
        experiment_name = os.path.join('_experiments', 
                                        f'{d}',
                                        'models', 
                                        f'{m}')
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


    # experiment_name = os.path.join('_experiments',
    #                                f'{args.data}',
    #                                f'models',
    #                                f'{args.model}',
    #                                f'{args.covtype}',
    #                                f'{args.seed}')
Notes on experiments on toy data and changes to make.

# General

- Done: ~~Distributed experiments: Probably the most valuable thing at the moment is to write a script which distributes experiments over GPUs, and executes several of them in parallel.~~

- Done: ~~Sampling functions for most models (GNP, AGNP, convGNP) fails. Must look into numerical stability of sampling, potentially add more jitter. (Added more jitter and converted to double).~~

- Done: ~~Log numbers of parameters: Add functionality to training script which prints the architecture name and total number of parameters.~~

- Done: ~~Log memory usage, wall-clock time and loss: Use tensorboardX for logging all of these quantities (and potentially images too).~~


# Observations

- Our models cannot model multi-modal distributions.



# Experiments

- Toy experiments:
    - Interpolation
    - Extrapolation within training range
    - Extrapolation beyond training range
    - Plots of successes and failure modes for each
    
    Datasets:
    - EQ
    - Matern
    - Weakly periodic
    - Noisy mixture
    - Sawtooth
    
    Models:
    - All models in the cartesian product
    - FullConvGNP
    
    Metrics:
    - Predictive log-likelihood
    - RMSE
    - Qualitative fits


- Percipitation
    - Coherent samples are better for downstream estimation. Sum of percipitations has more accurate error bars. Time-series threshold estimates: what is the probability that a region will have more percipitation than a threshold for a prolonged period of time?
    
    Datasets:
    - Percipitation data sliced temporally
    - Percipitation data sliced spatially
        
    Models:
    ** Requires 2D convCNP
    - AGNP
    - convGNP
    - TEGNP
    
    
    Metrics
    - Predictive log-likelihood
    - RMSE
    - Qualitative samples
    
    Extended experiments:
    - Total of percipitation in a region: show that correlated models give better error bars
    - Probability that percipitation remains over threshold for sufficiently long: only correlated models can do this
    
    
    
    
    

- Image completion:
    - Sample coherent images without approximate inference. Trainable with exact log-likelihood, avoiding: (i) dodgy training objective; (ii) issues with KL due to discretisation resolution.
    - Demonstrates inductive bias of convolutions -- ConvCNP expected to do better than TEGNP.
    
    Datasets:
        - MNIST image completion, generalisation to larger canvases
        - Face image completion
    
- Renewable energy
    - Coherent samples are better for downstream estimation. Sum of energies at different locations comes with more accurate error bars.

- Bayesopt
    - Our models enable predictive entropy search. Entropy search (hopefully) does better than GPUCB.

- Computation and memory complexity:
    - Compare FullConvGNP, convGNP and TEGNP on toy data -- both in terms of compute and memory.
    - Easy toy GP-data that all three models can learn.
    
    Datasets:
        - EQ/Matern in 1D for all three models
        - EQ/Matern in 2D and 3D for convGNP and TEGNP

- Large numbers of datapoints

- Sim-to-real



# Model todos and baselines

- Add 2D (and 3D?) convCNP
- FullConvGNP
- Leave out ~~ConvNP~~
- Deep Kernel Transfer (Storkey group)
- Move sampling and log-prob evaluation into covariance layers


# Extensions

- Pseudopoint sets (grid, collection of points)
- Process set of points all at once for grids/pseudopoint-sets of fixed size
- Normalising flows on top
- Gaussian mixture model
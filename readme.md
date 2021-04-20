# Experiments
- Toy experiments on 1D data
    - Requirements:
        - GNP
        - ANP
        - ConvNP
    - Things to measure:
        - Predictive LL (end of training), in-in, in-out, out-out
        - Parameter count
        - Predictive LL against wallclock time during training (line plot)
        - Predictive LL against forward pass time (scatter plot)
        - Plots of the covariance
        - Plots of the posteriors (in-in, in-out, out-out)


- Toy experiments on 2D/3D data
    - Requirements:
        - ANP
        - ConvNP
        - Multidimensional ConvGNP
    - Things to measure:
        - Predictive LL (end of training), in-in, in-out, out-out
        - Parameter count
        - Predictive LL against wallclock time during training (line plot)
        - Predictive LL against forward pass time (scatter plot)


- Toy experiments for multimodal data
    - Requirements:
        - Multimodal (mixture of GPs) MGNP, AMGNP, ConvMGNP
    - Things to measure:
        - A few plots showing these models can capture multimodalities


- Environmental data
    - Experiment description:
        - Spatial slice: Train on part of Central, Test on held-out Central, Test convolutional models on other three regions. Inputs & outputs, contexts & targets as in the ConvNP paper
        - **Active learning**
    - Requirements:
        - All of the models used in the toy experiments
        - Script for spatial experiments
    - Things to measure:
        - Predictive LL (end of training) and MSE, on Central held-out and on other three test regions
        - Spatial slices: Predictive LL on sums of percipitation over a region. Predictive LL on probability of exceeding a threshold everywhere on a region, calibration and TP, TN, FP, FN
        - Plots of posterior samples


# Tasks
- Implement: 2D and 3D convolutional models
- Implement: ANP, ConvNP


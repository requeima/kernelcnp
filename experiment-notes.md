Notes on experiments on toy data and changes to make.

# General

- Distributed experiments: Probably the most valuable thing at the moment is to write a script which distributes experiments over GPUs, and executes several of them in parallel.

- Log numbers of parameters: Add functionality to training script which prints the architecture name and total number of parameters.

- Log memory usage, wall-clock time and loss: Use tensorboardX for logging all of these quantities (and potentially images too).

# GNP

The current GNP model does not seem to produce very good fits. It doesn't get great predictions even close to the context set (sometimes the uncertainties are too large). The error bars blow up far from the data due to tanh activations.

    - We should check the number of parameters it uses and verify the architecture is sensible, and comparable to the other models. Should also consider switching to using FullyConnectedNetwork internally.
    - Replace ReLU by Tanh activations.

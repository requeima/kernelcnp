#!/bin/bash

kernels=("eq" "matern" "noisy-mixture" "weakly-periodic" "sawtooth")
# kernels=("matern" "noisy-mixture")

for data in "${kernels[@]}"; do
    python plot_all.py $data 
done
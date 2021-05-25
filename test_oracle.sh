#!/bin/bash
kernels=("eq" "matern" "noisy-mixture" "noisy-mixture-slow" "weakly-periodic" "weakly-periodic-slow")

for data in "${kernels[@]}"; do
    python test_oracle.py $data 
done



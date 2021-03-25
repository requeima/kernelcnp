#!/bin/bash

MODE="--train"
EPOCHS="100"
kernels=("eq" "matern" "noisy-mixture" "weakly-periodic" "sawtooth")
# kernels=("eq")

models=("GNP" "AGNP" "convGNP" "TEGNP")
covs=("innerprod-homo" "innerprod-hetero" "kvv-homo" "kvv-hetero"  "meanfield")

for data in "${kernels[@]}"; do
    for model in "${models[@]}"; do
        for cov in "${covs[@]}"; do
                python train.py $data $model $cov $MODE --learning_rate 3e-4 --epochs $EPOCHS  --test_context_num 10
        done
    done
done
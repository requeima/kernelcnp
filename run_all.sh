#!/bin/bash

MODE="--train"
EPOCHS="10000"
kernels=("eq" "matern" "noisy-mixture" "weakly-periodic" "sawtooth")
# kernels=("eq")

models=("GNP" "AGNP" "convGNP" "TEGNP")
covs=("innerprod-homo" "innerprod-hetero" "kvv-homo" "kvv-hetero"  "meanfield")

for data in "${kernels[@]}"; do
    for model in "${models[@]}"; do
        for cov in "${covs[@]}"; do
                python train.py $data $model $cov $MODE --epochs $EPOCHS --learning_rate 1e-4 --num_train_iters 5
        done
    done
done

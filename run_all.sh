#!/bin/bash

MODE="--train"
EPOCHS="2"
# kernels=("eq" "matern" "noisy-mixture" "weakly-periodic" "sawtooth")
kernels=("eq")
models=("GNP" "AGNP" "convGNP")
# covs=("innerprod homo" "innerprod hetero" "kvv homo" "kvv hetero"  "meanfield none")
covs=("meanfield none")

for data in "${kernels[@]}"; do
    for model in "${models[@]}"; do
        for cov in "${covs[@]}"; do
                python train.py $data $model $cov $MODE --learning_rate 3e-4 --epochs $EPOCHS
        done
    done
done
#!/bin/bash
kernels=("eq" "matern" "noisy-mixture" "weakly-periodic" "sawtooth")
# kernels=("sawtooth")

seeds=("0")
models=("GNP" "AGNP" "convGNP" "TEGNP" "MeanTEGNP" "MeanTEAGNP")
# models=("TEGNP")

covs=("innerprod-homo" "kvv-homo" "meanfield")
# covs=("kvv-homo")


for data in "${kernels[@]}"; do
    for model in "${models[@]}"; do
        for cov in "${covs[@]}"; do
            for seed in "${seeds[@]}"; do
                python plot_all.py $data $model $cov --seed $seed
            done
        done
    done
done



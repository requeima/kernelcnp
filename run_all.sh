#!/bin/bash

MODE="--train"
<<<<<<< Updated upstream
EPOCHS="10000"
=======
>>>>>>> Stashed changes
kernels=("eq" "matern" "noisy-mixture" "weakly-periodic" "sawtooth")

models=("GNP" "AGNP" "convGNP" "TEGNP")
covs=("innerprod-homo" "innerprod-hetero" "kvv-homo" "kvv-hetero")
basisdims=("4", "512")

for data in "${kernels[@]}"; do
    for model in "${models[@]}"; do
        for cov in "${covs[@]}"; do
<<<<<<< Updated upstream
                python train.py $data $model $cov $MODE --epochs $EPOCHS --learning_rate 1e-4 --num_train_iters 5
        done
    done
done
=======
            for dim in "${basisdims[@]}"; do
                    python train.py $data $model $cov $MODE --num_basis_dim $dim
        done
    done
done


# meanfield
for data in "${kernels[@]}"; do
    for cov in "${covs[@]}"; do
            python train.py $data meanfield $cov $MODE
    done
done
>>>>>>> Stashed changes

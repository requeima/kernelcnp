#!/bin/bash

MODE="--train --test"

# kernels=("eq" "matern" "noisy-mixture" "weakly-periodic" "sawtooth")
kernels=("matern" "noisy-mixture")

for data in "${kernels[@]}"; do
    python train.py $data InnerProdHomoNoiseKernelCNP $MODE --root _experiments/InnerProdHomoNoiseKernelCNP-$data --learning_rate 3e-4  --epochs 200
    python train.py $data InnerProdHeteroNoiseKernelCNP $MODE --root _experiments/InnerProdHeteroNoiseKernelCNP-$data --learning_rate 3e-4 --epochs 200
    python train.py $data KvvHomoNoiseKernelCNP $MODE --root _experiments/KvvHomoNoiseKernelCNP-$data --learning_rate 3e-4 --epochs 200
    python train.py $data KvvHeteroNoiseKernelCNP $MODE --root _experiments/KvvHeteroNoiseKernelCNP-$data --learning_rate 3e-4 --epochs 200
    
    python train.py $data cnp $MODE --root _experiments/cnp-$data --learning_rate 3e-4 --epochs 200
    python train.py $data anp $MODE  --root _experiments/anp-$data --learning_rate 3e-4 --epochs 200
    python train.py $data convcnp $MODE --root _experiments/convcnp-$data --learning_rate 3e-4 --epochs 200
    python train.py $data convcnpxl $MODE --root _experiments/convcnpxl-$data --learning_rate 1e-3 --epochs 200

done
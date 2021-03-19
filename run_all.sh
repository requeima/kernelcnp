#!/bin/bash

MODE="--train --test"
EPOCHS="200"

kernels=("eq" "matern" "noisy-mixture" "weakly-periodic" "sawtooth")
# kernels=("matern" "noisy-mixture")

for data in "${kernels[@]}"; do
    # python train.py $data cnp $MODE --root _experiments/cnp-$data --learning_rate 3e-4 --epochs $EPOCHS
    # python train.py $data anp $MODE  --root _experiments/anp-$data --learning_rate 3e-4 --epochs $EPOCHS
    # python train.py $data convcnp $MODE --root _experiments/convcnp-$data --learning_rate 3e-4 --epochs $EPOCHS
    # python train.py $data convcnpxl $MODE --root _experiments/convcnpxl-$data --learning_rate 1e-3 --epochs $EPOCHS

    # # convCNP kernel models
    # python train.py $data InnerProdHomoNoiseKernelCNP $MODE --root _experiments/InnerProdHomoNoiseKernelCNP-$data --learning_rate 3e-4  --epochs $EPOCHS
    # python train.py $data InnerProdHeteroNoiseKernelCNP $MODE --root _experiments/InnerProdHeteroNoiseKernelCNP-$data --learning_rate 3e-4 --epochs $EPOCHS
    # python train.py $data KvvHomoNoiseKernelCNP $MODE --root _experiments/KvvHomoNoiseKernelCNP-$data --learning_rate 3e-4 --epochs $EPOCHS
    # python train.py $data KvvHeteroNoiseKernelCNP $MODE --root _experiments/KvvHeteroNoiseKernelCNP-$data --learning_rate 3e-4 --epochs $EPOCHS
    
    # (no conv) CNP kernel models
    python train.py $data InnerProdHomoNoiseNoConvKernelCNP $MODE --root _experiments/InnerProdHomoNoiseNoConvKernelCNP-$data --learning_rate 3e-4  --epochs $EPOCHS
    python train.py $data InnerProdHeteroNoiseNoConvKernelCNP $MODE --root _experiments/InnerProdHeteroNoiseNoConvKernelCNP-$data --learning_rate 3e-4 --epochs $EPOCHS
    python train.py $data KvvHomoNoiseNoConvKernelCNP $MODE --root _experiments/KvvHomoNoiseNoConvKernelCNP-$data --learning_rate 3e-4 --epochs $EPOCHS
    python train.py $data KvvHeteroNoiseNoConvKernelCNP $MODE --root _experiments/KvvHeteroNoiseNoConvKernelCNP-$data --learning_rate 3e-4 --epochs $EPOCHS
    
    # (no conv) ANP kernel models
    python train.py $data InnerProdHomoNoiseNoConvKernelANP $MODE --root _experiments/InnerProdHomoNoiseNoConvKernelANP-$data --learning_rate 3e-4  --epochs $EPOCHS
    python train.py $data InnerProdHeteroNoiseNoConvKernelANP $MODE --root _experiments/InnerProdHeteroNoiseNoConvKernelANP-$data --learning_rate 3e-4 --epochs $EPOCHS
    python train.py $data KvvHomoNoiseNoConvKernelANP $MODE --root _experiments/KvvHomoNoiseNoConvKernelANP-$data --learning_rate 3e-4 --epochs $EPOCHS
    python train.py $data KvvHeteroNoiseNoConvKernelANP $MODE --root _experiments/KvvHeteroNoiseNoConvKernelANP-$data --learning_rate 3e-4 --epochs $EPOCHS

done
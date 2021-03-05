#!/bin/bash

kernels=("eq" "matern" "noisy-mixture" "weakly-periodic" "sawtooth")

for data in "${kernels[@]}"; do
    python train.py $data InnerProdHomoNoiseKernelCNP --train --root _experiments/InnerProdHomoNoiseKernelCNP-$data --learning_rate 3e-4
    python train.py $data InnerProdHeteroNoiseKernelCNP --train --root _experiments/InnerProdHeteroNoiseKernelCNP-$data --learning_rate 3e-4
    python train.py $data KvvHomoNoiseKernelCNP --train --root _experiments/KvvHomoNoiseKernelCNP-$data --learning_rate 3e-4
    python train.py $data KvvHeteroNoiseKernelCNP --train --root _experiments/KvvHeteroNoiseKernelCNP-$data --learning_rate 3e-4

    python train.py $data cnp --train --root _experiments/cnp-$data --learning_rate 3e-4
    python train.py $data anp --train  --root _experiments/anp-$data --learning_rate 3e-4
    python train.py $data convcnp --train --root _experiments/convcnp-$data --learning_rate 3e-4
    python train.py $data convcnpxl --train --root _experiments/convcnpxl-$data --learning_rate 1e-3

done
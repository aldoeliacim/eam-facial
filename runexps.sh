#!/bin/bash
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/opt/cuda

runpath=runs
echo "Storing results in $runpath"
python eam.py -n --runpath="$runpath" && \
    python eam.py -f --runpath="$runpath" && \
    python eam.py -e 1 --runpath="$runpath" && \
    python eam.py -r --runpath="$runpath" && \
    python eam.py -d --runpath="$runpath"

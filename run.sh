#!/bin/bash
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/opt/cuda

./run_first.sh

python parse_history.py | tee model_history.txt
python nnet_stats.py | tee classifier_accuracy.txt
python mcols_stdevs.py | tee memory_performance.txt

for n in 128 256 512 1024; do # Same as in run_first.sh and run_second.sh
  python noised_classif.py --domain=$n
  python choose.py --domain=$n
  python check_chosen.py --domain=$n
done

python system_stats.py | tee mem_nsd_prtl_performance.txt

./run_second.sh

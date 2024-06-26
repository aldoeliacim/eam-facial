#!/bin/bash

for n in 128 256 512 1024; do
  dir="runs-$n"
  if [ ! -d "$dir" ]; then
    echo "$dir not found, creating and copying default mem_params.csv"
    mkdir -p "$dir"
    cp mem_params.csv "$dir"  # Copia parámetros por default
  fi

  python eam.py -n --domain=$n --runpath=$dir  # Train NN
  python eam.py -f --domain=$n --runpath=$dir  # Extract features
  
  echo "Running experiment with domain size $n"
  python eam.py -e 1 --domain=$n --runpath=$dir

  if [ $? -ne 0 ]; then
    echo "Failed at domain size $n"
    break
  fi
done

#!/bin/bash

for n in 32; do
  python eam.py -n --domain=$n --runpath=runs-$n # && # Train NN
  #python eam.py -f --domain=$n --runpath=runs-$n && # Extract features
  #echo "Running experiment with domain size $n"
  #python eam.py -e 1 --domain=$n --runpath=runs-$n

  if [ $? -ne 0 ]; then
    echo "Failed at domain size $n"
    break
  fi
done

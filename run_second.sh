#!/bin/bash

for n in 32 64 128 256 512; do
    dir="runs-$n"
    python eam.py -r --domain=$n --runpath=runs-$n && \
    python eam.py -d --domain=$n --runpath=runs-$n
    
    if [ $? -ne 0 ]; then
        echo "Failed at domain size $n"
        break
    fi
done

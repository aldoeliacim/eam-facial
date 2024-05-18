#!/bin/bash
n=256
python eam.py -r --domain=$n --runpath=runs-$n && \
python eam.py -d --domain=$n --runpath=runs-$n
done

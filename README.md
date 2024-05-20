## Entropic Associative Memory experiments for facial emotion recognition

Used Python 3.12.2 and Tensorflow 2.16.1 on a laptop computer with specifications:
* OS: Arch Linux x86_64, Kernel: Linux 6.8.9-zen1-2-zen
* CPU: 12th Gen Intel(R) Core(TM) i7-12700H (20) @ 4.70 GHz
* GPU: NVIDIA GeForce RTX 3070 Ti Laptop GPU
* RAM: SODIMM 32GB DDR5 @ 4800 MT/s

#### Dataset
FER2013 (source: https://www.kaggle.com/datasets/msambare/fer2013) with the following modifications:
- Removed disgust folder and images from test and train subsets
- Converted every image into png

data.zip is the result of these changes, data folder must be unpacked in project root

### Usage:
Run ```run.sh``` to begin experiment 1

import numpy as np

import constants

## Find the classification of the dreamed (normal|noised) images for 
## the sigma value defined in index sigma_index. The classification was stored in
## a csv file for each association chain of depth (n_depth) for all sigma values
## in constants.sigma_values taking the image in chosen.csv corresponding to the
## current fold

## This is used in experiments 4 and 5, and figures 6 and 7, respectively.
## First you have to run: python eam.py -d --domain=$n --runpath=runs-$n

## Set domain
domain = 1024
constants.domain = domain

## Set run path
dirname = f'runs-{domain}'
constants.run_path=dirname

## Experimental settings
prefix = constants.memory_parameters_prefix
filename = constants.csv_filename(prefix)
parameters = \
             np.genfromtxt(filename, dtype=float, delimiter=',', skip_header=1)
es = constants.ExperimentSettings(parameters)

fname = constants.csv_filename(
    constants.chosen_prefix,es)

chosen = np.genfromtxt(fname, dtype=int, delimiter=',')
msize = 16
sigma_index = 0
n_depths = constants.dreaming_cycles

#
dict_class = {
    0:"airplane",
    1:"auto",
    2:"bird",
    3:"cat",
    4:"deer",
    5:"dog",
    6:"frog",
    7:"horse",
    8:"ship",
    9:"truck"}

for fold in range(constants.n_folds):
    label = chosen[fold,0]
    label_name = dict_class.get(label,'-')
    #
    idx = chosen[fold,1]
    # Uncomment for regular dreaming
    #prefix = constants.classification_name(es)
    # Uncomment for noised dreaming
    prefix = constants.noised_classification_name(es)
    fname = constants.data_filename(prefix, es, fold)
    classif = np.load(fname) # Labels of testing (normal|noised) data according to classifier
    class_num = classif[idx]
    class_name = dict_class.get(class_num,'-')
    print(f'"{label_name} {class_name}', end='')
    # Uncomment for noised dreaming
    prefix = constants.classification_name(es) + constants.noised_suffix
    prefix += constants.int_suffix(msize, 'msz')
    fname = constants.csv_filename(prefix, es, fold)
    classif = np.genfromtxt(fname, delimiter=',')
    start = sigma_index*6
    classif = classif[start:start+n_depths]
    for i in range(len(classif)):
        c = dict_class.get(int(classif[i]),'-')
        print(f' {c}', end='')
    print('" ')


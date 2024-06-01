import numpy as np

import constants

## Find the classification of the chosen (normal|noised) images for all
## sigma values defined in constants.sigma_values

## This is used in experiments 2 and 3, and figures 4 and 5, respectively.
## First you have to run: python eam.py -r --domain=$n --runpath=runs-$n

## Set domain
domain = 256
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

#
fname = constants.csv_filename(
    constants.chosen_prefix,es)
chosen = np.genfromtxt(fname, dtype=int, delimiter=',')
msize = 16 ## This is the memory size chosen to run experiment 2 and on

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
    # Uncomment for regular input
    #prefix = constants.classification_name(es)
    # Uncomment for noised input
    prefix = constants.noised_classification_name(es)
    fname = constants.data_filename(prefix, es, fold)
    classif = np.load(fname) # Labels of testing (normal|noised) data according to classifier
    class_num = classif[idx]
    class_name = dict_class.get(class_num,'-')
    print(f'"{label_name} {class_name}', end='')
    sigmas = constants.sigma_values
    prefix += constants.int_suffix(msize, 'msz')
    for s in sigmas:
        suffix = constants.float_suffix(s, 'sgm')
        fname = prefix + suffix
        fname = constants.data_filename(fname, es, fold)
        classif = np.load(fname)
        class_num = classif[idx]
        class_name = dict_class.get(class_num,'-')
        print(f' {class_name}', end = '')
    print('" ')

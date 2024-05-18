import random

import numpy as np

import constants

# This function randomly finds an image of each class
# (each chosen image belons to a different testing data segment)
# that was correctly classified

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

# We can do this because in Cifar10 there are the same 
# number of labels as folds.
chosen = np.zeros((constants.n_folds, 2), dtype=int)
classes = [*range(constants.n_labels)]
random.shuffle(classes)
print(classes)

for fold in range(constants.n_folds):
    prefix = constants.labels_name(es) + constants.testing_suffix
    fname = constants.data_filename(prefix, es, fold)
    labels = np.load(fname) # Labels of testing data according to the dataset
    
    prefix = constants.classification_name(es)
    fname = constants.data_filename(prefix, es, fold) 
    classif = np.load(fname) # Labels of testing data according to classifier

    label = classes[fold]
    n = 0
    for l, c in zip(labels, classif):
        # Choose randomly an element that shares the label given by the dataset
        # and the classifier. That label has to be equal to 'label'.
        if (random.randrange(10) == 0) and (l == label) and (l == c):
            chosen[fold,0] = label
            chosen[fold,1] = n
            break
        n += 1

prefix = constants.chosen_prefix
fname = constants.csv_filename(prefix, es)
np.savetxt(fname, chosen, fmt='%d', delimiter=',')


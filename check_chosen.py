import numpy as np

import constants

## Check that the classification of the chosen images is correct 

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

msg = ""

for fold in range(constants.n_folds):
    prefix = constants.classification_name(es)
    fname = constants.data_filename(prefix, es, fold) 
    classif = np.load(fname) # Labels of testing data according to classifier

    label = chosen[fold,0]
    n = chosen[fold,1]

    if (classif[n] != label):
        msg += "The image " + str(n) + " selected in fold " + str(fold) + " is assigned by the classifier to " + str(classif[n]) + ", but its correct class is " + str(label) + ".\n"
    
print(msg) if msg != "" else print("All labels are correctly assigned by the classifier.")

import numpy as np
import tensorflow as tf

import constants

## Set domain
domain = 512
constants.domain = domain
msize = 16

## Set run path
dirname = f'runs-{domain}'
constants.run_path=dirname

## Experimental settings
prefix = constants.memory_parameters_prefix
filename = constants.csv_filename(prefix)
parameters = np.genfromtxt(filename, dtype=float, delimiter=',', skip_header=1)
es = constants.ExperimentSettings(parameters)
sigmas = constants.sigma_values
n_sigmas = len(sigmas)

# Some prefixes ans suffixes
msize_suffix = constants.msize_suffix(msize)
testing_labels_prefix = constants.labels_prefix + constants.testing_suffix

# Arrays to store intermediate values obtained from original images
sigma_precision_memories = []
sigma_recall_memories = []

# Arrays to store intermediate values obtained from noised images
sigma_precision_noised = []
sigma_recall_noised = []

# Arrays to store intermediate values obtained from partial images
sigma_precision_partial = []
sigma_recall_partial = []

sgm_str = '\t\t  '

def print_row(fname, data):
    print(f'{fname}', end='')
    for d in data:
        print(f', {d:.3f}', end='')
    print('')

for fold in range(constants.n_folds):
    precisions_memories = []
    recalls_memories = []
    #
    precisions_noised = []
    recalls_noised = []
    #
    precisions_partial = []
    recalls_partial = []

    # Load original testing labels
    testing_labels_filename = constants.data_filename(
        testing_labels_prefix, es, fold)
    testing_labels = np.load(testing_labels_filename)
    n = len(testing_labels)
    
    for sgm in sigmas:
        sigma_suffix = constants.sigma_suffix(sgm)
        suffix = msize_suffix + sigma_suffix

        # Memories stats
        memories_lbls_filename = constants.classification_name(es) + suffix
        # Load testing labels according to memories
        fname = constants.data_filename(memories_lbls_filename, es, fold)
        classif_memories = np.load(fname)
        # True positives
        tp = (testing_labels == classif_memories).sum()
        # Recall
        recalls_memories.append(tp/n)
        # Precision
        fn = np.count_nonzero(classif_memories == 10) # No response
        precisions_memories.append(tp/(n-fn)) if n-fn>0 else precisions_memories.append(0.0)

        # Noised stats
        noised_lbls_filename = constants.noised_classification_name(es) + suffix
        # Load testing labels according to noised
        fname = constants.data_filename(noised_lbls_filename, es, fold)
        classif_noised = np.load(fname)
        # True positives
        tp = (testing_labels == classif_noised).sum()
        # Recall
        recalls_noised.append(tp/n)
        # Precision
        fn = np.count_nonzero(classif_noised == 10) # No response
        precisions_noised.append(tp/(n-fn)) if n-fn>0 else precisions_noised.append(0.0)

        # # Partial stats
        # partial_lbls_filename = constants.partial_classification_name(es) + suffix
        # # Load testing labels according to partial
        # fname = constants.data_filename(partial_lbls_filename, es, fold)
        # classif_partial = np.load(fname)
        # # True positives
        # tp = (testing_labels == classif_partial).sum()
        # # Recall
        # recalls_partial.append(tp/n)
        # # Precision
        # fn = np.count_nonzero(classif_partial == 10) # No response
        # print(f'Fold {fold} no response {fn}')
        # precisions_partial.append(tp/(n-fn)) if n-fn>0 else precisions_partial.append(0.0)

    #
    sigma_precision_memories.append(precisions_memories)
    sigma_recall_memories.append(recalls_memories)
    #
    sigma_precision_noised.append(precisions_noised)
    sigma_recall_noised.append(recalls_noised)
    #
    # sigma_precision_partial.append(precisions_partial)
    # sigma_recall_partial.append(recalls_partial)
        

# Compute mean for memories results
precision_memories = np.array(sigma_precision_memories)
recall_memories = np.array(sigma_recall_memories)
#
avg_precision_memories = np.mean(precision_memories, axis=0)
avg_recall_memories = np.mean(recall_memories, axis=0)
#
print('=====> Results obtained from memories with orginal images <=====')
print(f'\t',end='')
print_row('Sigmas\t\t ',sigmas)
print(f'\t',end='')
print_row('Mean precision is',avg_precision_memories)
print(f'\t',end='')
print_row('Mean recall is\t ', avg_recall_memories)

# Compute mean for noised results
precision_noised = np.array(sigma_precision_noised)
recall_noised = np.array(sigma_recall_noised)
#
avg_precision_noised = np.mean(precision_noised, axis=0)
avg_recall_noised = np.mean(recall_noised, axis=0)
#
print('\n=====> Results obtained from memories with noised images <=====')
print(f'\t',end='')
print_row('Sigmas\t\t ',sigmas)
print(f'\t',end='')
print_row('Mean precision is',avg_precision_noised)
print(f'\t',end='')
print_row('Mean recall is\t ', avg_recall_noised)


# Compute mean for partial results
# precision_partial = np.array(sigma_precision_partial)
# recall_partial = np.array(sigma_recall_partial)
# #
# avg_precision_partial = np.mean(precision_partial, axis=0)
# avg_recall_partial = np.mean(recall_partial, axis=0)
# #
# print('\n=====> Results obtained from memories with partial images <=====')
# print(f'Precision for all folds and sigma values:\n{precision_partial}\n')
# print(f'\t',end='')
# print_row('Sigmas\t\t ',sigmas)
# print(f'\t',end='')
# print_row('Mean precision is',avg_precision_partial)
# print(f'\t',end='')
# print_row('Mean recall is\t ', avg_recall_partial)


import numpy as np

import constants

## Prints accuracy of the classifier for each domain size

domain_sizes = [128, 256, 512, 1024]

def accuracy(labels, predictions):
    n = 0
    for l, p in zip(labels, predictions):
        n += (l == p)
    return n/len(labels)

if __name__ == "__main__":
    dir_prefix = 'runs-'
    es = constants.ExperimentSettings()
    for domain in domain_sizes:
        dirname = f'{dir_prefix}{domain}'
        constants.run_path=dirname
        print(f'Domain size: {domain}')
        for fold in range(constants.n_folds):
            prefix = constants.labels_prefix + constants.testing_suffix
            filename = constants.data_filename(prefix, es, fold)
            labels = np.load(filename) # Labels of testing data according to the dataset
            
            prefix = constants.classification_name(es)
            filename = constants.data_filename(prefix, es, fold)
            predictions = np.load(filename) # Labels of testing data according to classifier
            
            acc = accuracy(labels, predictions)
            print(f'\tFold: {fold}, {acc:.3f}')
        print('')

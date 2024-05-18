import numpy as np

import constants

## Prints mean and standar deviation values for the entropy, precision and
## recall of memories for all domain sizes

domain_sizes = [32, 64, 128, 256, 512]
names = ['memory_entropy', 'memory_precision', 'memory_recall']

def print_row(fname, data):
    print(f'\t\t{fname}', end='')
    for d in data:
        print(f', {d:.3f}', end='')
    print('')

if __name__ == "__main__":
    dir_prefix = 'runs-'
    es = constants.ExperimentSettings()
    for domain in domain_sizes:
        dirname = f'{dir_prefix}{domain}'
        constants.run_path=dirname
        print(f'Domain size: {domain}')
        for fname in names:
            print(f'\t{fname}')
            filename = constants.csv_filename(fname, es)
            data = np.genfromtxt(filename, delimiter=',')
            means = np.mean(data, axis=0)
            stdvs = np.std(data, axis=0)
            print_row('Mean values', means)
            print_row('Std dev val', stdvs)
        print('\n')

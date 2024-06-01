import numpy as np
import constants

## Imprime los valores medios y la desviación estándar para la entropía, precisión y
## recuperación de memorias para todos los tamaños de dominio

domain_sizes = [128, 256, 512, 1024]
names = ['memory_entropy', 'memory_precision', 'memory_recall']

def print_row(fname, data):
    print(f'\t\t{fname}', end='')
    if np.isscalar(data):
        print(f', {data:.3f}', end='')
    else:
        for d in data:
            print(f', {d:.3f}', end='')
    print('')

if __name__ == "__main__":
    dir_prefix = 'runs-'
    es = constants.ExperimentSettings()
    for domain in domain_sizes:
        dirname = f'{dir_prefix}{domain}'
        constants.run_path = dirname
        print(f'Tamaño de dominio: {domain}')
        for fname in names:
            print(f'\t{fname}')
            filename = constants.csv_filename(fname, es)
            data = np.genfromtxt(filename, delimiter=',')
            means = np.mean(data, axis=0)
            stdvs = np.std(data, axis=0)
            print_row('Valores medios', means)
            print_row('Valores de desv. estándar', stdvs)
        print('\n')

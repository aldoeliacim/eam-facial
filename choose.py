import argparse
import random
import numpy as np
import constants

# Configurar argumentos de línea de comandos
parser = argparse.ArgumentParser(description='Process domain parameter.')
parser.add_argument('--domain', type=int, required=True, help='Domain value')
args = parser.parse_args()

# Establecer el dominio
domain = args.domain
constants.domain = domain

# Establecer ruta de ejecución
dirname = f'runs-{constants.domain}'
constants.run_path = dirname

# Configuraciones experimentales
prefix = constants.memory_parameters_prefix
filename = constants.csv_filename(prefix)
parameters = np.genfromtxt(filename, dtype=float, delimiter=',', skip_header=1)
es = constants.ExperimentSettings(parameters)

# Podemos hacer esto porque hay el mismo número de etiquetas que de folds.
chosen = np.zeros((constants.n_folds, 2), dtype=int)
classes = [*range(constants.n_labels)]
random.shuffle(classes)
print(classes)

for fold in range(constants.n_folds):
    prefix = constants.labels_name(es) + constants.testing_suffix
    fname = constants.data_filename(prefix, es, fold)
    labels = np.load(fname)  # Etiquetas de datos de prueba según el conjunto de datos
    
    prefix = constants.classification_name(es)
    fname = constants.data_filename(prefix, es, fold)
    classif = np.load(fname)  # Etiquetas de datos de prueba según el clasificador

    label = classes[fold]
    n = 0
    for l, c in zip(labels, classif):
        # Elegir aleatoriamente un elemento que comparta la etiqueta dada por el conjunto de datos
        # y el clasificador. Esa etiqueta tiene que ser igual a 'label'.
        if (random.randrange(10) == 0) and (l == label) and (l == c):
            chosen[fold, 0] = label
            chosen[fold, 1] = n
            break
        n += 1

prefix = constants.chosen_prefix
fname = constants.csv_filename(prefix, es)
np.savetxt(fname, chosen, fmt='%d', delimiter=',')

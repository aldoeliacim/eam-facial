import argparse
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

# Cargar los elementos elegidos
fname = constants.csv_filename(constants.chosen_prefix, es)
chosen = np.genfromtxt(fname, dtype=int, delimiter=',')

msg = ""

for fold in range(constants.n_folds):
    prefix = constants.classification_name(es)
    fname = constants.data_filename(prefix, es, fold)
    classif = np.load(fname)  # Etiquetas de datos de prueba según el clasificador

    label = chosen[fold, 0]
    n = chosen[fold, 1]

    if classif[n] != label:
        msg += f"The image {n} selected in fold {fold} is assigned by the classifier to {classif[n]}, but its correct class is {label}.\n"

print(msg) if msg != "" else print("All labels are correctly assigned by the classifier.")

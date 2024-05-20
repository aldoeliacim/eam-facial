import argparse
import numpy as np
import tensorflow as tf

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

# Prefijo del modelo
model_prefix = constants.model_name(es)

for fold in range(constants.n_folds):
    # Cargar el encoder
    filename = constants.classifier_filename(model_prefix, es, fold)
    model = tf.keras.models.load_model(filename)
    model.summary()

    # Características con ruido
    suffix = constants.noised_suffix
    features_filename = constants.data_filename(constants.features_prefix + suffix, es, fold)
    noised_features = np.load(features_filename)

    prefix = constants.noised_classification_name(es)
    labels_filename = constants.data_filename(prefix, es, fold)
    labels = np.argmax(model.predict(noised_features), axis=1)

    np.save(labels_filename, labels)

    # # Características parciales
    # suffix = constants.partial_suffix
    # features_filename = constants.data_filename(constants.features_prefix + suffix, es, fold)
    # partial_features = np.load(features_filename)

    # prefix = constants.partial_classification_name(es)
    # labels_filename = constants.data_filename(prefix, es, fold)
    # labels = np.argmax(model.predict(partial_features), axis=1)

    # np.save(labels_filename, labels)

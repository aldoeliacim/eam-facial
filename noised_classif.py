import numpy as np
import tensorflow as tf

import constants
import dataset

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

## Model prefix
model_prefix = constants.model_name(es)

for fold in range(constants.n_folds):
    # Load de encoder
    filename = constants.classifier_filename(model_prefix, es, fold)
    model = tf.keras.models.load_model(filename)
    model.summary()

    # Noised features
    suffix = constants.noised_suffix
    features_filename = \
        constants.data_filename(constants.features_prefix + suffix, es, fold)
    noised_features = np.load(features_filename)

    prefix = constants.noised_classification_name(es)
    labels_filename = constants.data_filename(prefix, es, fold)
    labels = np.argmax(model.predict(noised_features), axis=1)

    np.save(labels_filename, labels)

    # # Partial features
    # suffix = constants.partial_suffix
    # features_filename = \
    #     constants.data_filename(constants.features_prefix + suffix, es, fold)
    # partial_features = np.load(features_filename)

    # prefix = constants.partial_classification_name(es)
    # labels_filename = constants.data_filename(prefix, es, fold)
    # labels = np.argmax(model.predict(partial_features), axis=1)

    # np.save(labels_filename, labels)

    

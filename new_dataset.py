import numpy as np
import tensorflow as tf

import random
import os

import constants

# Definir constantes para columnas y filas
columns = 48  # Asume un tamaño de imagen de 48x48
rows = 48

# Función para añadir ruido a los datos
def noised(data, percent):
    """
    Añadir ruido a los datos.

    Args:
    data (np.ndarray): Conjunto de datos de entrada.
    percent (float): Porcentaje de ruido a añadir.

    Returns:
    np.ndarray: Conjunto de datos con ruido añadido.
    """
    print(f'Adding {percent}% noise to data...', end='')
    copy = np.zeros(data.shape, dtype=float)
    n = 0
    for i in range(len(copy)):
        copy[i] = _noised(data[i], percent)
        n += 1
    return copy

def _noised(image, percent):
    """
    Añadir ruido a una imagen individual.

    Args:
    image (np.ndarray): Imagen de entrada.
    percent (float): Porcentaje de ruido a añadir.

    Returns:
    np.ndarray: Imagen con ruido añadido.
    """
    copy = np.array([row[:] for row in image])
    total = round(columns * rows * percent / 100.0)
    noised = []

    while len(noised) < total:
        i = random.randrange(rows)
        j = random.randrange(columns)
        if (i, j) in noised:
            continue
        value = random.randrange(256)
        copy[i, j] = value
        noised.append((i, j))

    return copy

# Función para dividir los datos
def split(dataset, train_size=0.7, test_size=0.2, fill_size=0.1, seed=123):
    np.random.seed(seed)
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)

    train_end = int(train_size * len(dataset))
    test_end = train_end + int(test_size * len(dataset))

    train_indices = indices[:train_end]
    test_indices = indices[train_end:test_end]
    fill_indices = indices[test_end:]

    train_data = tf.data.Dataset.from_tensor_slices([dataset[i] for i in train_indices])
    test_data = tf.data.Dataset.from_tensor_slices([dataset[i] for i in test_indices])
    fill_data = tf.data.Dataset.from_tensor_slices([dataset[i] for i in fill_indices])

    return train_data, test_data, fill_data

# Función para cargar y dividir los datos
def load_and_split_data(path, image_size=(48, 48), batch_size=32, seed=123):
    ds = tf.keras.utils.image_dataset_from_directory(
        path,
        labels='inferred',
        label_mode='int',
        color_mode='grayscale',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=seed,
        interpolation='bilinear'
    )
    return split(list(ds.as_numpy_iterator()), seed=seed)

# Función para preprocesar y almacenar los datos para cada pliegue
def preprocess_data(fold):
    train_data, test_data, fill_data = load_and_split_data(constants.data_path, seed=123 + fold)
    
    # Normalización de datos
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    normalized_train = train_data.map(lambda x, y: (normalization_layer(x), y))
    normalized_test = test_data.map(lambda x, y: (normalization_layer(x), y))
    
    # Añadir ruido a los datos de entrenamiento si no existen
    noised_filename = f'train_data_noised_fold_{fold}.tfrecord'
    if not os.path.exists(noised_filename):
        train_data_noised = noised(np.array(list(train_data.as_numpy_iterator())), constants.noise_percent)
        noised_ds = tf.data.Dataset.from_tensor_slices(train_data_noised)
        save_data(noised_ds, noised_filename)
    
    # Almacenar los datos preprocesados
    save_data(normalized_train, f'{constants.prep_data_fname}{constants.original_suffix}.tfrecord')
    save_data(normalized_test, f'{constants.prep_data_fname}{constants.noised_suffix}.tfrecord')
    save_data(fill_data, f'{constants.prep_data_fname}{constants.filling_suffix}.tfrecord')

# Función para guardar los datos en formato TFRecord
def save_data(dataset, filename):
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()]))
    
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    
    with tf.io.TFRecordWriter(filename) as writer:
        for image, label in dataset:
            feature = {
                'image': _bytes_feature(image),
                'label': _int64_feature(label)
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

# Función para cargar datos desde TFRecord
def load_data_from_tfrecord(filename):
    raw_dataset = tf.data.TFRecordDataset(filename)
    
    def _parse_function(proto):
        keys_to_features = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }
        parsed_features = tf.io.parse_single_example(proto, keys_to_features)
        image = tf.io.decode_jpeg(parsed_features['image'])
        label = parsed_features['label']
        return image, label
    
    return raw_dataset.map(_parse_function)

# Funciones para obtener los diferentes conjuntos de datos
def get_train_data(fold):
    return load_data_from_tfrecord(f'{constants.data_prefix}{constants.data_suffix}.tfrecord')

def get_test_data(fold):
    return load_data_from_tfrecord(f'{constants.pred_noised_data_fname}.tfrecord')

def get_noised_data(fold):
    return load_data_from_tfrecord(f'train_data_noised_fol.tfrecord')

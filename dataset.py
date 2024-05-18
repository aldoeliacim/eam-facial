# Copyright [2020] Luis Alberto Pineda Cortés, Rafael Morales Gamboa.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gzip
import os
import tensorflow as tf
import numpy as np
import random
from PIL import Image

import numpy as np

import constants

# Tamaño de imágenes
columns = 48
rows = 48

# Definir constantes de segmento
_TRAINING_SEGMENT = 0
_FILLING_SEGMENT = 1
_TESTING_SEGMENT = 2

def get_training(fold): return _get_segment(_TRAINING_SEGMENT, fold)
def get_filling(fold): return _get_segment(_FILLING_SEGMENT, fold)
def get_testing(fold, noised = False): return _get_segment(_TESTING_SEGMENT, fold, noised)

def _get_segment(segmento, fold, noised=False):
    """
    Obtener un segmento específico del conjunto de datos.

    Args:
    segmento (int): Tipo de segmento a obtener (entrenamiento, relleno, prueba).
    fold (int): Número del fold para la validación cruzada.
    noised (bool): Si se deben usar datos con ruido.

    Returns:
    tuple: Datos y etiquetas del segmento especificado.
    """
    if _get_segment.data is None or _get_segment.noised is None or _get_segment.labels is None:
        # Cargar el conjunto de datos si no se ha cargado aún
        _get_segment.data, _get_segment.noised, _get_segment.labels = _load_dataset(constants.data_path)

    print('Delimiting segment of data.')
    
    total = len(_get_segment.labels)
    training = total * constants.nn_training_percent
    filling = total * constants.am_filling_percent
    testing = total * constants.am_testing_percent
    step = total / constants.n_folds

    # Calcular índices de los segmentos
    i = int(fold * step)
    j = int((i + training) % total)
    k = int((j + filling) % total)
    l = int((k + testing) % total)

    # Seleccionar los índices según el segmento
    n, m = None, None
    if segmento == _TRAINING_SEGMENT: n, m = i, j
    elif segmento == _FILLING_SEGMENT: n, m = j, k
    elif segmento == _TESTING_SEGMENT: n, m = k, l

    # Obtener datos y etiquetas en el rango especificado
    data = constants.get_data_in_range(_get_segment.noised, n, m) if noised else constants.get_data_in_range(_get_segment.data, n, m)
    labels = constants.get_data_in_range(_get_segment.labels, n, m)
    
    return data, labels

# Inicializar atributos de la función
_get_segment.data = None
_get_segment.noised = None
_get_segment.labels = None

def noised(data, percent):
    """
    Añadir ruido a los datos.

    Args:
    data (np.ndarray): Conjunto de datos de entrada.
    percent (float): Porcentaje de ruido a añadir.

    Returns:
    np.ndarray: Conjunto de datos con ruido añadido.
    """
    print(f'Adding {percent}% noise to data.')
    copy = np.zeros(data.shape, dtype=float)
    n = 0
    for i in range(len(copy)):
        copy[i] = _noised(data[i], percent)
        n += 1
        constants.print_counter(n, 10000, step=100)
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

def _load_dataset(path):
    """
    Cargar el conjunto de datos, preprocesado o sin preprocesar.

    Args:
    path (str): Ruta al conjunto de datos.

    Returns:
    tuple: Datos, datos con ruido y etiquetas.
    """
    data, noised_data, labels = _preprocessed_dataset(path)
    if (data is None) or (noised_data is None) or (labels is None):
        data_train, labels_train = _load_fer2013(path, 'train')
        data_test, labels_test = _load_fer2013(path, 'test')
        
        data = np.concatenate((data_train, data_test), axis=0).astype(dtype=float)
        noised_data = noised(data, constants.noise_percent)
        labels = np.concatenate((labels_train, labels_test), axis=0)
        
        data, noised_data, labels = _shuffle(data, noised_data, labels)
        _save_dataset(data, noised_data, labels, path)
        
    return data, noised_data, labels

def _preprocessed_dataset(path):
    """
    Cargar el conjunto de datos preprocesados si existen.

    Args:
    path (str): Ruta al conjunto de datos.

    Returns:
    tuple: Datos, datos con ruido y etiquetas.
    """
    data_fname = os.path.join(path, constants.prep_data_fname)
    noised_fname = os.path.join(path, constants.pred_noised_data_fname)
    labels_fname = os.path.join(path, constants.prep_labels_fname)
    
    data, noised, labels = None, None, None
    try:
        data = np.load(data_fname)
        noised = np.load(noised_fname)
        labels = np.load(labels_fname).astype('int')
        print('Preprocessed dataset exists, it will be used.')
        print('El conjunto de datos preprocesado existe y se utilizará.')
    except FileNotFoundError:
        print('Preprocessed dataset does not exist.')
        print('El conjunto de datos preprocesado no existe.')
        
    return data, noised, labels

def _save_dataset(data, noised_data, labels, path):
    """
    Guardar el conjunto de datos preprocesado.

    Args:
    data (np.ndarray): Datos originales.
    noised_data (np.ndarray): Datos con ruido.
    labels (np.ndarray): Etiquetas.
    path (str): Ruta para guardar los datos.
    """
    print('Saving preprocessed dataset... ', end="")
    data_fname = os.path.join(path, constants.prep_data_fname)
    noised_fname = os.path.join(path, constants.pred_noised_data_fname)
    labels_fname = os.path.join(path, constants.prep_labels_fname)
    
    np.save(data_fname, data)
    np.save(noised_fname, noised_data)
    np.save(labels_fname, labels)
    print('Conjunto de datos preprocesado guardado.')

# def _shuffle(data, noised, labels):
#     print('Shuffling data and labels')
#     tuples = [(data[i], noised[i], labels[i]) for i in range(len(labels))]
#     random.shuffle(tuples)
#     data = np.array([p[0] for p in tuples])
#     noised = np.array([p[1] for p in tuples])
#     labels = np.array([p[2] for p in tuples], dtype=int)
#     return data, noised, labels

def _shuffle(data, noised_data, labels):
    print('Shuffling data and labels')
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    return data[indices], noised_data[indices], labels[indices]

def _load_fer2013(path, subset):
    """
    Load FER-2013 dataset from `path` using TensorFlow.

    Args:
    path (str): Ruta al conjunto de datos.
    subset (str): 'train' or 'test'.

    Returns:
    tuple: Imágenes y etiquetas.
    """
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(path, subset),
        labels='inferred',
        label_mode='int',
        color_mode='grayscale',
        batch_size=None,
        image_size=(rows, columns),
        seed=123
    )

    images = []
    labels = []

    for image, label in dataset:
        images.append(image.numpy())
        labels.append(label.numpy())

    images = np.array(images)
    labels = np.array(labels)

    return images, labels

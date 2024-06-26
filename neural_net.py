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

import numpy as np
import tensorflow as tf
from keras import Model
from keras.callbacks import Callback
from keras.layers import (
    BatchNormalization, Conv2D, Dense, Dropout, Flatten, Input, 
    LayerNormalization, MaxPool2D, Rescaling, Reshape, 
    SpatialDropout2D, UpSampling2D
)
from keras.models import load_model
from keras.utils import to_categorical

import constants
import dataset

epochs = 300
patience = 7 # class: EarlyStopping()
batch_size = 32
truly_training_percentage = 0.80

def conv_block(entry, layers, filters, dropout, first_block = False):
    conv = None
    for i in range(layers):
        if first_block:
            conv = Conv2D(kernel_size =3, padding ='same', activation='relu', 
                filters = filters, input_shape = (dataset.columns, dataset.rows, 1))(entry)
            first_block = False
        else:
            conv = Conv2D(kernel_size =3, padding ='same', activation='relu', 
                filters = filters)(entry)
        entry = BatchNormalization()(conv)
    pool = MaxPool2D(pool_size = 3, strides =2, padding ='same')(entry)
    drop = SpatialDropout2D(0.4)(pool)
    return drop

# The number of layers defined in get_encoder.
encoder_nlayers = 40

def get_encoder():
    # Entrada de la red con tamaño de las imágenes del dataset
    input_data = Input(shape=(dataset.columns, dataset.rows, 1))

    dropout = 0.1 # Dropout inicial
    filters = constants.domain // 32  # Filtros iniciales (16 en dominio 256)

    # Primer bloque convolucional con 2 capas
    # Tamaño de la imagen: 48x48 -> 48x48 (sin cambios por padding='same')
    # Filtros: 16
    output = conv_block(input_data, 2, filters, dropout, first_block=True)
    
    # Duplicamos los filtros y aumentamos el dropout
    filters *= 2
    dropout += 0.7
    # Segundo bloque convolucional con 2 capas
    # Tamaño de la imagen: 48x48 -> 24x24 (por MaxPool2D con pool_size=3 y strides=2)
    # Filtros: 32
    output = conv_block(output, 2, filters, dropout)
    
    # Duplicamos los filtros y aumentamos el dropout
    filters *= 2
    dropout += 0.7
    # Tercer bloque convolucional con 3 capas
    # Tamaño de la imagen: 24x24 -> 12x12 (por MaxPool2D)
    # Filtros: 64
    output = conv_block(output, 3, filters, dropout)
    
    # Duplicamos los filtros y aumentamos el dropout
    filters *= 2
    dropout += 0.7
    # Cuarto bloque convolucional con 3 capas
    # Tamaño de la imagen: 12x12 -> 6x6 (por MaxPool2D)
    # Filtros: 128
    output = conv_block(output, 3, filters, dropout)
    
    # Duplicamos los filtros y aumentamos el dropout
    filters *= 2
    dropout += 0.9
    # Quinto bloque convolucional con 3 capas
    # Tamaño de la imagen: 6x6 -> 3x3 (por MaxPool2D)
    # Filtros: 256
    output = conv_block(output, 3, filters, dropout)
    
    # Duplicamos los filtros y aumentamos el dropout
    filters *= 2
    dropout += 0.9
    # Sexto bloque convolucional con 3 capas adicionales
    # Tamaño de la imagen: 3x3 -> 1x1 (por MaxPool2D)
    # Filtros: 512
    output = conv_block(output, 3, filters, dropout)
    
    # Aplanamos las características extraídas
    output = Flatten()(output)
    # Normalizamos las características aplanadas
    output = LayerNormalization(name='encoded')(output)
    return input_data, output
# def get_encoder():
#     """Define el codificador VGGNet con 4 etapas convolucionales.
    
#     Returns:
#         tuple: Entrada y salida del codificador.
#     """
#     input_img = Input(shape=(48, 48, 1))

#     # Etapa 1
#     conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
#     bn1_1 = BatchNormalization()(conv1_1)
#     conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(bn1_1)
#     bn1_2 = BatchNormalization()(conv1_2)
#     pool1 = MaxPool2D((2, 2), padding='same')(bn1_2)  # 48x48x64 -> 24x24x64

#     # Etapa 2
#     conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
#     bn2_1 = BatchNormalization()(conv2_1)
#     conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(bn2_1)
#     bn2_2 = BatchNormalization()(conv2_2)
#     pool2 = MaxPool2D((2, 2), padding='same')(bn2_2)  # 24x24x128 -> 12x12x128

#     # Etapa 3
#     conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
#     bn3_1 = BatchNormalization()(conv3_1)
#     conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same')(bn3_1)
#     bn3_2 = BatchNormalization()(conv3_2)
#     pool3 = MaxPool2D((2, 2), padding='same')(bn3_2)  # 12x12x256 -> 6x6x256

#     # Etapa 4
#     conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
#     bn4_1 = BatchNormalization()(conv4_1)
#     conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same')(bn4_1)
#     bn4_2 = BatchNormalization()(conv4_2)
#     pool4 = MaxPool2D((2, 2), padding='same')(bn4_2)  # 6x6x512 -> 3x3x512

#     # Aplanar y capas completamente conectadas
#     flat = Flatten()(pool4)
#     fc1 = Dense(4096, activation='relu')(flat)
#     fc2 = Dense(4096, activation='relu')(fc1)
#     encoded = Dense(constants.domain, activation='relu')(fc2)

#     return input_img, encoded

def get_decoder():
    input_mem = Input(shape=(constants.domain,))
    width = dataset.columns // 4  # Aquí usamos 12 como width
    filters = constants.domain // 2
    dense = Dense(width * width * filters, activation='relu')(input_mem)
    output = Reshape((width, width, filters))(dense)

    # Ajuste de filtros y dropout
    filters *= 2
    dropout = 0.4

    # Bloques de UpSampling para reconstruir la imagen de vuelta a 48x48
    for i in range(2):
        trans = Conv2D(kernel_size=3, strides=1, padding='same', activation='relu', filters=filters)(output)
        pool = UpSampling2D(size=2)(trans)
        output = SpatialDropout2D(dropout)(pool)
        dropout /= 2.0
        filters //= 2
        output = BatchNormalization()(output)
        
    # Capa final de convolución para obtener la imagen de salida
    output = Conv2D(filters=1, kernel_size=3, strides=1, activation='sigmoid', padding='same')(output)
    output_img = Rescaling(255.0, name='decoded')(output)
    return input_mem, output_img
# def get_decoder():
#     """Define el decodificador para reconstruir la imagen a partir del codificador VGGNet.
    
#     Returns:
#         tuple: Entrada y salida del decodificador.
#     """
#     encoded_input = Input(shape=(constants.domain,))

#     # Aumentar la dimensionalidad
#     fc3 = Dense(4096, activation='relu')(encoded_input)
#     fc4 = Dense(4096, activation='relu')(fc3)
#     fc5 = Dense(3 * 3 * 512, activation='relu')(fc4)
#     reshaped = Reshape((3, 3, 512))(fc5)

#     # Etapa 4 inversa
#     upsample4 = UpSampling2D((2, 2))(reshaped)  # 3x3x512 -> 6x6x512
#     deconv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same')(upsample4)
#     deconv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same')(deconv4_2)
#     bn4_1 = BatchNormalization()(deconv4_1)

#     # Etapa 3 inversa
#     upsample3 = UpSampling2D((2, 2))(bn4_1)  # 6x6x512 -> 12x12x512
#     deconv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same')(upsample3)
#     deconv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same')(deconv3_2)
#     bn3_1 = BatchNormalization()(deconv3_1)

#     # Etapa 2 inversa
#     upsample2 = UpSampling2D((2, 2))(bn3_1)  # 12x12x256 -> 24x24x256
#     deconv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(upsample2)
#     deconv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same')(deconv2_2)
#     bn2_1 = BatchNormalization()(deconv2_1)

#     # Etapa 1 inversa
#     upsample1 = UpSampling2D((2, 2))(bn2_1)  # 24x24x128 -> 48x48x128
#     deconv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(upsample1)
#     deconv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(deconv1_2)
#     bn1_1 = BatchNormalization()(deconv1_1)

#     # Capa final para reconstruir la imagen
#     decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(bn1_1)  # 48x48x64 -> 48x48x1

#     return encoded_input, decoded

# The number of layers defined in get_classifier.
classifier_nlayers = 6

def get_classifier():
    input_mem = Input(shape=(constants.domain,))
    
    # Primera capa densa con activación ReLU y aplicar Dropout para regularización
    dense = Dense(constants.domain, activation='relu', input_shape=(constants.domain,))(input_mem)
    drop = Dropout(0.4)(dense)
    
    # Segunda capa densa con activación ReLU y aplicar Dropout para regularización
    dense = Dense(constants.domain, activation='relu')(drop)
    drop = Dropout(0.4)(dense)
    
    # Capa final densa con activación softmax para clasificación
    classification = Dense(constants.n_labels, activation='softmax', name='classified')(drop)
    
    return input_mem, classification

class EarlyStopping(Callback):
    """ Stop training when the loss gets lower than val_loss.

        Arguments:
            patience: Number of epochs to wait after condition has been hit.
            After this number of no reversal, training stops.
            It starts working after 10% of epochs have taken place.
    """

    def __init__(self):
        super(EarlyStopping, self).__init__()
        self.patience = patience
        self.prev_val_loss = float('inf')
        self.prev_val_accuracy = 0.0
        self.prev_val_rmse = float('inf')

        # Almacenar los mejores pesos cuando ocurra la mejora.
        # best_weights to store the weights at which the loss crossing occurs.
        self.best_weights = None
        self.start = min(epochs // 20, 3)
        self.wait = 0

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited since loss crossed val_loss.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        accuracy = logs.get('classifier_accuracy')
        val_accuracy = logs.get('val_classifier_accuracy')
        rmse = logs.get('decoder_root_mean_squared_error')
        val_rmse = logs.get('val_decoder_root_mean_squared_error')

        # Si estamos en las primeras épocas, solo guardamos los pesos actuales.
        if epoch < self.start:
            self.best_weights = self.model.get_weights()
        # Verificamos las condiciones para el early stopping.
        elif (loss < val_loss) or (accuracy > val_accuracy) or (rmse < val_rmse):
            self.wait += 1
        elif (val_accuracy > self.prev_val_accuracy):
            self.wait = 0
            self.prev_val_accuracy = val_accuracy
            self.best_weights = self.model.get_weights()
        elif (val_rmse < self.prev_val_rmse):
            self.wait = 0
            self.prev_val_rmse = val_rmse
            self.best_weights = self.model.get_weights()            
        elif (val_loss < self.prev_val_loss):
            self.wait = 0
            self.prev_val_loss = val_loss
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1

        print(f'Epochs waiting: {self.wait}')

        # Si hemos esperado más de `patience` épocas, detenemos el entrenamiento.
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            print("Restaurando los pesos del modelo desde el final de la mejor época.")
            print("Restoring model weights from the end of the best epoch.")
            self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


def train_network(prefix, es):
    # Inicializar la matriz de confusión y el historial de entrenamiento
    confusion_matrix = np.zeros((constants.n_labels, constants.n_labels))
    histories = []
    
    for fold in range(constants.n_folds):
        # Cargar los datos de entrenamiento y prueba para el pliegue actual
        training_data, training_labels = dataset.get_training(fold)
        testing_data, testing_labels = dataset.get_testing(fold)

        # Dividir los datos de entrenamiento en conjuntos de entrenamiento y validación
        truly_training = int(len(training_labels) * truly_training_percentage)
        validation_data = training_data[truly_training:]
        validation_labels = training_labels[truly_training:]
        training_data = training_data[:truly_training]
        training_labels = training_labels[:truly_training]

        # Convertir las etiquetas a formato categórico
        training_labels = to_categorical(training_labels)
        validation_labels = to_categorical(validation_labels)
        testing_labels = to_categorical(testing_labels)

        # Definir las métricas
        rmse = tf.keras.metrics.RootMeanSquaredError()
        accuracy = 'accuracy'

        # Crear las entradas del modelo
        input_data = Input(shape=(dataset.columns, dataset.rows, 1))

        # Definir y compilar el codificador
        input_enc, encoded = get_encoder()
        encoder = Model(input_enc, encoded, name='encoder')
        encoder.compile(optimizer = 'adam')
        encoder.summary()

        # Definir y compilar el clasificador
        input_cla, classified = get_classifier()
        classifier = Model(input_cla, classified, name='classifier')
        classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[accuracy])
        classifier.summary()

        # Definir y compilar el decodificador
        input_dec, decoded = get_decoder()
        decoder = Model(input_dec, decoded, name='decoder')
        decoder.compile(optimizer='adam', loss='mean_squared_error', metrics=[rmse])
        decoder.summary()

        # Conectar el codificador, el decodificador y el clasificador
        encoded = encoder(input_data)
        decoded = decoder(encoded)
        classified = classifier(encoded)

        # Definir y compilar el clasificador completo
        full_classifier = Model(inputs=input_data, outputs=classified, name='full_classifier')
        full_classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = [accuracy]) 

        # Definir y compilar el autoencoder
        autoencoder = Model(inputs = input_data, outputs=decoded, name='autoencoder')
        autoencoder.compile(loss='huber', optimizer='adam', metrics=[rmse])

        # Definir y compilar el modelo completo
        model = Model(inputs=input_data, outputs=[classified, decoded])
        model.compile(loss=['categorical_crossentropy', 'mean_squared_error'],
                      optimizer='adam',
                      metrics={'classifier': [accuracy], 'decoder' : [rmse]})
        model.summary()

        # Entrenar el modelo
        history = model.fit(
            training_data,
            (training_labels, training_data),
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(validation_data, {'classifier': validation_labels, 'decoder': validation_data}),
            callbacks=[EarlyStopping()],
            verbose=2
        )
        histories.append(history)
        
        # Evaluar el clasificador completo
        history = full_classifier.evaluate(testing_data, testing_labels, return_dict=True)
        histories.append(history)

        # Predecir las etiquetas y actualizar la matriz de confusión
        predicted_labels = np.argmax(full_classifier.predict(testing_data), axis=1)
        confusion_matrix += tf.math.confusion_matrix(np.argmax(testing_labels, axis=1),
                                                     predicted_labels, num_classes=constants.n_labels)

        # Evaluar el autoencoder
        history = autoencoder.evaluate(testing_data, testing_data, return_dict=True)
        histories.append(history)

        # Guardar los modelos
        encoder.save(constants.encoder_filename(prefix, es, fold))
        decoder.save(constants.decoder_filename(prefix, es, fold))
        classifier.save(constants.classifier_filename(prefix, es, fold))

        # Guardar las predicciones
        prediction_prefix = constants.classification_name(es)
        prediction_filename = constants.data_filename(prediction_prefix, es, fold)
        np.save(prediction_filename, predicted_labels)

    # Normalizar la matriz de confusión
    confusion_matrix = confusion_matrix.numpy()
    totals = confusion_matrix.sum(axis=1).reshape(-1, 1)
    return histories, confusion_matrix / totals


def obtain_features(model_prefix, features_prefix, labels_prefix, data_prefix, es):
    """ Generate features for sound segments, corresponding to phonemes.
    
    Uses the previously trained neural networks for generating the features.
    """
    for fold in range(constants.n_folds):
        # Cargar el modelo del encoder para el pliegue actual
        filename = constants.encoder_filename(model_prefix, es, fold)
        model = load_model(filename)
        model.summary()

        # Cargar los diferentes conjuntos de datos para el pliegue actual
        training_data, training_labels = dataset.get_training(fold)
        filling_data, filling_labels = dataset.get_filling(fold)
        testing_data, testing_labels = dataset.get_testing(fold)
        noised_data, noised_labels = dataset.get_testing(fold, noised = True)

        # Definir los conjuntos de datos y sus respectivos sufijos
        settings = [
            (training_data, training_labels, constants.training_suffix),
            (filling_data, filling_labels, constants.filling_suffix),
            (testing_data, testing_labels, constants.testing_suffix),
            (noised_data, noised_labels, constants.noised_suffix),
        ]

        # Extraer y guardar las características para cada conjunto de datos
        for data, labels, suffix in settings:
            features_filename = constants.data_filename(features_prefix + suffix, es, fold)
            labels_filename = constants.data_filename(labels_prefix + suffix, es, fold)

            # Predecir las características utilizando el modelo del codificador
            features = model.predict(data)
            
            # Guardar las características y las etiquetas
            np.save(features_filename, features)
            np.save(labels_filename, labels)

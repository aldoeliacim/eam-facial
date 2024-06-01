import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, saving

import constants
import dataset

def train_network(prefix, es):
    # Inicializar la matriz de confusión y el historial de entrenamiento
    confusion_matrix = np.zeros((constants.n_labels, constants.n_labels))
    histories = []

    for fold in range(constants.n_folds):
        # Preprocesar y almacenar los datos para este pliegue
        dataset.preprocess_data(fold)

        # Cargar los datos preprocesados
        train_data = ds.get_train_data(fold)
        test_data = ds.get_test_data(fold)
        noised_data = ds.get_noised_data(fold)
        
        # Dimensión del codificador
        encoding_dim = 64

        # Esta es nuestra imagen de entrada
        input_img = keras.Input(shape=(ds.columns, ds.rows, 1))

        # Codificador
        # La imagen de entrada de 48x48x1 pasa a través de una capa Conv2D y MaxPooling2D,
        # reduciendo las dimensiones y aumentando la profundidad de los canales.
        conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)  # 48x48x1 -> 48x48x64
        pool1 = layers.MaxPooling2D((2, 2), padding='same')(conv1)  # 48x48x64 -> 24x24x64
        conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)  # 24x24x64 -> 24x24x128
        pool2 = layers.MaxPooling2D((2, 2), padding='same')(conv2)  # 24x24x128 -> 12x12x128
        conv3 = layers.Conv2D(encoding_dim, (3, 3), activation='relu', padding='same')(pool2)  # 12x12x128 -> 12x12x64
        encoded = layers.MaxPooling2D((2, 2), padding='same')(conv3)  # 12x12x64 -> 6x6x64

        # Decodificador
        # La imagen codificada 6x6x64 pasa a través de una serie de capas Conv2D y UpSampling2D para reconstruir la imagen original de 48x48x1.
        deconv1 = layers.Conv2D(encoding_dim, (3, 3), activation='relu', padding='same')(encoded)  # 6x6x64 -> 6x6x64
        upsample1 = layers.UpSampling2D((2, 2))(deconv1)  # 6x6x64 -> 12x12x64
        deconv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(upsample1)  # 12x12x64 -> 12x12x128
        upsample2 = layers.UpSampling2D((2, 2))(deconv2)  # 12x12x128 -> 24x24x128
        deconv3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(upsample2)  # 24x24x128 -> 24x24x64
        upsample3 = layers.UpSampling2D((2, 2))(deconv3)  # 24x24x64 -> 48x48x64
        decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(upsample3)  # 48x48x64 -> 48x48x1
        
        # Este modelo mapea una entrada a su reconstrucción
        autoencoder = keras.Model(input_img, decoded)

        # Modelo de Codificador
        # Este modelo mapea una entrada a su representación codificada
        encoder = keras.Model(input_img, encoded)

        # Modelo de Decodificador
        # Esta es nuestra entrada codificada (dimensionalidad de 64)
        encoded_input = keras.Input(shape=(6, 6, encoding_dim))
        # Recuperar las últimas capas del modelo autoencoder
        decoder_layer = autoencoder.layers[-7:]  # Tomamos las últimas capas necesarias para el decodificador
        # Crear el modelo de decodificador
        x = encoded_input
        for layer in decoder_layer:
            x = layer(x)
        decoded_output = x
        decoder = keras.Model(encoded_input, decoded_output)

        # Compilar el modelo autoencoder
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

        # Cargar y dividir los datos
        ds = keras.utils.image_dataset_from_directory(
            constants.data_path,
            labels='inferred',
            label_mode='int',
            class_names=['angry', 'neutral'],
            color_mode='grayscale',
            batch_size=32,
            image_size=(48, 48),
            shuffle=True,
            seed=123,
            interpolation='bilinear'
        )

        train_data, test_data, fill_data = ds.split(list(ds.as_numpy_iterator()))

        # Normalización de datos
        normalization_layer = layers.Rescaling(1./255)
        normalized_train = train_data.map(lambda x, y: (normalization_layer(x), normalization_layer(x)))
        normalized_test = test_data.map(lambda x, y: (normalization_layer(x), normalization_layer(x)))

        # Callback para EarlyStopping
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

        # Entrenar el modelo autoencoder con EarlyStopping
        autoencoder.fit(train_data, epochs=100, validation_data=test_data, callbacks=[early_stop])

        # Obtener las representaciones codificadas y decodificadas
        test_images = next(iter(test_data.take(1)))[0]
        encoded_imgs = encoder.predict(test_images)
        decoded_imgs = decoder.predict(encoded_imgs)

        # Verificar las dimensiones de las imágenes originales y reconstruidas
        print("Dimensiones de las imágenes originales:", test_images.shape)
        print("Dimensiones de las imágenes reconstruidas:", decoded_imgs.shape)

        # Guardar el modelo autoencoder
        autoencoder.save(constants.data_path + '/autoencoder.keras')
        encoder.save(constants.encoder_filename(prefix, es, fold) + '.keras')
        decoder.save(constants.decoder_filename(prefix, es, fold) + '.keras')

        # Crear el modelo de clasificador
        encoded_output = encoder.output
        flatten = layers.Flatten()(encoded_output)
        dense1 = layers.Dense(128, activation='relu')(flatten)
        output = layers.Dense(len(constants.class_names), activation='softmax')(dense1)
        classifier = keras.Model(encoder.input, output)

        # Compilar el modelo de clasificador
        classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Entrenar el modelo de clasificador
        classifier.fit(train_data, epochs=100, validation_data=test_data, callbacks=[early_stop])

        # Evaluar el clasificador y actualizar la matriz de confusión
        test_labels = np.concatenate([y for x, y in test_data], axis=0)
        predictions = np.argmax(classifier.predict(test_images), axis=1)
        confusion_matrix += tf.math.confusion_matrix(test_labels, predictions, num_classes=len(constants.class_names))

        # Guardar el modelo de clasificador
        classifier.save(constants.classifier_filename(prefix, es, fold))
    
    # Normalizar la matriz de confusión
    confusion_matrix = confusion_matrix.numpy()
    totals = confusion_matrix.sum(axis=1).reshape(-1, 1)
    normalized_confusion_matrix = confusion_matrix / totals

    return histories, normalized_confusion_matrix

def obtain_features(model_prefix, features_prefix, labels_prefix, data_prefix, es):
    for fold in range(constants.n_folds):
        # Cargar el modelo del encoder para el pliegue actual
        filename = constants.encoder_filename(model_prefix, es, fold)
        model = saving.load_model(filename)
        model.summary()

        # Cargar los diferentes conjuntos de datos para el pliegue actual
        training_data, training_labels = dataset.get_training(fold)
        filling_data, filling_labels = dataset.get_filling(fold)
        testing_data, testing_labels = dataset.get_testing(fold)
        noised_data, noised_labels = dataset.get_testing(fold, noised=True)

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


# Llamar a la función para entrenar la red

# filename = 'mem_params.csv'
# parameters = np.genfromtxt(filename, dtype=float, delimiter=',', skip_header=1)
# es = constants.ExperimentSettings(np.genfromtxt(filename, dtype=float, delimiter=',', skip_header=1))
# model_prefix = constants.model_name(es)
# exp_settings = constants.ExperimentSettings(parameters)

# print('Training the network.')
# print('Model prefix:', model_prefix, 'Experiment settings:', es)
# train_network(model_prefix, es)

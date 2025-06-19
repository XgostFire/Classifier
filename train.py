import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--optimizer', type=str, default='adam')
args = parser.parse_args()

# Chargement des données Edge Impulse
X_train = np.load('X_split_train.npy')
Y_train = np.load('Y_split_train.npy')
X_test = np.load('X_split_test.npy')
Y_test = np.load('Y_split_test.npy')

num_classes = len(np.unique(Y_train))
input_shape = X_train.shape[1:]  # Exemple: (96,96,1)

# Préparation des labels
Y_train = tf.keras.utils.to_categorical(Y_train, num_classes)
Y_test = tf.keras.utils.to_categorical(Y_test, num_classes)

# Modèle simplifié
model = Sequential([
    Conv2D(4, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D(2, 2),
    Conv2D(8, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(16, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Optimiseur
if args.optimizer == 'adam':
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
elif args.optimizer == 'sgd':
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.learning_rate)
else:
    raise ValueError("Unsupported optimizer")

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Entraînement
model.fit(X_train, Y_train, epochs=args.epochs, batch_size=32, validation_data=(X_test, Y_test))

# Sauvegarde au format Keras
model.save('model.h5')

# Conversion TFLite quantifiée
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_quant_model)

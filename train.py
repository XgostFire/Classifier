import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os
import numpy as np

# Données
X_train = np.load(os.path.join(os.environ['EI_DATA_DIRECTORY'], 'X_train.npy'))
Y_train = np.load(os.path.join(os.environ['EI_DATA_DIRECTORY'], 'Y_train.npy'))
X_test = np.load(os.path.join(os.environ['EI_DATA_DIRECTORY'], 'X_test.npy'))
Y_test = np.load(os.path.join(os.environ['EI_DATA_DIRECTORY'], 'Y_test.npy'))

# Normalisation
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Récupérer le nombre de classes
num_classes = Y_train.shape[1]

# Modèle très léger
model = Sequential([
    Conv2D(8, (3,3), activation='relu', input_shape=(96, 96, 1)),
    MaxPooling2D(2,2),
    Conv2D(16, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entraînement
model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_data=(X_test, Y_test))

# Exportation du modèle TFLite
model.save('model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

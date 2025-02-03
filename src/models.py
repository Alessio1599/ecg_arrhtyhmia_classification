import tensorflow as tf
from tensorflow.keras import layers, models

def build_CNN(input_shape, output_shape, layer_1_size, layer_2_size, layer_3_size, layer_FC_size, dropout_rate):
    """
    Builds and returns a CNN model.
    """
    model = models.Sequential()

    # Input layer
    model.add(layers.InputLayer(shape=input_shape))

    # Conv Layers
    model.add(layers.Conv1D(filters=layer_1_size, kernel_size=3, activation='relu', padding='same'))
    model.add(layers.MaxPooling1D(pool_size=2))

    model.add(layers.Conv1D(filters=layer_2_size, kernel_size=3, activation='relu', padding='same'))
    model.add(layers.MaxPooling1D(pool_size=2))

    model.add(layers.Conv1D(filters=layer_3_size, kernel_size=3, activation='relu', padding='same'))
    model.add(layers.MaxPooling1D(pool_size=2))

    # Fully connected layer
    model.add(layers.Flatten())
    model.add(layers.Dense(layer_FC_size, activation='relu'))
    model.add(layers.Dropout(dropout_rate))

    # Output layer
    model.add(layers.Dense(output_shape, activation='softmax'))  

    return model

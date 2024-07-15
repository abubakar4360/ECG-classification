from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LSTM
import tensorflow as tf
from typing import Tuple

def build_model(input_shape: Tuple[int]) -> Sequential:
    """
    Builds a Convolutional Neural Network (CNN) model for ECG classification.

    Args:
        input_shape (tuple): Shape of the input data (number of features).

    Returns:
        Sequential: Compiled Keras Sequential model.
    """
    model = Sequential()

    model.add(Conv1D(32, 11, padding='same', input_shape=input_shape))
    model.add(Conv1D(32, 11, padding='same'))
    model.add(MaxPooling1D(pool_size=10))

    model.add(Conv1D(64, 9, padding='same'))
    model.add(Conv1D(64, 9, padding='same'))
    model.add(MaxPooling1D(pool_size=7))

    model.add(Conv1D(128, 7, padding='same'))
    model.add(Conv1D(128, 7, padding='same'))
    model.add(MaxPooling1D(pool_size=5))

    model.add(Conv1D(256, 5, padding='same'))
    model.add(Conv1D(256, 5, padding='same'))
    model.add(MaxPooling1D(pool_size=3))

    model.add(Conv1D(512, 3, padding='same'))
    model.add(Conv1D(512, 3, padding='same'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(LSTM(512))

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])

    return model

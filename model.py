import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout

def audio_encoder(input_shape):
    model = tf.keras.Sequential([
        Dense(128, activation='relu', input_shape=input_shape),
        Dropout(0.3),
        Dense(64, activation='relu')
    ])
    return model

def video_encoder(input_shape):
    model = tf.keras.Sequential([
        Dense(256, activation='relu', input_shape=input_shape),
        Dropout(0.4),
        Dense(64, activation='relu')
    ])
    return model

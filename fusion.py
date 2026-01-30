import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense

def fusion_model(audio_feat, video_feat):
    fused = Concatenate()([audio_feat, video_feat])
    x = Dense(128, activation='relu')(fused)
    x = Dense(64, activation='relu')(x)
    output = Dense(3, activation='softmax')(x)
    return output

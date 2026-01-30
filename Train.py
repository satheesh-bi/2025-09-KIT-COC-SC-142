import numpy as np
import pandas as pd
import tensorflow as tf

from model import audio_encoder, video_encoder
from fusion import fusion_model
from evaluation import evaluate

# Load data
audio_features = np.load("../data/audio_features.npy")
video_features = np.load("../data/video_features.npy")
labels_df = pd.read_csv("../data/ef_labels.csv")

labels = labels_df["label"].values

# Build model
audio_input = tf.keras.Input(shape=(audio_features.shape[1],))
video_input = tf.keras.Input(shape=(video_features.shape[1],))

audio_model = audio_encoder((audio_features.shape[1],))
video_model = video_encoder((video_features.shape[1],))

audio_out = audio_model(audio_input)
video_out = video_model(video_input)

output = fusion_model(audio_out, video_out)

model = tf.keras.Model(inputs=[audio_input, video_input], outputs=output)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(
    [audio_features, video_features],
    labels,
    epochs=30,
    batch_size=16,
    validation_split=0.2
)

# Evaluate
preds = model.predict([audio_features, video_features])
y_pred = preds.argmax(axis=1)

results = evaluate(labels, y_pred)
print(results)

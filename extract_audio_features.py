import numpy as np
import librosa
from datasets import load_dataset
import os

os.makedirs("../data", exist_ok=True)

dataset = load_dataset("BAAI/ChildMandarin", split="train")

features = []
ids = []

for idx, sample in enumerate(dataset):
    y = sample["audio"]["array"]
    sr = sample["audio"]["sampling_rate"]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    features.append(mfcc_mean)
    ids.append(idx)

audio_features = np.array(features)

np.save("../data/audio_features.npy", audio_features)

print("Audio features saved:", audio_features.shape)

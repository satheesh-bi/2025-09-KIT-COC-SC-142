import numpy as np
from sklearn.model_selection import train_test_split

X = np.load("../data/audio_features.npy")
ids = np.arange(len(X))

train, temp = train_test_split(ids, test_size=0.3, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

np.savetxt("../data/train_ids.txt", train, fmt="%d")
np.savetxt("../data/val_ids.txt", val, fmt="%d")
np.savetxt("../data/test_ids.txt", test, fmt="%d")

print("Splits saved")

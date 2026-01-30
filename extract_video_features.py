import numpy as np

# Placeholder for CNN video embeddings
video_features = np.random.rand(100, 256)

np.save("../data/video_features.npy", video_features)
print("Video features saved")

import numpy as np
import os

from dinov2_numpy import Dinov2Numpy
from preprocess_image import center_crop

if not os.path.exists("vit-dinov2-base.npz"):
    print("Error: 'vit-dinov2-base.npz' not found. Please verify the file path.")
    exit()

print("Loading model weights...")
weights = np.load("vit-dinov2-base.npz")
vit = Dinov2Numpy(weights)
print("Model loaded.")

print("Processing cat.jpg...")
cat_pixel_values = center_crop("./demo_data/cat.jpg")
cat_feat = vit(cat_pixel_values)

print("Processing dog.jpg...")
dog_pixel_values = center_crop("./demo_data/dog.jpg")
dog_feat = vit(dog_pixel_values)

ref_path = "./demo_data/cat_dog_feature.npy"
if os.path.exists(ref_path):
    print(f"Loading reference features from {ref_path}...")
    ref_feat = np.load(ref_path)
    
    my_feat = np.concatenate([cat_feat, dog_feat], axis=0)
    
    print(f"\nMy features shape: {my_feat.shape}")
    print(f"Ref features shape: {ref_feat.shape}")

    diff = np.linalg.norm(my_feat - ref_feat)
    print(f"\nDiff (L2 Norm): {diff:.6f}")

weights = np.load("vit-dinov2-base.npz")
vit = Dinov2Numpy(weights)
ref_features = np.load("./demo_data/cat_dog_feature.npy")
cat_img = center_crop("./demo_data/cat.jpg")
my_cat_feat = vit(cat_img).flatten()
ref_cat_feat = ref_features[0].flatten()
similarity = np.dot(my_cat_feat, ref_cat_feat) / (np.linalg.norm(my_cat_feat) * np.linalg.norm(ref_cat_feat))
print(f"Cosine Similarity: {similarity:.6f}")
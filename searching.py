import numpy as np
import os
import glob
import time
from PIL import Image
from dinov2_numpy import Dinov2Numpy
from preprocess_image import resize_short_side

WEIGHTS_PATH = "vit-dinov2-base.npz"

GALLERY_DIR = "gallery_images"

INDEX_FILE = "gallery_features.npy"

MAX_INDEX_SIZE = 10000  

DEFAULT_QUERY_IMG = os.path.join(GALLERY_DIR, "0.jpg")

def load_model():
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Weight file '{WEIGHTS_PATH}' not found!")
    
    print(f"Loading model weights from {WEIGHTS_PATH}...")
    weights = np.load(WEIGHTS_PATH)
    model = Dinov2Numpy(weights)
    print("Model loaded successfully.")
    return model

def extract_feature(model, img_path):
    try:
        pixel_values = resize_short_side(img_path)

        feat = model(pixel_values) # (1, 768)
        
        norm = np.linalg.norm(feat, axis=1, keepdims=True)
        feat = feat / (norm + 1e-6)
        
        return feat
    except Exception as e:
        return None

def build_index(model, gallery_dir, save_path):
    extensions = ['*.jpg', '*.jpeg', '*.png']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(gallery_dir, ext)))
    
    try:
        image_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    except:
        pass 

    if not image_paths:
        print(f"No images found in '{gallery_dir}'.")
        return None, None

    total_found = len(image_paths)
    if MAX_INDEX_SIZE is not None and total_found > MAX_INDEX_SIZE:
        print(f"Found {total_found} images, but limiting index to top {MAX_INDEX_SIZE} for speed.")
        image_paths = image_paths[:MAX_INDEX_SIZE]
    else:
        print(f"Found {total_found} images. Using all of them.")

    print(f"\nBuilding index for {len(image_paths)} images...")
    
    feats_list = []
    valid_paths = []
    
    start_time = time.time()

    for i, path in enumerate(image_paths):
        feat = extract_feature(model, path)
        if feat is not None:
            feats_list.append(feat)
            valid_paths.append(path)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed
            remaining_sec = (len(image_paths) - i) / speed
            print(f"Processed {i+1}/{len(image_paths)} | Speed: {speed:.1f} img/s | ETA: {remaining_sec/60:.1f} min")

    if not feats_list:
        return None, None

    gallery_feats = np.concatenate(feats_list, axis=0)

    data = {"feats": gallery_feats, "paths": valid_paths}
    np.save(save_path, data)
    print(f"\nIndex built and saved to '{save_path}'")
    
    return gallery_feats, valid_paths

def search(model, query_img_path, gallery_feats, gallery_paths, top_k=5):
    if not os.path.exists(query_img_path):
        print(f"Query image '{query_img_path}' not found.")
        return

    print(f"\n{'-'*60}")
    print(f"Query Image: {query_img_path}")
    print(f"{'-'*60}")

    query_feat = extract_feature(model, query_img_path)
    if query_feat is None: return

    scores = np.dot(query_feat, gallery_feats.T).flatten()

    top_indices = np.argsort(scores)[-top_k:][::-1]

    print(f"{'Rank':<5} | {'Score':<8} | {'Image Path'}")
    print(f"{'-'*60}")
    
    for rank, idx in enumerate(top_indices):
        score = scores[idx]
        path = gallery_paths[idx]
        print(f"{rank+1:<5} | {score:.4f}   | {path}")
        
        if rank == 0 and score > 0.99:
            print(f"       (Found exact match!)")

if __name__ == "__main__":
    vit_model = load_model()

    if os.path.exists(INDEX_FILE):
        print(f"Loading existing index from {INDEX_FILE}...")
        try:
            data = np.load(INDEX_FILE, allow_pickle=True).item()
            gallery_feats = data["feats"]
            gallery_paths = data["paths"]
            print(f"Loaded {len(gallery_paths)} images from index.")
        except:
            print("Index file corrupted. Rebuilding...")
            gallery_feats, gallery_paths = build_index(vit_model, GALLERY_DIR, INDEX_FILE)
    else:
        gallery_feats, gallery_paths = build_index(vit_model, GALLERY_DIR, INDEX_FILE)

    extra_test = "demo_data\cat.jpg" 
    if os.path.exists(extra_test):
        search(vit_model, extra_test, gallery_feats, gallery_paths)
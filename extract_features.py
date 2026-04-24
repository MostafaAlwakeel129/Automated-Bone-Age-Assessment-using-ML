"""
Bone age feature extraction script for the RSNA Bone Age dataset.

Note: Due to the size of the raw RSNA Bone Age image dataset, this script 
is designed to be executed in a cloud environment (e.g., Kaggle/Colab). 

It processes the raw X-rays using OpenCV (CLAHE, Morphological transformations) 
and extracts 189 distinct features (SIFT Bag of Visual Words, HOG, Global Stats).
The output consists of lightweight .npy arrays that are downloaded for local 
model training in the main pipeline. They are currently placed in the processed
data folder for organizational purposes.
=============================================================================
"""

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.feature import hog
from sklearn.cluster import MiniBatchKMeans

# Define paths (Adjust these to your Kaggle environment when running there)
TRAIN_CSV_PATH = '/kaggle/input/datasets/kmader/rsna-bone-age/boneage-training-dataset.csv'
TRAIN_IMG_DIR = '/kaggle/input/datasets/kmader/rsna-bone-age/boneage-training-dataset/boneage-training-dataset/'

sift = cv2.SIFT_create()
DICT_SIZE = 150

def extract_ROI(img_path, target_size=(1024, 1024)):
    """Isolates the hand, knuckles, and wrist from raw X-rays."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, target_size)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_contrast = clahe.apply(img)

    blurred = cv2.GaussianBlur(img_contrast, (7, 7), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    h, w = mask.shape
    cv2.rectangle(mask, (0, 0), (w - 1, h - 1), 0, thickness=2)

    k1 = np.ones((5, 5), np.uint8)
    k2 = np.ones((21, 21), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hands = [c for c in contours if cv2.contourArea(c) >= 5000 and (float(cv2.boundingRect(c)[3]) / cv2.boundingRect(c)[2]) > 0.5]

    if not hands:
        return None

    hand = max(hands, key=cv2.contourArea)
    canvas = np.zeros_like(mask)
    cv2.drawContours(canvas, [hand], -1, 255, -1)
    hand_img = cv2.bitwise_and(img_contrast, img_contrast, mask=canvas)

    M = cv2.moments(hand)
    if M["m00"] == 0:
        return None

    cy = int(M["m01"] / M["m00"])
    x, y, cw, ch = cv2.boundingRect(hand)
    finger_len = cy - y

    knuckle_y = cy - int(finger_len * 0.38)
    wrist_y = cy + int(ch * 0.18)
    crop_h = int(ch * 0.14)

    knuckles = hand_img[max(0, knuckle_y - crop_h):min(1024, knuckle_y + crop_h), x:x + cw]
    wrist = hand_img[max(0, wrist_y - crop_h):min(1024, wrist_y + crop_h), x:x + cw]

    return hand_img, knuckles, wrist

def extract_sift_features(crop):
    """Extracts SIFT descriptors from a specific image crop."""
    img = cv2.normalize(crop, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    _, descs = sift.detectAndCompute(img, None)
    return descs

def build_histogram(descs, dictionary, dict_size):
    """Converts SIFT descriptors into a Bag of Visual Words histogram."""
    if descs is None or len(descs) == 0:
        return np.zeros(dict_size)
    words = dictionary.predict(descs.astype(np.float32))
    hist, _ = np.histogram(words, bins=np.arange(dict_size + 1), density=True)
    return hist

def extract_global_features(hand_img, target_size=(128, 128)):
    """Extracts HOG features and global intensity statistics."""
    img = cv2.resize(hand_img, target_size) / 255.0
    feats = [np.mean(img), np.std(img)]
    hist, _ = np.histogram(img, bins=16, range=(0, 1))
    feats.extend(hist / np.sum(hist))
    hog_feats = hog(img, pixels_per_cell=(16, 16), cells_per_block=(1, 1), feature_vector=True)
    feats.extend(hog_feats[:20])
    return np.array(feats)

def main():
    print(" Loading Dataset ")
    df = pd.read_csv(TRAIN_CSV_PATH)
    
    print(" Building Visual Dictionary ")
    sample_df = df.sample(500, random_state=42)
    all_features = []

    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Learning Bone Shapes"):
        img_path = os.path.join(TRAIN_IMG_DIR, f"{row['id']}.png")
        result = extract_ROI(img_path)
        if result is None: continue

        _, knuckles, wrist = result
        kn_descs = extract_sift_features(knuckles)
        wr_descs = extract_sift_features(wrist)

        if kn_descs is not None: all_features.extend(kn_descs)
        if wr_descs is not None: all_features.extend(wr_descs)

    all_features = np.array(all_features, dtype=np.float32)
    dictionary = MiniBatchKMeans(n_clusters=DICT_SIZE, batch_size=2048, random_state=42, n_init=3)
    dictionary.fit(all_features)
    print("Dictionary built")

    print("\nTranslating X-Rays into Feature Vectors")
    X, y = [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Full Dataset"):
        img_path = os.path.join(TRAIN_IMG_DIR, f"{row['id']}.png")
        is_male = 1.0 if row['male'] else 0.0

        result = extract_ROI(img_path)
        if result is None: continue

        full_hand, knuckles, wrist = result
        kn_descs = extract_sift_features(knuckles)
        wr_descs = extract_sift_features(wrist)

        if kn_descs is not None and wr_descs is not None:
            combined = np.vstack((kn_descs, wr_descs))
        elif kn_descs is not None:
            combined = kn_descs
        elif wr_descs is not None:
            combined = wr_descs
        else:
            combined = None

        bow = build_histogram(combined, dictionary, DICT_SIZE)
        global_feats = extract_global_features(full_hand)
        vec = np.concatenate((bow, global_feats, [is_male]))

        X.append(vec)
        y.append(row['boneage'])

    np.save('X_sift_full.npy', np.array(X))
    np.save('y_sift_full.npy', np.array(y))
    print(f"Saved {len(X)} processed vectors.")

if __name__ == "__main__":
    main()
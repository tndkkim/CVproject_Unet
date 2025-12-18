import os
import pydicom
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import glob
from pathlib import Path

# ===== 경로 설정 =====
root = "./pneumothorax_data"
train_rle_path = os.path.join(root, "train-rle.csv")
dicom_train_folder = os.path.join(root, "dicom-images-train")

output_base = "./data"
train_png_folder = os.path.join(output_base, "train_png")
test_png_folder = os.path.join(output_base, "test_png")

os.makedirs(train_png_folder, exist_ok=True)
os.makedirs(test_png_folder, exist_ok=True)


df_rle = pd.read_csv(train_rle_path)
print("\nOriginal columns:", df_rle.columns.tolist())

df_rle.columns = df_rle.columns.str.strip()
print("Cleaned columns:", df_rle.columns.tolist())

dicom_file_paths = {}

all_dcm_files = glob.glob(os.path.join(dicom_train_folder, "**/*.dcm"), recursive=True)
for dcm_path in tqdm(all_dcm_files, desc="Mapping files"):
    basename = os.path.basename(dcm_path).replace('.dcm', '')
    dicom_file_paths[basename] = dcm_path

image_ids = df_rle["ImageId"].unique()
print(f"\n📊 Splitting {len(image_ids)} images...")

labels = []
for img_id in image_ids:
    rle = df_rle[df_rle['ImageId'] == img_id]['EncodedPixels'].values[0]
    has_pneumothorax = 0 if (rle == ' -1' or pd.isna(rle)) else 1
    labels.append(has_pneumothorax)

train_ids, test_ids = train_test_split(
    image_ids, 
    test_size=0.0, 
    random_state=42,
    stratify=labels
)

train_labels = [labels[i] for i, img_id in enumerate(image_ids) if img_id in train_ids]
test_labels = [labels[i] for i, img_id in enumerate(image_ids) if img_id in test_ids]

print(f"\nTrain set: {sum(train_labels)} positive, {len(train_labels)-sum(train_labels)} negative")
print(f"Test set:  {sum(test_labels)} positive, {len(test_labels)-sum(test_labels)} negative")

def convert_dicom_to_png(dicom_path):
    """DICOM 파일을 PNG로 변환"""
    try:
        dicom = pydicom.dcmread(dicom_path)
        img = dicom.pixel_array
        
        # Normalize to 0-255
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype(np.uint8)
        
        # Convert grayscale to BGR
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        return img
    except Exception as e:
        print(f"Error reading DICOM {dicom_path}: {e}")
        return None

def process_images(image_ids, output_folder, split_name):
    success_count = 0
    fail_count = 0

    
    for img_id in tqdm(image_ids, desc=split_name):
        if img_id not in dicom_file_paths:
            fail_count += 1
            if fail_count <= 5:
                print(f"File not found: {img_id}")
            continue
        
        dicom_path = dicom_file_paths[img_id]
        
        # DICOM → PNG 변환
        img = convert_dicom_to_png(dicom_path)
        if img is None:
            fail_count += 1
            continue

        output_path = os.path.join(output_folder, f"{img_id}.png")
        cv2.imwrite(output_path, img)
        
        success_count += 1

    return success_count, fail_count

train_success, train_fail = process_images(train_ids, train_png_folder, "Train")
test_success, test_fail = process_images(test_ids, test_png_folder, "Test")


train_pngs = len(glob.glob(os.path.join(train_png_folder, "*.png")))

output_ids = {
    'train_ids': train_ids.tolist(),
    'test_ids': test_ids.tolist()
}

import json
with open(os.path.join(output_base, 'split_ids.json'), 'w') as f:
    json.dump(output_ids, f, indent=2)

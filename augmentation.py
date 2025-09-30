import os
import cv2
import pandas as pd
import albumentations as A
from tqdm import tqdm
import csv

# ---------------- CONFIG ---------------- #
INPUT_IMAGES_DIR = "dataset/images/all_images"       # original images
INPUT_CSV = "dataset/images/all_annotations.csv"    # original annotations
OUTPUT_DIR = "augmented_dataset/train/images"            # where augmented images will go
OUTPUT_CSV = "augmented_dataset/train/annotations.csv"   # updated annotations
N_AUGMENTATIONS = 3  # how many augmented copies per original image
# ---------------------------------------- #

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load CSV
df = pd.read_csv(INPUT_CSV)

# Define augmentation pipeline
transform = A.Compose([
    A.RandomBrightnessContrast(p=0.6),
    A.MotionBlur(blur_limit=7, p=0.3),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.RandomFog(fog_coef_lower=0.2, fog_coef_upper=0.5, p=0.2),
    A.RandomRain(p=0.2),
],
bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

# Prepare output CSV
with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "class_id", "class_name", "xmin", "ymin", "xmax", "ymax"])

    # Loop over all images
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = os.path.join(INPUT_IMAGES_DIR, row["filename"])
        image = cv2.imread(img_path)

        bbox = [row["xmin"], row["ymin"], row["xmax"], row["ymax"]]
        label = row["class_id"]

        # Save original as well
        writer.writerow([row["filename"], row["class_id"], row["class_name"],
                         bbox[0], bbox[1], bbox[2], bbox[3]])

        # Generate augmented copies
        for i in range(N_AUGMENTATIONS):
            augmented = transform(image=image, bboxes=[bbox], class_labels=[label])
            aug_img = augmented["image"]
            aug_bbox = augmented["bboxes"][0]

            # Save augmented image
            base_name, ext = os.path.splitext(row["filename"])
            aug_filename = f"{base_name}_aug{i}{ext}"
            cv2.imwrite(os.path.join(OUTPUT_DIR, aug_filename), aug_img)

            # Write augmented annotation
            writer.writerow([aug_filename, row["class_id"], row["class_name"],
                             int(aug_bbox[0]), int(aug_bbox[1]),
                             int(aug_bbox[2]), int(aug_bbox[3])])

print(f"✅ Augmentation complete! Augmented images saved to {OUTPUT_DIR}")
print(f"📑 Updated annotations saved to {OUTPUT_CSV}")
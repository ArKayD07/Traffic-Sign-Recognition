import os
import csv
import random
import shutil
from pathlib import Path
from PIL import Image

# ---------------- CONFIG ---------------- #
TEMPLATES_DIR = "dataset/templates"
BACKGROUNDS_DIR = "dataset/backgrounds"
OCCLUDERS_DIR = "dataset/occluders"
CLASSES_CSV = "dataset/classes.csv"
OUTPUT_DIR = "dataset/images"
N_IMAGES = 50000  # total number of synthetic images to generate
SPLIT_RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}
# ---------------------------------------- #

random.seed(42)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load class mapping ---
template_to_class = {}
with open(CLASSES_CSV, newline='', encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        tpl = row["template_filename"].strip()
        cid = int(row["class_id"])
        name = row["class_name"].strip()
        template_to_class[tpl] = {"id": cid, "name": name}

# --- Collect assets ---
templates = list(template_to_class.keys())
backgrounds = [os.path.join(BACKGROUNDS_DIR, f) for f in os.listdir(BACKGROUNDS_DIR)]
occluders = [os.path.join(OCCLUDERS_DIR, f) for f in os.listdir(OCCLUDERS_DIR)]

# --- Temporary folder for all images ---
all_images_dir = os.path.join(OUTPUT_DIR, "all_images")
os.makedirs(all_images_dir, exist_ok=True)

annotations_file = os.path.join(OUTPUT_DIR, "all_annotations.csv")
with open(annotations_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "class_id", "class_name", "xmin", "ymin", "xmax", "ymax"])

    for i in range(N_IMAGES):
        bg = Image.open(random.choice(backgrounds)).convert("RGB")
        tpl_name = random.choice(templates)
        tpl_path = os.path.join(TEMPLATES_DIR, tpl_name)
        tpl_img = Image.open(tpl_path).convert("RGBA")

        # Resize template randomly
        scale = random.uniform(0.2, 0.6)
        tpl_w, tpl_h = int(tpl_img.width * scale), int(tpl_img.height * scale)
        tpl_img = tpl_img.resize((tpl_w, tpl_h), Image.LANCZOS)

        # Place sign at random position
        x = random.randint(0, bg.width - tpl_w)
        y = random.randint(0, bg.height - tpl_h)

        # Composite onto background
        bg.paste(tpl_img, (x, y), tpl_img)

        # Optional occluder
        if random.random() < 0.3:
            occ_path = random.choice(occluders)
            occ_img = Image.open(occ_path).convert("RGBA")
            occ_scale = random.uniform(0.3, 0.7)
            occ_w, occ_h = int(occ_img.width * occ_scale), int(occ_img.height * occ_scale)
            occ_img = occ_img.resize((occ_w, occ_h), Image.LANCZOS)
            ox = random.randint(0, bg.width - occ_w)
            oy = random.randint(0, bg.height - occ_h)
            bg.paste(occ_img, (ox, oy), occ_img)

        # Save final image
        filename = f"synthetic_{i:04d}.jpg"
        filepath = os.path.join(all_images_dir, filename)
        bg.save(filepath, "JPEG", quality=90)

        # Write bounding box
        cls = template_to_class[tpl_name]
        writer.writerow([filename, cls["id"], cls["name"], x, y, x + tpl_w, y + tpl_h])

print(f"✅ Generated {N_IMAGES} images and saved to {all_images_dir}")
print(f"📑 Annotations saved to {annotations_file}")

# --- Split into train/val/test ---
def stratified_split(csv_path, output_dir, split_ratios):
    with open(csv_path, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Group by class
    by_class = {}
    for row in rows:
        by_class.setdefault(row["class_id"], []).append(row)

    # Create split folders
    for split in split_ratios:
        Path(os.path.join(output_dir, split, "images")).mkdir(parents=True, exist_ok=True)
        open(os.path.join(output_dir, split, "annotations.csv"), "w", newline="", encoding="utf-8").close()

    # Split stratified
    for cls, samples in by_class.items():
        random.shuffle(samples)
        n = len(samples)
        n_train = int(n * split_ratios["train"])
        n_val = int(n * split_ratios["val"])

        splits = {
            "train": samples[:n_train],
            "val": samples[n_train:n_train+n_val],
            "test": samples[n_train+n_val:]
        }

        for split, rows_split in splits.items():
            ann_file = os.path.join(output_dir, split, "annotations.csv")
            with open(ann_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for row in rows_split:
                    if f.tell() == 0:
                        writer.writerow(row.keys())  # header
                    writer.writerow(row.values())

            for row in rows_split:
                src = os.path.join(all_images_dir, row["filename"])
                dst = os.path.join(output_dir, split, "images", row["filename"])
                shutil.copy(src, dst)

stratified_split(annotations_file, OUTPUT_DIR, SPLIT_RATIOS)

print("✅ Stratified split complete! Folders created:")
print("   synthetic_dataset/train, synthetic_dataset/val, synthetic_dataset/test")

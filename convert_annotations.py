#!/usr/bin/env python3
"""
convert_annotations.py (prefer dataset/all_images/all_annotations.csv)

Usage:
    python convert_annotations.py --dataset_dir dataset --output_dir converted_dataset
    python convert_annotations.py --dataset_dir dataset --output_dir converted_dataset --master_csv dataset/all_images/all_annotations.csv

Requirements:
    pip install pillow tqdm
"""

import os
import shutil
import csv
import json
import argparse
import random
from collections import defaultdict, OrderedDict
from xml.etree.ElementTree import Element, SubElement, ElementTree
from PIL import Image
from tqdm import tqdm

random.seed(42)

# ------------------ Utilities ------------------ #

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def read_csv_rows(csv_path):
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        hdr = reader.fieldnames
        for r in reader:
            rows.append({k: (v.strip() if isinstance(v, str) else v) for k, v in r.items()})
    return rows, hdr

def write_csv_rows(csv_path, header, rows):
    ensure_dir(os.path.dirname(csv_path))
    with open(csv_path, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

# ------------------ Master CSV discovery ------------------ #

def prefer_master_csv(dataset_dir, explicit_master=None):
    """
    Preference order:
     1) explicit_master if provided and exists
     2) dataset/all_images/all_annotations.csv
     3) dataset/annotations.csv
     4) any CSV in dataset containing 'annot' in its name
     5) None -> error
    """
    if explicit_master and os.path.exists(explicit_master):
        return explicit_master

    cand1 = os.path.join(dataset_dir, "all_images", "all_annotations.csv")
    if os.path.exists(cand1):
        return cand1

    cand2 = os.path.join(dataset_dir, "annotations.csv")
    if os.path.exists(cand2):
        return cand2

    # search for any CSV with 'annot' in the filename
    for fname in os.listdir(dataset_dir):
        if fname.lower().endswith(".csv") and "annot" in fname.lower():
            return os.path.join(dataset_dir, fname)

    return None

# ------------------ Matching & parsing helpers ------------------ #

def parse_row_fields(row):
    """Normalize a CSV row to {filename, class_id, class_name, bbox} or return None."""
    # find filename-like field
    fname_keys = ['filename', 'file', 'image', 'image_name', 'img', 'path']
    fname = None
    for k in fname_keys:
        if k in row and row[k]:
            fname = os.path.basename(row[k].strip())
            break
    if not fname:
        # fallback: find any field that looks like a jpg/png path
        for k, v in row.items():
            if isinstance(v, str) and v.lower().endswith(('.jpg', '.jpeg', '.png')):
                fname = os.path.basename(v.strip())
                break
    if not fname:
        return None

    # class id
    cid = None
    if 'class_id' in row and row['class_id'] != '':
        try:
            cid = int(float(row['class_id']))
        except Exception:
            cid = None

    # class name
    cname = row.get('class_name') or row.get('class') or row.get('label') or ''
    try:
        cname = cname.strip()
    except Exception:
        cname = str(cname)

    # bbox
    try:
        xmin = int(float(row.get('xmin', row.get('x1', 0))))
        ymin = int(float(row.get('ymin', row.get('y1', 0))))
        xmax = int(float(row.get('xmax', row.get('x2', xmin))))
        ymax = int(float(row.get('ymax', row.get('y2', ymin))))
    except Exception:
        return None

    return {'filename': fname, 'class_id': cid if cid is not None else -1, 'class_name': cname, 'bbox': [xmin, ymin, xmax, ymax], 'orig': row}

# ------------------ Image discovery & heuristic matching ------------------ #

def discover_images_map(dataset_dir):
    """Return mapping basename -> list of full paths (in case duplicate names exist in different folders)."""
    mapping = {}
    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                mapping.setdefault(f, []).append(os.path.join(root, f))
    return mapping

def match_master_rows_to_images(master_rows, image_map):
    """
    Use multi-step heuristics to match master CSV rows to discovered images.
    Returns:
      matched_rows: list of parsed rows with filename replaced by matched basename
      unmatched: list of parsed rows that couldn't be matched
    """
    parsed = []
    unmatched = []
    # Build lower-case key map for case-insensitive matching
    lc_map = {k.lower(): k for k in image_map.keys()}

    for r in master_rows:
        p = parse_row_fields(r)
        if not p:
            continue
        base = p['filename']
        matched_key = None

        # heuristic 1: exact basename
        if base in image_map:
            matched_key = base
        else:
            # heuristic 2: case-insensitive basename
            if base.lower() in lc_map:
                matched_key = lc_map[base.lower()]
            else:
                # heuristic 3: stem match (filename without extension)
                stem = os.path.splitext(base)[0]
                for key in image_map.keys():
                    if os.path.splitext(key)[0] == stem:
                        matched_key = key
                        break
                # heuristic 4: substring matches
                if matched_key is None:
                    for key in image_map.keys():
                        if (base in key) or (key in base):
                            matched_key = key
                            break
                # heuristic 5: startswith/endswith on stems
                if matched_key is None:
                    for key in image_map.keys():
                        s_key = os.path.splitext(key)[0]
                        if s_key.startswith(stem) or s_key.endswith(stem) or stem.startswith(s_key) or stem.endswith(s_key):
                            matched_key = key
                            break

        if matched_key:
            p['filename'] = matched_key
            parsed.append(p)
        else:
            unmatched.append(p)

    return parsed, unmatched

# ------------------ Splitting & writing ------------------ #

def stratified_split_and_copy(dataset_dir, parsed_rows, splits=('train', 'val', 'test'), ratios=(0.7, 0.2, 0.1)):
    """
    parsed_rows: list of parsed rows (filename basenames standardized)
    Copies images into dataset/<split>/images and returns a dict split->rows (rows are original-style dicts).
    """
    # group parsed rows by filename (preserve multiple objects)
    by_file = defaultdict(list)
    for p in parsed_rows:
        by_file[p['filename']].append(p)

    # pick a primary class per image for stratified sampling (mode or first)
    primary = {}
    for fname, anns in by_file.items():
        # choose the most common class_id among annotations for that image
        counts = defaultdict(int)
        for a in anns:
            counts[a['class_id']] += 1
        primary[fname] = max(counts.items(), key=lambda kv: kv[1])[0]

    filenames = list(by_file.keys())
    labels = [primary[f] for f in filenames]

    # simple stratified split using deterministic approach (we'll shuffle per class)
    class_groups = defaultdict(list)
    for fn in filenames:
        class_groups[primary[fn]].append(fn)

    train_files, val_files, test_files = [], [], []
    for cls, fns in class_groups.items():
        random.shuffle(fns)
        n = len(fns)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        train_files.extend(fns[:n_train])
        val_files.extend(fns[n_train:n_train + n_val])
        test_files.extend(fns[n_train + n_val:])

    split_map = {'train': train_files, 'val': val_files, 'test': test_files}

    # prepare directories and copy images
    for split in splits:
        out_dir = os.path.join(dataset_dir, split, 'images')
        ensure_dir(out_dir)
        for fn in split_map[split]:
            # choose first discovered path for that filename (image discovery may have multiple)
            # master image should be found in discover_images_map earlier
            # find full path by walking dataset_dir again (or pass a map). We'll search quickly.
            src = None
            for root, _, files in os.walk(dataset_dir):
                if fn in files:
                    src = os.path.join(root, fn)
                    break
            if src is None:
                # this should not happen; but skip if missing
                print(f"[WARN] Could not find source file for {fn} when copying to split {split}")
                continue
            dst = os.path.join(out_dir, fn)
            if os.path.abspath(src) != os.path.abspath(dst):
                shutil.copyfile(src, dst)

    # build per-split row lists in CSV-friendly format
    split_rows = {s: [] for s in splits}
    for split in splits:
        for fn in split_map[split]:
            for ann in by_file[fn]:
                # flatten to standard CSV record; prefer original 'orig' row mapping if present
                orig = ann.get('orig', None)
                if orig:
                    # ensure filename field uses the basename only
                    # find first filename-like key in orig and set to basename
                    newrow = dict(orig)
                    for key in ('filename', 'file', 'image', 'image_name', 'path'):
                        if key in newrow:
                            newrow[key] = ann['filename']
                    # ensure bbox fields exist
                    for key in ('xmin','ymin','xmax','ymax'):
                        if key not in newrow:
                            newrow[key] = ann['bbox'][('xmin','ymin','xmax','ymax').index(key)]
                    # ensure class fields exist
                    if 'class_id' not in newrow or newrow['class_id']=='':
                        newrow['class_id'] = ann['class_id']
                    if 'class_name' not in newrow or newrow['class_name']=='':
                        newrow['class_name'] = ann['class_name']
                    split_rows[split].append(newrow)
                else:
                    split_rows[split].append({
                        'filename': ann['filename'],
                        'class_id': ann['class_id'],
                        'class_name': ann['class_name'],
                        'xmin': ann['bbox'][0],
                        'ymin': ann['bbox'][1],
                        'xmax': ann['bbox'][2],
                        'ymax': ann['bbox'][3]
                    })

    return split_rows

# ------------------ Conversion (VOC/YOLO/COCO) ------------------ #

def write_voc_xml(image_info, annotations, out_xml_path):
    root = Element('annotation')
    SubElement(root, 'folder').text = image_info.get('folder', '')
    SubElement(root, 'filename').text = image_info['filename']
    SubElement(root, 'path').text = image_info.get('path', '')
    source = SubElement(root, 'source'); SubElement(source, 'database').text = 'Unknown'
    size = SubElement(root, 'size'); SubElement(size, 'width').text = str(image_info['width']); SubElement(size, 'height').text = str(image_info['height']); SubElement(size, 'depth').text = str(image_info.get('depth', 3))
    SubElement(root, 'segmented').text = '0'
    for ann in annotations:
        obj = SubElement(root, 'object')
        SubElement(obj, 'name').text = ann['class_name']
        SubElement(obj, 'pose').text = 'Unspecified'
        SubElement(obj, 'truncated').text = '0'
        SubElement(obj, 'difficult').text = '0'
        bnd = SubElement(obj, 'bndbox')
        SubElement(bnd, 'xmin').text = str(ann['bbox'][0])
        SubElement(bnd, 'ymin').text = str(ann['bbox'][1])
        SubElement(bnd, 'xmax').text = str(ann['bbox'][2])
        SubElement(bnd, 'ymax').text = str(ann['bbox'][3])
    ensure_dir(os.path.dirname(out_xml_path))
    ElementTree(root).write(out_xml_path, encoding='utf-8', xml_declaration=True)

def convert_to_yolo_line(ann, img_w, img_h, classid_map):
    xmin, ymin, xmax, ymax = ann['bbox']
    xmin = max(0, min(xmin, img_w - 1)); xmax = max(0, min(xmax, img_w - 1))
    ymin = max(0, min(ymin, img_h - 1)); ymax = max(0, min(ymax, img_h - 1))
    bw = xmax - xmin; bh = ymax - ymin
    if bw <= 0 or bh <= 0:
        return None
    x_c = (xmin + bw / 2.0) / img_w
    y_c = (ymin + bh / 2.0) / img_h
    w_n = bw / img_w; h_n = bh / img_h
    cls_idx = classid_map.get(ann['class_id'], 0)
    return f"{cls_idx} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}"

def make_coco_annotation(ann_id, image_id, ann, classid_map):
    xmin, ymin, xmax, ymax = ann['bbox']
    w = xmax - xmin; h = ymax - ymin
    if w <= 0 or h <= 0:
        return None
    bbox = [xmin, ymin, w, h]
    return {"id": ann_id, "image_id": image_id, "category_id": int(classid_map.get(ann['class_id'], 0)), "bbox": [float(x) for x in bbox], "area": float(w * h), "iscrowd": 0, "segmentation": []}

# ------------------ Main pipeline ------------------ #

def convert_annotations(dataset_dir, output_dir, master_csv_arg=None, splits=('train', 'val', 'test')):
    dataset_dir = os.path.abspath(dataset_dir)
    ensure_dir(output_dir)

    master_csv = prefer_master_csv(dataset_dir, explicit_master=master_csv_arg)
    if not master_csv:
        raise RuntimeError(f"No master annotations CSV found (looked for --master_csv, dataset/all_images/all_annotations.csv, dataset/annotations.csv, or any CSV with 'annot' in name).")
    print(f"[INFO] Using master CSV: {master_csv}")

    # Discover images
    image_map = discover_images_map(dataset_dir)
    if len(image_map) == 0:
        raise RuntimeError("No image files found under dataset directory.")

    # If no per-split images exist, create splits using master CSV rows matched to discovered images
    any_split_images = any(os.path.isdir(os.path.join(dataset_dir, s, 'images')) and len([f for f in os.listdir(os.path.join(dataset_dir, s, 'images')) if f.lower().endswith(('.jpg','.jpeg','.png'))]) > 0 for s in splits)

    if not any_split_images:
        master_rows, _ = read_csv_rows(master_csv)
        parsed, unmatched = match_master_rows_to_images(master_rows, image_map)
        print(f"[INFO] Matched {len(parsed)} rows to discovered images; {len(unmatched)} rows unmatched.")
        if len(parsed) == 0:
            raise RuntimeError("No master CSV rows matched available images. Inspect CSV filenames vs image filenames.")
        # create stratified splits and copy images into dataset/<split>/images/
        split_rows = stratified_split_and_copy(dataset_dir, parsed, splits=splits)
        # write per-split CSVs (CSV header standardized)
        header = ["filename", "class_id", "class_name", "xmin", "ymin", "xmax", "ymax"]
        for s in splits:
            rows = split_rows.get(s, [])
            if not rows:
                continue
            write_csv_rows(os.path.join(dataset_dir, s, "annotations.csv"), header, rows)
            print(f"[INFO] Wrote {len(rows)} rows to {os.path.join(dataset_dir, s, 'annotations.csv')}")
    else:
        print("[INFO] Per-split image folders detected; will use existing split folders and create split CSVs if missing.")
        # if split CSVs missing, build them by filtering master CSV for images in each split folder
        for s in splits:
            csv_path = os.path.join(dataset_dir, s, "annotations.csv")
            if os.path.exists(csv_path):
                print(f"[OK] Found existing CSV for split '{s}': {csv_path}")
                continue
            # find images in split
            img_dir1 = os.path.join(dataset_dir, s, "images")
            img_dir2 = os.path.join(dataset_dir, "images", s)
            img_dir = img_dir1 if os.path.isdir(img_dir1) else (img_dir2 if os.path.isdir(img_dir2) else None)
            if not img_dir:
                continue
            imgs = set([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            if not imgs:
                continue
            master_rows, header = read_csv_rows(master_csv)
            filtered = []
            for r in master_rows:
                # attempt to extract basename from row
                fname = None
                for key in ('filename', 'file', 'image', 'image_name', 'path'):
                    if key in r and r[key]:
                        fname = os.path.basename(r[key].strip())
                        break
                if fname and fname in imgs:
                    filtered.append(r)
            if filtered:
                write_csv_rows(csv_path, header or ["filename","class_id","class_name","xmin","ymin","xmax","ymax"], filtered)
                print(f"[INFO] Wrote {len(filtered)} rows to {csv_path}")

    # Now collect classes from per-split CSVs (or master)
    split_csvs = []
    for s in splits:
        p = os.path.join(dataset_dir, s, "annotations.csv")
        if os.path.exists(p):
            split_csvs.append(p)
    csvs_for_classes = split_csvs if len(split_csvs) > 0 else [master_csv]

    classid_name = OrderedDict()
    for p in csvs_for_classes:
        rows, _ = read_csv_rows(p)
        for r in rows:
            cid = None
            if 'class_id' in r and r['class_id'] != '':
                try:
                    cid = int(float(r['class_id']))
                except:
                    cid = None
            if cid is None:
                continue
            cname = r.get('class_name') or r.get('class') or r.get('label') or ''
            classid_name[cid] = cname.strip() if cname else str(cid)

    if len(classid_name) == 0:
        raise RuntimeError("No classes found in dataset. Make sure annotations CSVs contain class_id/class_name fields.")

    dataset_class_ids = list(classid_name.keys())
    classid_map = {cid: idx for idx, cid in enumerate(dataset_class_ids)}
    class_names = [classid_name[cid] for cid in dataset_class_ids]

    # write classes.names
    ensure_dir(os.path.join(output_dir, 'yolo'))
    class_names_path = os.path.join(output_dir, 'yolo', 'classes.names')
    with open(class_names_path, 'w', encoding='utf-8') as f:
        for n in class_names:
            f.write(n + '\n')
    print(f"[INFO] Wrote classes.names ({len(class_names)} classes) -> {class_names_path}")

    # Convert each split
    for s in splits:
        csv_path = os.path.join(dataset_dir, s, "annotations.csv")
        img_dir1 = os.path.join(dataset_dir, s, "images")
        img_dir2 = os.path.join(dataset_dir, "images", s)
        images_dir = img_dir1 if os.path.isdir(img_dir1) else (img_dir2 if os.path.isdir(img_dir2) else None)
        if not csv_path or not images_dir:
            print(f"[SKIP] Missing CSV or images for split '{s}' -> skipping")
            continue

        print(f"\n[PROCESS] Converting split: {s}")
        rows, _ = read_csv_rows(csv_path)
        by_file = defaultdict(list)
        for r in rows:
            p = parse_row_fields(r)
            if p:
                by_file[p['filename']].append({'class_id': p['class_id'], 'class_name': p['class_name'], 'bbox': p['bbox']})

        voc_out = os.path.join(output_dir, 'voc', s); ensure_dir(voc_out)
        yolo_images_out = os.path.join(output_dir, 'yolo', s, 'images'); ensure_dir(yolo_images_out)
        yolo_labels_out = os.path.join(output_dir, 'yolo', s, 'labels'); ensure_dir(yolo_labels_out)
        coco_out_dir = os.path.join(output_dir, 'coco'); ensure_dir(coco_out_dir)

        coco_images = []
        coco_annotations = []
        ann_id = 1
        img_id = 1

        for fname, ann_list in tqdm(by_file.items(), desc=f"Processing {s}", unit="image"):
            src_img = os.path.join(images_dir, fname)
            if not os.path.exists(src_img):
                print(f"  [WARN] image not found: {src_img} -> skipping")
                continue
            try:
                with Image.open(src_img) as im:
                    w, h = im.size
            except Exception as ex:
                print(f"  [ERROR] cannot open {src_img}: {ex} -> skipping")
                continue

            # VOC
            voc_annotations = [{'class_name': a['class_name'] or str(a['class_id']), 'bbox': a['bbox']} for a in ann_list]
            xml_path = os.path.join(voc_out, os.path.splitext(fname)[0] + '.xml')
            write_voc_xml({'filename': fname, 'width': w, 'height': h, 'depth': 3, 'folder': os.path.basename(voc_out), 'path': os.path.abspath(src_img)}, voc_annotations, xml_path)

            # YOLO: copy image + write label file
            dst_img = os.path.join(yolo_images_out, fname)
            if os.path.abspath(src_img) != os.path.abspath(dst_img):
                shutil.copyfile(src_img, dst_img)
            yolo_lines = []
            for a in ann_list:
                line = convert_to_yolo_line(a, w, h, classid_map)
                if line:
                    yolo_lines.append(line)
            with open(os.path.join(yolo_labels_out, os.path.splitext(fname)[0] + '.txt'), 'w', encoding='utf-8') as lf:
                lf.write("\n".join(yolo_lines))

            # COCO entries
            coco_images.append({"id": img_id, "file_name": fname, "width": w, "height": h})
            for a in ann_list:
                coco_ann = make_coco_annotation(ann_id, img_id, a, classid_map)
                if coco_ann:
                    coco_annotations.append(coco_ann)
                    ann_id += 1
            img_id += 1

        # write COCO json
        coco_path = os.path.join(coco_out_dir, f"{s}.json")
        coco_dict = {"images": coco_images, "annotations": coco_annotations, "categories": [{"id": classid_map[cid], "name": classid_name[cid]} for cid in dataset_class_ids]}
        with open(coco_path, 'w', encoding='utf-8') as cf:
            json.dump(coco_dict, cf, indent=2)
        print(f"[OK] Split '{s}' converted. VOC:{voc_out}, YOLO images:{yolo_images_out}, labels:{yolo_labels_out}, COCO:{coco_path}")

    print("\n[COMPLETE] All done.")

# ------------------ CLI ------------------ #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", type=str, default="dataset", help="Root dataset directory.")
    p.add_argument("--output_dir", type=str, default="converted_dataset", help="Directory for converted outputs.")
    p.add_argument("--master_csv", type=str, default=None, help="Optional explicit master CSV path (overrides defaults).")
    p.add_argument("--splits", type=str, default="train,val,test", help="Comma-separated split names to process.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    convert_annotations(args.dataset_dir, args.output_dir, master_csv_arg=args.master_csv, splits=splits)

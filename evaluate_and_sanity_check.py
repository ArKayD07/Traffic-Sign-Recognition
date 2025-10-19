#!/usr/bin/env python3
"""
evaluate_and_sanity_check.py

Evaluate trained model(s) and produce sanity-check artifacts:
 - predictions.csv (filename, true_class_id, true_class_name, pred_id, pred_name, prob, top3_ids, top3_probs)
 - classification_report.csv (per-class metrics)
 - confusion_matrix.png
 - misclassified_samples/ (cropped images) + misclassified.csv
 - eval_summary.json (accuracy, top1/top3, ECE, sample counts)

Usage:
python evaluate_and_sanity_check.py --dataset_dir dataset --model_path results/best_model.pth --output_dir results/eval --batch_size 64

If you want to evaluate an external real dataset (e.g., GTSRB) provide --other_dir <path> where a similar layout (images/ + annotations.csv) exists.
"""

import os
import json
import argparse
import itertools
import shutil
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------- helpers ----------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def safe_int(v, default=0):
    try:
        return int(float(v))
    except Exception:
        return default

def checkpoint_load_info(path):
    """Load checkpoint and return dict. If path is raw state_dict returns {'model_state_dict': state_dict}."""
    ck = torch.load(path, map_location='cpu')
    if isinstance(ck, dict):
        # if it contains nested state, return as-is
        return ck
    else:
        # raw state_dict
        return {'model_state_dict': ck}

# ---------------- dataset ----------------
class BBoxDatasetEval(Dataset):
    """
    Loads bounding-box CSV and yields cropped images and labels.
    expects CSV header containing: filename,class_id,class_name,xmin,ymin,xmax,ymax (var names tolerated)
    """
    def __init__(self, images_dir, annotations_csv, classid_to_idx, transform=None, bbox_pad=0):
        self.images_dir = images_dir
        self.transform = transform
        self.bbox_pad = int(bbox_pad)
        self.classid_to_idx = classid_to_idx
        self.rows = []

        if not os.path.exists(annotations_csv):
            raise RuntimeError(f"Annotations CSV not found: {annotations_csv}")
        df = pd.read_csv(annotations_csv)

        # case-insensitive mapping
        cols = {c.lower(): c for c in df.columns}
        def col(*names):
            for n in names:
                if n in cols:
                    return cols[n]
            return None

        file_col = col('filename','file','image','image_name') or 'filename'
        cid_col  = col('class_id','class') or 'class_id'
        cname_col= col('class_name','label') or 'class_name'
        xmin_col = col('xmin','x1') or 'xmin'
        ymin_col = col('ymin','y1') or 'ymin'
        xmax_col = col('xmax','x2') or 'xmax'
        ymax_col = col('ymax','y2') or 'ymax'

        for _, r in df.iterrows():
            fname = r.get(file_col, None) if file_col in df.columns else r.get('filename', None)
            if pd.isna(fname) or fname is None:
                continue
            fname = os.path.basename(str(fname))
            img_path = os.path.join(images_dir, fname)
            if not os.path.exists(img_path):
                # try nested
                found = None
                for root, _, files in os.walk(images_dir):
                    if fname in files:
                        found = os.path.join(root, fname); break
                if found:
                    img_path = found
            if not os.path.exists(img_path):
                # skip
                continue
            xmin = safe_int(r.get(xmin_col, r.get('xmin', 0)))
            ymin = safe_int(r.get(ymin_col, r.get('ymin', 0)))
            xmax = safe_int(r.get(xmax_col, r.get('xmax', 0)))
            ymax = safe_int(r.get(ymax_col, r.get('ymax', 0)))
            cid = safe_int(r.get(cid_col, -1))
            cname = r.get(cname_col, '') if cname_col in df.columns else r.get('class_name', '')
            self.rows.append({'img_path': img_path, 'fname': fname, 'bbox': [xmin,ymin,xmax,ymax], 'class_id': cid, 'class_name': str(cname)})

        if len(self.rows) == 0:
            raise RuntimeError(f"No valid annotation rows found in {annotations_csv}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        img = Image.open(r['img_path']).convert('RGB')
        xmin,ymin,xmax,ymax = r['bbox']
        if self.bbox_pad > 0:
            xmin = max(0, xmin - self.bbox_pad)
            ymin = max(0, ymin - self.bbox_pad)
            xmax = min(img.width - 1, xmax + self.bbox_pad)
            ymax = min(img.height - 1, ymax + self.bbox_pad)
        xmin = max(0, xmin); ymin = max(0, ymin)
        xmax = min(img.width - 1, xmax); ymax = min(img.height - 1, ymax)
        if xmax <= xmin or ymax <= ymin:
            crop = img
        else:
            crop = img.crop((xmin,ymin,xmax,ymax))
        if self.transform:
            tensor = self.transform(crop)
        else:
            tensor = transforms.ToTensor()(crop)
        # label: map class_id to index if available
        label_idx = self.classid_to_idx.get(r['class_id'], None) if hasattr(self, 'classid_to_idx') else None
        return tensor, r

# ---------------- metrics & utils ----------------
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def topk_from_logits(logits, k=3):
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy() if isinstance(logits, np.ndarray) else torch.softmax(logits, dim=1).cpu().numpy()
    topk_ids = np.argsort(-probs, axis=1)[:, :k]
    topk_probs = -np.sort(-probs, axis=1)[:, :k]
    return topk_ids, topk_probs

def expected_calibration_error(confs, correct, n_bins=10):
    """
    Simple ECE: partition confidences into n_bins, compute weighted avg |acc - conf|
    confs: list/np array of confidence (float in [0,1])
    correct: boolean array 1 if correct else 0
    """
    confs = np.array(confs)
    correct = np.array(correct).astype(int)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confs > bins[i]) & (confs <= bins[i+1]) if i>0 else (confs >= bins[i]) & (confs <= bins[i+1])
        if mask.sum() == 0:
            continue
        avg_conf = confs[mask].mean()
        avg_acc = correct[mask].mean()
        ece += (mask.sum() / confs.size) * abs(avg_acc - avg_conf)
    return float(ece)

def plot_confusion(cm, labels, outpath):
    if cm.size == 0:
        return
    cm_sum = cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.divide(cm, cm_sum, where=cm_sum!=0)
    plt.figure(figsize=(10,8))
    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix (normalized by row)")
    plt.colorbar(fraction=0.045, pad=0.02)
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90, fontsize=8)
    plt.yticks(tick_marks, labels, fontsize=8)
    fmt = '.2f'
    thresh = cm_norm.max() / 2.0 if cm_norm.max() != 0 else 0.5
    for i, j in itertools.product(range(cm_norm.shape[0]), range(cm_norm.shape[1])):
        plt.text(j, i, format(cm_norm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm_norm[i, j] > thresh else "black",
                 fontsize=6)
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches='tight', dpi=200)
    plt.close()

# ---------------- main evaluation pipeline ----------------
def evaluate_split(model, device, dataset_dir, split, classid_to_idx, idx_to_name, image_size, batch_size, bbox_pad, num_workers, output_dir):
    images_dir = os.path.join(dataset_dir, split, 'images')
    csv_path = os.path.join(dataset_dir, split, 'annotations.csv')
    if not os.path.exists(images_dir) or not os.path.exists(csv_path):
        raise RuntimeError(f"Missing {split} data (images or annotations): {images_dir}, {csv_path}")

    # transforms: use validation-style preprocessing
    val_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    ds = BBoxDatasetEval(images_dir, csv_path, classid_to_idx, transform=val_tf, bbox_pad=bbox_pad)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type=='cuda'))

    preds_all = []
    gts_all = []
    fnames = []
    true_names = []
    true_ids = []
    top3_all = []
    top3_probs_all = []
    confidences = []
    all_logits = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Eval {split}"):
            # Each batch should be (tensor_batch, rows), but rows can be:
            # - a list of dicts [{...}, {...}, ...] (usual)
            # - a dict of lists {'fname': [...], 'class_id': [...], ...} (collated)
            tensors, rows = batch
            tensors = tensors.to(device)
            logits = model(tensors)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            top3 = np.argsort(-probs, axis=1)[:, :3]
            top3p = -np.sort(-probs, axis=1)[:, :3]

            # Normalize rows -> rows_list: list of dicts
            if isinstance(rows, dict):
                # rows is dict of lists: convert to list of dicts
                # derive batch size from first value
                first_val = next(iter(rows.values()))
                batch_len = len(first_val)
                rows_list = []
                for i in range(batch_len):
                    entry = {}
                    for k, v in rows.items():
                        try:
                            entry[k] = v[i]
                        except Exception:
                            # if value is scalar, reuse it
                            entry[k] = v
                    rows_list.append(entry)
            else:
                # assume it's already a list of dicts
                rows_list = rows

            for i, r in enumerate(rows_list):
                # guard for missing expected keys
                fname = r.get('fname') or r.get('filename') or r.get('file') or os.path.basename(r.get('img_path', ''))
                preds_all.append(int(preds[i]))
                # map true class id to class index if possible, else -1
                true_cid = r.get('class_id') if 'class_id' in r else (r.get('label') if 'label' in r else None)
                try:
                    true_cid_int = int(true_cid) if true_cid is not None else -1
                except Exception:
                    true_cid_int = -1
                true_ids.append(true_cid_int)
                # class index mapping
                gt_idx = classid_to_idx.get(true_cid_int, -1)
                gts_all.append(int(gt_idx))
                fnames.append(fname)
                true_names.append(r.get('class_name') or r.get('label') or str(true_cid_int))
                top3_all.append([int(x) for x in top3[i].tolist()])
                top3_probs_all.append([float(x) for x in top3p[i].tolist()])
                confidences.append(float(probs[i, preds[i]]))
                all_logits.append(probs[i].tolist())

    # map preds idx->class id (original class ids)
    idx_to_cid = {v:k for k,v in classid_to_idx.items()}
    pred_cids = [ idx_to_cid.get(p, None) for p in preds_all ]
    pred_names = [ idx_to_name[p] if p < len(idx_to_name) else str(pred_cids[i]) for i,p in enumerate(preds_all) ]

    # compute metrics
    gt_idxs = np.array(gts_all)
    preds_array = np.array(preds_all)

    valid_mask = gt_idxs >= 0
    accuracy = float((preds_array[valid_mask] == gt_idxs[valid_mask]).mean()) if valid_mask.sum()>0 else 0.0

    top3_ok = 0
    for i in range(len(preds_all)):
        if gt_idxs[i] in top3_all[i]:
            top3_ok += 1
    top3_acc = float(top3_ok / len(preds_all)) if len(preds_all)>0 else 0.0

    cr = classification_report(gt_idxs[valid_mask], preds_array[valid_mask], output_dict=True, zero_division=0)
    cm = confusion_matrix(gt_idxs[valid_mask], preds_array[valid_mask], labels=list(range(len(idx_to_name)))) if valid_mask.sum()>0 else np.zeros((len(idx_to_name), len(idx_to_name)))

    correct_flags = (preds_array == gt_idxs)
    ece = expected_calibration_error(confidences, correct_flags, n_bins=10)

    # Save predictions
    pred_rows = []
    for i in range(len(fnames)):
        pred_rows.append({
            'filename': fnames[i],
            'true_class_id': int(true_ids[i]) if true_ids[i] is not None else -1,
            'true_class_name': true_names[i],
            'pred_class_idx': int(preds_array[i]),
            'pred_class_id': int(idx_to_cid.get(preds_array[i], -1)) if preds_array[i] in idx_to_cid else -1,
            'pred_class_name': pred_names[i],
            'confidence': float(confidences[i]),
            'top3_pred_idxs': "|".join(str(x) for x in top3_all[i]),
            'top3_pred_probs': "|".join(f"{x:.4f}" for x in top3_probs_all[i])
        })
    preds_df = pd.DataFrame(pred_rows)
    preds_csv = os.path.join(output_dir, f"{split}_predictions.csv")
    preds_df.to_csv(preds_csv, index=False)

    # classification_report CSV
    cr_rows = []
    for k, v in cr.items():
        if k == 'accuracy':
            continue
        row = {'label': k}
        row.update(v)
        cr_rows.append(row)
    cr_df = pd.DataFrame(cr_rows)
    cr_csv = os.path.join(output_dir, f"{split}_classification_report.csv")
    cr_df.to_csv(cr_csv, index=False)

    # confusion matrix image
    cm_png = os.path.join(output_dir, f"{split}_confusion_matrix.png")
    plot_confusion(cm, idx_to_name, cm_png)

    # misclassified samples: save cropped images
    mis_dir = os.path.join(output_dir, f"{split}_misclassified_samples")
    ensure_dir(mis_dir)
    mis_rows = []

    ann_df = pd.read_csv(os.path.join(dataset_dir, split, 'annotations.csv'))
    file_map = defaultdict(list)
    for _, a in ann_df.iterrows():
        fname = os.path.basename(str(a.get('filename', a.get('file', ''))))
        file_map[fname].append(a.to_dict())

    for i, pr in preds_df.iterrows():
        fname = pr['filename']
        gt = pr['true_class_id']
        pred_idx = pr['pred_class_idx']
        pred_cid = pr['pred_class_id']
        correct = (pred_idx == classid_to_idx.get(gt, -999))
        if not correct:
            ann_list = file_map.get(fname, [])
            ann = None
            for a in ann_list:
                if int(safe_int(a.get('class_id', -1))) == int(gt):
                    ann = a; break
            if ann is None and len(ann_list)>0:
                ann = ann_list[0]
            if ann is None:
                continue
            img_path = os.path.join(dataset_dir, split, 'images', fname)
            if not os.path.exists(img_path):
                found = None
                for root, _, files in os.walk(os.path.join(dataset_dir, split, 'images')):
                    if fname in files:
                        found = os.path.join(root, fname); break
                if found:
                    img_path = found
            try:
                im = Image.open(img_path).convert('RGB')
                xmin = safe_int(ann.get('xmin', ann.get('x1', 0)))
                ymin = safe_int(ann.get('ymin', ann.get('y1', 0)))
                xmax = safe_int(ann.get('xmax', ann.get('x2', im.width-1)))
                ymax = safe_int(ann.get('ymax', ann.get('y2', im.height-1)))
                crop = im.crop((xmin, ymin, xmax, ymax))
                outname = f"mis_{i}_{fname}"
                outpath = os.path.join(mis_dir, outname)
                crop.save(outpath)
                mis_rows.append({'filename': fname, 'saved_crop': outpath, 'true_class_id': gt, 'true_class_name': pr['true_class_name'], 'pred_class_id': pred_cid, 'pred_class_name': pr['pred_class_name'], 'confidence': pr['confidence']})
            except Exception:
                continue

    mis_df = pd.DataFrame(mis_rows)
    mis_csv = os.path.join(output_dir, f"{split}_misclassified.csv")
    mis_df.to_csv(mis_csv, index=False)

    summary = {
        'split': split,
        'num_samples': int(len(preds_df)),
        'accuracy': float(accuracy),
        'top3_accuracy': float(top3_acc),
        'ece': float(ece),
        'predictions_csv': preds_csv,
        'classification_report_csv': cr_csv,
        'confusion_matrix_png': cm_png,
        'misclassified_csv': mis_csv
    }

    return summary

# ---------------- main ----------------
def collect_class_map_from_splits(dataset_dir):
    """Read train/val/test annotation CSVs and return OrderedDict class_id -> class_name (first occurrence order)"""
    mapping = OrderedDict()
    for split in ('train','val','test'):
        csvp = os.path.join(dataset_dir, split, 'annotations.csv')
        if not os.path.exists(csvp):
            continue
        df = pd.read_csv(csvp)
        cols = {c.lower(): c for c in df.columns}
        id_col = cols.get('class_id') or cols.get('class') or None
        name_col = cols.get('class_name') or cols.get('label') or None
        if id_col is None:
            continue
        for _, r in df.iterrows():
            try:
                cid = int(float(r[id_col]))
            except Exception:
                continue
            cname = str(r[name_col]) if (name_col in df.columns and not pd.isna(r[name_col])) else str(cid)
            if cid not in mapping:
                mapping[cid] = cname
    return mapping

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint (.pth).")
    parser.add_argument("--output_dir", type=str, default="results/eval")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--bbox_pad", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--other_dir", type=str, default=None, help="Optional: evaluate a second dataset dir (same layout) - e.g., GTSRB.")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    device = torch.device(args.device if torch.cuda.is_available() or args.device=='cpu' else 'cpu')
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Loading checkpoint: {args.model_path}")
    ck = checkpoint_load_info(args.model_path)

    # collect class map (prefer checkpoint, then dataset splits)
    classid_to_idx = None
    idx_to_name = None
    if isinstance(ck, dict) and 'classid_to_idx' in ck:
        classid_to_idx = ck['classid_to_idx']
    if isinstance(ck, dict) and 'class_map' in ck:
        # class_map is mapping classid->name
        mapping = OrderedDict(ck['class_map'])
        classid_to_idx = {cid: idx for idx, cid in enumerate(mapping.keys())}
        idx_to_name = [mapping[cid] for cid in mapping.keys()]

    if classid_to_idx is None:
        # build from dataset
        mapping = collect_class_map_from_splits(args.dataset_dir)
        if len(mapping) == 0:
            raise RuntimeError("Could not determine class map from checkpoint or dataset splits.")
        classid_to_idx = {cid: idx for idx, cid in enumerate(mapping.keys())}
        idx_to_name = [mapping[cid] for cid in classid_to_idx.keys()]

    if idx_to_name is None:
        # create list from classid_to_idx
        sorted_pairs = sorted(classid_to_idx.items(), key=lambda kv: kv[1])
        idx_to_name = [ str(kv[0]) for kv in sorted_pairs ]
        # if we can find names in dataset, prefer them
        mapping = collect_class_map_from_splits(args.dataset_dir)
        for i, (cid, idx) in enumerate(sorted_pairs):
            if cid in mapping:
                idx_to_name[idx] = mapping[cid]

    num_classes = len(classid_to_idx)
    print(f"[INFO] Number of classes: {num_classes}")

    # build model
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    # load weights
    if 'model_state_dict' in ck:
        msd = ck['model_state_dict']
        # some checkpoints saved full ckpt, some only state_dict
        try:
            model.load_state_dict(msd)
            print("[INFO] Loaded model_state_dict from checkpoint.")
        except Exception as e:
            # sometimes checkpoint has nested structures; try common keys
            if 'state_dict' in msd:
                model.load_state_dict(msd['state_dict'])
            else:
                raise
    else:
        # ck itself might be state_dict
        try:
            model.load_state_dict(ck)
        except Exception as e:
            raise RuntimeError(f"Could not load model weights from checkpoint: {e}")

    model = model.to(device)

    # Evaluate on synthetic test split
    print("[INFO] Evaluating on synthetic test split (dataset/test)...")
    summary_test = evaluate_split(model, device, args.dataset_dir, 'test', classid_to_idx, idx_to_name, args.image_size, args.batch_size, args.bbox_pad, args.num_workers, args.output_dir)
    print("[INFO] Synthetic test summary:", summary_test)

    outputs = {'synthetic_test': summary_test}

    # Optionally evaluate other_dir (e.g., GTSRB prepared folder)
    if args.other_dir:
        print(f"[INFO] Evaluating on other dataset at {args.other_dir} (assumes same layout)...")
        summary_other = evaluate_split(model, device, args.other_dir, 'test', classid_to_idx, idx_to_name, args.image_size, args.batch_size, args.bbox_pad, args.num_workers, args.output_dir)
        outputs['other_test'] = summary_other
        print("[INFO] Other test summary:", summary_other)

    # save summary
    with open(os.path.join(args.output_dir, 'eval_summary.json'), 'w', encoding='utf-8') as jf:
        json.dump(outputs, jf, indent=2)

    print("[DONE] Evaluation artifacts saved to:", args.output_dir)

if __name__ == "__main__":
    main()

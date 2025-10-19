#!/usr/bin/env python3
"""
evaluate_and_save_named_preds.py

Evaluate a trained classifier on the dataset/test split and save outputs with a model-specific name.

Usage example:
python evaluate_and_save_named_preds.py --dataset_dir dataset --model_path results/model_C.pth --output_dir results/eval --model_name C --batch_size 64
"""
import os
import argparse
import json
from collections import defaultdict, OrderedDict

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
def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def safe_int(v, default=0):
    try:
        return int(float(v))
    except Exception:
        return default

def load_checkpoint(path):
    ck = torch.load(path, map_location='cpu')
    if isinstance(ck, dict):
        return ck
    else:
        return {'model_state_dict': ck}

# ---------------- dataset ----------------
class BBoxEvalDataset(Dataset):
    def __init__(self, images_dir, annotations_csv, transform=None, bbox_pad=0):
        self.images_dir = images_dir
        self.transform = transform
        self.bbox_pad = int(bbox_pad)
        if not os.path.exists(annotations_csv):
            raise RuntimeError(f"Annotations CSV not found: {annotations_csv}")
        self.df = pd.read_csv(annotations_csv)
        cols = {c.lower(): c for c in self.df.columns}
        def col(*names):
            for n in names:
                if n in cols:
                    return cols[n]
            return None
        self.file_col = col('filename','file','image','image_name') or 'filename'
        self.cid_col  = col('class_id','class') or 'class_id'
        self.cname_col= col('class_name','label') or 'class_name'
        self.xmin = col('xmin','x1') or 'xmin'
        self.ymin = col('ymin','y1') or 'ymin'
        self.xmax = col('xmax','x2') or 'xmax'
        self.ymax = col('ymax','y2') or 'ymax'

        # build rows
        rows = []
        for _, r in self.df.iterrows():
            fname = r.get(self.file_col, None)
            if pd.isna(fname) or fname is None:
                continue
            fname = os.path.basename(str(fname))
            img_path = os.path.join(images_dir, fname)
            if not os.path.exists(img_path):
                # search nested directories
                found = None
                for root, _, files in os.walk(images_dir):
                    if fname in files:
                        found = os.path.join(root, fname)
                        break
                if found:
                    img_path = found
            if not os.path.exists(img_path):
                # skip if image missing
                continue
            xmin = safe_int(r.get(self.xmin, 0))
            ymin = safe_int(r.get(self.ymin, 0))
            xmax = safe_int(r.get(self.xmax, 0))
            ymax = safe_int(r.get(self.ymax, 0))
            cid = r.get(self.cid_col, -1)
            cname = r.get(self.cname_col, '') if self.cname_col in self.df.columns else ''
            rows.append({'img_path': img_path, 'fname': fname, 'bbox': [xmin,ymin,xmax,ymax], 'class_id': int(safe_int(cid, -1)), 'class_name': str(cname)})
        if len(rows) == 0:
            raise RuntimeError(f"No valid rows found in annotations: {annotations_csv}")
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        im = Image.open(r['img_path']).convert('RGB')
        xmin,ymin,xmax,ymax = r['bbox']
        if self.bbox_pad > 0:
            xmin = max(0, xmin - self.bbox_pad)
            ymin = max(0, ymin - self.bbox_pad)
            xmax = min(im.width-1, xmax + self.bbox_pad)
            ymax = min(im.height-1, ymax + self.bbox_pad)
        xmin = max(0, xmin); ymin = max(0, ymin)
        xmax = min(im.width-1, xmax); ymax = min(im.height-1, ymax)
        if xmax <= xmin or ymax <= ymin:
            crop = im
        else:
            crop = im.crop((xmin,ymin,xmax,ymax))
        if self.transform:
            tensor = self.transform(crop)
        else:
            tensor = transforms.ToTensor()(crop)
        return tensor, r

# ---------------- transforms ----------------
def make_val_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

# ---------------- evaluation ----------------
def evaluate_model(model, device, loader, classid_to_idx, idx_to_name, output_dir, model_name):
    preds = []
    gts = []
    filenames = []
    true_names = []
    confidences = []
    top3 = []

    model.eval()
    with torch.no_grad():
        for tensors, rows in tqdm(loader, desc=f"Eval {model_name}"):
            tensors = tensors.to(device)
            logits = model(tensors)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            pred_idxs = np.argmax(probs, axis=1)
            topk_ids = np.argsort(-probs, axis=1)[:, :3]
            for i, r in enumerate(rows):
                filenames.append(r['fname'])
                gt_cid = int(r['class_id']) if ('class_id' in r and r['class_id'] is not None) else -1
                gts.append(gt_cid)
                pred_idx = int(pred_idxs[i])
                # map pred idx back to original class id via idx->cid
                # idx_to_name maps index positions to names; we need idx->cid map created outside
                preds.append(pred_idx)
                true_names.append(r.get('class_name',''))
                confidences.append(float(probs[i, pred_idx]))
                top3.append([int(x) for x in topk_ids[i].tolist()])

    # merged dataframe - note: pred values are model-internal class idx (0..K-1)
    # but we saved classid_to_idx mapping earlier; we'll map back to original class_id if available
    df = pd.DataFrame({
        'filename': filenames,
        'true_class_id': gts,
        'true_class_name': true_names,
        'pred_internal_idx': preds,
        'confidence': confidences,
        'top3_internal_idxs': ["|".join(str(x) for x in t) for t in top3]
    })
    return df

# ---------------- plotting helpers ----------------
def plot_confusion(cm, labels, outpath):
    plt.figure(figsize=(10,8))
    sns = __import__('seaborn')
    sns.heatmap(cm, cmap='Blues', xticklabels=labels, yticklabels=labels, square=True)
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout(); plt.savefig(outpath, dpi=200); plt.close()

# ---------------- main ----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", type=str, default="dataset", help="Dataset root containing test/ subdir.")
    p.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint (.pth).")
    p.add_argument("--output_dir", type=str, default="results", help="Where to save outputs (csvs/images).")
    p.add_argument("--model_name", type=str, required=True, help="Short name for model (e.g. A, B, C).")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--bbox_pad", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ensure_dir(args.output_dir)
    model_name = args.model_name
    model_out_dir = os.path.join(args.output_dir, f"{model_name}_eval")
    ensure_dir(model_out_dir)

    device = torch.device(args.device if torch.cuda.is_available() or args.device=='cpu' else "cpu")
    print("[INFO] Device:", device)
    print("[INFO] Loading checkpoint:", args.model_path)
    ck = load_checkpoint(args.model_path)

    # try to load class map from checkpoint (classid_to_idx or class_map)
    classid_to_idx = None
    class_map = None
    if isinstance(ck, dict):
        if 'classid_to_idx' in ck:
            classid_to_idx = ck['classid_to_idx']
        if 'class_map' in ck:
            class_map = OrderedDict(ck['class_map'])
            # build classid_to_idx from class_map order if needed
            if classid_to_idx is None:
                classid_to_idx = {cid: idx for idx, cid in enumerate(class_map.keys())}

    # if not found, try dataset/classes.csv
    if classid_to_idx is None:
        classes_csv = os.path.join("dataset","classes.csv")
        if os.path.exists(classes_csv):
            dfc = pd.read_csv(classes_csv)
            if 'class_id' in dfc.columns and 'class_name' in dfc.columns:
                class_map = OrderedDict()
                for _, r in dfc.iterrows():
                    cid = int(safe_int(r['class_id']))
                    class_map[cid] = str(r['class_name'])
                classid_to_idx = {cid: idx for idx, cid in enumerate(class_map.keys())}

    if classid_to_idx is None:
        raise RuntimeError("Could not determine class mapping. Provide classes.csv or checkpoint containing class_map/classid_to_idx.")

    # create idx->cid and idx->name helpers
    idx_to_cid = {v:k for k,v in classid_to_idx.items()}
    idx_to_name = [ class_map.get(idx_to_cid[i], str(idx_to_cid[i])) if class_map else str(idx_to_cid[i]) for i in range(len(idx_to_cid)) ]

    # build model
    num_classes = len(classid_to_idx)
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # load weights
    if 'model_state_dict' in ck:
        msd = ck['model_state_dict']
        try:
            model.load_state_dict(msd)
            print("[INFO] Loaded model_state_dict from checkpoint.")
        except Exception:
            # try nested state dict key names
            if 'state_dict' in msd:
                model.load_state_dict(msd['state_dict'])
            else:
                model.load_state_dict(msd)
    else:
        # if checkpoint is raw state dict
        try:
            model.load_state_dict(ck)
        except Exception as e:
            raise RuntimeError(f"Could not load model weights: {e}")

    # prepare evaluation dataset (test split)
    test_images_dir = os.path.join(args.dataset_dir, 'test', 'images')
    test_csv = os.path.join(args.dataset_dir, 'test', 'annotations.csv')
    if not os.path.exists(test_images_dir) or not os.path.exists(test_csv):
        # try alternative: all_images + all_annotations.csv
        alt_img = os.path.join(args.dataset_dir, 'all_images')
        alt_csv = os.path.join(args.dataset_dir, 'all_annotations.csv')
        if os.path.exists(alt_img) and os.path.exists(alt_csv):
            test_images_dir = alt_img
            test_csv = alt_csv
        else:
            raise RuntimeError("Test split not found under dataset/test or dataset/all_images.")

    val_tf = make_val_transform(args.image_size)
    ds = BBoxEvalDataset(test_images_dir, test_csv, transform=val_tf, bbox_pad=args.bbox_pad)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type=='cuda'))

    # run evaluation
    print(f"[INFO] Evaluating model {model_name} on {len(ds)} samples...")
    preds_df = evaluate_model(model, device, loader, classid_to_idx, idx_to_name, model_out_dir, model_name)

    # map internal preds (0..K-1) back to original class_id (cid)
    preds_df['pred_class_id'] = preds_df['pred_internal_idx'].apply(lambda x: int(idx_to_cid.get(x, -1)))
    # also map top3 internal idxs -> top3 cids
    def map_top3(s):
        try:
            parts = [int(x) for x in str(s).split('|') if x!='']
            return "|".join(str(int(idx_to_cid.get(p, -1))) for p in parts)
        except Exception:
            return ""
    preds_df['top3_pred_cids'] = preds_df['top3_internal_idxs'].apply(map_top3)

    # compute metrics and save outputs
    out_preds_csv = os.path.join(model_out_dir, f"{model_name}_test_predictions.csv")
    preds_df[['filename','true_class_id','true_class_name','pred_class_id','confidence','top3_pred_cids']].to_csv(out_preds_csv, index=False)
    print("[INFO] Saved predictions ->", out_preds_csv)

    # classification report & confusion matrix
    y_true = preds_df['true_class_id'].astype(int).values
    y_pred = preds_df['pred_class_id'].astype(int).values
    # only include labels present in mapping order
    labels_sorted = sorted(list(classid_to_idx.keys()))
    rep = classification_report(y_true, y_pred, labels=labels_sorted, output_dict=True, zero_division=0)
    cr_rows = []
    for k,v in rep.items():
        cr_rows.append({'label': k, **(v if isinstance(v, dict) else {'value': v})})
    # save classification_report.csv in human-readable form
    # prefer sklearn text -> convert to DataFrame using keys present
    rep_df = []
    for k, v in rep.items():
        if k in ('accuracy','macro avg','weighted avg'):
            rep_df.append({'class': k, 'precision': v if not isinstance(v, dict) else (v.get('precision','')), 'recall': v if not isinstance(v, dict) else (v.get('recall','')), 'f1-score': v if not isinstance(v, dict) else (v.get('f1-score','')), 'support': v if not isinstance(v, dict) else (v.get('support',''))})
        else:
            rep_df.append({'class': str(k), 'precision': v.get('precision',0.0), 'recall': v.get('recall',0.0), 'f1-score': v.get('f1-score',0.0), 'support': int(v.get('support',0))})
    rep_df = pd.DataFrame(rep_df)
    rep_csv = os.path.join(model_out_dir, f"{model_name}_classification_report.csv")
    rep_df.to_csv(rep_csv, index=False)
    print("[INFO] Saved classification report ->", rep_csv)

    # confusion matrix and image
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    cm_df = pd.DataFrame(cm, index=[str(x) for x in labels_sorted], columns=[str(x) for x in labels_sorted])
    cm_csv = os.path.join(model_out_dir, f"{model_name}_confusion_matrix.csv")
    cm_df.to_csv(cm_csv)
    # plot with readable names if class_map available
    labels_for_plot = [class_map.get(l, str(l)) if class_map else str(l) for l in labels_sorted]
    try:
        import seaborn as sns
        plt.figure(figsize=(10,8))
        sns.heatmap(cm, cmap='Blues', xticklabels=labels_for_plot, yticklabels=labels_for_plot, square=True)
        plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout()
        cm_png = os.path.join(model_out_dir, f"{model_name}_confusion_matrix.png")
        plt.savefig(cm_png, dpi=200); plt.close()
        print("[INFO] Saved confusion matrix plot ->", cm_png)
    except Exception:
        print("[WARN] seaborn not available - skipped confusion plot image (csv saved).")

    print("[DONE] Evaluation for model", model_name, "written to", model_out_dir)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
train_and_eval.py

Train a ResNet-18 classifier on bbox-cropped traffic-sign samples (from annotations CSVs).
Produces model, metrics, confusion matrix, and logs in output_dir.

Expected dataset layout:
dataset/
  train/
    images/
    annotations.csv    (filename,class_id,class_name,xmin,ymin,xmax,ymax)
  val/
    images/
    annotations.csv
  test/
    images/
    annotations.csv

Requirements:
pip install torch torchvision pandas scikit-learn pillow matplotlib tqdm

Run example:
python train_and_eval.py --dataset_dir dataset --output_dir results --epochs 12 --batch_size 64 --image_size 224 --pretrained
"""

import os
import json
import argparse
import shutil
import itertools
from collections import OrderedDict

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

# ---------------- utilities ----------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def safe_int(v, default=0):
    try:
        return int(float(v))
    except Exception:
        return default

# ---------------- dataset ----------------
class BBoxClassificationDataset(Dataset):
    def __init__(self, images_dir, annotations_csv, classid_to_idx, transform=None, bbox_pad=0):
        self.images_dir = images_dir
        self.transform = transform
        self.bbox_pad = int(bbox_pad)
        self.classid_to_idx = classid_to_idx
        self.samples = []

        if not os.path.exists(annotations_csv):
            raise RuntimeError(f"Annotations CSV not found: {annotations_csv}")
        df = pd.read_csv(annotations_csv)
        if df.shape[0] == 0:
            raise RuntimeError(f"Annotations CSV empty: {annotations_csv}")

        # case-insensitive column resolution
        cols = {c.lower(): c for c in df.columns}
        def col(*names):
            for n in names:
                if n in cols:
                    return cols[n]
            return None

        file_col = col('filename', 'file', 'image', 'image_name') or 'filename'
        cid_col = col('class_id', 'class') or 'class_id'
        cname_col = col('class_name', 'label') or 'class_name'
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
                # recursive search
                found = None
                for root, _, files in os.walk(images_dir):
                    if fname in files:
                        found = os.path.join(root, fname)
                        break
                if found:
                    img_path = found
            if not os.path.exists(img_path):
                print(f"[WARN] Image not found for annotation: {fname} (expected under {images_dir})")
                continue
            xmin = safe_int(r.get(xmin_col, r.get('xmin', 0)))
            ymin = safe_int(r.get(ymin_col, r.get('ymin', 0)))
            xmax = safe_int(r.get(xmax_col, r.get('xmax', 0)))
            ymax = safe_int(r.get(ymax_col, r.get('ymax', 0)))
            cid = safe_int(r.get(cid_col, -1))
            cname = r.get(cname_col, '') if cname_col in df.columns else r.get('class_name', '')
            class_idx = classid_to_idx.get(cid, 0)
            self.samples.append({'img_path': img_path, 'bbox': [xmin, ymin, xmax, ymax], 'class_id': cid, 'class_name': str(cname), 'class_idx': class_idx})

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples found in {annotations_csv}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s['img_path']).convert('RGB')
        xmin, ymin, xmax, ymax = s['bbox']

        # pad bbox
        if self.bbox_pad > 0:
            xmin = max(0, xmin - self.bbox_pad)
            ymin = max(0, ymin - self.bbox_pad)
            xmax = min(img.width - 1, xmax + self.bbox_pad)
            ymax = min(img.height - 1, ymax + self.bbox_pad)

        xmin = max(0, xmin); ymin = max(0, ymin)
        xmax = min(img.width - 1, xmax); ymax = min(img.height - 1, ymax)

        if xmax <= xmin or ymax <= ymin:
            crop = img  # fallback to whole image
        else:
            crop = img.crop((xmin, ymin, xmax, ymax))

        if self.transform:
            crop = self.transform(crop)
        label = int(s['class_idx'])
        return crop, label

# ---------------- transforms and loaders ----------------
def make_transforms(image_size):
    train_tf = transforms.Compose([
        transforms.Resize((int(image_size * 1.2), int(image_size * 1.2))),
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return train_tf, val_tf

def collect_class_map(csv_paths):
    mapping = OrderedDict()
    for p in csv_paths:
        if not p or not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        cols = {c.lower(): c for c in df.columns}
        id_col = cols.get('class_id') or cols.get('class') or None
        name_col = cols.get('class_name') or cols.get('label') or None
        if id_col is None:
            for c in df.columns:
                if c.lower().startswith('class') and 'name' not in c.lower():
                    id_col = c; break
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

def build_dataloaders(dataset_dir, classid_to_idx, image_size, batch_size, bbox_pad, num_workers, device):
    train_tf, val_tf = make_transforms(image_size)
    loaders = {}
    datasets = {}
    for split in ('train','val','test'):
        images_dir = os.path.join(dataset_dir, split, 'images')
        csv_path = os.path.join(dataset_dir, split, 'annotations.csv')
        if not os.path.exists(images_dir) or not os.path.exists(csv_path):
            print(f"[WARN] Missing split {split}: images_dir={images_dir}, csv={csv_path}")
            loaders[split] = None
            continue
        tf = train_tf if split == 'train' else val_tf
        ds = BBoxClassificationDataset(images_dir, csv_path, classid_to_idx, transform=tf, bbox_pad=bbox_pad)
        pin_memory = (device.type == 'cuda')
        loader = DataLoader(ds, batch_size=batch_size, shuffle=(split == 'train'), num_workers=num_workers, pin_memory=pin_memory)
        loaders[split] = loader
        datasets[split] = ds
        print(f"[INFO] {split}: {len(ds)} samples, batches={len(loader)}")
    return loaders, datasets

# ---------------- evaluation helpers ----------------
def evaluate_model(model, dataloader, device):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Eval", leave=False):
            images = images.to(device); labels = labels.to(device)
            outputs = model(images)
            _, p = torch.max(outputs, 1)
            preds.extend(p.cpu().numpy().tolist())
            gts.extend(labels.cpu().numpy().tolist())
    return np.array(preds), np.array(gts)

def plot_confusion_matrix(cm, classes, outpath, normalize=True):
    if normalize:
        cm_sum = cm.sum(axis=1)[:, np.newaxis]
        cm = np.divide(cm, cm_sum, where=cm_sum!=0)
    plt.figure(figsize=(10,8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=8)
    plt.yticks(tick_marks, classes, fontsize=8)
    fmt = '.2f' if normalize else 'd'
    thresh = (cm.max() / 2.0) if cm.max() != 0 else 0.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=6)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(outpath, bbox_inches='tight', dpi=200)
    plt.close()

# ---------------- main training + eval ----------------
def train_and_eval(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"[INFO] Device: {device}")

    # collect class map from CSVs (train/val/test)
    csv_paths = [os.path.join(args.dataset_dir, s, 'annotations.csv') for s in ('train','val','test')]
    class_map = collect_class_map(csv_paths)
    if len(class_map) == 0:
        raise RuntimeError("No classes found in annotation CSVs.")
    print(f"[INFO] {len(class_map)} classes discovered.")
    classid_to_idx = {cid: idx for idx, cid in enumerate(class_map.keys())}
    idx_to_name = [class_map[cid] for cid in classid_to_idx.keys()]

    # dataloaders
    loaders, datasets = build_dataloaders(args.dataset_dir, classid_to_idx, args.image_size, args.batch_size, args.bbox_pad, args.num_workers, device)
    if loaders.get('train') is None:
        raise RuntimeError("Training data not found.")

    # model
    model = models.resnet18(pretrained=args.pretrained)
    model.fc = nn.Linear(model.fc.in_features, len(classid_to_idx))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    ensure_dir(args.output_dir)
    best_val_acc = 0.0
    best_model_path = os.path.join(args.output_dir, 'best_model.pth')
    train_log = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(loaders['train'], desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for images, labels in pbar:
            images = images.to(device); labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix({'loss': f"{running_loss/total:.4f}", 'acc': f"{correct/total:.4f}"})

        train_loss = running_loss / total
        train_acc = correct / total

        # validation
        val_loss = None; val_acc = None
        if loaders.get('val') is not None:
            model.eval()
            v_loss = 0.0; v_total = 0; v_correct = 0
            with torch.no_grad():
                for images, labels in loaders['val']:
                    images = images.to(device); labels = labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    v_loss += loss.item() * images.size(0)
                    _, preds = torch.max(outputs, 1)
                    v_correct += (preds == labels).sum().item()
                    v_total += labels.size(0)
            val_loss = (v_loss / v_total) if v_total > 0 else 0.0
            val_acc = (v_correct / v_total) if v_total > 0 else 0.0
            scheduler.step(val_acc if val_acc is not None else 0.0)

        val_loss_str = f"{val_loss:.4f}" if val_loss is not None else "None"
        val_acc_str  = f"{val_acc:.4f}" if val_acc is not None else "None"
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss_str}, val_acc={val_acc_str}")

        train_log.append({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc})

        # save best
        if val_acc is not None and val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({'model_state_dict': model.state_dict(), 'classid_to_idx': classid_to_idx, 'class_map': class_map}, best_model_path)
            print(f"[INFO] New best model (val_acc={best_val_acc:.4f}) saved to {best_model_path}")

    # final save if none saved
    if not os.path.exists(best_model_path):
        torch.save({'model_state_dict': model.state_dict(), 'classid_to_idx': classid_to_idx, 'class_map': class_map}, best_model_path)
        print(f"[INFO] Saved final model to {best_model_path}")

    pd.DataFrame(train_log).to_csv(os.path.join(args.output_dir, 'train_log.csv'), index=False)

    # final evaluation on test split
    if loaders.get('test') is None:
        print("[WARN] No test set found; skipping final evaluation.")
        return

    ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)

    preds, gts = evaluate_model(model, loaders['test'], device)
    test_acc = float(accuracy_score(gts, preds))
    cls_report = classification_report(gts, preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(gts, preds)

    # write classification report
    rows = []
    for label_idx, metrics in cls_report.items():
        if label_idx == 'accuracy': continue
        rows.append({'label': label_idx, **metrics})
    pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, 'classification_report.csv'), index=False)

    # build ordered idx->name list for plotting
    idx_to_name = [None] * len(classid_to_idx)
    for cid, idx in classid_to_idx.items():
        idx_to_name[idx] = class_map.get(cid, str(cid))
    plot_confusion_matrix(cm, idx_to_name, os.path.join(args.output_dir, 'confusion_matrix.png'), normalize=True)

    results = {'test_accuracy': test_acc, 'num_test_samples': int(len(gts)), 'best_val_acc': float(best_val_acc)}
    with open(os.path.join(args.output_dir, 'results.json'), 'w', encoding='utf-8') as jf:
        json.dump(results, jf, indent=2)

    print(f"[RESULT] Test accuracy: {test_acc:.4f} on {len(gts)} samples")
    print(f"[INFO] Artifacts saved to: {args.output_dir}")

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Train ResNet-18 on bbox-cropped traffic sign dataset and evaluate.")
    p.add_argument("--dataset_dir", type=str, default="dataset", help="Dataset root (train/val/test subfolders).")
    p.add_argument("--output_dir", type=str, default="results", help="Where to store outputs.")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--pretrained", action='store_true', help="Use ImageNet pretrained weights.")
    p.add_argument("--bbox_pad", type=int, default=0, help="Pixels to pad around bbox when cropping.")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--no_cuda", action='store_true', help="Disable CUDA even if available.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    ensure_dir(args.output_dir)
    try:
        train_and_eval(args)
    except Exception as e:
        print("[ERROR]", str(e))
        raise

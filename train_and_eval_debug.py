import os, sys, traceback, json, argparse, shutil
from collections import OrderedDict, defaultdict
import itertools

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def safe_int(v, default=0):
    try:
        return int(float(v))
    except Exception:
        return default

def log(msg, logfile=None):
    print(msg)
    if logfile:
        with open(logfile, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

def list_folder_sample(path, n=10):
    try:
        items = os.listdir(path)
        return items[:n]
    except Exception as e:
        return f"<error listing {path}: {e}>"

def read_csv_head(path, n=6):
    try:
        df = pd.read_csv(path)
        return df.head(n).to_dict(orient='records'), list(df.columns)
    except Exception as e:
        return f"<error reading {path}: {e}>", None

class BBoxClassificationDataset(Dataset):
    def __init__(self, images_dir, annotations_csv, classid_to_idx, transform=None, bbox_pad=0, debug_log=None):
        self.images_dir = images_dir
        self.transform = transform
        self.bbox_pad = int(bbox_pad)
        self.classid_to_idx = classid_to_idx
        self.samples = []
        self.debug_log = debug_log

        if not os.path.exists(annotations_csv):
            raise RuntimeError(f"Annotations CSV not found: {annotations_csv}")
        df = pd.read_csv(annotations_csv)
        if df.shape[0] == 0:
            raise RuntimeError(f"Annotations CSV is empty: {annotations_csv}")

        cols = {c.lower(): c for c in df.columns}
        def col(name):
            return cols.get(name, None)

        file_col = col('filename') or col('file') or col('image') or col('image_name') or 'filename'
        cid_col = col('class_id') or col('class') or 'class_id'
        cname_col = col('class_name') or col('label') or 'class_name'
        xmin_col = col('xmin') or col('x1') or 'xmin'
        ymin_col = col('ymin') or col('y1') or 'ymin'
        xmax_col = col('xmax') or col('x2') or 'xmax'
        ymax_col = col('ymax') or col('y2') or 'ymax'

        for _, r in df.iterrows():
            fname = r.get(file_col, None) if file_col in df.columns else r.get('filename', None)
            if pd.isna(fname) or fname is None:
                continue
            fname = os.path.basename(str(fname))
            img_path = os.path.join(images_dir, fname)
            if not os.path.exists(img_path):
                found = None
                for root, _, files in os.walk(images_dir):
                    if fname in files:
                        found = os.path.join(root, fname); break
                if found:
                    img_path = found
            if not os.path.exists(img_path):
                log(f"[WARN] image not found for annotation: {fname} (expected under {images_dir})", self.debug_log)
                continue
            xmin = safe_int(r.get(xmin_col, r.get('xmin', 0)))
            ymin = safe_int(r.get(ymin_col, r.get('ymin', 0)))
            xmax = safe_int(r.get(xmax_col, r.get('xmax', 0)))
            ymax = safe_int(r.get(ymax_col, r.get('ymax', 0)))
            cid = safe_int(r.get(cid_col, -1))
            cname = r.get(cname_col, '') if cname_col in df.columns else r.get('class_name', '')
            class_idx = classid_to_idx.get(cid, 0)
            self.samples.append({'img_path': img_path, 'bbox': [xmin,ymin,xmax,ymax], 'class_id': cid, 'class_name': str(cname), 'class_idx': class_idx})
        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples found in {annotations_csv}")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s['img_path']).convert('RGB')
        xmin,ymin,xmax,ymax = s['bbox']
        if self.bbox_pad>0:
            xmin = max(0, xmin - self.bbox_pad); ymin = max(0, ymin - self.bbox_pad)
            xmax = min(img.width-1, xmax + self.bbox_pad); ymax = min(img.height-1, ymax + self.bbox_pad)
        xmin = max(0, xmin); ymin = max(0, ymin)
        xmax = min(img.width-1, xmax); ymax = min(img.height-1, ymax)
        if xmax <= xmin or ymax <= ymin:
            crop = img
        else:
            crop = img.crop((xmin,ymin,xmax,ymax))
        if self.transform:
            crop = self.transform(crop)
        label = int(s['class_idx'])
        return crop, label

def collect_class_map(csv_paths):
    mapping = OrderedDict()
    for p in csv_paths:
        if not p or not os.path.exists(p): continue
        df = pd.read_csv(p)
        cols = {c.lower():c for c in df.columns}
        id_col = cols.get('class_id') or cols.get('class') or None
        name_col = cols.get('class_name') or cols.get('label') or None
        if id_col is None: 
            for c in df.columns:
                if c.lower().startswith('class') and 'name' not in c.lower():
                    id_col = c; break
        if id_col is None: continue
        for _, r in df.iterrows():
            try: cid = int(float(r[id_col])); 
            except: continue
            cname = str(r[name_col]) if (name_col in df.columns and not pd.isna(r[name_col])) else str(cid)
            if cid not in mapping: mapping[cid] = cname
    return mapping

def make_transforms(image_size):
    from torchvision import transforms
    train_tf = transforms.Compose([transforms.Resize((int(image_size*1.2), int(image_size*1.2))),
                                   transforms.RandomResizedCrop(image_size),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
    val_tf = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.CenterCrop(image_size), transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
    return train_tf, val_tf

def build_loaders(dataset_dir, classid_to_idx, image_size, batch_size, bbox_pad, num_workers, device):
    train_tf, val_tf = make_transforms(image_size)
    loaders = {}
    for split in ('train','val','test'):
        images_dir = os.path.join(dataset_dir, split, 'images')
        csv_path = os.path.join(dataset_dir, split, 'annotations.csv')
        if not os.path.exists(images_dir) or not os.path.exists(csv_path):
            log(f"[INFO] Missing split {split}: images_dir={images_dir}, csv={csv_path}", debug_log)
            loaders[split] = None
            continue
        tf = train_tf if split=='train' else val_tf
        ds = BBoxClassificationDataset(images_dir, csv_path, classid_to_idx, transform=tf, bbox_pad=bbox_pad, debug_log=debug_log)
        pin_memory = (device.type == 'cuda')
        loader = DataLoader(ds, batch_size=batch_size, shuffle=(split=='train'), num_workers=num_workers, pin_memory=pin_memory)
        loaders[split] = loader
        log(f"[INFO] {split}: {len(ds)} samples, {len(loader)} batches", debug_log)
    return loaders

def train_and_eval_debug(args, debug_log):
    log(f"[START] train_and_eval_debug (pid={os.getpid()})", debug_log)
    log(f"Args: {args}", debug_log)

    if not os.path.isdir(args.dataset_dir):
        raise RuntimeError(f"dataset_dir not found: {args.dataset_dir}")
    possible = [os.path.join(args.dataset_dir,'all_images','all_annotations.csv'),
                os.path.join(args.dataset_dir,'annotations.csv')]
    master = args.master_csv if args.master_csv and os.path.exists(args.master_csv) else None
    if not master:
        for p in possible:
            if os.path.exists(p):
                master = p; break
    if not master:
        for f in os.listdir(args.dataset_dir):
            if f.lower().endswith('.csv') and 'annot' in f.lower():
                master = os.path.join(args.dataset_dir, f); break
    log(f"[INFO] Master CSV: {master}", debug_log)

    log(f"Top-level dataset files: {list_folder_sample(args.dataset_dir, 20)}", debug_log)
    log(f"train/images sample: {list_folder_sample(os.path.join(args.dataset_dir,'train','images'), 10)}", debug_log)
    log(f"val/images sample: {list_folder_sample(os.path.join(args.dataset_dir,'val','images'), 10)}", debug_log)
    log(f"test/images sample: {list_folder_sample(os.path.join(args.dataset_dir,'test','images'), 10)}", debug_log)

    if master:
        head, hdr = read_csv_head(master, 6)
        log(f"Master CSV header: {hdr}", debug_log)
        log(f"Master CSV sample rows: {head}", debug_log)

    csv_paths = [os.path.join(args.dataset_dir, s, 'annotations.csv') for s in ('train','val','test')]
    class_map = collect_class_map(csv_paths)
    log(f"Collected class map (from splits): {class_map}", debug_log)
    if len(class_map)==0 and master:
        try:
            df = pd.read_csv(master)
            cols = {c.lower():c for c in df.columns}
            id_col = cols.get('class_id') or cols.get('class') or None
            name_col = cols.get('class_name') or cols.get('label') or None
            if id_col:
                for _, r in df.iterrows():
                    try: cid = int(float(r[id_col]))
                    except: continue
                    cname = str(r[name_col]) if (name_col in df.columns and not pd.isna(r[name_col])) else str(cid)
                    if cid not in class_map: class_map[cid] = cname
            log(f"[INFO] Inferred class_map from master CSV: {class_map}", debug_log)
        except Exception as e:
            log(f"[WARN] Could not read master CSV for class map: {e}", debug_log)

    if len(class_map) == 0:
        raise RuntimeError("No classes found. Ensure annotations CSVs contain class_id/class_name.")

    classid_to_idx = {cid: idx for idx, cid in enumerate(class_map.keys())}
    log(f"[INFO] classid_to_idx: {classid_to_idx}", debug_log)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    loaders = build_loaders(args.dataset_dir, classid_to_idx, args.image_size, args.batch_size, args.bbox_pad, args.num_workers, device)
    if loaders.get('train', None) is None:
        raise RuntimeError("Train loader not available. Check your train/images and train/annotations.csv.")

    log("[INFO] Starting a single training epoch (debug) to test pipeline", debug_log)
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(classid_to_idx))
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for images, labels in tqdm(loaders['train'], desc="DebugTrainEpoch", total=min(50, len(loaders['train']))):
        images = images.to(device); labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        break

    log("[INFO] Single-batch training step finished.", debug_log)

    eval_loader = loaders.get('val') or loaders.get('train')
    if eval_loader is None:
        log("[WARN] No eval loader available", debug_log)
        return
    model.eval()
    with torch.no_grad():
        for images, labels in eval_loader:
            images = images.to(device); labels = labels.to(device)
            outs = model(images)
            _, preds = torch.max(outs, 1)
            log(f"[INFO] Eval batch preds sample: {preds[:10].cpu().numpy().tolist()}", debug_log)
            log(f"[INFO] Eval batch gts sample: {labels[:10].cpu().numpy().tolist()}", debug_log)
            break

    log("[COMPLETE] Debug run succeeded. Proceed with full training script next.", debug_log)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--master_csv", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--bbox_pad", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    debug_log = os.path.join(args.output_dir, "debug.log")
    ensure_dir(args.output_dir)
    with open(debug_log, "w", encoding="utf-8") as f: 
        f.write(f"Debug log started\n")

    try:
        train_and_eval_debug(args, debug_log)
    except Exception as e:
        tb = traceback.format_exc()
        log(f"[ERROR] Exception occurred:\n{tb}", debug_log)
        print("[ERROR] Exception occurred during debug run. See debug.log for details:", debug_log)
        sys.exit(1)
    print("Debug run finished successfully. Check debug.log for details:", debug_log)

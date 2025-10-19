#!/usr/bin/env python3
"""
compare_models.py

Compute comparative performance tables & statistics for three classifier outputs
(Model A, Model B, Model C) on the same test set.

Outputs:
 - model_comparison_table.csv  (accuracy, macro_f1, etc + bootstrap CIs)
 - per_class_metrics_{A,B,C}.csv
 - confusion_matrix_{A,B,C}.png
 - paired_stats.csv (McNemar p-values and bootstrap diffs)
 - plots: per_class_f1_{model}.png

Usage:
  python compare_models.py --preds_a path/to/A_preds.csv --preds_b path/to/B_preds.csv --preds_c path/to/C_preds.csv \
      --gt path/to/gt_annotations.csv --class_map path/to/classes.csv --output_dir results/comparison --n_boot 5000
"""

import os
import argparse
import json
from collections import OrderedDict

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from statsmodels.stats.contingency_tables import mcnemar

# ---------------- utils ----------------
def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def read_preds(path, filename_col='filename', pred_col='pred_class_id'):
    df = pd.read_csv(path)
    # normalize column names (case-insensitive)
    cols = {c.lower(): c for c in df.columns}
    fn = cols.get(filename_col.lower())
    pc = cols.get(pred_col.lower())
    if fn is None or pc is None:
        # attempt smart detection
        fn = cols.get('filename') or cols.get('file') or cols.get('image') or list(df.columns)[0]
        # find integer-like column for predictions
        possible_preds = [c for c in df.columns if c.lower() in ('pred','pred_class','pred_class_id','prediction','predicted')]
        pc = possible_preds[0] if possible_preds else list(df.columns)[1] if len(df.columns) > 1 else list(df.columns)[0]
    out = df[[fn, pc]].rename(columns={fn: 'filename', pc: 'pred_class_id'})
    # coerce numeric preds if possible
    try:
        out['pred_class_id'] = out['pred_class_id'].astype(int)
    except Exception:
        # keep as-is
        pass
    out['filename'] = out['filename'].astype(str)
    return out

def read_gt(gt_path, filename_col='filename', class_col='class_id'):
    df = pd.read_csv(gt_path)
    cols = {c.lower(): c for c in df.columns}
    fn = cols.get(filename_col.lower())
    cl = cols.get(class_col.lower())
    if fn is None or cl is None:
        # attempt detection
        fn = cols.get('filename') or cols.get('file') or list(df.columns)[0]
        # find class id col
        cl_candidates = [c for c in df.columns if 'class' in c.lower() or 'label' in c.lower()]
        cl = cl_candidates[0] if cl_candidates else list(df.columns)[1] if len(df.columns) > 1 else list(df.columns)[0]
    out = df[[fn, cl]].rename(columns={fn: 'filename', cl: 'class_id'})
    out['filename'] = out['filename'].astype(str)
    try:
        out['class_id'] = out['class_id'].astype(int)
    except Exception:
        pass
    return out

def load_class_map(path):
    if path is None:
        return None
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        if 'class_id' in cols and 'class_name' in cols:
            mapping = {int(r[cols['class_id']]): str(r[cols['class_name']]) for _, r in df.iterrows()}
            return mapping
        # fallback: if only single column of names
        if 'class_name' in cols:
            names = list(df[cols['class_name']].astype(str))
            return {i: names[i] for i in range(len(names))}
    except Exception:
        try:
            j = json.load(open(path,'r'))
            if isinstance(j, dict):
                return {int(k): v for k,v in j.items()}
            elif isinstance(j, list):
                return {i: j[i] for i in range(len(j))}
        except Exception:
            return None
    return None

# ---------------- metrics ----------------
def per_class_report(y_true, y_pred, labels=None):
    # classification_report returns strings for keys; use sklearn's report to build DataFrame
    rep = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    rows = []
    for k, v in rep.items():
        if k in ('accuracy','macro avg','weighted avg'):
            continue
        # k might be string of label index
        try:
            label_idx = int(k)
        except Exception:
            label_idx = k
        row = {'label': label_idx,
               'precision': v.get('precision', 0.0),
               'recall': v.get('recall', 0.0),
               'f1-score': v.get('f1-score', 0.0),
               'support': int(v.get('support', 0))}
        rows.append(row)
    df = pd.DataFrame(rows).sort_values('label').reset_index(drop=True)
    return df, rep

def plot_confusion(cm, labels, outpath, figsize=(10,10)):
    plt.figure(figsize=figsize)
    sns.heatmap(cm, cmap='Blues', annot=False, xticklabels=labels, yticklabels=labels, square=True)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_per_class_metric(df, label_map, outpath, metric='f1-score', figsize=(12,6)):
    # df contains columns: label (int or str), metric
    df2 = df.copy()
    # map labels to names if map present
    if label_map:
        df2['label_name'] = df2['label'].apply(lambda x: label_map.get(int(x), str(x)) if isinstance(x,(int,np.integer)) else label_map.get(str(x), str(x)))
    else:
        df2['label_name'] = df2['label'].astype(str)
    df2 = df2.sort_values(metric)
    plt.figure(figsize=figsize)
    sns.barplot(x=metric, y='label_name', data=df2, orient='h')
    plt.xlabel(metric)
    plt.ylabel('class')
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

# bootstrap diff of accuracy: returns mean diff and 95% CI
def bootstrap_accuracy_diff(correct_x, correct_y, n_boot=5000, seed=0):
    rng = np.random.RandomState(seed)
    n = len(correct_x)
    diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        diffs.append(correct_x[idx].mean() - correct_y[idx].mean())
    arr = np.array(diffs)
    return arr.mean(), np.percentile(arr, 2.5), np.percentile(arr, 97.5)

# McNemar test given arrays of booleans: correct_x, correct_y
def paired_mcnemar(correct_x, correct_y):
    # n01: x correct, y wrong ; n10: x wrong, y correct
    n01 = int(((correct_x == 1) & (correct_y == 0)).sum())
    n10 = int(((correct_x == 0) & (correct_y == 1)).sum())
    table = [[0, n01], [n10, 0]]
    # if table all zeros -> pval=1.0
    try:
        res = mcnemar(table, exact=False, correction=True)
        pval = float(res.pvalue)
    except Exception:
        pval = 1.0
    return n01, n10, pval

# ---------------- main pipeline ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds_a', required=True, help='CSV with predictions for Model A (filename,pred_class_id)')
    parser.add_argument('--preds_b', required=True, help='CSV with predictions for Model B')
    parser.add_argument('--preds_c', required=True, help='CSV with predictions for Model C')
    parser.add_argument('--gt', required=True, help='Ground-truth CSV (filename,class_id)')
    parser.add_argument('--class_map', required=False, default=None, help='Optional CSV or JSON mapping class_id->class_name')
    parser.add_argument('--output_dir', required=False, default='results/comparison', help='Where to write results')
    parser.add_argument('--n_boot', type=int, default=5000, help='Bootstrap iterations for CI')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for bootstrap')
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    # read predictions and ground truth
    A = read_preds(args.preds_a)
    B = read_preds(args.preds_b)
    C = read_preds(args.preds_c)
    gt = read_gt(args.gt)

    # Merge on filename - keep only filenames present in GT and in all preds
    merged = gt[['filename','class_id']].merge(A, on='filename', how='inner').merge(B, on='filename', how='inner', suffixes=('_A','_B')).merge(C[['filename','pred_class_id']], on='filename', how='inner')
    merged = merged.rename(columns={'pred_class_id':'pred_class_id_C'})
    # after merges columns: filename, class_id, pred_class_id_A, pred_class_id_B, pred_class_id_C
    # ensure ints
    for col in ['class_id','pred_class_id_A','pred_class_id_B','pred_class_id_C']:
        try:
            merged[col] = merged[col].astype(int)
        except Exception:
            pass

    if len(merged) == 0:
        raise RuntimeError("No overlapping filenames between GT and all three prediction files. Check filenames.")

    # load class map if present
    class_map = load_class_map(args.class_map)

    # compute per-model metrics and per-class metrics, confusion matrices
    results = {}
    models = {'A':'pred_class_id_A','B':'pred_class_id_B','C':'pred_class_id_C'}
    labels_sorted = sorted(list(set(merged['class_id'].astype(int).unique())))
    label_names = [class_map.get(l, str(l)) if class_map else str(l) for l in labels_sorted]

    for m, col in models.items():
        y_true = merged['class_id'].astype(int).values
        y_pred = merged[col].astype(int).values
        acc = float(accuracy_score(y_true, y_pred))
        macrof1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        per_df, rep = per_class_report(y_true, y_pred, labels=labels_sorted)
        cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
        # save outputs
        per_df.to_csv(os.path.join(args.output_dir, f'per_class_metrics_{m}.csv'), index=False)
        # plot per-class f1
        plot_per_class_metric(per_df, class_map, os.path.join(args.output_dir, f'per_class_f1_{m}.png'), metric='f1-score', figsize=(8, max(6, len(per_df)*0.2)))
        # save confusion matrix csv
        cm_df = pd.DataFrame(cm, index=[str(x) for x in labels_sorted], columns=[str(x) for x in labels_sorted])
        cm_df.to_csv(os.path.join(args.output_dir, f'confusion_matrix_{m}.csv'))
        # plot confusion matrix image with names if available
        labels_for_plot = [class_map.get(l, str(l)) if class_map else str(l) for l in labels_sorted]
        plot_confusion(cm, labels_for_plot, os.path.join(args.output_dir, f'confusion_matrix_{m}.png'), figsize=(10,8))
        results[m] = {'accuracy': acc, 'macro_f1': macrof1, 'n_samples': int(len(y_true))}
        print(f"[INFO] Model {m}: accuracy={acc:.4f}, macro_f1={macrof1:.4f}, samples={len(y_true)}")

    # build comparison table (point estimates)
    comp_rows = []
    for m in ['A','B','C']:
        comp_rows.append({'model': m, 'accuracy': results[m]['accuracy'], 'macro_f1': results[m]['macro_f1'], 'n_samples': results[m]['n_samples']})
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(os.path.join(args.output_dir, 'model_comparison_table.csv'), index=False)

    # bootstrapped CI and McNemar tests pairwise
    pairs = [('A','B'), ('A','C'), ('B','C')]
    paired_rows = []
    for x,y in pairs:
        col_x = models[x]; col_y = models[y]
        correct_x = (merged[col_x].astype(int).values == merged['class_id'].astype(int).values).astype(int)
        correct_y = (merged[col_y].astype(int).values == merged['class_id'].astype(int).values).astype(int)
        # bootstrap diff (x - y)
        mean_diff, lo, hi = bootstrap_accuracy_diff(correct_x, correct_y, n_boot=args.n_boot, seed=args.seed)
        # mcnemar: counts where x correct y wrong and vice versa
        n01, n10, pval = paired_mcnemar(correct_x, correct_y)
        paired_rows.append({
            'model_x': x, 'model_y': y,
            'acc_x': float(correct_x.mean()), 'acc_y': float(correct_y.mean()),
            'mean_diff_acc_x_minus_y': float(mean_diff),
            'ci_lo': float(lo), 'ci_hi': float(hi),
            'n_x_correct_y_wrong': n01, 'n_x_wrong_y_correct': n10,
            'mcnemar_pvalue': float(pval)
        })
        print(f"[INFO] Pair {x} vs {y}: Δacc={mean_diff:.4f} 95%CI=[{lo:.4f},{hi:.4f}] McNemar p={pval:.4e} n01={n01} n10={n10}")

    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(os.path.join(args.output_dir, 'paired_stats.csv'), index=False)

    # Save aligned per-sample predictions for further analysis
    merged.to_csv(os.path.join(args.output_dir, 'aligned_predictions_gt.csv'), index=False)

    # produce a summary JSON-style CSV
    summary = {
        'A_accuracy': results['A']['accuracy'],
        'B_accuracy': results['B']['accuracy'],
        'C_accuracy': results['C']['accuracy'],
        'A_macro_f1': results['A']['macro_f1'],
        'B_macro_f1': results['B']['macro_f1'],
        'C_macro_f1': results['C']['macro_f1'],
        'n_samples': len(merged)
    }
    pd.DataFrame([summary]).to_csv(os.path.join(args.output_dir, 'summary_metrics.csv'), index=False)

    print(f"[DONE] All comparative outputs saved in {args.output_dir}")

if __name__ == '__main__':
    main()

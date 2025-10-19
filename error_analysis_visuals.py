import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def plot_confusion_matrix(cm_path, out_path, class_mapping=None):
    cm = pd.read_csv(cm_path, index_col=0)
    plt.figure(figsize=(12, 10))

    # Prepare tick labels: try to map the matrix's columns/indices to human-readable names
    xticklabels = True
    yticklabels = True
    if class_mapping is not None:
        def map_label(x):
            # Try string key first, then integer-like
            s = str(x)
            if s in class_mapping:
                return class_mapping[s]
            try:
                si = str(int(float(x)))
                if si in class_mapping:
                    return class_mapping[si]
            except Exception:
                pass
            return x

        mapped_cols = [map_label(c) for c in cm.columns]
        mapped_idx = [map_label(i) for i in cm.index]

        # Only use as tick labels if sizes match (safety)
        if len(mapped_cols) == cm.shape[1] and len(mapped_idx) == cm.shape[0]:
            xticklabels = mapped_cols
            yticklabels = mapped_idx

    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=xticklabels, yticklabels=yticklabels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_classification_report(csv_path, out_path, class_mapping=None):
    df = pd.read_csv(csv_path)

    # identify the column that contains class labels: 'class' or 'label'
    label_col = 'class' if 'class' in df.columns else ('label' if 'label' in df.columns else None)
    if label_col is None:
        raise ValueError("classification report CSV must contain a 'label' or 'class' column")

    # remove aggregate rows commonly named 'accuracy', 'macro avg', 'weighted avg'
    aggregates = {'accuracy', 'macro avg', 'weighted avg'}
    df = df[~df[label_col].isin(aggregates)]

    # map numeric labels to human-friendly names if mapping provided
    def map_label_val(v):
        s = str(v)
        if class_mapping and s in class_mapping:
            return class_mapping[s]
        try:
            si = str(int(float(v)))
            if class_mapping and si in class_mapping:
                return class_mapping[si]
        except Exception:
            pass
        return s

    df[label_col] = df[label_col].apply(map_label_val)

    plt.figure(figsize=(14, 6))
    df_melted = df.melt(id_vars=[label_col], value_vars=["precision", "recall", "f1-score"], var_name="metric", value_name="score")
    sns.barplot(data=df_melted, x=label_col, y="score", hue="metric")
    plt.title("Classification Report Metrics per Class")
    plt.xticks(rotation=90)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Error analysis and visualization")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory with evaluation results")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save plots")
    parser.add_argument("--class_names", type=str, required=True, help="CSV (preferred) or JSON file mapping class ids to names (e.g. dataset/classes.csv)")
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    # Paths
    cm_path = os.path.join(args.results_dir, "confusion_matrix.csv")
    report_csv_path = os.path.join(args.results_dir, "classification_report.csv")

    # Load class names mapping. Prefer CSV with columns 'class_id' and 'class_name'.
    class_mapping = None
    if args.class_names:
        path = args.class_names
        if path.lower().endswith('.csv'):
            try:
                classes_df = pd.read_csv(path)
                if 'class_id' in classes_df.columns and 'class_name' in classes_df.columns:
                    # map numeric IDs to names, store keys as strings for flexible matching
                    class_mapping = {str(int(cid)): name for cid, name in zip(classes_df['class_id'], classes_df['class_name'])}
                elif 'class_name' in classes_df.columns:
                    # fallback: use order-based mapping
                    class_mapping = {str(i): name for i, name in enumerate(classes_df['class_name'])}
                else:
                    print(f"[WARN] CSV provided but expected columns 'class_id' and 'class_name' not found. Columns: {list(classes_df.columns)}")
            except Exception as e:
                print(f"[WARN] Failed to read classes CSV: {e}")
        elif path.lower().endswith('.json'):
            try:
                with open(path, 'r') as f:
                    j = json.load(f)
                if isinstance(j, dict):
                    class_mapping = {str(k): v for k, v in j.items()}
                elif isinstance(j, list):
                    class_mapping = {str(i): name for i, name in enumerate(j)}
            except Exception as e:
                print(f"[WARN] Failed to read class names JSON: {e}")
        else:
            # try CSV then JSON heuristically
            try:
                classes_df = pd.read_csv(path)
                if 'class_id' in classes_df.columns and 'class_name' in classes_df.columns:
                    class_mapping = {str(int(cid)): name for cid, name in zip(classes_df['class_id'], classes_df['class_name'])}
            except Exception:
                try:
                    with open(path, 'r') as f:
                        j = json.load(f)
                    if isinstance(j, dict):
                        class_mapping = {str(k): v for k, v in j.items()}
                    elif isinstance(j, list):
                        class_mapping = {str(i): name for i, name in enumerate(j)}
                except Exception as e:
                    print(f"[WARN] Could not parse class names file: {e}")

    # Plot confusion matrix
    if os.path.exists(cm_path):
        plot_confusion_matrix(cm_path, os.path.join(args.output_dir, "confusion_matrix.png"), class_mapping=class_mapping)
        print(f"[INFO] Confusion matrix saved to {args.output_dir}")
    else:
        print("[WARN] Confusion matrix CSV not found!")

    # Plot classification report (from CSV)
    if os.path.exists(report_csv_path):
        plot_classification_report(report_csv_path, os.path.join(args.output_dir, "classification_report.png"), class_mapping=class_mapping)
        print(f"[INFO] Classification report plots saved to {args.output_dir}")
    else:
        print("[WARN] Classification report CSV not found!")

if __name__ == "__main__":
    main()

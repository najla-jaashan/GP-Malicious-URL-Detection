"""Cross-dataset (out-of-distribution) evaluation.

The headline in-distribution accuracy is measured on a held-out slice of the
*same* Kaggle dataset the model trained on. That number tends to be optimistic
because train and test share collection artifacts. This script scores the
model on an *independent* CSV so you can report the honest generalization gap
-- the single most credible robustness result for a defense.

Expected input: a CSV with the same two columns as the training data:
    url,type
where ``type`` uses the same class names as ``label_map.json``.

Usage (from the repository root):
    python -m src.cross_eval path/to/external_urls.csv
"""

import sys

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from . import config, data
from .predict import load_model, predict_proba


def evaluate_csv(csv_path: str):
    """Score the fine-tuned model on an external labelled URL CSV."""
    ext = pd.read_csv(csv_path).dropna(subset=["url", "type"])
    label_map = data.load_label_map(config.OUTPUT_DIR)

    # Keep only rows whose label the model actually knows about.
    known = set(label_map)
    unknown = set(ext["type"]) - known
    if unknown:
        print(f"Warning: dropping rows with unknown labels: {sorted(unknown)}")
        ext = ext[ext["type"].isin(known)]

    tokenizer, model = load_model()
    id2label = model.config.id2label

    y_true, y_pred = [], []
    for _, row in ext.iterrows():
        scores = predict_proba(row["url"], tokenizer, model)
        # Argmax over the raw calibrated scores (no "uncertain" collapsing
        # here -- we want a class prediction to compare against the label).
        pred = max(scores.items(), key=lambda kv: kv[1])[0]
        y_true.append(row["type"])
        y_pred.append(pred)

    labels_sorted = [id2label[i] for i in range(len(id2label))]
    acc = accuracy_score(y_true, y_pred)

    print(f"\nExternal dataset: {csv_path}")
    print(f"Rows evaluated:   {len(y_true)}")
    print(f"Cross-dataset accuracy: {acc:.4f}\n")
    print(classification_report(y_true, y_pred, labels=labels_sorted, zero_division=0))
    return acc


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.cross_eval <external_csv_path>")
        sys.exit(1)
    evaluate_csv(sys.argv[1])


if __name__ == "__main__":
    main()

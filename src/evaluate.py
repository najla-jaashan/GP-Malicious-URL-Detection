"""Evaluate the fine-tuned model on the held-out test split.

Usage (from the repository root):
    python -m src.evaluate

Recreates the exact same stratified split used during training (fixed
random seed), scores the test set, and prints a per-class metrics table
plus a confusion matrix.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
)

from . import config, data


def main():
    # Rebuild the identical test split (same seed + stratification as training).
    df = data.load_dataframe()
    label_map = data.load_label_map(config.OUTPUT_DIR)
    train_df, val_df, test_df = data.stratified_split(df)
    _, _, test_df = data.encode_labels((train_df, val_df, test_df), label_map)

    # Load the fine-tuned artifacts saved by src/train.py.
    tokenizer = DistilBertTokenizerFast.from_pretrained(str(config.OUTPUT_DIR))
    model = DistilBertForSequenceClassification.from_pretrained(str(config.OUTPUT_DIR))
    model.eval()

    _, _, test_ds = data.tokenize_splits(tokenizer, train_df, val_df, test_df)

    # Batch prediction via the Trainer API (handles device placement).
    trainer = Trainer(model=model)
    predictions = trainer.predict(test_ds)
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids

    # ----------------------------------------------------------- metrics
    acc = accuracy_score(labels, preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )

    class_names = [name for name, _ in sorted(label_map.items(), key=lambda x: x[1])]
    results = pd.DataFrame(
        {
            "Class": class_names,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Support": support,
        }
    )
    results.loc[len(results)] = ["Macro Avg", macro_p, macro_r, macro_f1, support.sum()]

    print(f"\nTest Accuracy: {acc:.4f}\n")
    print(results.round(4).to_string(index=False))

    print("\nConfusion Matrix (rows = true, columns = predicted):")
    cm = pd.DataFrame(
        confusion_matrix(labels, preds), index=class_names, columns=class_names
    )
    print(cm.to_string())


if __name__ == "__main__":
    with torch.no_grad():
        main()

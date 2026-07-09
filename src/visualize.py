"""Project figures: class distribution and the test-set confusion matrix.

Usage (from the repository root):
    python -m src.visualize                 # class-distribution figure
    python -m src.visualize --confusion      # confusion matrix (needs a model)

Figures are written to ``docs/`` so they can be embedded in the README/report.
"""

import argparse

import matplotlib

matplotlib.use("Agg")  # allow running on headless machines/servers
import matplotlib.pyplot as plt
import numpy as np

from . import config, data


def plot_class_distribution():
    """Bar chart of class counts across the train/val/test splits."""
    df = data.load_dataframe()
    train_df, val_df, test_df = data.stratified_split(df)

    classes = sorted(df["type"].unique())
    train_counts = train_df["type"].value_counts().reindex(classes)
    val_counts = val_df["type"].value_counts().reindex(classes)
    test_counts = test_df["type"].value_counts().reindex(classes)

    x = np.arange(len(classes))
    width = 0.25

    plt.figure(figsize=(8, 4))
    plt.bar(x - width, train_counts, width, label="Train")
    plt.bar(x, val_counts, width, label="Validation")
    plt.bar(x + width, test_counts, width, label="Test")

    plt.xticks(x, classes)
    plt.title("Class Distribution Across Splits")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()

    out_path = config.PROJECT_ROOT / "docs" / "class_distribution.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved figure to {out_path}")


def plot_confusion_matrix():
    """Confusion matrix on the held-out test split (requires a trained model).

    Imports the heavy ML stack lazily so the (fast, dependency-light) class
    distribution figure can be produced without a model present.
    """
    import torch
    from sklearn.metrics import confusion_matrix
    from transformers import (
        DistilBertForSequenceClassification,
        DistilBertTokenizerFast,
        Trainer,
    )

    df = data.load_dataframe()
    label_map = data.load_label_map(config.OUTPUT_DIR)
    train_df, val_df, test_df = data.stratified_split(df)
    _, _, test_df = data.encode_labels((train_df, val_df, test_df), label_map)

    tokenizer = DistilBertTokenizerFast.from_pretrained(str(config.OUTPUT_DIR))
    model = DistilBertForSequenceClassification.from_pretrained(str(config.OUTPUT_DIR))
    model.eval()
    _, _, test_ds = data.tokenize_splits(tokenizer, train_df, val_df, test_df)

    with torch.no_grad():
        preds = Trainer(model=model).predict(test_ds)
    y_pred = np.argmax(preds.predictions, axis=1)
    y_true = preds.label_ids

    class_names = [name for name, _ in sorted(label_map.items(), key=lambda x: x[1])]
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (row-normalized)")

    # Annotate each cell with the raw count.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > 0.5 else "black",
                fontsize=9,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    out_path = config.PROJECT_ROOT / "docs" / "confusion_matrix.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved figure to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate project figures")
    parser.add_argument(
        "--confusion",
        action="store_true",
        help="plot the test-set confusion matrix (requires a trained model)",
    )
    args = parser.parse_args()

    if args.confusion:
        plot_confusion_matrix()
    else:
        plot_class_distribution()


if __name__ == "__main__":
    main()

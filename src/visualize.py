"""Plot the class distribution across the train/validation/test splits.

Usage (from the repository root):
    python -m src.visualize

Saves the figure to ``docs/class_distribution.png`` and also shows it if a
display is available. Useful sanity check that stratification preserved
class proportions in every split.
"""

import matplotlib

matplotlib.use("Agg")  # allow running on headless machines/servers
import matplotlib.pyplot as plt
import numpy as np

from . import config, data


def main():
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


if __name__ == "__main__":
    main()

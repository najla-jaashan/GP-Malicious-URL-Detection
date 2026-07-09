"""Dataset loading, splitting, label encoding, and tokenization.

The raw Kaggle CSV has two columns:
    url   -- the raw URL string
    type  -- one of {benign, defacement, malware, phishing}

We treat each URL as a short text sequence and let DistilBERT's sub-word
tokenizer handle all text processing (no handcrafted features).
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def load_dataframe(csv_path: Path = config.DATASET_CSV) -> pd.DataFrame:
    """Load the raw dataset and drop rows with missing values."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["url", "type"]).reset_index(drop=True)
    return df


def build_label_map(df: pd.DataFrame) -> dict:
    """Create a deterministic class -> integer mapping.

    NOTE: the original notebook used ``df['type'].unique()``, whose order
    depends on row order in the CSV. Sorting alphabetically makes the
    mapping stable across runs and machines, and we persist it to disk so
    inference always uses the exact mapping the model was trained with.
    """
    return {label: i for i, label in enumerate(sorted(df["type"].unique()))}


def save_label_map(label_map: dict, output_dir: Path) -> None:
    """Persist the label map next to the model weights."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)


def load_label_map(model_dir: Path) -> dict:
    """Load the label map saved during training (class name -> id)."""
    with open(Path(model_dir) / "label_map.json") as f:
        return json.load(f)


def stratified_split(df: pd.DataFrame):
    """Split into 70% train / 15% validation / 15% test, stratified by class.

    Stratification preserves the (imbalanced) class proportions in every
    split, which keeps evaluation honest for the minority classes.
    """
    train_df, temp_df = train_test_split(
        df,
        test_size=config.TEST_VAL_FRACTION,
        stratify=df["type"],
        random_state=config.RANDOM_SEED,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=config.TEST_FRACTION,
        stratify=temp_df["type"],
        random_state=config.RANDOM_SEED,
    )
    return train_df, val_df, test_df


def encode_labels(dfs, label_map: dict):
    """Map string class names to integer labels in each split."""
    encoded = []
    for df in dfs:
        df = df.copy()
        df["label"] = df["type"].map(label_map)
        encoded.append(df)
    return encoded


def get_tokenizer():
    """Load the DistilBERT sub-word tokenizer.

    ``transformers`` is imported lazily so that importing this module (e.g. in
    unit tests or the fast class-distribution plot) does not require the heavy
    ML stack.
    """
    from transformers import DistilBertTokenizerFast

    return DistilBertTokenizerFast.from_pretrained(config.BASE_MODEL)


def tokenize_splits(tokenizer, train_df, val_df, test_df):
    """Convert pandas splits to tokenized HuggingFace ``Dataset`` objects.

    Every URL is padded/truncated to ``MAX_SEQ_LENGTH`` sub-word tokens and
    formatted as PyTorch tensors ready for the Trainer API.
    """

    from datasets import Dataset

    def tokenize(batch):
        return tokenizer(
            batch["url"],
            padding="max_length",
            truncation=True,
            max_length=config.MAX_SEQ_LENGTH,
        )

    datasets = []
    for df in (train_df, val_df, test_df):
        ds = Dataset.from_pandas(df[["url", "label"]])
        ds = ds.map(tokenize, batched=True)
        ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        datasets.append(ds)
    return tuple(datasets)

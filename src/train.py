"""Fine-tune DistilBERT for 4-class malicious URL classification.

Usage (from the repository root):
    python -m src.train

The script:
  1. loads and splits the Kaggle dataset (70/15/15, stratified),
  2. tokenizes URLs with the DistilBERT tokenizer,
  3. fine-tunes ``distilbert-base-uncased`` with a classification head,
  4. saves the model, tokenizer, and label map to ``models/``.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from . import config, data


def compute_metrics(eval_pred):
    """Report accuracy and macro P/R/F1 during evaluation epochs."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }


def main():
    # ------------------------------------------------------------------ data
    print("Loading dataset...")
    df = data.load_dataframe()
    label_map = data.build_label_map(df)
    print(f"Label map: {label_map}")

    train_df, val_df, test_df = data.stratified_split(df)
    train_df, val_df, test_df = data.encode_labels(
        (train_df, val_df, test_df), label_map
    )
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    tokenizer = data.get_tokenizer()
    train_ds, val_ds, _ = data.tokenize_splits(tokenizer, train_df, val_df, test_df)

    # ----------------------------------------------------------------- model
    model = DistilBertForSequenceClassification.from_pretrained(
        config.BASE_MODEL,
        num_labels=config.NUM_LABELS,
        # Embed human-readable class names in the model config itself,
        # so inference never depends on a hardcoded mapping.
        id2label={v: k for k, v in label_map.items()},
        label2id=label_map,
    )

    training_args = TrainingArguments(
        output_dir=str(config.OUTPUT_DIR / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=config.LOGGING_STEPS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        num_train_epochs=config.NUM_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        load_best_model_at_end=True,  # keep the best epoch, not the last
        metric_for_best_model="macro_f1",
        report_to=[],  # disable wandb/tensorboard logging
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    # ----------------------------------------------------------------- train
    print("Starting fine-tuning...")
    trainer.train()

    # ------------------------------------------------------------------ save
    print(f"Saving model to {config.OUTPUT_DIR}")
    trainer.save_model(str(config.OUTPUT_DIR))
    tokenizer.save_pretrained(str(config.OUTPUT_DIR))
    data.save_label_map(label_map, config.OUTPUT_DIR)
    print("Done. Run `python -m src.evaluate` to score the held-out test set.")


if __name__ == "__main__":
    main()

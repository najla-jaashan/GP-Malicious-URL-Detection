"""Confidence calibration via temperature scaling.

Neural classifiers are typically over-confident: a predicted probability of
0.99 does not mean the model is right 99% of the time. Temperature scaling
(Guo et al., 2017) fits a single scalar T that divides the logits before the
softmax, minimizing negative log-likelihood on the validation set. It leaves
the predicted class unchanged (so accuracy is unaffected) but makes the
reported probabilities trustworthy.

Usage (from the repository root, after training):
    python -m src.calibrate

Writes the learned temperature to ``models/.../temperature.json``, which
``src/predict.py`` then applies automatically.
"""

import json

import numpy as np
import torch
from torch import nn, optim
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
)

from . import config, data


def _collect_logits(model, dataset):
    """Run the model over a dataset and return (logits, labels) as tensors."""
    trainer = Trainer(model=model)
    output = trainer.predict(dataset)
    logits = torch.tensor(output.predictions)
    labels = torch.tensor(output.label_ids)
    return logits, labels


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Learn the temperature T that minimizes NLL on (logits, labels)."""
    # T is optimized in log-space via a single learnable parameter so it
    # stays strictly positive.
    log_t = nn.Parameter(torch.zeros(1))
    optimizer = optim.LBFGS([log_t], lr=0.01, max_iter=100)
    nll = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = nll(logits / torch.exp(log_t), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_t).item())


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins=10):
    """Compute ECE: the gap between confidence and accuracy across bins."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.any():
            bin_conf = confidences[mask].mean()
            bin_acc = accuracies[mask].mean()
            ece += (mask.mean()) * abs(bin_conf - bin_acc)
    return float(ece)


def main():
    # Rebuild the same validation split used during training.
    df = data.load_dataframe()
    label_map = data.load_label_map(config.OUTPUT_DIR)
    train_df, val_df, test_df = data.stratified_split(df)
    _, val_df, _ = data.encode_labels((train_df, val_df, test_df), label_map)

    tokenizer = DistilBertTokenizerFast.from_pretrained(str(config.OUTPUT_DIR))
    model = DistilBertForSequenceClassification.from_pretrained(str(config.OUTPUT_DIR))
    model.eval()

    _, val_ds, _ = data.tokenize_splits(tokenizer, train_df, val_df, test_df)
    logits, labels = _collect_logits(model, val_ds)

    # ECE before and after, for the report.
    probs_before = torch.softmax(logits, dim=1).numpy()
    ece_before = expected_calibration_error(probs_before, labels.numpy())

    temperature = fit_temperature(logits, labels)

    probs_after = torch.softmax(logits / temperature, dim=1).numpy()
    ece_after = expected_calibration_error(probs_after, labels.numpy())

    with open(config.TEMPERATURE_FILE, "w") as f:
        json.dump({"temperature": temperature}, f, indent=2)

    print(f"Learned temperature: {temperature:.4f}")
    print(f"Expected Calibration Error before: {ece_before:.4f}")
    print(f"Expected Calibration Error after:  {ece_after:.4f}")
    print(f"Saved to {config.TEMPERATURE_FILE}")


if __name__ == "__main__":
    main()

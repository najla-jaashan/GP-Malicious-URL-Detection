"""URL classification: model loading, temperature-calibrated inference, CLI.

Usage (from the repository root):
    python -m src.predict                      # interactive loop
    python -m src.predict http://example.com   # classify one or more URLs

Class names are read from the model config (``id2label``) saved at training
time, so there is no hardcoded label mapping that could drift out of sync
with the trained weights. If a calibrated temperature has been learned by
``src/calibrate.py`` it is applied automatically; otherwise T=1.0 (no change).
"""

import json
import sys

import torch

from . import config


def load_temperature() -> float:
    """Load the calibrated softmax temperature, defaulting to 1.0."""
    try:
        with open(config.TEMPERATURE_FILE) as f:
            return float(json.load(f)["temperature"])
    except (FileNotFoundError, KeyError, ValueError):
        return 1.0


def load_model():
    """Load the fine-tuned tokenizer + model from ``models/``.

    Returns (tokenizer, model). The learned temperature is attached to the
    model as ``model.temperature`` so downstream callers stay simple.
    """
    from transformers import (
        DistilBertForSequenceClassification,
        DistilBertTokenizerFast,
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained(str(config.OUTPUT_DIR))
    model = DistilBertForSequenceClassification.from_pretrained(str(config.OUTPUT_DIR))
    model.eval()
    model.temperature = load_temperature()
    return tokenizer, model


def predict_proba(url: str, tokenizer, model) -> dict[str, float]:
    """Return a calibrated {class_name: probability} map for one URL."""
    inputs = tokenizer(
        url,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=config.MAX_SEQ_LENGTH,
    ).to(model.device)

    temperature = getattr(model, "temperature", 1.0)
    with torch.no_grad():
        # Temperature scaling: divide logits by T before softmax.
        logits = model(**inputs).logits / temperature
        probs = torch.softmax(logits, dim=1)[0]

    return {model.config.id2label[i]: float(p) for i, p in enumerate(probs)}


def predict_url(url: str, tokenizer, model) -> tuple[str, float]:
    """Classify a single URL; returns (class_name, confidence).

    If the top calibrated probability falls below
    ``config.UNCERTAIN_THRESHOLD`` the label is reported as ``"uncertain"``
    so borderline inputs are flagged for review rather than force-classified.
    """
    scores = predict_proba(url, tokenizer, model)
    label, confidence = max(scores.items(), key=lambda kv: kv[1])
    if confidence < config.UNCERTAIN_THRESHOLD:
        return "uncertain", confidence
    return label, confidence


def interactive_loop(tokenizer, model):
    """Simple REPL: type a URL, get a prediction; type 'exit' to quit."""
    print("Type a URL to classify it. Type 'exit' to stop.\n")
    while True:
        url = input("Enter URL: ").strip()
        if url.lower() == "exit":
            print("Exiting.")
            break
        if not url:
            print("Empty input. Try again.\n")
            continue
        label, conf = predict_url(url, tokenizer, model)
        print(f"Prediction: {label}  (confidence: {conf:.2%})\n")


def main():
    tokenizer, model = load_model()
    urls = sys.argv[1:]
    if urls:
        # Non-interactive mode: classify all URLs passed as arguments.
        for url in urls:
            label, conf = predict_url(url, tokenizer, model)
            print(f"{url}\t->\t{label}\t({conf:.2%})")
    else:
        interactive_loop(tokenizer, model)


if __name__ == "__main__":
    main()

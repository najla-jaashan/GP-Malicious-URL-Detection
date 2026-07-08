"""Interactive URL classification tool.

Usage (from the repository root):
    python -m src.predict                      # interactive loop
    python -m src.predict http://example.com   # classify one or more URLs

Loads the fine-tuned model saved by ``src/train.py`` and prints the
predicted class for each URL. The class names are read from the model
config (saved at training time), so there is no hardcoded label mapping
that could silently drift out of sync with the trained weights.
"""

import sys

import torch
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
)

from . import config


def load_model():
    """Load the fine-tuned tokenizer + model from ``models/``."""
    tokenizer = DistilBertTokenizerFast.from_pretrained(str(config.OUTPUT_DIR))
    model = DistilBertForSequenceClassification.from_pretrained(str(config.OUTPUT_DIR))
    model.eval()
    return tokenizer, model


def predict_url(url: str, tokenizer, model) -> tuple[str, float]:
    """Classify a single URL; returns (class_name, confidence)."""
    inputs = tokenizer(
        url,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=config.MAX_SEQ_LENGTH,
    ).to(model.device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)

    pred_id = int(torch.argmax(probs, dim=1).item())
    confidence = float(probs[0, pred_id].item())
    # id2label was persisted in the model config during training.
    return model.config.id2label[pred_id], confidence


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

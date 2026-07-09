"""Gradio web interface for the malicious URL detector.

Reuses the inference logic in ``src/predict.py`` (single source of truth) so
the UI never duplicates model handling. Class names and colors are derived
dynamically from the model config -- nothing is hardcoded, so retraining with
a different label set requires no changes here.

Run from the repository root (requires a trained model in ``models/``):

    python app.py
"""

import gradio as gr

from src.predict import load_model, predict_proba, predict_url
from src import config

# Load the fine-tuned model ONCE at startup, not per request.
tokenizer, model = load_model()

# Class names come straight from the trained model's config.
CLASS_NAMES = [model.config.id2label[i] for i in range(model.config.num_labels)]

# A small palette; classes not listed fall back to a neutral gray. "uncertain"
# is not a model class but can be returned by predict_url below the threshold.
_PALETTE = {
    "benign": "#16a34a",
    "phishing": "#dc2626",
    "malware": "#b91c1c",
    "defacement": "#ea580c",
    "uncertain": "#6b7280",
}


def _color_for(label: str) -> str:
    return _PALETTE.get(label.lower(), "#374151")


def classify(url: str):
    """Return (HTML verdict badge, {class: probability} for the Label widget)."""
    url = (url or "").strip()
    if not url:
        return "<em>Please enter a URL.</em>", {}

    # Full calibrated probability distribution for the bar display...
    scores = predict_proba(url, tokenizer, model)
    # ...and the final verdict (may be "uncertain" below the threshold).
    label, confidence = predict_url(url, tokenizer, model)

    color = _color_for(label)
    note = (
        " &nbsp;·&nbsp; below confidence threshold, treat as needs-review"
        if label == "uncertain"
        else ""
    )
    verdict = (
        f"<div style='padding:14px;border-radius:10px;background:{color};"
        f"color:white;font-size:20px;font-weight:600;text-align:center'>"
        f"{label.upper()} &nbsp;·&nbsp; {confidence:.1%} confidence{note}</div>"
    )
    return verdict, scores


with gr.Blocks(title="Malicious URL Detection — DistilBERT") as demo:
    gr.Markdown(
        "# Malicious URL Detection\n"
        "Fine-tuned **DistilBERT** classifier. Enter a URL to classify it as "
        f"one of: {', '.join(f'**{c}**' for c in CLASS_NAMES)}. "
        f"Predictions below {config.UNCERTAIN_THRESHOLD:.0%} confidence are "
        "flagged as **uncertain** rather than force-classified."
    )
    with gr.Row():
        url_in = gr.Textbox(
            label="URL", placeholder="http://example.com/login", scale=4
        )
        btn = gr.Button("Classify", variant="primary", scale=1)

    verdict_out = gr.HTML()
    scores_out = gr.Label(num_top_classes=len(CLASS_NAMES), label="Class probabilities")

    gr.Examples(
        examples=[
            "https://www.google.com",
            "http://secure-paypa1.account-verify.ru/login",
            "http://192.168.1.1/wp-admin/setup-config.php",
        ],
        inputs=url_in,
    )

    btn.click(classify, inputs=url_in, outputs=[verdict_out, scores_out])
    url_in.submit(classify, inputs=url_in, outputs=[verdict_out, scores_out])

    gr.Markdown(
        "<small>Research prototype. A prediction is one signal, not a "
        "guarantee of safety; use within a layered security workflow.</small>"
    )

if __name__ == "__main__":
    # share=True prints a temporary public URL — convenient for a live demo.
    demo.launch(share=True)

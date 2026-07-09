# Malicious URL Detection with DistilBERT

A transformer-based NLP system that classifies URLs into four categories — **benign**, **phishing**, **malware**, and **defacement** — by fine-tuning **DistilBERT** directly on raw URL strings, with no handcrafted feature engineering. The model achieves **98.89% accuracy** on a held-out test set of ~97K URLs.

This repository is the **engineering and productionization work of Najla Jaashan**: a research prototype was re-architected into a clean, modular, reproducible Python package, with a latent label-mapping bug identified and fixed, model persistence and evaluation tooling added, and full documentation authored. The underlying model research is credited below. The original research notebook, paper, and presentation are preserved under `notebooks/` and `docs/`.

## Project Structure

```
malicious-url-detection-distilbert/
├── src/
│   ├── config.py        # All hyperparameters, paths, and constants
│   ├── data.py          # Loading, stratified splitting, labels, tokenization
│   ├── train.py         # Fine-tuning pipeline (python -m src.train)
│   ├── evaluate.py      # Test-set metrics + confusion matrix
│   ├── predict.py       # Interactive / CLI URL classifier
│   └── visualize.py     # Class-distribution plot across splits
├── notebooks/
│   └── NLP_Project.ipynb    # Original research notebook
├── docs/
│   ├── paper.pdf            # IEEE-format project paper
│   ├── presentation.pdf     # Project presentation
│   └── TECHNICAL_REPORT.md  # Detailed analysis & recommendations
├── data/                # Place malicious_phish.csv here (not tracked by git)
├── models/              # Fine-tuned model artifacts saved here
├── requirements.txt
└── README.md
```

## How It Works

1. **Dataset** — the [Malicious URLs Dataset (Kaggle)](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset): ~651,000 labeled URLs across 4 classes.
2. **Split** — stratified 70% train / 15% validation / 15% test (fixed seed 42), preserving class proportions in every split.
3. **Tokenization** — URLs are treated as text and encoded with the DistilBERT WordPiece tokenizer (max 64 tokens). No manual feature extraction.
4. **Model** — `distilbert-base-uncased` + a 4-way classification head (~40% fewer parameters than BERT, well suited to short sequences).
5. **Training** — 3 epochs, AdamW, learning rate 2e-5, batch size 16, weight decay 0.01, via the HuggingFace `Trainer` API.
6. **Inference** — an interactive tool classifies any URL in real time and reports a confidence score.

## Results

| Class      | Precision | Recall | F1-Score |
|------------|-----------|--------|----------|
| Benign     | 0.9922    | 0.9944 | 0.9933   |
| Phishing   | 0.9641    | 0.9596 | 0.9619   |
| Malware    | 0.9908    | 0.9705 | 0.9805   |
| Defacement | 0.9974    | 0.9991 | 0.9982   |
| **Macro Avg** | **0.9861** | **0.9809** | **0.9835** |

**Overall test accuracy: 98.89%**

## Setup

### 1. Environment

```bash
git clone <your-repo-url>
cd malicious-url-detection-distilbert

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

A CUDA-capable GPU is strongly recommended for training (~450K training URLs × 3 epochs). Google Colab's free T4 GPU works fine. Inference runs comfortably on CPU.

### 2. Dataset

Download `malicious_phish.csv` from the [Kaggle dataset page](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset) and place it in `data/`.

### 3. Train

```bash
python -m src.train
```

Saves the fine-tuned model, tokenizer, and label map to `models/distilbert-url-classifier/`.

### 4. Evaluate

```bash
python -m src.evaluate
```

Prints the per-class metrics table and a confusion matrix on the held-out test split.

### 5. Predict

```bash
# Interactive loop
python -m src.predict

# Or classify URLs directly
python -m src.predict "http://secure-login.paypa1-account.example/verify"
```

## Improvements Over the Original Notebook

- **Deterministic, persisted label map** — the notebook derived labels from `df['type'].unique()` (row-order dependent) and hardcoded a separate `id2label` dictionary in the prediction cell, a mismatch risk. Here the mapping is sorted, saved to `label_map.json`, and embedded in the model config.
- **Model persistence** — the notebook never saved the fine-tuned model; retraining was required for every session. `train.py` saves all artifacts for reuse.
- **Best-epoch selection** — `load_best_model_at_end` with macro-F1 replaces "keep whatever the last epoch produced".
- **Metrics during training** — a `compute_metrics` callback reports accuracy/macro-F1 each epoch instead of loss only.
- **Confidence scores and confusion matrix** added at inference/evaluation time.
- **Modular, documented code** with centralized configuration.

See `docs/TECHNICAL_REPORT.md` for the full analysis, identified challenges, and future-work recommendations.

## Disclaimer

This model is a research prototype. Predictions should not be treated as a guarantee that a URL is safe; use it as one signal within a layered security workflow.

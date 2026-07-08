# Technical Report: Malicious URL Detection with DistilBERT

**Prepared as part of a code review, restructuring, and repository reorganization of the original project by Alhamli, Aljomuh & Alkhalis (Imam Abdulrahman Bin Faisal University).**

---

## 1. Project Concept

Malicious URLs are the primary delivery mechanism for phishing campaigns, malware distribution, and defaced-site redirection. Traditional detectors depend on blacklists or handcrafted lexical features (URL length, keyword presence, special-character counts), which attackers evade through obfuscation: homoglyph substitution, brand impersonation inside subdomains, randomized paths, and disguised redirects.

This project reframes URL classification as a **natural language processing task**. A URL is treated as a short text sequence, and a pretrained transformer — **DistilBERT** — is fine-tuned to learn contextual and structural patterns directly from raw URL strings. The result is a four-class classifier (benign / phishing / malware / defacement) that requires essentially no manual preprocessing or feature engineering.

The accompanying paper (`docs/paper.pdf`) situates the work against related transformer-based URL detectors (URLTran, DomURLs-BERT, SecureNet) and motivates DistilBERT specifically as a lightweight alternative suitable for real-time deployment.

## 2. Original Repository Contents

| File | Description |
|---|---|
| `NLP_Project.ipynb` | The entire codebase: a 14-cell Colab-style notebook covering the full pipeline |
| `Nlp-Project.pdf` | 5-page IEEE-format paper |
| `Transformer-Based NLP Model..._presentation.pdf` | Slide deck |
| `README.md` | Project summary with results table |

The project is notebook-only — there was no package structure, no dependency manifest, no model persistence, and no separation between training, evaluation, and inference. This report documents the restructuring into a standard Python package.

## 3. Architecture and Pipeline

### 3.1 Data

- **Source:** Kaggle "Malicious URLs Dataset" (`malicious_phish.csv`), ~651,000 URLs.
- **Classes:** benign (~428K), defacement (~96K), phishing (~94K), malware (~32K) — a notably imbalanced distribution.
- **Split:** stratified 70/15/15 train/validation/test with a fixed random seed (42). Stratification preserves class proportions in every split, which keeps minority-class evaluation honest.

### 3.2 Tokenization

URLs are encoded with the standard `distilbert-base-uncased` WordPiece tokenizer, padded/truncated to **64 sub-word tokens**. No URL-specific normalization is applied — the design bet, consistent with the URLTran literature, is that the transformer learns structural regularities (protocol, domain, path, parameter patterns) from sub-word co-occurrence statistics.

### 3.3 Model

`DistilBertForSequenceClassification`: 6 transformer layers, 66M parameters (~40% smaller than BERT-base), with a linear classification head over the pooled `[CLS]` representation producing 4 logits.

### 3.4 Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs | 3 |
| Batch size | 16 |
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Loss | Cross-entropy |
| Frameworks | PyTorch + HuggingFace Transformers/Datasets |

### 3.5 Reported Results

Test accuracy **98.89%**, macro-F1 **0.9835**. Phishing is the weakest class (F1 0.9619) — expected, since phishing URLs deliberately mimic benign structure. Defacement is nearly perfectly separable (F1 0.9982), likely because those URLs share strong surface signatures (e.g., characteristic CMS paths).

## 4. Code Review: Issues Identified in the Original Notebook

1. **Fragile label mapping (highest-risk issue).** The training label map was built from `df['type'].unique()` — an ordering that depends on CSV row order — while the prediction cell hardcoded a *separate* dictionary `{0: phishing, 1: benign, 2: defacement, 3: malware}`. If the CSV row order ever changes (or a different copy of the dataset is used), training silently uses one mapping while inference assumes another, producing systematically wrong class names with no error raised.
2. **No model persistence.** The fine-tuned model was never saved; every session required ~hours of retraining before the prediction tool could be used.
3. **No `compute_metrics` in the Trainer.** Only loss was visible during training; accuracy/F1 appeared only in a post-hoc cell.
4. **Last-epoch model kept by default.** Without `load_best_model_at_end`, the final checkpoint is used even if an earlier epoch generalized better.
5. **Pandas `SettingWithCopy` risk.** Labels were assigned to slices of the split dataframes without `.copy()`.
6. **Prediction tokenization mismatch.** The interactive cell tokenized with `padding=True` and no `max_length`, subtly different from training (`max_length=64`); harmless for short URLs but inconsistent.
7. **No dependency pinning, no `.gitignore`, no seed control beyond the split**, and setup cells (`!pip install`) embedded in the notebook.
8. **Uncased model choice.** `distilbert-base-uncased` lowercases input, discarding case information that can carry signal in URLs (e.g., mixed-case obfuscation, certain homoglyph patterns).

## 5. Restructured Repository

The notebook was decomposed into a conventional package:

```
src/
├── config.py     # single source of truth for hyperparameters & paths
├── data.py       # loading, deterministic label map, stratified split, tokenization
├── train.py      # fine-tuning + artifact persistence (model, tokenizer, label_map.json)
├── evaluate.py   # test metrics table + confusion matrix
├── predict.py    # interactive & CLI inference with confidence scores
└── visualize.py  # class-distribution figure
```

Key corrections applied during restructuring:

- Label map is now **sorted alphabetically** (deterministic), **saved to `label_map.json`**, and additionally **embedded in the model config** (`id2label`/`label2id`), eliminating the training/inference mismatch class entirely.
- `train.py` saves the model and tokenizer; `evaluate.py` and `predict.py` load the saved artifacts rather than depending on in-memory notebook state.
- `load_best_model_at_end=True` with macro-F1 as the selection metric.
- A `compute_metrics` callback surfaces accuracy and macro P/R/F1 every epoch.
- `predict.py` reports a softmax **confidence score** and supports both interactive and batch CLI modes.
- `evaluate.py` adds a **confusion matrix**, which the original evaluation lacked.
- `requirements.txt` and `.gitignore` added; large artifacts (CSV, model weights) excluded from version control.

**Note on reproducibility:** because the label mapping was changed from row-order-dependent to alphabetical, integer label IDs differ from the original notebook run. This does not affect metrics or behavior — only the internal ID assignment — and the persisted mapping guarantees internal consistency.

## 6. Environment Setup and Run Instructions

1. **Clone and install**
   ```bash
   git clone <repo-url> && cd malicious-url-detection-distilbert
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Get the data** — download `malicious_phish.csv` from the Kaggle "Malicious URLs Dataset" page into `data/`.
3. **Train** — `python -m src.train` (GPU strongly recommended; a Colab T4 completes 3 epochs on ~455K URLs in a few hours).
4. **Evaluate** — `python -m src.evaluate` (prints per-class metrics + confusion matrix on the held-out 15% test split).
5. **Use** — `python -m src.predict` for the interactive loop, or pass URLs as CLI arguments.

## 7. Recommendations for Future Work

**Modeling**
- **Cased or character-aware tokenization.** Compare `distilbert-base-cased`, CharBERT-style character models, or URL-aware tokenizers with structural markers (`[DOMAIN]`, `[PATH]`) as in DomURLs-BERT — case and character-level signals matter for obfuscated URLs.
- **Class imbalance handling.** Malware is ~5% of the data; weighted cross-entropy or focal loss could lift its recall (currently the lowest at 0.9705).
- **Adversarial robustness testing.** Evaluate against homoglyph substitution, URL shorteners, and typosquatting perturbations before claiming deployment readiness.
- **Cross-dataset generalization.** The 98.89% figure is within-dataset. Testing on an independent corpus (e.g., PhishTank feeds, newer URL dumps) would reveal how much of the performance reflects dataset-specific artifacts — a known pitfall in this literature.

**Engineering**
- **Serve the model** behind a lightweight API (FastAPI) with batching, and consider ONNX/quantization export — DistilBERT quantizes well and would enable low-latency CPU inference.
- **Calibration and thresholds.** Softmax confidence is uncalibrated; temperature scaling plus a "suspicious/uncertain" band would make the tool safer in practice than a hard four-way decision.
- **CI + tests.** Unit tests for the split determinism, label-map round-trip, and a smoke test on a tiny data sample.
- **Data versioning** (DVC or HuggingFace Datasets Hub) so the exact CSV used for the reported numbers is pinned.

**Reporting**
- Add the confusion matrix and per-class support counts to the paper's results section; accuracy alone is dominated by the benign majority class.
- Report inference latency (ms/URL on CPU and GPU) to substantiate the "real-time deployment" claim, which is the paper's core motivation for choosing DistilBERT.

## 8. Conclusion

The project is a well-executed demonstration that a compact transformer can learn discriminative URL structure directly from raw strings, matching or exceeding handcrafted-feature baselines with minimal preprocessing. Its main weaknesses were engineering ones — a notebook-only codebase with a latent label-mapping bug and no persistence — all of which are addressed in this restructured repository. The most valuable next scientific step is cross-dataset and adversarial evaluation, which would test whether the impressive within-dataset numbers survive contact with real-world URL drift.

# Contributing & Setup Guide

Thanks for your interest in this project. This guide covers how to set up a
development environment, the conventions the codebase follows, and how to run
the checks that CI enforces.

## Development Setup

```bash
# 1. Clone and enter the project
git clone https://github.com/najla-jaashan/GP-Malicious-URL-Detection.git
cd GP-Malicious-URL-Detection

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install runtime dependencies
pip install -r requirements.txt

# 4. Install development tooling
pip install ruff
```

## Getting the Dataset

The dataset is **not** tracked in git (it is ~651K rows and excluded via
`.gitignore`). Download `malicious_phish.csv` from the
[Kaggle Malicious URLs Dataset](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset)
and place it in the `data/` directory:

```
data/malicious_phish.csv
```

## Project Layout

```
src/
├── config.py     # All hyperparameters, paths, and constants live here
├── data.py       # Loading, splitting, label encoding, tokenization
├── train.py      # Fine-tuning pipeline + artifact persistence
├── evaluate.py   # Held-out test metrics + confusion matrix
├── predict.py    # Interactive / CLI inference
└── visualize.py  # Class-distribution plotting
```

If you change a hyperparameter or path, change it in `config.py` — nothing
should be hardcoded elsewhere.

## Running the Pipeline

```bash
python -m src.train       # fine-tune (GPU strongly recommended)
python -m src.evaluate    # score the held-out test set
python -m src.predict     # interactive URL classifier
python -m src.visualize   # save the class-distribution figure
```

## Code Style & Checks

This project uses [ruff](https://docs.astral.sh/ruff/) for both linting and
formatting. Before opening a pull request, run the same checks CI runs:

```bash
ruff check src/           # lint
ruff format src/          # auto-format (use --check to only verify)
python -m compileall src/ # byte-compile / syntax check
```

CI (see `.github/workflows/ci.yml`) runs these on Python 3.10, 3.11, and 3.12
for every push and pull request to `main`. Note that CI intentionally does
**not** download the dataset or train the model — that requires a GPU and
significant compute — so it only validates code quality and importability.

## Commit & Pull Request Conventions

- Keep commits focused and write imperative-mood subject lines
  (e.g. "Add confidence threshold to predict.py").
- Ensure `ruff check`, `ruff format --check`, and `compileall` all pass locally
  before pushing.
- For substantive changes, briefly describe the motivation in the PR body.

## Reporting Issues

When filing an issue, please include your Python version, OS, the command you
ran, and the full error traceback. For model-behavior questions, mention
whether you retrained locally or used pre-saved artifacts.

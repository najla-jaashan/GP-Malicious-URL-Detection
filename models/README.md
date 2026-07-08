# models/

Fine-tuned model artifacts are saved here. This directory is kept in version
control, but the model weights themselves are **not** committed (they are
excluded via `.gitignore` because of their size).

## What gets written here

Running `python -m src.train` saves the following into
`models/distilbert-url-classifier/`:

- the fine-tuned DistilBert weights and config,
- the tokenizer files,
- `label_map.json` (the class-name → id mapping used for inference).

`src/evaluate.py` and `src/predict.py` load these artifacts from this
directory.

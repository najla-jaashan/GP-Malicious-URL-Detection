# data/

Place the dataset file here. This directory is intentionally kept in version
control, but the dataset itself is **not** committed (it is excluded via
`.gitignore` because of its size, ~651K rows).

## Required file

Download `malicious_phish.csv` from the
[Kaggle Malicious URLs Dataset](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset)
and save it here:

```
data/malicious_phish.csv
```

The training and evaluation scripts read the path from `src/config.py`
(`DATASET_CSV`).

"""Central configuration for the malicious URL detection project.

All hyperparameters, paths, and constants live here so that experiments
can be reproduced or modified from a single location instead of hunting
through the training code.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Kaggle "Malicious URLs Dataset" CSV (columns: url, type)
# https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset
DATASET_CSV = DATA_DIR / "malicious_phish.csv"

# Directory where the fine-tuned model, tokenizer, and label map are saved
OUTPUT_DIR = MODELS_DIR / "distilbert-url-classifier"

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
BASE_MODEL = "distilbert-base-uncased"
NUM_LABELS = 4  # benign, defacement, malware, phishing

# URLs are short sequences; 64 sub-word tokens covers the vast majority.
MAX_SEQ_LENGTH = 64

# ---------------------------------------------------------------------------
# Data split (stratified by class)
# ---------------------------------------------------------------------------
TEST_VAL_FRACTION = 0.30  # 70% train, then the remaining 30% is split...
TEST_FRACTION = 0.50  # ...50/50 into validation (15%) and test (15%)
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Training hyperparameters (as reported in the accompanying paper)
# ---------------------------------------------------------------------------
NUM_EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 100

# ---------------------------------------------------------------------------
# Inference / calibration
# ---------------------------------------------------------------------------
# A calibrated temperature (>1 softens over-confident logits) is learned on the
# validation set by src/calibrate.py and written here. Default 1.0 = no change.
TEMPERATURE_FILE = OUTPUT_DIR / "temperature.json"

# If the top calibrated probability is below this threshold, inference returns
# an "uncertain" verdict instead of a hard class. A security tool should not
# pretend certainty on borderline inputs.
UNCERTAIN_THRESHOLD = 0.60

"""Subtask 1b: Evidence Extraction — Config."""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, "train.xlsx")
LEADERBOARD_PATH = os.path.join(DATA_DIR, "leaderboard.xlsx")
OVERSAMPLE_PATH = os.path.join(DATA_DIR, "oversample.json")

MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LENGTH = 768

BATCH_SIZE = 4           # BIO needs more memory per sample; halved for 768-token context
GRAD_ACCUM = 2           # effective = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
EPOCHS = 32
WARMUP_RATIO = 0.1
FP16 = False
SEED = 42
N_FOLDS = 5
MAX_CHECKPOINTS = 5
PATIENCE = 4

RISK_LEVELS = ["indicator", "ideation", "behavior", "attempt"]

# BIO tags
BIO_TAGS = ["O", "B-EVD", "I-EVD"]
BIO2ID = {t: i for i, t in enumerate(BIO_TAGS)}
ID2BIO = {i: t for i, t in enumerate(BIO_TAGS)}
NUM_BIO = len(BIO_TAGS)

# Up-weight B/I tokens heavily (they're <5% of all tokens)
BIO_WEIGHTS = [0.1, 8.0, 8.0]

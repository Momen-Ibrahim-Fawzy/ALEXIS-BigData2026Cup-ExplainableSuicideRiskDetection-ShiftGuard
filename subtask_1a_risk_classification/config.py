"""Subtask 1a: Risk Classification — Config."""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, "train.xlsx")
LEADERBOARD_PATH = os.path.join(DATA_DIR, "leaderboard.xlsx")
OVERSAMPLE_PATH = os.path.join(DATA_DIR, "oversample.json")

# Model — DeBERTa-v3-large for A100
MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LENGTH = 512

# Training — tuned for A100 30GB VRAM
BATCH_SIZE = 16
GRAD_ACCUM = 1          # effective batch = 16
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
EPOCHS = 8
WARMUP_RATIO = 0.1
FP16 = True
SEED = 42
N_FOLDS = 5
MAX_CHECKPOINTS = 5     # keep at most this many, always keep best

# 4-class risk levels
RISK_LEVELS = ["indicator", "ideation", "behavior", "attempt"]
RISK2ID = {r: i for i, r in enumerate(RISK_LEVELS)}
ID2RISK = {i: r for i, r in enumerate(RISK_LEVELS)}
NUM_CLASSES = len(RISK_LEVELS)

# Inverse-frequency class weights (from EDA: 611, 519, 391, 114)
CLASS_WEIGHTS = [0.669, 0.788, 1.045, 3.586]

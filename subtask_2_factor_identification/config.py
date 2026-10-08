"""Subtask 2: Factor Identification — Config."""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, "train.xlsx")
LEADERBOARD_PATH = os.path.join(DATA_DIR, "leaderboard.xlsx")
OVERSAMPLE_PATH = os.path.join(DATA_DIR, "oversample.json")

MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LENGTH = 512

BATCH_SIZE = 16
GRAD_ACCUM = 1
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
EPOCHS = 32
WARMUP_RATIO = 0.1
FP16 = False
SEED = 42
N_FOLDS = 5
MAX_CHECKPOINTS = 5
PATIENCE = 4

FACTOR_LIST = [
    "mental health issues", "physical health/characteristic", "substance use",
    "hopelessness", "emotion dysregulation", "low self-esteem",
    "poor school performance", "low socio-economic status",
    "interpersonal violence", "prior self-harm or suicidal thought/attempt",
    "poor social support", "interpersonal difficulty", "dysfunctional family",
    "exposure to others' suicide", "stressful life event", "traumatic experience",
    "cognitive deficits", "suicide means (with access)",
    "sexual orientation related issues", "social support", "coping strategy",
    "psychological capital", "sense of responsibility", "meaning in life",
]
NUM_FACTORS = len(FACTOR_LIST)
FACTOR2ID = {f: i for i, f in enumerate(FACTOR_LIST)}

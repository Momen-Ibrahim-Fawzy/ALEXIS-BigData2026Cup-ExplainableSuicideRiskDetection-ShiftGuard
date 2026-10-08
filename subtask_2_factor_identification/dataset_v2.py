"""Sub2 v2 dataset: prepends a special risk-level token (see model_v2.py)
and attaches SBert similarity features per post.

Critical design choice for the risk token: TRAINING rows use the TRUE risk
label (strongest possible training signal), but VALIDATION rows (both the
GroupKFold OOF fold and the final leaderboard prediction) use Sub1a's own
PREDICTED risk label, not ground truth -- ground truth doesn't exist at real
inference time, so scoring OOF with true labels would make the OOF estimate
optimistic relative to what actually happens when Sub1a's ~20% error rate is
present. This is standard "train with teacher forcing, evaluate under
realistic conditions" practice, adapted from sequence modeling.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer

from config import *
from dataset import load_train as _load_train_base, load_leaderboard
from model_v2 import RISK_TOKEN_MAP

SBERT_DIR = os.path.join(os.path.dirname(__file__), "sbert_cache")
SBERT_MODEL = "all-mpnet-base-v2"

_sbert_model = None  # lazy singleton, only loaded if an on-the-fly fallback is actually needed


def _get_sbert_model():
    global _sbert_model
    if _sbert_model is None:
        _sbert_model = SentenceTransformer(SBERT_MODEL)
    return _sbert_model


def load_sbert_lookup():
    """Returns (text -> sims array) dicts for train and leaderboard, built
    from the precomputed cache (sbert_features.py must have been run first).
    Keyed by post text (robust to any row_id inconsistencies across
    oversample sources)."""
    train_df = pd.read_excel(TRAIN_PATH, engine="openpyxl")
    train_sims = np.load(os.path.join(SBERT_DIR, "train_sims.npy"))
    lb_df = pd.read_excel(LEADERBOARD_PATH, engine="openpyxl")
    lb_sims = np.load(os.path.join(SBERT_DIR, "leaderboard_sims.npy"))

    lookup = {}
    for text, sim in zip(train_df["post"].astype(str), train_sims):
        lookup[text] = sim
    for text, sim in zip(lb_df["post"].astype(str), lb_sims):
        lookup[text] = sim
    return lookup


def get_sims_for_texts(texts, lookup):
    """Cache hit for real train/leaderboard posts; on-the-fly SBert encoding
    (lazy-loaded model) for anything else (synthetic oversample rows only)."""
    missing = [t for t in texts if t not in lookup]
    if missing:
        model = _get_sbert_model()
        from factor_descriptions import FACTOR_DESCRIPTIONS
        factor_embeds_path = os.path.join(SBERT_DIR, "factor_embeds.npy")
        factor_embeds = np.load(factor_embeds_path) if os.path.exists(factor_embeds_path) \
            else model.encode(FACTOR_DESCRIPTIONS, normalize_embeddings=True, convert_to_numpy=True)
        embeds = model.encode(missing, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        sims = embeds @ factor_embeds.T
        for t, s in zip(missing, sims):
            lookup[t] = s.astype(np.float32)
    return np.stack([lookup[t] for t in texts])


def load_train_with_risk(oversample_source=None):
    """Same as dataset.load_train, but risk_norm is normalized/lowercased
    consistently (already the case in the base loader) -- returned as-is,
    this wrapper exists so train_v2.py has one obvious import to reach for."""
    return _load_train_base(oversample_source=oversample_source)


class FactorDatasetV2(Dataset):
    def __init__(self, df, risk_labels, tokenizer, sbert_lookup, max_length=MAX_LENGTH, is_test=False):
        """df: has 'post' and (if not is_test) 'factors_list' columns.
        risk_labels: list/array of risk_norm strings, SAME LENGTH as df,
        aligned by position -- caller decides whether these are true labels
        (training split) or Sub1a-predicted labels (validation/leaderboard)."""
        assert len(risk_labels) == len(df)
        self.texts = df["post"].astype(str).tolist()
        self.risk_tokens = [RISK_TOKEN_MAP[r] for r in risk_labels]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        self.sims = get_sims_for_texts(self.texts, sbert_lookup)

        if not is_test:
            self.labels = []
            for _, row in df.iterrows():
                vec = torch.zeros(NUM_FACTORS)
                for f in row["factors_list"]:
                    fl = f.lower().strip()
                    if fl in FACTOR2ID:
                        vec[FACTOR2ID[fl]] = 1.0
                self.labels.append(vec)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = f"{self.risk_tokens[idx]} {self.texts[idx]}"
        enc = self.tokenizer(text, max_length=self.max_length, padding="max_length",
                             truncation=True, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["sbert_sims"] = torch.tensor(self.sims[idx], dtype=torch.float32)
        if not self.is_test:
            item["labels"] = self.labels[idx]
        return item

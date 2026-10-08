"""
Dataset for the cross-task-context Sub2 variant (v15): same posts/factor
labels as dataset.py, plus a per-row context feature vector = [4 risk-level
softmax probs from exp09's OOF/leaderboard predictions, has_evidence flag].

No oversampling here — exp09's OOF logits only cover the 1635 original
train rows (synthetic posts were never scored by it), so this experiment
is deliberately scoped to the plain original data, isolating the "does
cross-task context help" question from any oversample-source effects.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from config import MODEL_NAME, MAX_LENGTH, NUM_FACTORS, FACTOR2ID, TRAIN_PATH, LEADERBOARD_PATH

SEQ_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "subtask_1a_risk_classification", "results", "exp09_deberta_large_lr_high", "sequence_data",
)


def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def load_train_with_context():
    df = pd.read_excel(TRAIN_PATH, engine="openpyxl")
    df["risk_norm"] = df["suicide risk"].str.strip().str.lower()
    import ast
    df["factors_list"] = df["factors"].apply(
        lambda x: sorted(set(ast.literal_eval(x))) if pd.notna(x) else [])

    logits = np.load(os.path.join(SEQ_DATA_DIR, "train_oof_logits.npy"))
    risk_probs = softmax(logits)  # (N, 4), true OOF — never saw these rows during exp09 training
    has_evidence = (df["evidence for suicide risk level"].astype(str).str.strip().str.lower() != "none").astype(np.float32).to_numpy()

    assert len(df) == risk_probs.shape[0], "train.xlsx row count must match exp09's OOF logits"
    context = np.concatenate([risk_probs, has_evidence[:, None]], axis=1).astype(np.float32)  # (N, 5)
    return df.reset_index(drop=True), context


def load_leaderboard_with_context():
    df = pd.read_excel(LEADERBOARD_PATH, engine="openpyxl")
    logits = np.load(os.path.join(SEQ_DATA_DIR, "leaderboard_logits.npy"))
    risk_probs = softmax(logits)  # (N, 4), 5-fold ensemble average

    evidence_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "subtask_1b_evidence_extraction", "outputs", "evidence_predictions.csv",
    )
    ev_df = pd.read_csv(evidence_path)
    ev_df = ev_df.set_index("row_id").reindex(df["row_id"])
    has_evidence = (ev_df["evidence"].astype(str).str.strip().str.lower() != "none").astype(np.float32).to_numpy()

    assert len(df) == risk_probs.shape[0], "leaderboard.xlsx row count must match exp09's leaderboard logits"
    context = np.concatenate([risk_probs, has_evidence[:, None]], axis=1).astype(np.float32)  # (N, 5)
    return df.reset_index(drop=True), context


def get_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


class ContextFactorDataset(Dataset):
    def __init__(self, texts, context, labels=None, tokenizer=None, max_length=MAX_LENGTH):
        self.texts = texts
        self.context = context
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(str(self.texts[idx]), max_length=self.max_length,
                             padding="max_length", truncation=True, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["context"] = torch.tensor(self.context[idx], dtype=torch.float32)
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item


def build_label_matrix(df):
    labels = np.zeros((len(df), NUM_FACTORS), dtype=np.float32)
    for i, row in df.iterrows():
        for f in row["factors_list"]:
            fl = f.lower().strip()
            if fl in FACTOR2ID:
                labels[i, FACTOR2ID[fl]] = 1.0
    return labels

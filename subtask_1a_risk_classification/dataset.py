"""Subtask 1a: Dataset — risk classification labels."""

import ast
import json
import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from config import TRAIN_PATH, LEADERBOARD_PATH, OVERSAMPLE_PATH, MODEL_NAME, MAX_LENGTH, RISK2ID


def load_train(use_oversample=False, filter_both_agree=True):
    """Load training data with normalized risk labels."""
    df = pd.read_excel(TRAIN_PATH, engine="openpyxl")
    df["risk_norm"] = df["suicide risk"].str.strip().str.lower()

    if use_oversample:
        with open(OVERSAMPLE_PATH, "r", encoding="utf-8") as f:
            samples = json.load(f)
        rows = []
        for s in samples:
            if filter_both_agree and not s.get("both_agree", True):
                continue
            rows.append({
                "row_id": s.get("id", ""),
                "anon_user_id": "SYN_OVERSAMPLE",
                "post": s["text"],
                "risk_norm": s["risk_level"],
            })
        if rows:
            ov_df = pd.DataFrame(rows)
            df = pd.concat([df[["row_id", "anon_user_id", "post", "risk_norm"]], ov_df],
                           ignore_index=True)

    return df


def load_leaderboard():
    return pd.read_excel(LEADERBOARD_PATH, engine="openpyxl")


def get_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


class RiskDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=MAX_LENGTH, is_test=False):
        self.texts = df["post"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        if not is_test:
            self.labels = [RISK2ID[r] for r in df["risk_norm"]]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if not self.is_test:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

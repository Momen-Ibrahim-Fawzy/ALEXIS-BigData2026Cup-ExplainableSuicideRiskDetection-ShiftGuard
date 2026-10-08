"""Subtask 2: Dataset — multi-label factor binary vectors."""

import ast, json
import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from config import *


OV_FILES = {
    # Legacy Sub1a risk-level oversample files
    "gpt":            "oversample_gpt_oss.json",
    "claude":         "oversample_claude.json",
    "llama":          "oversample_llama.json",
    "all_sources":    "oversample_all_sources.json",
    "verified":       "oversample_gpt_oss.json",        # filter by both_agree
    "all":            "oversample.json",
    # Sub2 factor-targeted oversample (1,943 posts for 8 rare factors)
    "factor_gpt":     "oversample_factor_gpt.json",
    "factor_claude":  "oversample_factor_claude.json",
    "factor_llama":   "oversample_factor_llama.json",
    "factor_all":     "oversample_factor_all.json",
    "factor_verified":"oversample_factor_annotated.json",  # filter by both_detect_target
}


def load_train(oversample_source=None):
    df = pd.read_excel(TRAIN_PATH, engine="openpyxl")
    df["risk_norm"] = df["suicide risk"].str.strip().str.lower()
    df["factors_list"] = df["factors"].apply(
        lambda x: sorted(set(ast.literal_eval(x))) if pd.notna(x) else [])

    if oversample_source and oversample_source in OV_FILES:
        ov_path = os.path.join(os.path.dirname(TRAIN_PATH), OV_FILES[oversample_source])
        with open(ov_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        rows = []
        for s in samples:
            if oversample_source == "verified" and not s.get("both_agree", True):
                continue
            if oversample_source == "factor_verified" and not s.get("both_detect_target", True):
                continue
            facs = s.get("factors", [])
            if isinstance(facs, str):
                facs = [f.strip() for f in facs.split(",") if f.strip()]
            rows.append({
                "row_id": s.get("id", ""), "anon_user_id": "SYN",
                "post": s["text"], "risk_norm": s["risk_level"],
                "factors_list": facs,
            })
        if rows:
            ov = pd.DataFrame(rows)
            df = pd.concat([df[["row_id","anon_user_id","post","risk_norm","factors_list"]],
                            ov], ignore_index=True)
            print(f"  Added {len(rows)} oversampled posts (source={oversample_source})")
    return df


def load_leaderboard():
    return pd.read_excel(LEADERBOARD_PATH, engine="openpyxl")


def get_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


class FactorDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=MAX_LENGTH, is_test=False):
        self.texts = df["post"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
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
        enc = self.tokenizer(str(self.texts[idx]), max_length=self.max_length,
                             padding="max_length", truncation=True, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if not self.is_test:
            item["labels"] = self.labels[idx]
        return item

"""Subtask 1b: Dataset — BIO-tagged evidence spans aligned to tokenizer offsets."""

import ast, json
import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from config import *


OV_FILES = {
    "gpt":         "oversample_gpt_oss.json",
    "claude":      "oversample_claude.json",
    "llama":       "oversample_llama.json",
    "all_sources": "oversample_all_sources.json",
    "verified":    "oversample_gpt_oss.json",   # filter by both_agree below
    "all":         "oversample.json",            # legacy GPT-OSS file
}


def load_train(oversample_source=None):
    df = pd.read_excel(TRAIN_PATH, engine="openpyxl")
    df["risk_norm"] = df["suicide risk"].str.strip().str.lower()

    def parse_ev(x):
        if pd.isna(x) or str(x).strip().lower() == "none":
            return []
        return [s.strip() for s in str(x).split(";") if s.strip() and s.strip().lower() != "none"]
    df["evidence_list"] = df["evidence for suicide risk level"].apply(parse_ev)

    if oversample_source and oversample_source in OV_FILES:
        ov_path = os.path.join(os.path.dirname(TRAIN_PATH), OV_FILES[oversample_source])
        with open(ov_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        rows = []
        for s in samples:
            if oversample_source == "verified" and not s.get("both_agree", True):
                continue
            ev = s.get("evidence", [])
            if isinstance(ev, str):
                ev = [e.strip() for e in ev.split(";") if e.strip() and e.lower() != "none"]
            else:
                ev = [e for e in (ev or []) if e.lower() != "none"]
            rows.append({
                "row_id": s.get("id", ""),
                "anon_user_id": "SYN",
                "post": s["text"],
                "risk_norm": s["risk_level"],
                "evidence_list": ev,
            })
        if rows:
            ov_df = pd.DataFrame(rows)
            df = pd.concat([df[["row_id", "anon_user_id", "post", "risk_norm", "evidence_list"]],
                            ov_df], ignore_index=True)
            print(f"  Added {len(rows)} oversampled posts (source={oversample_source})")
    return df


def load_leaderboard():
    return pd.read_excel(LEADERBOARD_PATH, engine="openpyxl")


def get_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


class EvidenceDataset(Dataset):
    """Token-level BIO labels aligned to DeBERTa tokenizer offsets."""

    def __init__(self, df, tokenizer, max_length=MAX_LENGTH, is_test=False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row["post"])

        enc = self.tokenizer(text, max_length=self.max_length, padding="max_length",
                             truncation=True, return_tensors="pt",
                             return_offsets_mapping=True)
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"].squeeze(0)

        offsets = enc["offset_mapping"].squeeze(0)

        if self.is_test:
            item["offset_mapping"] = offsets
            return item

        # Build BIO labels
        bio = torch.zeros(self.max_length, dtype=torch.long)  # all O
        evidence_list = row.get("evidence_list", [])

        if evidence_list:
            text_lower = text.lower()
            for span_text in evidence_list:
                span_lower = span_text.lower().strip()
                if not span_lower or span_lower == "none":
                    continue
                start_char = text_lower.find(span_lower)
                if start_char == -1:
                    continue
                end_char = start_char + len(span_lower)

                first = True
                for tok_i in range(self.max_length):
                    ts, te = offsets[tok_i].tolist()
                    if ts == 0 and te == 0:
                        continue
                    if te <= start_char:
                        continue
                    if ts >= end_char:
                        break
                    bio[tok_i] = BIO2ID["B-EVD"] if first else BIO2ID["I-EVD"]
                    first = False

        item["labels"] = bio
        return item

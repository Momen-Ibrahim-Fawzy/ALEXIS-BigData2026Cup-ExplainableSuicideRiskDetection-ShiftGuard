"""
Extract per-post logits (+ user/sequence metadata) for exp09, for both the
true-OOF train set and the leaderboard set. Feeds sequence_rescore.py.

Train logits are genuine out-of-fold: each fold's own held-out checkpoint
scores only its own held-out rows, so no row is ever scored by a model that
saw it during training.

Usage:
    python extract_logits.py
"""

import os, sys, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from trainer import Classifier, TextDataset, RISK2ID, BASE_DIR, DATA_DIR, RESULTS_DIR
from experiment_config import EXPERIMENTS

EXP_NAME = "exp09_deberta_large_lr_high"
N_FOLDS = 5
OUT_DIR = os.path.join(RESULTS_DIR, EXP_NAME, "sequence_data")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = EXPERIMENTS[EXP_NAME]
    exp_dir = os.path.join(RESULTS_DIR, EXP_NAME)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    # ── Train: true OOF logits ────────────────────────────────────────────
    df = pd.read_excel(os.path.join(DATA_DIR, "train.xlsx"), engine="openpyxl")
    df["risk_norm"] = df["suicide risk"].str.strip().str.lower()
    df = df.reset_index(drop=True)

    gkf = GroupKFold(n_splits=N_FOLDS)
    groups = df["anon_user_id"].values

    oof_logits = np.zeros((len(df), 4), dtype=np.float64)
    filled = np.zeros(len(df), dtype=bool)

    for fold, (tr_i, va_i) in enumerate(gkf.split(df, groups=groups)):
        va_df = df.iloc[va_i]
        ckpt_path = os.path.join(exp_dir, f"best_fold{fold}.pt")
        print(f"  [train OOF] fold {fold}: {len(va_i)} rows <- {ckpt_path}")

        model = Classifier(cfg["model_name"]).to(device).float()
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        va_ds = TextDataset(va_df["post"].tolist(),
                             [RISK2ID[r] for r in va_df["risk_norm"]],
                             tokenizer, cfg["max_length"])
        va_ld = DataLoader(va_ds, batch_size=cfg["batch_size"] * 2, shuffle=False,
                            num_workers=2, pin_memory=True)

        logits_list = []
        with torch.no_grad():
            for batch in va_ld:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                logits = model(ids, mask)
                logits_list.append(logits.float().cpu().numpy())
        oof_logits[va_i] = np.concatenate(logits_list, axis=0)
        filled[va_i] = True
        del model
        torch.cuda.empty_cache()

    assert filled.all(), "Not all train rows received OOF logits"

    np.save(os.path.join(OUT_DIR, "train_oof_logits.npy"), oof_logits)
    df[["row_id", "anon_user_id", "post_id", "risk_norm"]].to_csv(
        os.path.join(OUT_DIR, "train_meta.csv"), index=False)
    print(f"  Saved train OOF logits: {oof_logits.shape} -> {OUT_DIR}/train_oof_logits.npy")

    # ── Leaderboard: 5-fold ensemble average logits ───────────────────────
    lb_df = pd.read_excel(os.path.join(DATA_DIR, "leaderboard.xlsx"), engine="openpyxl")
    lb_ds = TextDataset(lb_df["post"].tolist(), [0] * len(lb_df), tokenizer, cfg["max_length"])
    lb_ld = DataLoader(lb_ds, batch_size=cfg["batch_size"] * 2, shuffle=False,
                        num_workers=2, pin_memory=True)

    fold_logits = []
    for fold in range(N_FOLDS):
        ckpt_path = os.path.join(exp_dir, f"best_fold{fold}.pt")
        print(f"  [leaderboard] fold {fold} <- {ckpt_path}")
        model = Classifier(cfg["model_name"]).to(device).float()
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        logits_list = []
        with torch.no_grad():
            for batch in lb_ld:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                logits = model(ids, mask)
                logits_list.append(logits.float().cpu().numpy())
        fold_logits.append(np.concatenate(logits_list, axis=0))
        del model
        torch.cuda.empty_cache()

    lb_logits = np.mean(fold_logits, axis=0)
    np.save(os.path.join(OUT_DIR, "leaderboard_logits.npy"), lb_logits)
    lb_df[["row_id", "anon_user_id", "post_id"]].to_csv(
        os.path.join(OUT_DIR, "leaderboard_meta.csv"), index=False)
    print(f"  Saved leaderboard logits: {lb_logits.shape} -> {OUT_DIR}/leaderboard_logits.npy")


if __name__ == "__main__":
    main()

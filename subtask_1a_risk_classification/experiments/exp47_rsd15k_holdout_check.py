"""Check exp47 (RSD-15K two-stage) against the exact 10% RSD-15K slice that
stage A held out for early stopping only -- never used for gradient updates
at any point in training (not stage A's weight updates, not stage B's,
since stage B trains only on train.xlsx). A genuine generalization check
using data this checkpoint has never learned from.

Reconstructs the identical GroupShuffleSplit(test_size=0.1, random_state=42)
used in train_rsd15k_two_stage.py's stage_a() to recover the same held-out
row indices.

Usage:
    python exp47_rsd15k_holdout_check.py
"""

import os, sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(__file__))
from trainer import Classifier, TextDataset, RISK2ID, RESULTS_DIR

RSD_PATH = "../../data/RSD-15K/RSD-15K.csv"
MODEL_NAME = "microsoft/deberta-v3-large"
EXP_DIR = os.path.join(RESULTS_DIR, "exp47_rsd15k_two_stage")
MAX_LENGTH = 512
N_FOLDS = 5
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict_5fold_avg(ds):
    ld = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2)
    fold_probs = []
    for fold in range(N_FOLDS):
        ckpt_path = os.path.join(EXP_DIR, f"best_fold{fold}.pt")
        model = Classifier(MODEL_NAME).to(DEVICE).float()
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        model.eval()
        probs_this_fold = []
        for batch in ld:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids, mask)
                p = torch.softmax(logits.float(), dim=-1)
            probs_this_fold.append(p.cpu().numpy())
        fold_probs.append(np.concatenate(probs_this_fold, axis=0))
        print(f"  fold {fold} done", flush=True)
        del model
        torch.cuda.empty_cache()
    return np.mean(fold_probs, axis=0)


def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    rsd = pd.read_csv(RSD_PATH)
    rsd["risk_norm"] = rsd["sentiment"].str.strip().str.lower()
    labels = np.array([RISK2ID[r] for r in rsd["risk_norm"]])

    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
    tr_idx, va_idx = next(gss.split(rsd, groups=rsd["users"].values))
    print(f"Reconstructed stage-A holdout: {len(va_idx)} rows, never used for any gradient update")

    hold_df = rsd.iloc[va_idx]
    hold_labels = labels[va_idx]

    ds = TextDataset(hold_df["text"].astype(str).tolist(), hold_labels.tolist(), tokenizer, MAX_LENGTH)
    exp47_probs = predict_5fold_avg(ds)
    exp47_wf1 = f1_score(hold_labels, exp47_probs.argmax(1), average="weighted", zero_division=0)

    exp09_probs_full = np.load(os.path.join(RESULTS_DIR, "exp09_rsd15k_probs.npy"))
    exp09_probs_hold = exp09_probs_full[va_idx]
    exp09_wf1 = f1_score(hold_labels, exp09_probs_hold.argmax(1), average="weighted", zero_division=0)

    print(f"\n{'='*70}\n  RESULT: exp47 vs exp09 on stage-A's held-out RSD-15K slice\n{'='*70}")
    print(f"  exp09 (no RSD training at all): WF1={exp09_wf1:.4f}")
    print(f"  exp47 (RSD-15K two-stage):      WF1={exp47_wf1:.4f}")
    print(f"  delta: {exp47_wf1 - exp09_wf1:+.4f}")


if __name__ == "__main__":
    main()

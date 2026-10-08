"""Evaluate the already-trained, already-trusted exp09 (5-fold average) on
RSD-15K -- a genuinely external population (1265 users, none overlapping
with this project's 153 training users), same 4-class taxonomy
(Indicator/Ideation/Behavior/Attempt). No retraining, just inference: does
exp09's local ~0.80 WF1 hold up on a completely different population, or
does it also collapse? This is the cheapest possible test of whether the
project's real-transfer problem is about this leaderboard specifically, or
about generalizing beyond the narrow 153-user training set at all.

Usage:
    python eval_exp09_on_rsd15k.py
"""

import os, sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report

sys.path.insert(0, os.path.dirname(__file__))
from trainer import Classifier, TextDataset, RISK2ID, ID2RISK, RESULTS_DIR

RSD_PATH = "../../data/RSD-15K/RSD-15K.csv"
MODEL_NAME = "microsoft/deberta-v3-large"
EXP_DIR = os.path.join(RESULTS_DIR, "exp09_deberta_large_lr_high")
MAX_LENGTH = 512
N_FOLDS = 5
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict_5fold_avg(ds):
    ld = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    fold_probs = []
    for fold in range(N_FOLDS):
        ckpt_path = os.path.join(EXP_DIR, f"best_fold{fold}.pt")
        model = Classifier(MODEL_NAME).to(DEVICE).float()
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        probs_this_fold = []
        for batch in ld:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids, mask)
                probs = torch.softmax(logits.float(), dim=-1)
            probs_this_fold.append(probs.cpu().numpy())
        fold_probs.append(np.concatenate(probs_this_fold, axis=0))
        print(f"  fold {fold} done")
        del model
        torch.cuda.empty_cache()

    return np.mean(fold_probs, axis=0)


def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    rsd = pd.read_csv(RSD_PATH)
    rsd["risk_norm"] = rsd["sentiment"].str.strip().str.lower()
    print(f"RSD-15K: {len(rsd)} posts, {rsd['users'].nunique()} users")
    print(rsd["risk_norm"].value_counts())

    true_labels = np.array([RISK2ID[r] for r in rsd["risk_norm"]])
    ds = TextDataset(rsd["text"].astype(str).tolist(), true_labels.tolist(), tokenizer, MAX_LENGTH)

    probs = predict_5fold_avg(ds)
    preds = probs.argmax(axis=1)

    wf1 = f1_score(true_labels, preds, average="weighted", zero_division=0)
    mf1 = f1_score(true_labels, preds, average="macro", zero_division=0)
    acc = (preds == true_labels).mean()

    print(f"\n{'='*70}")
    print(f"  exp09 (5-fold avg, trained on 1635 posts / 153 users) on RSD-15K")
    print(f"  ({len(rsd)} posts / {rsd['users'].nunique()} users, ZERO overlap)")
    print(f"{'='*70}")
    print(f"  WF1:  {wf1:.4f}   (vs {0.8014:.4f} on this project's own true-OOF)")
    print(f"  MF1:  {mf1:.4f}")
    print(f"  Acc:  {acc:.4f}")
    print()
    print(classification_report(true_labels, preds, target_names=["indicator", "ideation", "behavior", "attempt"], zero_division=0))

    np.save(os.path.join(RESULTS_DIR, "exp09_rsd15k_probs.npy"), probs)
    rsd[["users", "text", "sentiment"]].assign(exp09_pred=[ID2RISK[i] for i in preds]).to_csv(
        os.path.join(RESULTS_DIR, "exp09_rsd15k_predictions.csv"), index=False)
    print(f"\nSaved probs/predictions to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()

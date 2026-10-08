"""Joint (mixed) training on train.xlsx + RSD-15K, as a different mechanism
from exp47's two-stage transfer learning (which regressed for real, -0.0089,
despite the best true-OOF margin of the session). exp47's own RSD-15K
holdout check found the likely cause: after stage B's sequential fine-tuning
on train.xlsx alone, the model scored WORSE than plain exp09 on RSD-15K's
held-out slice -- meaning stage B overwrote whatever cross-population
generalization stage A's RSD-15K exposure built, rather than RSD-15K itself
being unhelpful.

This trains on BOTH datasets in every single epoch (balanced sampling, since
RSD-15K is ~9x train.xlsx's size and would otherwise dominate the gradient),
so the model never gets the chance to forget RSD-15K's diversity the way
sequential fine-tuning did. Same 5-fold GroupKFold split on train.xlsx as
exp09, for direct true-OOF comparability; RSD-15K's 1,265 users don't
overlap with train.xlsx's 153, so including all of it in every fold's
training set carries no leakage risk.

Usage:
    python train_rsd15k_joint.py
"""

import os, sys, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(__file__))
from trainer import Classifier, TextDataset, RISK2ID, DATA_DIR, RESULTS_DIR

RSD_PATH = "../../data/RSD-15K/RSD-15K.csv"
MODEL_NAME = "microsoft/deberta-v3-large"
EXP_NAME = "exp49_rsd15k_joint"
OUT_DIR = os.path.join(RESULTS_DIR, EXP_NAME)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 512
N_FOLDS = 5
SEED = 42

LR = 2e-5
EPOCHS = 15
PATIENCE = 3
BATCH_SIZE = 16
LABEL_SMOOTHING = 0.05
CLASS_WEIGHTS = [0.669, 0.788, 1.045, 3.586]  # exp09's, kept identical for comparability


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, labels_all = [], []
    for batch in loader:
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids, mask)
        preds.append(logits.float().argmax(-1).cpu().numpy())
        labels_all.append(batch["labels"].numpy())
    preds = np.concatenate(preds)
    labels_all = np.concatenate(labels_all)
    return f1_score(labels_all, preds, average="weighted", zero_division=0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    df = pd.read_excel(os.path.join(DATA_DIR, "train.xlsx"), engine="openpyxl")
    df["risk_norm"] = df["suicide risk"].str.strip().str.lower()
    df = df.reset_index(drop=True)
    labels = np.array([RISK2ID[r] for r in df["risk_norm"]])
    groups = df["anon_user_id"].values

    rsd = pd.read_csv(RSD_PATH)
    rsd["risk_norm"] = rsd["sentiment"].str.strip().str.lower()
    rsd_labels = np.array([RISK2ID[r] for r in rsd["risk_norm"]])
    print(f"train.xlsx: {len(df)} rows, RSD-15K: {len(rsd)} rows (joint every epoch)")

    gkf = GroupKFold(n_splits=N_FOLDS)
    class_w = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(DEVICE)
    fold_wf1s = []

    rsd_ds_full = TextDataset(rsd["text"].astype(str).tolist(), rsd_labels.tolist(), tokenizer, MAX_LENGTH)

    for fold, (tr_i, va_i) in enumerate(gkf.split(df, groups=groups)):
        print(f"\n{'#'*70}\n  FOLD {fold+1}/{N_FOLDS}\n{'#'*70}")
        tr_df, va_df = df.iloc[tr_i], df.iloc[va_i]

        tr_ds = TextDataset(tr_df["post"].tolist(), labels[tr_i].tolist(), tokenizer, MAX_LENGTH)
        va_ds = TextDataset(va_df["post"].tolist(), labels[va_i].tolist(), tokenizer, MAX_LENGTH)

        combined = ConcatDataset([tr_ds, rsd_ds_full])
        # balanced sampling: each epoch, train.xlsx and RSD-15K rows are equally
        # likely to be drawn, regardless of RSD-15K being ~9x larger -- keeps
        # RSD-15K present in every epoch without letting it dominate the gradient
        n_tr, n_rsd = len(tr_ds), len(rsd_ds_full)
        sample_weights = np.concatenate([
            np.full(n_tr, 1.0 / n_tr),
            np.full(n_rsd, 1.0 / n_rsd),
        ])
        sampler = WeightedRandomSampler(sample_weights, num_samples=2 * n_tr, replacement=True)
        tr_ld = DataLoader(combined, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)
        va_ld = DataLoader(va_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2)

        torch.manual_seed(SEED + fold)
        model = Classifier(MODEL_NAME).to(DEVICE).float()
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        total_steps = len(tr_ld) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.06), total_steps)
        criterion = nn.CrossEntropyLoss(weight=class_w, label_smoothing=LABEL_SMOOTHING)

        best_wf1, best_state, patience_ctr = 0, None, 0
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0.0
            for batch in tr_ld:
                ids = batch["input_ids"].to(DEVICE)
                mask = batch["attention_mask"].to(DEVICE)
                lbls = batch["labels"].to(DEVICE)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(ids, mask)
                    loss = criterion(logits.float(), lbls)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            val_wf1 = evaluate(model, va_ld)
            print(f"  epoch {epoch+1}: train_loss={total_loss/len(tr_ld):.4f} val_wf1={val_wf1:.4f}", flush=True)
            if val_wf1 > best_wf1:
                best_wf1, patience_ctr = val_wf1, 0
                best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE:
                    print(f"  early stop at epoch {epoch+1}")
                    break

        torch.save(best_state, os.path.join(OUT_DIR, f"best_fold{fold}.pt"))
        print(f"  fold {fold} best val WF1: {best_wf1:.4f}")
        fold_wf1s.append(best_wf1)
        del model
        torch.cuda.empty_cache()

    mean_wf1 = float(np.mean(fold_wf1s))
    std_wf1 = float(np.std(fold_wf1s))
    print(f"\n{'='*70}\n  {EXP_NAME}: WF1 = {mean_wf1:.4f} +/- {std_wf1:.4f}\n"
          f"  (vs exp09's 0.8003+/-0.0241, vs exp47 two-stage's 0.8177+/-0.0387)\n{'='*70}")
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump({"experiment": EXP_NAME, "mean_wf1": mean_wf1, "std_wf1": std_wf1,
                    "fold_wf1": fold_wf1s}, f, indent=2)


if __name__ == "__main__":
    main()

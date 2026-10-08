"""Two-stage transfer learning: pretrain the Sub1a classifier on RSD-15K
(14,613 posts, 1,265 users -- 9x the competition's own training set, same
4-class taxonomy), then fine-tune on the competition's own 1,635-post
train.xlsx with the SAME 5-fold GroupKFold split exp09 uses, for a directly
comparable true-OOF number.

Different in kind from domain-adaptive MLM pretraining (which came back flat,
0.7992 vs 0.8003): this transfers actual task-relevant LABELED signal, not
just unlabeled language modeling, and reuses the RSD-15K-trained
classification head too (valid here since it's the exact same task/labels,
unlike MLM's head which had to be discarded).

Stage A: train on RSD-15K with its own held-out validation split (by user)
for early stopping. Class weights recomputed for RSD-15K's own distribution
(Ideation 48.8%, Indicator 31.6%, Behavior 14.1%, Attempt 5.5%).

Stage B: for each of train.xlsx's 5 GroupKFold folds, initialize from stage
A's full checkpoint (backbone + head), fine-tune with a SHORTER, LOWER-LR
schedule than exp09's from-scratch recipe -- standard practice when starting
from an already-converged checkpoint rather than raw pretrained weights, to
avoid overwriting the RSD-15K-learned signal too aggressively on a much
smaller (≈1,300-row) per-fold training set.

Usage:
    python train_rsd15k_two_stage.py            # stage A + stage B
    python train_rsd15k_two_stage.py --stage-b-only   # reuse existing stage-A checkpoint
"""

import os, sys, json, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(__file__))
from trainer import Classifier, TextDataset, RISK2ID, DATA_DIR, RESULTS_DIR

RSD_PATH = "../../data/RSD-15K/RSD-15K.csv"
MODEL_NAME = "microsoft/deberta-v3-large"
EXP_NAME = "exp47_rsd15k_two_stage"
OUT_DIR = os.path.join(RESULTS_DIR, EXP_NAME)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 512
SEED = 42

# Stage A (RSD-15K pretrain)
STAGE_A_LR = 2e-5
STAGE_A_EPOCHS = 15
STAGE_A_PATIENCE = 3
STAGE_A_BATCH = 16
LABEL_SMOOTHING = 0.05

# Stage B (fine-tune on train.xlsx, starting from stage A's checkpoint)
STAGE_B_LR = 7e-6          # lower than exp09's 2e-5 -- already-converged start point
STAGE_B_EPOCHS = 20
STAGE_B_PATIENCE = 4
STAGE_B_BATCH = 16
N_FOLDS = 5


def compute_class_weights(labels, n_classes=4):
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    inv_freq = counts.sum() / (n_classes * counts)
    return inv_freq / inv_freq.mean()


def run_epoch_train(model, loader, optimizer, scheduler, criterion):
    model.train()
    total_loss = 0.0
    for batch in loader:
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids, mask)
            loss = criterion(logits.float(), labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


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
    wf1 = f1_score(labels_all, preds, average="weighted", zero_division=0)
    return wf1


def stage_a(tokenizer):
    print(f"\n{'#'*70}\n  STAGE A: pretrain on RSD-15K\n{'#'*70}")
    rsd = pd.read_csv(RSD_PATH)
    rsd["risk_norm"] = rsd["sentiment"].str.strip().str.lower()
    labels = np.array([RISK2ID[r] for r in rsd["risk_norm"]])

    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
    tr_idx, va_idx = next(gss.split(rsd, groups=rsd["users"].values))
    print(f"  RSD-15K: {len(tr_idx)} train / {len(va_idx)} val (user-grouped)")

    class_weights = compute_class_weights(labels[tr_idx])
    print(f"  RSD-15K class weights (indicator, ideation, behavior, attempt): {class_weights.round(3)}")

    tr_ds = TextDataset(rsd["text"].astype(str).values[tr_idx].tolist(), labels[tr_idx].tolist(), tokenizer, MAX_LENGTH)
    va_ds = TextDataset(rsd["text"].astype(str).values[va_idx].tolist(), labels[va_idx].tolist(), tokenizer, MAX_LENGTH)
    tr_ld = DataLoader(tr_ds, batch_size=STAGE_A_BATCH, shuffle=True, num_workers=2)
    va_ld = DataLoader(va_ds, batch_size=STAGE_A_BATCH * 2, shuffle=False, num_workers=2)

    torch.manual_seed(SEED)
    model = Classifier(MODEL_NAME).to(DEVICE).float()
    optimizer = AdamW(model.parameters(), lr=STAGE_A_LR, weight_decay=0.01)
    total_steps = len(tr_ld) * STAGE_A_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.06), total_steps)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32).to(DEVICE),
        label_smoothing=LABEL_SMOOTHING,
    )

    best_wf1, best_state, patience_ctr = 0, None, 0
    for epoch in range(STAGE_A_EPOCHS):
        train_loss = run_epoch_train(model, tr_ld, optimizer, scheduler, criterion)
        val_wf1 = evaluate(model, va_ld)
        print(f"  epoch {epoch+1}: train_loss={train_loss:.4f} val_wf1={val_wf1:.4f}", flush=True)
        if val_wf1 > best_wf1:
            best_wf1, patience_ctr = val_wf1, 0
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= STAGE_A_PATIENCE:
                print(f"  early stop at epoch {epoch+1}")
                break

    os.makedirs(OUT_DIR, exist_ok=True)
    ckpt_path = os.path.join(OUT_DIR, "stage_a_rsd15k.pt")
    torch.save(best_state, ckpt_path)
    print(f"  Stage A best val WF1: {best_wf1:.4f} -- saved {ckpt_path}")
    return ckpt_path


def stage_b(tokenizer, stage_a_ckpt):
    print(f"\n{'#'*70}\n  STAGE B: fine-tune on train.xlsx (5-fold, same split as exp09)\n{'#'*70}")
    df = pd.read_excel(os.path.join(DATA_DIR, "train.xlsx"), engine="openpyxl")
    df["risk_norm"] = df["suicide risk"].str.strip().str.lower()
    df = df.reset_index(drop=True)
    labels = np.array([RISK2ID[r] for r in df["risk_norm"]])
    class_weights = [0.669, 0.788, 1.045, 3.586]  # exp09's, kept identical for comparability
    groups = df["anon_user_id"].values
    gkf = GroupKFold(n_splits=N_FOLDS)

    fold_wf1s = []
    for fold, (tr_i, va_i) in enumerate(gkf.split(df, groups=groups)):
        print(f"\n  --- fold {fold} ---")
        tr_df, va_df = df.iloc[tr_i], df.iloc[va_i]
        tr_ds = TextDataset(tr_df["post"].tolist(), labels[tr_i].tolist(), tokenizer, MAX_LENGTH)
        va_ds = TextDataset(va_df["post"].tolist(), labels[va_i].tolist(), tokenizer, MAX_LENGTH)
        tr_ld = DataLoader(tr_ds, batch_size=STAGE_B_BATCH, shuffle=True, num_workers=2)
        va_ld = DataLoader(va_ds, batch_size=STAGE_B_BATCH * 2, shuffle=False, num_workers=2)

        torch.manual_seed(SEED + fold)
        model = Classifier(MODEL_NAME).to(DEVICE).float()
        model.load_state_dict(torch.load(stage_a_ckpt, map_location=DEVICE, weights_only=True))

        optimizer = AdamW(model.parameters(), lr=STAGE_B_LR, weight_decay=0.01)
        total_steps = len(tr_ld) * STAGE_B_EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.06), total_steps)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32).to(DEVICE),
            label_smoothing=LABEL_SMOOTHING,
        )

        best_wf1, best_state, patience_ctr = 0, None, 0
        for epoch in range(STAGE_B_EPOCHS):
            train_loss = run_epoch_train(model, tr_ld, optimizer, scheduler, criterion)
            val_wf1 = evaluate(model, va_ld)
            print(f"    epoch {epoch+1}: train_loss={train_loss:.4f} val_wf1={val_wf1:.4f}", flush=True)
            if val_wf1 > best_wf1:
                best_wf1, patience_ctr = val_wf1, 0
                best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            else:
                patience_ctr += 1
                if patience_ctr >= STAGE_B_PATIENCE:
                    print(f"    early stop at epoch {epoch+1}")
                    break

        torch.save(best_state, os.path.join(OUT_DIR, f"best_fold{fold}.pt"))
        print(f"  fold {fold} best val WF1: {best_wf1:.4f}")
        fold_wf1s.append(best_wf1)
        del model
        torch.cuda.empty_cache()

    mean_wf1 = float(np.mean(fold_wf1s))
    std_wf1 = float(np.std(fold_wf1s))
    print(f"\n{'='*70}\n  exp47_rsd15k_two_stage: WF1 = {mean_wf1:.4f} +/- {std_wf1:.4f}\n"
          f"  (vs exp09's 0.8003+/-0.0241, vs exp46 domain-adaptive's 0.7992+/-0.0322)\n{'='*70}")
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump({"experiment": EXP_NAME, "mean_wf1": mean_wf1, "std_wf1": std_wf1,
                    "fold_wf1": fold_wf1s}, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-b-only", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    os.makedirs(OUT_DIR, exist_ok=True)

    stage_a_ckpt = os.path.join(OUT_DIR, "stage_a_rsd15k.pt")
    if args.stage_b_only:
        assert os.path.exists(stage_a_ckpt), f"missing {stage_a_ckpt}"
    else:
        stage_a_ckpt = stage_a(tokenizer)

    stage_b(tokenizer, stage_a_ckpt)


if __name__ == "__main__":
    main()

"""
Train the CORAL dual-head Sub1a classifier (backlog item #2), isolated from
trainer.py's 24 existing experiments to avoid any risk to exp09/exp14.

Same hyperparameters as exp09 (best known config: DeBERTa-v3-large, lr=2e-5,
weighted CE with label smoothing 0.05, linear warmup, 5-fold GroupKFold,
patience=4) — the ONLY change is the model head (standard 4-way softmax ->
standard + CORAL dual head) and loss (CE -> CE + CORAL BCE). Isolating this
one variable lets us tell whether ordinal-aware training helps on its own,
before combining with the already-validated user-sequence re-scoring.

Usage:
    python train_coral.py
"""

import os, sys, csv, json, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(__file__))
from trainer import TextDataset, RISK2ID, ID2RISK, RISK_LEVELS, BASE_DIR, DATA_DIR, RESULTS_DIR, SEED
from coral_model import CoralDualHeadClassifier

import pandas as pd

EXP_NAME = "exp25_deberta_coral_ordinal"
MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LENGTH = 512
LR = 2e-5
LABEL_SMOOTHING = 0.05
CLASS_WEIGHTS = [0.669, 0.788, 1.045, 3.586]
EPOCHS = 32
BATCH_SIZE = 16
GRAD_ACCUM = 1
WARMUP_RATIO = 0.06
PATIENCE = 4
N_FOLDS = 5
CORAL_LOSS_WEIGHT = 1.0  # combined_loss = CE + CORAL_LOSS_WEIGHT * coral_bce

EXP_DIR = os.path.join(RESULTS_DIR, EXP_NAME)
os.makedirs(EXP_DIR, exist_ok=True)


def load_data():
    df = pd.read_excel(os.path.join(DATA_DIR, "train.xlsx"), engine="openpyxl")
    df["risk_norm"] = df["suicide risk"].str.strip().str.lower()
    return df.reset_index(drop=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    df = load_data()
    print(f"  Data: {len(df)} posts, {N_FOLDS} folds")

    log_path = os.path.join(EXP_DIR, "training_log.csv")
    log_fields = ["fold", "epoch", "train_loss", "val_loss", "val_wf1", "val_mf1",
                  "best_wf1", "lr", "elapsed"]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=log_fields).writeheader()

    gkf = GroupKFold(n_splits=N_FOLDS)
    groups = df["anon_user_id"].values
    class_w = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(device)
    all_wf1 = []

    for fold, (tr_i, va_i) in enumerate(gkf.split(df, groups=groups)):
        tr_df, va_df = df.iloc[tr_i], df.iloc[va_i]
        print(f"\n  Fold {fold+1}/{N_FOLDS}  Train={len(tr_df)} Val={len(va_df)}")

        tr_ds = TextDataset(tr_df["post"].tolist(), [RISK2ID[r] for r in tr_df["risk_norm"]],
                             tokenizer, MAX_LENGTH)
        va_ds = TextDataset(va_df["post"].tolist(), [RISK2ID[r] for r in va_df["risk_norm"]],
                             tokenizer, MAX_LENGTH)
        tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        va_ld = DataLoader(va_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2, pin_memory=True)

        model = CoralDualHeadClassifier(MODEL_NAME).to(device).float()
        model.backbone.gradient_checkpointing_enable()

        ce_criterion = nn.CrossEntropyLoss(weight=class_w, label_smoothing=LABEL_SMOOTHING)
        coral_criterion = nn.BCEWithLogitsLoss()

        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        total_steps = len(tr_ld) * EPOCHS // GRAD_ACCUM
        warmup_steps = int(total_steps * WARMUP_RATIO)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        use_amp = device.type == "cuda"
        best_wf1, patience_counter, best_epoch = 0, 0, 0

        for epoch in range(EPOCHS):
            t0 = time.time()
            model.train()
            train_loss, steps = 0, 0
            for batch in tr_ld:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    standard_logits, coral_logits = model(ids, mask)

                ce_loss = ce_criterion(standard_logits.float(), labels)
                coral_targets = model.coral_targets(labels, num_classes=4)
                coral_loss = coral_criterion(coral_logits.float(), coral_targets)
                loss = (ce_loss + CORAL_LOSS_WEIGHT * coral_loss) / GRAD_ACCUM

                loss.backward()
                if (steps + 1) % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                train_loss += loss.item() * GRAD_ACCUM
                steps += 1
            train_loss /= max(steps, 1)

            model.eval()
            all_preds, all_labels, val_loss = [], [], 0
            with torch.no_grad():
                for batch in va_ld:
                    ids = batch["input_ids"].to(device)
                    mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    standard_logits, coral_logits = model(ids, mask)
                    ce_loss = ce_criterion(standard_logits.float(), labels)
                    coral_targets = model.coral_targets(labels, num_classes=4)
                    coral_loss = coral_criterion(coral_logits.float(), coral_targets)
                    val_loss += (ce_loss + CORAL_LOSS_WEIGHT * coral_loss).item()

                    probs = model.combined_probs(standard_logits.float(), coral_logits.float())
                    all_preds.extend(probs.argmax(-1).cpu().tolist())
                    all_labels.extend(labels.cpu().tolist())

            n = max(len(all_preds), 1)
            val_loss /= n
            ps = [ID2RISK[p] for p in all_preds]
            ls = [ID2RISK[l] for l in all_labels]
            wf1 = f1_score(ls, ps, labels=RISK_LEVELS, average="weighted", zero_division=0)
            mf1 = f1_score(ls, ps, labels=RISK_LEVELS, average="macro", zero_division=0)
            elapsed = time.time() - t0

            is_best = wf1 > best_wf1
            if is_best:
                best_wf1, best_epoch, patience_counter = wf1, epoch + 1, 0
                torch.save(model.state_dict(), os.path.join(EXP_DIR, f"best_fold{fold}.pt"))
            else:
                patience_counter += 1

            lr_now = scheduler.get_last_lr()[0]
            print(f"    E{epoch+1:02d} ({elapsed:.0f}s) loss={train_loss:.4f}/{val_loss:.4f} "
                  f"WF1={wf1:.4f} MF1={mf1:.4f}{'  **BEST**' if is_best else ''}")

            with open(log_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=log_fields).writerow({
                    "fold": fold, "epoch": epoch + 1, "train_loss": f"{train_loss:.4f}",
                    "val_loss": f"{val_loss:.4f}", "val_wf1": f"{wf1:.4f}", "val_mf1": f"{mf1:.4f}",
                    "best_wf1": f"{best_wf1:.4f}", "lr": f"{lr_now:.2e}", "elapsed": f"{elapsed:.0f}"})

            if patience_counter >= PATIENCE:
                print(f"    Early stopping at epoch {epoch+1} (patience={PATIENCE})")
                break

        print(f"  Fold {fold+1}: best WF1={best_wf1:.4f} at epoch {best_epoch}")
        all_wf1.append(best_wf1)
        del model, optimizer, scheduler
        torch.cuda.empty_cache()

    mean_wf1, std_wf1 = np.mean(all_wf1), np.std(all_wf1)
    print(f"\n  {EXP_NAME}: WF1 = {mean_wf1:.4f} +/- {std_wf1:.4f}")
    summary = {
        "experiment": EXP_NAME, "model": MODEL_NAME, "mean_wf1": float(mean_wf1),
        "std_wf1": float(std_wf1), "fold_wf1": [float(w) for w in all_wf1],
    }
    with open(os.path.join(EXP_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

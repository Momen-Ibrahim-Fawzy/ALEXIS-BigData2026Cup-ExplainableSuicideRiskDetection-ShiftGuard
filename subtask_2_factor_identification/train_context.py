"""
Train the cross-task-context Sub2 model (v15): FactorClassifierWithContext,
injecting exp09's predicted risk-level probabilities + a has_evidence flag
as an auxiliary feature. Plain BCEWithLogitsLoss (no pos_weight) — isolating
the "does cross-task context help" question from the oversample-source and
loss-weighting questions already explored elsewhere.

Usage:
    python train_context.py
"""

import os, csv, json, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

from config import *
from dataset_context import (load_train_with_context, get_tokenizer,
                              ContextFactorDataset, build_label_matrix)
from model_context import FactorClassifierWithContext

OUTPUT_DIR_CTX = os.path.join(BASE_DIR, "outputs_v15_context")
os.makedirs(OUTPUT_DIR_CTX, exist_ok=True)

EPOCHS = 32
PATIENCE = 4

LOG_FIELDS = ["fold", "epoch", "train_loss", "val_loss", "val_macro_f1",
              "best_mf1", "lr", "elapsed"]


def compute_macro_f1(probs, labels, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    per_factor = [f1_score(labels[:, i], preds[:, i], zero_division=0) for i in range(NUM_FACTORS)]
    return float(np.mean(per_factor)), per_factor


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer()

    df, context = load_train_with_context()
    labels_all = build_label_matrix(df)
    print(f"  Data: {len(df)} posts, {N_FOLDS} folds, context_dim={context.shape[1]}")

    log_path = os.path.join(OUTPUT_DIR_CTX, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=LOG_FIELDS).writeheader()

    gkf = GroupKFold(n_splits=N_FOLDS)
    groups = df["anon_user_id"].values

    oof_probs = np.zeros((len(df), NUM_FACTORS))
    completed = {}

    for fold, (tr_i, va_i) in enumerate(gkf.split(df, groups=groups)):
        print(f"\n  Fold {fold+1}/{N_FOLDS}  Train={len(tr_i)} Val={len(va_i)}")

        tr_ds = ContextFactorDataset(df["post"].values[tr_i], context[tr_i], labels_all[tr_i], tokenizer, MAX_LENGTH)
        va_ds = ContextFactorDataset(df["post"].values[va_i], context[va_i], labels_all[va_i], tokenizer, MAX_LENGTH)
        tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        va_ld = DataLoader(va_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2, pin_memory=True)

        model = FactorClassifierWithContext(MODEL_NAME).to(device).float()
        model.backbone.gradient_checkpointing_enable()

        criterion = nn.BCEWithLogitsLoss()
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        total_steps = len(tr_ld) * EPOCHS
        warmup_steps = int(total_steps * WARMUP_RATIO)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        use_amp = device.type == "cuda"
        best_mf1, patience_counter, best_epoch = 0, 0, 0
        fold_dir = os.path.join(OUTPUT_DIR_CTX, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        for epoch in range(EPOCHS):
            t0 = time.time()
            model.train()
            train_loss, steps = 0, 0
            for batch in tr_ld:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                ctx = batch["context"].to(device)
                labels = batch["labels"].to(device)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    logits = model(ids, mask, ctx)
                loss = criterion(logits.float(), labels)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                steps += 1
            train_loss /= max(steps, 1)

            model.eval()
            val_probs, val_labels, val_loss = [], [], 0
            with torch.no_grad():
                for batch in va_ld:
                    ids = batch["input_ids"].to(device)
                    mask = batch["attention_mask"].to(device)
                    ctx = batch["context"].to(device)
                    labels = batch["labels"].to(device)
                    logits = model(ids, mask, ctx)
                    val_loss += criterion(logits.float(), labels).item()
                    val_probs.append(torch.sigmoid(logits.float()).cpu().numpy())
                    val_labels.append(labels.cpu().numpy())
            val_loss /= max(len(va_ld), 1)
            val_probs = np.concatenate(val_probs)
            val_labels = np.concatenate(val_labels)
            mf1, _ = compute_macro_f1(val_probs, val_labels)

            elapsed = time.time() - t0
            is_best = mf1 > best_mf1
            if is_best:
                best_mf1, best_epoch, patience_counter = mf1, epoch + 1, 0
                torch.save({"model": model.state_dict()}, os.path.join(fold_dir, "best.pt"))
            else:
                patience_counter += 1

            lr_now = scheduler.get_last_lr()[0]
            print(f"    E{epoch+1:02d} ({elapsed:.0f}s) loss={train_loss:.4f}/{val_loss:.4f} "
                  f"MacroF1={mf1:.4f}{'  **BEST**' if is_best else ''}")

            with open(log_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=LOG_FIELDS).writerow({
                    "fold": fold, "epoch": epoch + 1, "train_loss": f"{train_loss:.4f}",
                    "val_loss": f"{val_loss:.4f}", "val_macro_f1": f"{mf1:.4f}",
                    "best_mf1": f"{best_mf1:.4f}", "lr": f"{lr_now:.2e}", "elapsed": f"{elapsed:.0f}"})

            if patience_counter >= PATIENCE:
                print(f"    Early stopping at epoch {epoch+1} (patience={PATIENCE})")
                break

        print(f"  Fold {fold+1}: best MacroF1={best_mf1:.4f} at epoch {best_epoch}")
        completed[str(fold)] = {"macro_f1": best_mf1}
        with open(os.path.join(OUTPUT_DIR_CTX, "completed_folds.json"), "w") as f:
            json.dump(completed, f, indent=2)

        ckpt = torch.load(os.path.join(fold_dir, "best.pt"), map_location=device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        probs_list = []
        with torch.no_grad():
            for batch in va_ld:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                ctx = batch["context"].to(device)
                logits = model(ids, mask, ctx)
                probs_list.append(torch.sigmoid(logits.float()).cpu().numpy())
        oof_probs[va_i] = np.concatenate(probs_list)

        del model, optimizer, scheduler
        torch.cuda.empty_cache()

    np.save(os.path.join(OUTPUT_DIR_CTX, "oof_probs.npy"), oof_probs)
    np.save(os.path.join(OUTPUT_DIR_CTX, "oof_labels.npy"), labels_all)
    mean_mf1, _ = compute_macro_f1(oof_probs, labels_all)
    print(f"\n  Context-aware Sub2 true OOF Macro F1 (threshold=0.5): {mean_mf1:.4f}")


if __name__ == "__main__":
    main()

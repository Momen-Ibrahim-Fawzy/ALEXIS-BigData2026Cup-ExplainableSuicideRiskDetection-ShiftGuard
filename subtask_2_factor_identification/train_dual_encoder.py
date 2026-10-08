"""
Train the label-semantic dual-encoder Sub2 model (v14 / backlog item,
inspired by DEXML, ICLR 2024). See dual_encoder_model.py for the
architecture and loss rationale.

Same base hyperparameters as the rest of Sub2 (DeBERTa-v3-large, lr=2e-5,
5-fold GroupKFold, patience=4) — only the model architecture (dual-encoder
scoring against factor-description embeddings instead of a plain
classification head) and loss (DecoupledSoftmax instead of BCE) differ.
This is a single unified model — no vanilla/weighted split needed, since
DecoupledSoftmax already handles co-occurring-positive interference
differently from the imbalance problem the split-ensemble addresses.

Usage:
    python train_dual_encoder.py
"""

import os, csv, json, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

from config import *
from dataset import load_train, get_tokenizer, FactorDataset
from dual_encoder_model import DualEncoderFactorModel, decoupled_softmax_loss
from factor_descriptions import FACTOR_DESCRIPTIONS

OUTPUT_DIR_DE = os.path.join(BASE_DIR, "outputs_v14_dual_encoder")
os.makedirs(OUTPUT_DIR_DE, exist_ok=True)

EMBED_DIM = 256
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

    factor_enc = tokenizer(FACTOR_DESCRIPTIONS, max_length=64, padding="max_length",
                            truncation=True, return_tensors="pt")
    factor_ids = factor_enc["input_ids"].to(device)
    factor_mask = factor_enc["attention_mask"].to(device)

    df = load_train(oversample_source=None)
    orig_df = df[df["anon_user_id"] != "SYN"].reset_index(drop=True)
    print(f"  Data: {len(orig_df)} posts, {N_FOLDS} folds, {NUM_FACTORS} factors")

    log_path = os.path.join(OUTPUT_DIR_DE, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=LOG_FIELDS).writeheader()

    gkf = GroupKFold(n_splits=N_FOLDS)
    groups = orig_df["anon_user_id"].values

    oof_probs = np.zeros((len(orig_df), NUM_FACTORS))
    oof_labels = np.zeros((len(orig_df), NUM_FACTORS))
    completed = {}

    for fold, (tr_i, va_i) in enumerate(gkf.split(orig_df, groups=groups)):
        tr_df, va_df = orig_df.iloc[tr_i], orig_df.iloc[va_i]
        print(f"\n  Fold {fold+1}/{N_FOLDS}  Train={len(tr_df)} Val={len(va_df)}")

        tr_ds = FactorDataset(tr_df, tokenizer, MAX_LENGTH)
        va_ds = FactorDataset(va_df, tokenizer, MAX_LENGTH)
        tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        va_ld = DataLoader(va_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2, pin_memory=True)

        model = DualEncoderFactorModel(MODEL_NAME, embed_dim=EMBED_DIM).to(device).float()
        model.backbone.gradient_checkpointing_enable()

        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        total_steps = len(tr_ld) * EPOCHS
        warmup_steps = int(total_steps * WARMUP_RATIO)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        use_amp = device.type == "cuda"
        best_mf1, patience_counter, best_epoch = 0, 0, 0
        fold_dir = os.path.join(OUTPUT_DIR_DE, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        for epoch in range(EPOCHS):
            t0 = time.time()
            model.train()
            train_loss, steps = 0, 0
            for batch in tr_ld:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    scores = model(ids, mask, factor_ids, factor_mask)
                loss = decoupled_softmax_loss(scores.float(), labels)

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
                    labels = batch["labels"].to(device)
                    scores = model(ids, mask, factor_ids, factor_mask)
                    val_loss += decoupled_softmax_loss(scores.float(), labels).item()
                    val_probs.append(torch.sigmoid(scores.float()).cpu().numpy())
                    val_labels.append(labels.cpu().numpy())
            val_loss /= max(len(va_ld), 1)
            val_probs = np.concatenate(val_probs)
            val_labels = np.concatenate(val_labels)
            mf1, _ = compute_macro_f1(val_probs, val_labels)

            elapsed = time.time() - t0
            is_best = mf1 > best_mf1
            if is_best:
                best_mf1, best_epoch, patience_counter = mf1, epoch + 1, 0
                torch.save({"model": model.state_dict(), "fold": fold, "epoch": epoch}, os.path.join(fold_dir, "best.pt"))
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
        with open(os.path.join(OUTPUT_DIR_DE, "completed_folds.json"), "w") as f:
            json.dump(completed, f, indent=2)

        # Re-load best checkpoint and fill OOF for this fold's held-out rows
        ckpt = torch.load(os.path.join(fold_dir, "best.pt"), map_location=device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        probs_list = []
        with torch.no_grad():
            for batch in va_ld:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                scores = model(ids, mask, factor_ids, factor_mask)
                probs_list.append(torch.sigmoid(scores.float()).cpu().numpy())
        oof_probs[va_i] = np.concatenate(probs_list)
        oof_labels[va_i] = va_df.apply(
            lambda row: [1.0 if f in row["factors_list"] else 0.0 for f in FACTOR_LIST], axis=1
        ).tolist()

        del model, optimizer, scheduler
        torch.cuda.empty_cache()

    np.save(os.path.join(OUTPUT_DIR_DE, "oof_probs.npy"), oof_probs)
    np.save(os.path.join(OUTPUT_DIR_DE, "oof_labels.npy"), oof_labels)
    mean_mf1, _ = compute_macro_f1(oof_probs, oof_labels)
    print(f"\n  Dual-encoder true OOF Macro F1 (threshold=0.5): {mean_mf1:.4f}")


if __name__ == "__main__":
    main()

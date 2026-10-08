"""Idea 1: seed-bag v20's validated NLI cross-encoder recipe (exact same
architecture/data/epoch caps as train_nli_finetune_kfold.py -- 3 frozen + 3
unfrozen epochs -- only the random seed differs). Fully independent file,
shares no code path with any currently-running job.

Rationale: seed-bagging is the safest lever established this project (same
architecture/data, only init/shuffling/dropout differ) -- but it was never
cleanly isolated on its own merits before. The one prior seed-bagging
attempt (factor_verified, v18) combined it WITH cross-source ensembling in
the same submission, so we don't actually know whether seed-bagging alone
would have generalized; the failure could have come from either ingredient.
The cross-encoder (v20, real Subtask2 0.6156) is a much stronger base than
what seed-bagging was tested on before, and GroupKFold splits are identical
across seeds (deterministic on anon_user_id, independent of the torch seed)
so this is a clean, leakage-free comparison.

Usage:
    python train_nli_seedbag.py --seed 101 --output-dir outputs_nli_seedbag_101
    python train_nli_seedbag.py --seed 202 --output-dir outputs_nli_seedbag_202
"""

import argparse, ast, os, time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import GroupKFold

from config import *
from dataset_nli_finetune import build_pairs, NLIPairDataset
from model_nli_finetune import NLIFactorClassifier, NLI_MODEL
from train_nli_finetune import run_epoch, pairs_to_matrix, compute_macro_f1

MAX_LENGTH = 256
BATCH_SIZE = 16
FROZEN_EPOCHS = 3    # matches v20's validated recipe exactly
UNFROZEN_EPOCHS = 3  # matches v20's validated recipe exactly
FROZEN_LR = 1e-4
UNFROZEN_LR = 1e-5
WARMUP_RATIO = 0.1
PATIENCE = 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--fold-ids", type=str, default=None)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--seed", type=int, required=True,
                     help="Different from config.SEED (42, used by v20) -- GroupKFold splits are "
                          "unaffected (deterministic on anon_user_id), only init/shuffling/dropout differ.")
    args = ap.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    tag = args.fold_ids.replace(",", "-") if args.fold_ids else "all"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Output: {output_dir}  Seed: {args.seed}  Fold subset: {args.fold_ids or 'all'}")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    df = pd.read_excel(TRAIN_PATH, engine="openpyxl")
    df["factors_list"] = df["factors"].apply(lambda x: sorted(set(ast.literal_eval(x))) if pd.notna(x) else [])
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)

    # GroupKFold split itself is deterministic on anon_user_id -- identical across seeds
    gkf = GroupKFold(n_splits=args.folds)
    groups = df["anon_user_id"].values
    all_splits = list(gkf.split(df, groups=groups))
    fold_ids = [int(x) for x in args.fold_ids.split(",")] if args.fold_ids else list(range(args.folds))

    oof_probs = np.zeros((len(df), NUM_FACTORS), dtype=np.float32)
    oof_filled = np.zeros(len(df), dtype=bool)
    oof_labels = np.zeros((len(df), NUM_FACTORS), dtype=np.float32)
    for i, row in df.iterrows():
        for f in row["factors_list"]:
            fl = f.lower().strip()
            if fl in FACTOR2ID:
                oof_labels[i, FACTOR2ID[fl]] = 1.0

    for fold in fold_ids:
        tr_i, va_i = all_splits[fold]
        tr_df = df.iloc[tr_i].reset_index(drop=False)
        va_df = df.iloc[va_i].reset_index(drop=False)
        print(f"\n{'='*60}\n  FOLD {fold+1}/{args.folds}  Train={len(tr_df)} posts  Val={len(va_df)} posts\n{'='*60}")

        train_pairs = build_pairs(tr_df)
        val_pairs = build_pairs(va_df)

        train_ds = NLIPairDataset(train_pairs, tokenizer, MAX_LENGTH)
        val_ds = NLIPairDataset(val_pairs, tokenizer, MAX_LENGTH)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2)

        model = NLIFactorClassifier().to(device).float()
        model.model.gradient_checkpointing_enable()

        log_rows = []

        print(f"\n  -- Stage A: frozen backbone, {FROZEN_EPOCHS} epochs --")
        model.freeze_backbone()
        optimizer = AdamW(model.trainable_parameters(), lr=FROZEN_LR, weight_decay=0.01)
        total_steps = len(train_loader) * FROZEN_EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * WARMUP_RATIO), total_steps)

        best_mf1, best_state, patience_ctr = 0.0, None, 0
        for epoch in range(FROZEN_EPOCHS):
            t0 = time.time()
            train_loss, *_ = run_epoch(model, train_loader, optimizer, scheduler, device, train=True)
            _, scores, labels_flat, row_idx, factor_idx = run_epoch(model, val_loader, None, None, device, train=False)
            val_probs = pairs_to_matrix(scores, row_idx, factor_idx, len(va_df))
            val_labels = pairs_to_matrix(labels_flat, row_idx, factor_idx, len(va_df))
            mf1 = compute_macro_f1(val_probs, val_labels, np.full(NUM_FACTORS, 0.5))
            elapsed = time.time() - t0
            print(f"  [A] Epoch {epoch+1}/{FROZEN_EPOCHS} ({elapsed:.0f}s) loss={train_loss:.4f} val_MacroF1(t=0.5)={mf1:.4f}")
            log_rows.append({"fold": fold, "stage": "A", "epoch": epoch + 1, "train_loss": train_loss, "val_mf1_t05": mf1})
            if mf1 > best_mf1:
                best_mf1, patience_ctr = mf1, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE:
                    print("  Early stopping Stage A"); break

        print(f"\n  -- Stage B: unfrozen, {UNFROZEN_EPOCHS} epochs --")
        model.unfreeze_backbone()
        optimizer = AdamW(model.trainable_parameters(), lr=UNFROZEN_LR, weight_decay=0.01)
        total_steps = len(train_loader) * UNFROZEN_EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * WARMUP_RATIO), total_steps)

        patience_ctr = 0
        for epoch in range(UNFROZEN_EPOCHS):
            t0 = time.time()
            train_loss, *_ = run_epoch(model, train_loader, optimizer, scheduler, device, train=True)
            _, scores, labels_flat, row_idx, factor_idx = run_epoch(model, val_loader, None, None, device, train=False)
            val_probs = pairs_to_matrix(scores, row_idx, factor_idx, len(va_df))
            val_labels = pairs_to_matrix(labels_flat, row_idx, factor_idx, len(va_df))
            mf1 = compute_macro_f1(val_probs, val_labels, np.full(NUM_FACTORS, 0.5))
            elapsed = time.time() - t0
            print(f"  [B] Epoch {epoch+1}/{UNFROZEN_EPOCHS} ({elapsed:.0f}s) loss={train_loss:.4f} val_MacroF1(t=0.5)={mf1:.4f}")
            log_rows.append({"fold": fold, "stage": "B", "epoch": epoch + 1, "train_loss": train_loss, "val_mf1_t05": mf1})
            if mf1 > best_mf1:
                best_mf1, patience_ctr = mf1, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE:
                    print("  Early stopping Stage B"); break

        model.load_state_dict(best_state)
        fold_dir = os.path.join(output_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        torch.save({"model": model.state_dict()}, os.path.join(fold_dir, "best.pt"))
        pd.DataFrame(log_rows).to_csv(os.path.join(fold_dir, "training_log.csv"), index=False)

        _, final_scores, _, final_row_idx, final_factor_idx = run_epoch(model, val_loader, None, None, device, train=False)
        final_probs = pairs_to_matrix(final_scores, final_row_idx, final_factor_idx, len(va_df))
        oof_probs[va_i] = final_probs
        oof_filled[va_i] = True
        print(f"  Fold {fold+1} final val MacroF1(t=0.5): {best_mf1:.4f}")

        del model
        torch.cuda.empty_cache()

    partial_path = os.path.join(output_dir, f"oof_probs_folds_{tag}.npy")
    np.save(partial_path, oof_probs)
    np.save(os.path.join(output_dir, f"oof_filled_folds_{tag}.npy"), oof_filled)
    np.save(os.path.join(output_dir, "oof_labels.npy"), oof_labels)
    df[["row_id", "anon_user_id"]].to_csv(os.path.join(output_dir, "oof_meta.csv"), index=False)
    print(f"\nSaved partial OOF ({oof_filled.sum()}/{len(df)} rows) -> {partial_path}")


if __name__ == "__main__":
    main()

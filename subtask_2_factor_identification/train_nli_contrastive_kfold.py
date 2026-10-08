"""Idea 2: v20's exact validated 5-fold NLI cross-encoder recipe, plus a
small-weight auxiliary supervised contrastive loss (PairSupConLoss,
losses_v2.py) on the pooled representation -- isolated on its own this
time, unlike the earlier "v2" architecture which combined contrastive
learning with Distribution-Balanced Loss and a special-token mechanism all
at once and failed; the diagnosis then pointed at DB Loss's untuned
hyperparameters and/or the special tokens as the more likely culprits, but
contrastive learning itself was never cleanly tested alone.

Total loss = BCEWithLogitsLoss(entailment_logit, label)
             + CONTRASTIVE_WEIGHT * PairSupConLoss(pooled_embedding, label)

CONTRASTIVE_WEIGHT is kept small (0.1) and fixed, not swept -- this project
has already seen one architecture fail from an untuned auxiliary loss
dominating the primary objective (DB Loss in "v2"), so the auxiliary term
is deliberately capped low enough that even a fully wrong contrastive
signal can only perturb, not overwhelm, the main entailment loss.

Fully independent from every other run: separate model file
(model_nli_contrastive.py), separate output directory, no shared model
state or in-flight edits to any currently-running job's files.

Usage (single GPU, all 5 folds):
    python train_nli_contrastive_kfold.py --output-dir outputs_nli_contrastive_kfold

Usage (2 GPUs in parallel):
    CUDA_VISIBLE_DEVICES=0 python train_nli_contrastive_kfold.py --fold-ids 0,1,2 --output-dir outputs_nli_contrastive_kfold &
    CUDA_VISIBLE_DEVICES=1 python train_nli_contrastive_kfold.py --fold-ids 3,4 --output-dir outputs_nli_contrastive_kfold &
    wait
    python merge_nli_contrastive_folds.py --output-dir outputs_nli_contrastive_kfold
"""

import argparse, ast, os, time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from config import *
from dataset_nli_finetune import build_pairs, NLIPairDataset
from model_nli_contrastive import NLIContrastiveClassifier, NLI_MODEL
from train_nli_finetune import pairs_to_matrix, compute_macro_f1
from losses_v2 import PairSupConLoss

MAX_LENGTH = 256
BATCH_SIZE = 16
FROZEN_EPOCHS = 3    # matches v20's validated recipe exactly
UNFROZEN_EPOCHS = 3  # matches v20's validated recipe exactly
FROZEN_LR = 1e-4
UNFROZEN_LR = 1e-5
WARMUP_RATIO = 0.1
PATIENCE = 2
CONTRASTIVE_WEIGHT = 0.1  # small, fixed -- see module docstring


def run_epoch_contrastive(model, loader, optimizer, scheduler, device, contrastive_fn, train=True):
    model.train() if train else model.eval()
    total_loss, total_bce, total_con, steps = 0.0, 0.0, 0.0, 0
    all_scores, all_labels, all_row_idx, all_factor_idx = [], [], [], []
    bce_fn = torch.nn.BCEWithLogitsLoss()
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        pbar = tqdm(loader, desc="  train" if train else "  eval", leave=False, dynamic_ncols=True)
        for batch in pbar:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            ttype = batch.get("token_type_ids")
            if ttype is not None: ttype = ttype.to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logit, embedding = model(ids, mask, ttype, return_embedding=True)
            bce_loss = bce_fn(logit.float(), labels)
            con_loss = contrastive_fn(embedding.float(), labels) if train else embedding.sum() * 0.0
            loss = bce_loss + CONTRASTIVE_WEIGHT * con_loss

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            total_bce += bce_loss.item()
            total_con += con_loss.item()
            steps += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", bce=f"{bce_loss.item():.4f}", con=f"{con_loss.item():.4f}")

            all_scores.append(torch.sigmoid(logit.float()).detach().cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_row_idx.append(batch["row_idx"].numpy())
            all_factor_idx.append(batch["factor_idx"].numpy())
    pbar.close()
    return (total_loss / max(steps, 1), total_bce / max(steps, 1), total_con / max(steps, 1),
            np.concatenate(all_scores), np.concatenate(all_labels),
            np.concatenate(all_row_idx), np.concatenate(all_factor_idx))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--fold-ids", type=str, default=None)
    ap.add_argument("--output-dir", type=str, required=True)
    args = ap.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    tag = args.fold_ids.replace(",", "-") if args.fold_ids else "all"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Output: {output_dir}  Fold subset: {args.fold_ids or 'all'}  "
          f"Contrastive weight: {CONTRASTIVE_WEIGHT}")
    torch.manual_seed(SEED); np.random.seed(SEED)

    df = pd.read_excel(TRAIN_PATH, engine="openpyxl")
    df["factors_list"] = df["factors"].apply(lambda x: sorted(set(ast.literal_eval(x))) if pd.notna(x) else [])
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    contrastive_fn = PairSupConLoss(temperature=0.1)

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

        model = NLIContrastiveClassifier().to(device).float()
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
            train_loss, train_bce, train_con, *_ = run_epoch_contrastive(
                model, train_loader, optimizer, scheduler, device, contrastive_fn, train=True)
            _, _, _, scores, labels_flat, row_idx, factor_idx = run_epoch_contrastive(
                model, val_loader, None, None, device, contrastive_fn, train=False)
            val_probs = pairs_to_matrix(scores, row_idx, factor_idx, len(va_df))
            val_labels = pairs_to_matrix(labels_flat, row_idx, factor_idx, len(va_df))
            mf1 = compute_macro_f1(val_probs, val_labels, np.full(NUM_FACTORS, 0.5))
            elapsed = time.time() - t0
            print(f"  [A] Epoch {epoch+1}/{FROZEN_EPOCHS} ({elapsed:.0f}s) loss={train_loss:.4f} "
                  f"(bce={train_bce:.4f} con={train_con:.4f}) val_MacroF1(t=0.5)={mf1:.4f}")
            log_rows.append({"fold": fold, "stage": "A", "epoch": epoch + 1, "train_loss": train_loss,
                              "train_bce": train_bce, "train_con": train_con, "val_mf1_t05": mf1})
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
            train_loss, train_bce, train_con, *_ = run_epoch_contrastive(
                model, train_loader, optimizer, scheduler, device, contrastive_fn, train=True)
            _, _, _, scores, labels_flat, row_idx, factor_idx = run_epoch_contrastive(
                model, val_loader, None, None, device, contrastive_fn, train=False)
            val_probs = pairs_to_matrix(scores, row_idx, factor_idx, len(va_df))
            val_labels = pairs_to_matrix(labels_flat, row_idx, factor_idx, len(va_df))
            mf1 = compute_macro_f1(val_probs, val_labels, np.full(NUM_FACTORS, 0.5))
            elapsed = time.time() - t0
            print(f"  [B] Epoch {epoch+1}/{UNFROZEN_EPOCHS} ({elapsed:.0f}s) loss={train_loss:.4f} "
                  f"(bce={train_bce:.4f} con={train_con:.4f}) val_MacroF1(t=0.5)={mf1:.4f}")
            log_rows.append({"fold": fold, "stage": "B", "epoch": epoch + 1, "train_loss": train_loss,
                              "train_bce": train_bce, "train_con": train_con, "val_mf1_t05": mf1})
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

        _, _, _, final_scores, _, final_row_idx, final_factor_idx = run_epoch_contrastive(
            model, val_loader, None, None, device, contrastive_fn, train=False)
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
    print(f"Once all fold subsets are done, run: python merge_nli_contrastive_folds.py --output-dir {output_dir}")


if __name__ == "__main__":
    main()

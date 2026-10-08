"""Train the ordinal cumulative-NLI reformulation of Subtask 1a, full 5-fold
GroupKFold, warm-started per-fold from Sub2's own fine-tuned NLI checkpoint
and fed Sub2's per-factor entailment probabilities as auxiliary input. See
model_nli_ordinal.py for the full rationale.

The GroupKFold split here is verified byte-identical to Sub2's and Sub1b's
(same train.xlsx, same anon_user_id column, same deterministic algorithm --
checked directly: every fold's validation user set matches exactly), so
warm-starting fold i from Sub2 fold i's checkpoint is leak-free: that
checkpoint never trained on this fold's validation users.

Usage (single GPU, all 5 folds):
    python train_nli_ordinal_kfold.py --output-dir outputs_nli_ordinal

Usage (2 GPUs in parallel):
    CUDA_VISIBLE_DEVICES=0 python train_nli_ordinal_kfold.py --fold-ids 0,1,2 --output-dir outputs_nli_ordinal &
    CUDA_VISIBLE_DEVICES=1 python train_nli_ordinal_kfold.py --fold-ids 3,4 --output-dir outputs_nli_ordinal &
    wait
    python merge_nli_ordinal_folds.py --output-dir outputs_nli_ordinal
"""

import argparse, os, time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # subtask_1a_risk_classification
BASELINES_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.xlsx")

SUB2_DIR = os.path.join(BASELINES_DIR, "subtask_2_factor_identification")
SUB2_WARMSTART_DIR = os.path.join(SUB2_DIR, "outputs_nli_finetune_kfold_moreepochs")  # v22, per-fold checkpoints
SUB2_AUX_OOF_PATH = os.path.join(SUB2_DIR, "outputs_nli_moreepochs_seedbag_combined", "oof_probs.npy")  # v24

from risk_hypotheses import RISK_LEVELS, THRESHOLD_NAMES, NUM_THRESHOLDS, decode_ordinal
from dataset_nli_ordinal import build_pairs, NLIOrdinalDataset, pairs_to_matrix
from model_nli_ordinal import NLIOrdinalRiskClassifier, NLI_MODEL, NUM_FACTORS

MAX_LENGTH = 256
BATCH_SIZE = 16
FROZEN_EPOCHS = 3
UNFROZEN_EPOCHS = 5
FROZEN_LR = 1e-4
UNFROZEN_LR = 1e-5
WARMUP_RATIO = 0.1
PATIENCE = 2
SEED = 42
N_FOLDS = 5


def run_epoch(model, loader, optimizer, scheduler, device, train=True):
    model.train() if train else model.eval()
    total_loss, steps = 0.0, 0
    all_scores, all_labels, all_row_idx, all_thresh_idx = [], [], [], []
    criterion = torch.nn.BCEWithLogitsLoss()
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        pbar = tqdm(loader, desc="  train" if train else "  eval", leave=False, dynamic_ncols=True)
        for batch in pbar:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            aux = batch["aux_features"].to(device)
            ttype = batch.get("token_type_ids")
            if ttype is not None: ttype = ttype.to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logit = model(ids, mask, aux, ttype)
            loss = criterion(logit.float(), labels)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            steps += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            all_scores.append(torch.sigmoid(logit.float()).detach().cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_row_idx.append(batch["row_idx"].numpy())
            all_thresh_idx.append(batch["thresh_idx"].numpy())
    pbar.close()
    return (total_loss / max(steps, 1), np.concatenate(all_scores), np.concatenate(all_labels),
            np.concatenate(all_row_idx), np.concatenate(all_thresh_idx))


def compute_wf1(prob_matrix, label_matrix):
    pred_ranks = [decode_ordinal(row >= 0.5) for row in prob_matrix]
    true_ranks = [decode_ordinal(row >= 0.5) for row in label_matrix]
    ps = [RISK_LEVELS[r] for r in pred_ranks]
    ls = [RISK_LEVELS[r] for r in true_ranks]
    return f1_score(ls, ps, labels=RISK_LEVELS, average="weighted", zero_division=0)


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
    print(f"Device: {device}  Output: {output_dir}  Fold subset: {args.fold_ids or 'all'}")
    torch.manual_seed(SEED); np.random.seed(SEED)

    df = pd.read_excel(TRAIN_PATH, engine="openpyxl")
    df["risk_norm"] = df["suicide risk"].str.strip().str.lower()

    if not os.path.exists(SUB2_AUX_OOF_PATH):
        raise FileNotFoundError(
            f"\n\nMissing Sub2 auxiliary factor-probability file: {SUB2_AUX_OOF_PATH}\n"
            f"This is small (~157KB, a (1635, 24) float array) and load-bearing for the "
            f"'aux features' design (point 3): the model's classifier head is shaped "
            f"hidden+{NUM_FACTORS} and cannot run without it.\n"
            f"subtask_2_factor_identification/ was deleted to free disk space -- if you have "
            f"a backup, the two files needed here are:\n"
            f"  outputs_nli_moreepochs_seedbag_combined/oof_probs.npy\n"
            f"  outputs_nli_moreepochs_seedbag_combined/leaderboard_probs.npy (needed later, by predict)\n"
            f"Restore just those two (tiny) files to subtask_2_factor_identification/"
            f"outputs_nli_moreepochs_seedbag_combined/ and re-run.")
    aux_oof = np.load(SUB2_AUX_OOF_PATH).astype(np.float32)
    assert aux_oof.shape == (len(df), NUM_FACTORS), \
        f"Sub2 aux OOF shape {aux_oof.shape} != expected ({len(df)}, {NUM_FACTORS})"
    print(f"Loaded Sub2 aux OOF factor probs: {aux_oof.shape}")

    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)

    gkf = GroupKFold(n_splits=args.folds)
    groups = df["anon_user_id"].values
    all_splits = list(gkf.split(df, groups=groups))
    fold_ids = [int(x) for x in args.fold_ids.split(",")] if args.fold_ids else list(range(args.folds))

    oof_probs = np.zeros((len(df), NUM_THRESHOLDS), dtype=np.float32)
    oof_labels = np.zeros((len(df), NUM_THRESHOLDS), dtype=np.float32)
    oof_filled = np.zeros(len(df), dtype=bool)
    from risk_hypotheses import RISK2ORDINAL, THRESHOLDS
    for i, row in df.iterrows():
        ordinal = RISK2ORDINAL[row["risk_norm"]]
        for t_idx, (_, min_ordinal, _) in enumerate(THRESHOLDS):
            oof_labels[i, t_idx] = 1.0 if ordinal >= min_ordinal else 0.0

    for fold in fold_ids:
        tr_i, va_i = all_splits[fold]
        tr_df = df.iloc[tr_i].reset_index(drop=False)  # keeps original index as column "index"
        va_df = df.iloc[va_i].reset_index(drop=False)
        print(f"\n{'='*60}\n  FOLD {fold+1}/{args.folds}  Train={len(tr_df)} posts  Val={len(va_df)} posts\n{'='*60}")

        train_pairs = build_pairs(tr_df, global_idx_col="index")
        val_pairs = build_pairs(va_df, global_idx_col="index")
        print(f"  Pairs: train={len(train_pairs)}  val={len(val_pairs)}")

        train_ds = NLIOrdinalDataset(train_pairs, tokenizer, MAX_LENGTH, aux_oof)
        val_ds = NLIOrdinalDataset(val_pairs, tokenizer, MAX_LENGTH, aux_oof)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2)

        model = NLIOrdinalRiskClassifier().to(device).float()
        warmstart_path = os.path.join(SUB2_WARMSTART_DIR, f"fold_{fold}", "best.pt")
        if os.path.exists(warmstart_path):
            n_copied = model.load_sub2_warm_start(warmstart_path, device)
            print(f"  Warm-started {n_copied} backbone/pooler tensors from Sub2 fold {fold}: {warmstart_path}")
        else:
            print(f"  WARNING: Sub2 warm-start checkpoint not found ({warmstart_path}) -- "
                  f"using generic off-the-shelf {NLI_MODEL} instead.")
        model.deberta.gradient_checkpointing_enable()

        best_wf1, best_state, patience_ctr = 0.0, None, 0

        print(f"\n  -- Stage A: frozen backbone, {FROZEN_EPOCHS} epochs --")
        model.freeze_backbone()
        optimizer = AdamW(model.trainable_parameters(), lr=FROZEN_LR, weight_decay=0.01)
        total_steps = len(train_loader) * FROZEN_EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * WARMUP_RATIO), total_steps)

        for epoch in range(FROZEN_EPOCHS):
            t0 = time.time()
            train_loss, *_ = run_epoch(model, train_loader, optimizer, scheduler, device, train=True)
            _, scores, labels_flat, row_idx, thresh_idx = run_epoch(model, val_loader, None, None, device, train=False)
            val_probs = pairs_to_matrix(scores, row_idx, thresh_idx, len(va_df))
            val_labels = pairs_to_matrix(labels_flat, row_idx, thresh_idx, len(va_df))
            wf1 = compute_wf1(val_probs, val_labels)
            elapsed = time.time() - t0
            print(f"  [A] Epoch {epoch+1}/{FROZEN_EPOCHS} ({elapsed:.0f}s) loss={train_loss:.4f} val_WF1={wf1:.4f}")
            if wf1 > best_wf1:
                best_wf1, patience_ctr = wf1, 0
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
            _, scores, labels_flat, row_idx, thresh_idx = run_epoch(model, val_loader, None, None, device, train=False)
            val_probs = pairs_to_matrix(scores, row_idx, thresh_idx, len(va_df))
            val_labels = pairs_to_matrix(labels_flat, row_idx, thresh_idx, len(va_df))
            wf1 = compute_wf1(val_probs, val_labels)
            elapsed = time.time() - t0
            print(f"  [B] Epoch {epoch+1}/{UNFROZEN_EPOCHS} ({elapsed:.0f}s) loss={train_loss:.4f} val_WF1={wf1:.4f}")
            if wf1 > best_wf1:
                best_wf1, patience_ctr = wf1, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE:
                    print("  Early stopping Stage B"); break

        model.load_state_dict(best_state)
        fold_dir = os.path.join(output_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        torch.save({"model": model.state_dict()}, os.path.join(fold_dir, "best.pt"))

        _, final_scores, _, final_row_idx, final_thresh_idx = run_epoch(model, val_loader, None, None, device, train=False)
        final_probs = pairs_to_matrix(final_scores, final_row_idx, final_thresh_idx, len(va_df))
        oof_probs[va_i] = final_probs
        oof_filled[va_i] = True
        print(f"  Fold {fold+1} final val WF1: {best_wf1:.4f}")

        del model
        torch.cuda.empty_cache()

    partial_path = os.path.join(output_dir, f"oof_probs_folds_{tag}.npy")
    np.save(partial_path, oof_probs)
    np.save(os.path.join(output_dir, f"oof_filled_folds_{tag}.npy"), oof_filled)
    np.save(os.path.join(output_dir, "oof_labels.npy"), oof_labels)
    df[["row_id", "anon_user_id"]].to_csv(os.path.join(output_dir, "oof_meta.csv"), index=False)
    print(f"\nSaved partial OOF ({oof_filled.sum()}/{len(df)} rows) -> {partial_path}")
    print(f"Once all fold subsets are done, run: python merge_nli_ordinal_folds.py --output-dir {output_dir}")


if __name__ == "__main__":
    main()

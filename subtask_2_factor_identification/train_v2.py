"""Sub2 v2 training: DeBERTa-v3-large + special risk-level tokens + SBert
similarity features + Distribution-Balanced Loss + weighted multi-label
contrastive auxiliary loss. See model_v2.py / dataset_v2.py / losses_v2.py
for the four components' individual rationale.

GroupKFold over the FULL 153 train users (matches every other Sub2
experiment's convention, so the resulting OOF number is directly comparable
to existing baselines: 0.5806 factor_verified, 0.5982 current best 2-way
ensemble, 0.6065 v18's naive OOF that FAILED on the real leaderboard at
0.5473 -- see pseudo_leaderboard_split.py, which is the number to actually
trust, not the naive GroupKFold OOF alone).

Supports --fold-ids to train only a subset of the 5 folds (e.g. "0,1,2"),
so two GPU processes can split the 5 folds between them and run in parallel
-- each writes fold checkpoints plus a uniquely-named partial OOF file
(no shared-file write conflicts), combined afterward by merge_v2_folds.py.

Checkpoints save model weights only (no optimizer/scheduler state) --
~1.7GB/fold instead of ~4.9GB, the leaner pattern already used by
train_chain.py/train_gcn.py after the disk-space incident earlier this
project.

Usage (single GPU, all 5 folds):
    python train_v2.py --oversample factor_verified --output-dir outputs_v2_factor_verified

Usage (2 GPUs in parallel):
    CUDA_VISIBLE_DEVICES=0 python train_v2.py --oversample factor_verified \\
        --fold-ids 0,1,2 --output-dir outputs_v2_factor_verified &
    CUDA_VISIBLE_DEVICES=1 python train_v2.py --oversample factor_verified \\
        --fold-ids 3,4 --output-dir outputs_v2_factor_verified &
    wait
    python merge_v2_folds.py --output-dir outputs_v2_factor_verified
"""

import os, csv, json, time, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from tqdm import tqdm

from config import *
from dataset_v2 import FactorDatasetV2, load_train_with_risk, load_sbert_lookup
from model_v2 import FactorClassifierV2, add_risk_tokens, init_new_token_embeddings
from losses_v2 import DistributionBalancedLoss, WeightedMultiLabelContrastiveLoss

LOG_FIELDS = ["fold", "epoch", "train_loss", "cls_loss", "con_loss", "val_macro_f1", "best_mf1", "lr", "elapsed"]


def init_log(output_dir, tag):
    path = os.path.join(output_dir, f"training_log_{tag}.csv")
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=LOG_FIELDS).writeheader()
    return path


def log_row(path, row):
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=LOG_FIELDS).writerow(row)


def compute_macro_f1(probs, labels, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    return float(np.mean([f1_score(labels[:, i], preds[:, i], zero_division=0) for i in range(NUM_FACTORS)]))


def get_sub1a_oof_risk_map():
    """row_id -> Sub1a's OOF-predicted risk label (argmax of exp09's true
    OOF logits) -- used for the VALIDATION split's risk token, never the
    training split's (see dataset_v2.py docstring)."""
    seq_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "subtask_1a_risk_classification", "results",
                            "exp09_deberta_large_lr_high", "sequence_data")
    meta = pd.read_csv(os.path.join(seq_dir, "train_meta.csv"))
    logits = np.load(os.path.join(seq_dir, "train_oof_logits.npy"))
    preds = logits.argmax(axis=1)
    risk_levels = ["indicator", "ideation", "behavior", "attempt"]
    return {rid: risk_levels[p] for rid, p in zip(meta["row_id"], preds)}


def train_epoch(model, loader, optimizer, scheduler, cls_criterion, con_criterion,
                 con_weight, device):
    model.train()
    total, total_cls, total_con, steps = 0.0, 0.0, 0.0, 0
    pbar = tqdm(loader, desc="  train", leave=False, unit="batch", dynamic_ncols=True)
    for batch in pbar:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        sims = batch["sbert_sims"].to(device)
        ttype = batch.get("token_type_ids")
        if ttype is not None: ttype = ttype.to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, pooled = model(ids, mask, sims, ttype)
        cls_loss = cls_criterion(logits.float(), labels)
        con_loss = con_criterion(pooled.float(), labels) if con_weight > 0 else torch.tensor(0.0, device=device)
        loss = cls_loss + con_weight * con_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total += loss.item(); total_cls += cls_loss.item(); total_con += con_loss.item()
        steps += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", cls=f"{cls_loss.item():.4f}", con=f"{con_loss.item():.4f}")
    pbar.close()
    n = max(steps, 1)
    return total / n, total_cls / n, total_con / n


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"]
        sims = batch["sbert_sims"].to(device)
        ttype = batch.get("token_type_ids")
        if ttype is not None: ttype = ttype.to(device)
        logits, _ = model(ids, mask, sims, ttype)
        all_probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        all_labels.append(labels.numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    return probs, labels, compute_macro_f1(probs, labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--fold-ids", type=str, default=None,
                        help="Comma-separated subset of fold indices to train, e.g. '0,1,2' "
                             "-- for splitting the 5 folds across 2 GPU processes.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--oversample", type=str, default=None,
                        choices=["factor_gpt", "factor_claude", "factor_llama",
                                 "factor_all", "factor_verified"])
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--contrastive-weight", type=float, default=0.3)
    parser.add_argument("--db-focal", action="store_true",
                        help="Enable DB Loss's optional focal-style modulation. Off by "
                             "default -- this project's own focal-loss experiment underperformed "
                             "weighted-BCE for Sub2 (0.5619 vs 0.5806), see BACKLOG.md.")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    tag = args.fold_ids.replace(",", "-") if args.fold_ids else "all"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Output: {output_dir}  Fold subset: {args.fold_ids or 'all'}")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    log_path = init_log(output_dir, tag)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    add_risk_tokens(tokenizer)  # deterministic -- predict_v2.py must mirror this exactly

    sbert_lookup = load_sbert_lookup()
    df = load_train_with_risk(oversample_source=args.oversample)
    print(f"Dataset: {len(df)} | Oversample: {args.oversample or 'none'}")

    orig_df = df[df["anon_user_id"] != "SYN"].reset_index(drop=True)
    syn_df = df[df["anon_user_id"] == "SYN"].reset_index(drop=True)

    sub1a_oof_risk = get_sub1a_oof_risk_map()
    missing = orig_df.loc[~orig_df["row_id"].isin(sub1a_oof_risk.keys())]
    if len(missing):
        raise ValueError(f"{len(missing)} train rows have no Sub1a OOF risk prediction -- "
                          f"re-run subtask_1a_risk_classification/experiments/extract_logits.py first.")

    gkf = GroupKFold(n_splits=args.folds)
    groups = orig_df["anon_user_id"].values
    all_splits = list(gkf.split(orig_df, groups=groups))

    fold_ids = [int(x) for x in args.fold_ids.split(",")] if args.fold_ids else list(range(args.folds))

    oof_probs = np.zeros((len(orig_df), NUM_FACTORS), dtype=np.float32)
    oof_filled = np.zeros(len(orig_df), dtype=bool)

    for fold in fold_ids:
        tr_i, va_i = all_splits[fold]
        tr_df = pd.concat([orig_df.iloc[tr_i], syn_df], ignore_index=True) if len(syn_df) else orig_df.iloc[tr_i].reset_index(drop=True)
        va_df = orig_df.iloc[va_i].reset_index(drop=True)
        print(f"\n{'='*60}\n  FOLD {fold+1}/{args.folds}  Train={len(tr_df)} Val={len(va_df)}\n{'='*60}")

        tr_risk = tr_df["risk_norm"].tolist()  # TRUE labels for training (incl. synthetic's own labels)
        va_risk = [sub1a_oof_risk[rid] for rid in va_df["row_id"]]  # Sub1a's OOF predictions for validation

        tr_ds = FactorDatasetV2(tr_df, tr_risk, tokenizer, sbert_lookup)
        va_ds = FactorDatasetV2(va_df, va_risk, tokenizer, sbert_lookup)
        tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2, pin_memory=True)

        model = FactorClassifierV2(tokenizer=tokenizer).to(device).float()
        model.backbone.gradient_checkpointing_enable()

        class_freq = torch.stack(tr_ds.labels).sum(0).to(device)
        cls_criterion = DistributionBalancedLoss(class_freq, torch.tensor(float(len(tr_df))),
                                                   use_focal=args.db_focal).to(device)
        con_criterion = WeightedMultiLabelContrastiveLoss().to(device)

        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        total_steps = len(tr_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * WARMUP_RATIO), total_steps)

        best, patience_counter, best_state = 0.0, 0, None
        for epoch in range(args.epochs):
            t0 = time.time()
            t_loss, c_loss, con_loss = train_epoch(model, tr_loader, optimizer, scheduler,
                                                     cls_criterion, con_criterion,
                                                     args.contrastive_weight, device)
            _, _, val_mf1 = eval_epoch(model, va_loader, device)
            elapsed = time.time() - t0
            is_best = val_mf1 > best
            if is_best:
                best, patience_counter = val_mf1, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
            lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch+1}/{args.epochs} ({elapsed:.0f}s) lr={lr:.2e}  "
                  f"loss={t_loss:.4f} (cls={c_loss:.4f} con={con_loss:.4f})  "
                  f"MacroF1={val_mf1:.4f}{'  **BEST**' if is_best else ''}")
            log_row(log_path, {"fold": fold, "epoch": epoch + 1, "train_loss": f"{t_loss:.4f}",
                     "cls_loss": f"{c_loss:.4f}", "con_loss": f"{con_loss:.4f}",
                     "val_macro_f1": f"{val_mf1:.4f}", "best_mf1": f"{best:.4f}",
                     "lr": f"{lr:.2e}", "elapsed": f"{elapsed:.0f}"})
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        model.load_state_dict(best_state)
        fold_dir = os.path.join(output_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        torch.save({"model": model.state_dict()}, os.path.join(fold_dir, "best.pt"))

        final_probs, final_labels, final_mf1 = eval_epoch(model, va_loader, device)
        oof_probs[va_i] = final_probs
        oof_filled[va_i] = True
        print(f"  Fold {fold+1} final: MacroF1={final_mf1:.4f}")

        del model
        torch.cuda.empty_cache()

    # Uniquely-named partial file -- safe for concurrent processes covering
    # disjoint fold subsets to write into the same output_dir simultaneously.
    partial_path = os.path.join(output_dir, f"oof_probs_folds_{tag}.npy")
    np.save(partial_path, oof_probs)
    np.save(os.path.join(output_dir, f"oof_filled_folds_{tag}.npy"), oof_filled)
    print(f"\nSaved partial OOF ({oof_filled.sum()}/{len(orig_df)} rows) -> {partial_path}")
    print("If training all folds in one process, or once all parallel fold-subsets are done, run:")
    print(f"  python merge_v2_folds.py --output-dir {output_dir}")


if __name__ == "__main__":
    main()

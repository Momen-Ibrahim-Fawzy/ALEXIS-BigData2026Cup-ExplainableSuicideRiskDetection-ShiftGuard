"""Stage 1: supervised LP-FT fine-tune of the NLI cross-encoder checkpoint
on our actual (post, factor) pairs.

Scoped as a SINGLE validation split, not full 5-fold CV -- exploding 1,635
posts x 24 factors into pairs makes a full 5-fold run prohibitively
expensive (~25h estimated). Split: the same 20 pseudo-leaderboard users
(never touched, per pseudo_leaderboard_split.py) held out entirely; of the
remaining 133 users, ~80% (106 users) train the model, ~20% (27 users) tune
per-factor thresholds -- a clean 3-way split with no leakage in either
direction. If this validates, a full 5-fold ensemble becomes worth the
larger compute investment; if not, we've spent ~2.5-3h finding out instead
of ~25h.

LP-FT (Kumar et al. 2022): Stage A trains only the pooler+classifier head
with the DeBERTa backbone frozen; Stage B unfreezes the backbone and
continues at a lower LR. See model_nli_finetune.py docstring for why.

Usage:
    python train_nli_finetune.py
"""

import ast, json, os, time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from tqdm import tqdm

from config import *
from dataset_nli_finetune import build_pairs, NLIPairDataset
from model_nli_finetune import NLIFactorClassifier, NLI_MODEL
from pseudo_leaderboard_split import get_or_create_split

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs_nli_finetune")
MAX_LENGTH = 256
BATCH_SIZE = 16
FROZEN_EPOCHS = 3
UNFROZEN_EPOCHS = 3
FROZEN_LR = 1e-4       # head-only, can afford a higher LR
UNFROZEN_LR = 1e-5     # whole model, low LR to limit feature distortion (Kumar et al.)
WARMUP_RATIO = 0.1
PATIENCE = 2


def run_epoch(model, loader, optimizer, scheduler, device, train=True):
    model.train() if train else model.eval()
    total_loss, steps = 0.0, 0
    all_scores, all_labels, all_row_idx, all_factor_idx = [], [], [], []
    criterion = torch.nn.BCEWithLogitsLoss()
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
                logit = model(ids, mask, ttype)
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
            all_factor_idx.append(batch["factor_idx"].numpy())
    pbar.close()
    return (total_loss / max(steps, 1), np.concatenate(all_scores), np.concatenate(all_labels),
            np.concatenate(all_row_idx), np.concatenate(all_factor_idx))


def pairs_to_matrix(scores, row_idx, factor_idx, n_rows):
    mat = np.zeros((n_rows, NUM_FACTORS), dtype=np.float32)
    mat[row_idx, factor_idx] = scores
    return mat


def tune_thresholds(probs, labels):
    candidates = np.arange(0.05, 0.95, 0.025)
    thresholds = np.full(NUM_FACTORS, 0.5)
    for i in range(NUM_FACTORS):
        best_f1, best_t = -1, 0.5
        for t in candidates:
            preds = (probs[:, i] >= t).astype(int)
            f1 = f1_score(labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[i] = best_t
    return thresholds


def compute_macro_f1(probs, labels, thresholds):
    preds = (probs >= thresholds).astype(int)
    return float(np.mean([f1_score(labels[:, i], preds[:, i], zero_division=0) for i in range(NUM_FACTORS)]))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED); np.random.seed(SEED)

    df = pd.read_excel(TRAIN_PATH, engine="openpyxl")
    df["factors_list"] = df["factors"].apply(lambda x: sorted(set(ast.literal_eval(x))) if pd.notna(x) else [])

    pseudo_lb_users = set(get_or_create_split())
    is_pseudo_lb = df["anon_user_id"].isin(pseudo_lb_users).values
    remaining_df = df[~is_pseudo_lb].reset_index(drop=False)  # keep original index as 'index'
    pseudo_lb_df = df[is_pseudo_lb].reset_index(drop=True)

    remaining_users = sorted(remaining_df["anon_user_id"].unique())
    rng = np.random.RandomState(SEED)
    rng.shuffle(remaining_users)
    n_thresh_users = max(1, int(0.2 * len(remaining_users)))
    thresh_users = set(remaining_users[:n_thresh_users])
    train_users = set(remaining_users[n_thresh_users:])

    train_df = remaining_df[remaining_df["anon_user_id"].isin(train_users)].reset_index(drop=True)
    thresh_df = remaining_df[remaining_df["anon_user_id"].isin(thresh_users)].reset_index(drop=True)

    print(f"Split: {len(train_users)} model-train users ({len(train_df)} posts) | "
          f"{len(thresh_users)} threshold-tune users ({len(thresh_df)} posts) | "
          f"{len(pseudo_lb_users)} pseudo-leaderboard users ({len(pseudo_lb_df)} posts, never touched)")

    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    train_pairs = build_pairs(train_df)
    thresh_pairs = build_pairs(thresh_df)
    pseudo_lb_pairs = build_pairs(pseudo_lb_df)
    print(f"Pairs: train={len(train_pairs)}  thresh={len(thresh_pairs)}  pseudo_lb={len(pseudo_lb_pairs)}")

    train_ds = NLIPairDataset(train_pairs, tokenizer, MAX_LENGTH)
    thresh_ds = NLIPairDataset(thresh_pairs, tokenizer, MAX_LENGTH)
    pseudo_lb_ds = NLIPairDataset(pseudo_lb_pairs, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    thresh_loader = DataLoader(thresh_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2)
    pseudo_lb_loader = DataLoader(pseudo_lb_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2)

    model = NLIFactorClassifier().to(device).float()
    model.model.gradient_checkpointing_enable()

    log_rows = []

    # ---- Stage A: linear probe (frozen backbone) ----
    print(f"\n{'='*60}\n  STAGE A: frozen backbone, head-only ({FROZEN_EPOCHS} epochs)\n{'='*60}")
    model.freeze_backbone()
    optimizer = AdamW(model.trainable_parameters(), lr=FROZEN_LR, weight_decay=0.01)
    total_steps = len(train_loader) * FROZEN_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * WARMUP_RATIO), total_steps)

    best_thresh_mf1, best_state, patience_ctr = 0.0, None, 0
    for epoch in range(FROZEN_EPOCHS):
        t0 = time.time()
        train_loss, *_ = run_epoch(model, train_loader, optimizer, scheduler, device, train=True)
        _, scores, labels_flat, row_idx, factor_idx = run_epoch(model, thresh_loader, None, None, device, train=False)
        thresh_probs = pairs_to_matrix(scores, row_idx, factor_idx, len(thresh_df))
        thresh_labels = pairs_to_matrix(labels_flat, row_idx, factor_idx, len(thresh_df))
        mf1_at_half = compute_macro_f1(thresh_probs, thresh_labels, np.full(NUM_FACTORS, 0.5))
        elapsed = time.time() - t0
        print(f"  [Stage A] Epoch {epoch+1}/{FROZEN_EPOCHS} ({elapsed:.0f}s) train_loss={train_loss:.4f} "
              f"thresh_pool_MacroF1(t=0.5)={mf1_at_half:.4f}")
        log_rows.append({"stage": "A", "epoch": epoch + 1, "train_loss": train_loss, "thresh_mf1_t05": mf1_at_half})
        if mf1_at_half > best_thresh_mf1:
            best_thresh_mf1, patience_ctr = mf1_at_half, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print("  Early stopping Stage A")
                break

    # ---- Stage B: unfreeze, low LR ----
    print(f"\n{'='*60}\n  STAGE B: unfrozen, low LR ({UNFROZEN_EPOCHS} epochs)\n{'='*60}")
    model.unfreeze_backbone()
    optimizer = AdamW(model.trainable_parameters(), lr=UNFROZEN_LR, weight_decay=0.01)
    total_steps = len(train_loader) * UNFROZEN_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * WARMUP_RATIO), total_steps)

    patience_ctr = 0
    for epoch in range(UNFROZEN_EPOCHS):
        t0 = time.time()
        train_loss, *_ = run_epoch(model, train_loader, optimizer, scheduler, device, train=True)
        _, scores, labels_flat, row_idx, factor_idx = run_epoch(model, thresh_loader, None, None, device, train=False)
        thresh_probs = pairs_to_matrix(scores, row_idx, factor_idx, len(thresh_df))
        thresh_labels = pairs_to_matrix(labels_flat, row_idx, factor_idx, len(thresh_df))
        mf1_at_half = compute_macro_f1(thresh_probs, thresh_labels, np.full(NUM_FACTORS, 0.5))
        elapsed = time.time() - t0
        print(f"  [Stage B] Epoch {epoch+1}/{UNFROZEN_EPOCHS} ({elapsed:.0f}s) train_loss={train_loss:.4f} "
              f"thresh_pool_MacroF1(t=0.5)={mf1_at_half:.4f}")
        log_rows.append({"stage": "B", "epoch": epoch + 1, "train_loss": train_loss, "thresh_mf1_t05": mf1_at_half})
        if mf1_at_half > best_thresh_mf1:
            best_thresh_mf1, patience_ctr = mf1_at_half, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print("  Early stopping Stage B")
                break

    pd.DataFrame(log_rows).to_csv(os.path.join(OUT_DIR, "training_log.csv"), index=False)
    model.load_state_dict(best_state)
    torch.save({"model": model.state_dict()}, os.path.join(OUT_DIR, "best.pt"))

    # ---- Final honest evaluation ----
    _, thresh_scores, thresh_labels_flat, t_row_idx, t_factor_idx = run_epoch(model, thresh_loader, None, None, device, train=False)
    thresh_probs = pairs_to_matrix(thresh_scores, t_row_idx, t_factor_idx, len(thresh_df))
    thresh_labels_mat = pairs_to_matrix(thresh_labels_flat, t_row_idx, t_factor_idx, len(thresh_df))
    thresholds = tune_thresholds(thresh_probs, thresh_labels_mat)

    _, lb_scores, lb_labels_flat, lb_row_idx, lb_factor_idx = run_epoch(model, pseudo_lb_loader, None, None, device, train=False)
    pseudo_probs = pairs_to_matrix(lb_scores, lb_row_idx, lb_factor_idx, len(pseudo_lb_df))
    pseudo_labels = pairs_to_matrix(lb_labels_flat, lb_row_idx, lb_factor_idx, len(pseudo_lb_df))
    honest_score = compute_macro_f1(pseudo_probs, pseudo_labels, thresholds)
    honest_score_t05 = compute_macro_f1(pseudo_probs, pseudo_labels, np.full(NUM_FACTORS, 0.5))

    print(f"\n{'='*60}\n  FINAL RESULT\n{'='*60}")
    print(f"  Honest pseudo-leaderboard Macro F1 (tuned thresholds): {honest_score:.4f}")
    print(f"  Honest pseudo-leaderboard Macro F1 (t=0.5, untuned):   {honest_score_t05:.4f}")
    print(f"  Compare to: zero-shot honest 0.2880, factor_verified baseline 0.5806, "
          f"current best 2-way ensemble 0.5982")

    with open(os.path.join(OUT_DIR, "result.json"), "w") as f:
        json.dump({"honest_score_tuned": float(honest_score), "honest_score_t05": float(honest_score_t05),
                    "zero_shot_honest": 0.2880, "baseline": 0.5806}, f, indent=2)

    if honest_score > 0.5806:
        print("\n*** BEATS BASELINE -- worth scaling to a full 5-fold ensemble and a real submission test. ***")
    else:
        print("\n*** Did not beat baseline. Fine-tuning improved on zero-shot "
              f"({0.2880:.4f} -> {honest_score:.4f}) but not enough yet -- "
              "consider more epochs, or treat this direction as exhausted for now. ***")


if __name__ == "__main__":
    main()

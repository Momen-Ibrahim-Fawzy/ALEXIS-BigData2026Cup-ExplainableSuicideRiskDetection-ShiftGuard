"""Subtask 2: Train factor classifier with GroupKFold, checkpoints, logging.

Saves out-of-fold probabilities for threshold tuning after training.
"""

import os, csv, json, time, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from tqdm import tqdm

from config import *
from dataset import load_train, get_tokenizer, FactorDataset
from model import FactorClassifier
from losses import FocalLossMultiLabel

# ── Logging & Checkpointing ─────────────────────────────────────────────────

LOG_FIELDS = ["fold","epoch","train_loss","val_loss","val_macro_f1",
              "best_mf1","lr","elapsed"]

def init_log():
    path = os.path.join(OUTPUT_DIR, "training_log.csv")
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=LOG_FIELDS).writeheader()

def log_row(row):
    with open(os.path.join(OUTPUT_DIR, "training_log.csv"), "a", newline="") as f:
        csv.DictWriter(f, fieldnames=LOG_FIELDS).writerow(row)

def save_checkpoint(fold, epoch, model, optimizer, scheduler, scaler, best, is_best):
    d = os.path.join(OUTPUT_DIR, f"fold_{fold}")
    os.makedirs(d, exist_ok=True)
    state = {"fold":fold,"epoch":epoch,"model":model.state_dict(),
             "optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
             "scaler":scaler.state_dict() if scaler else None,"best":best}
    torch.save(state, os.path.join(d, f"epoch_{epoch}.pt"))
    if is_best:
        torch.save(state, os.path.join(d, "best.pt"))
    fs = sorted([f for f in os.listdir(d) if f.startswith("epoch_")],
                key=lambda x: int(x.split("_")[1].split(".")[0]))
    while len(fs) > MAX_CHECKPOINTS - 1:
        os.remove(os.path.join(d, fs.pop(0)))

def load_checkpoint(fold):
    d = os.path.join(OUTPUT_DIR, f"fold_{fold}")
    if not os.path.isdir(d): return None
    fs = sorted([f for f in os.listdir(d) if f.startswith("epoch_")],
                key=lambda x: int(x.split("_")[1].split(".")[0]))
    return torch.load(os.path.join(d, fs[-1]), weights_only=False) if fs else None

def fold_completed(fold):
    meta = os.path.join(OUTPUT_DIR, "completed_folds.json")
    if os.path.exists(meta):
        with open(meta) as f: return str(fold) in json.load(f)
    return False

def mark_fold_done(fold, metrics):
    meta = os.path.join(OUTPUT_DIR, "completed_folds.json")
    data = json.load(open(meta)) if os.path.exists(meta) else {}
    data[str(fold)] = metrics
    json.dump(data, open(meta,"w"), indent=2)
    # Delete epoch checkpoints — only best.pt is needed after fold completes
    d = os.path.join(OUTPUT_DIR, f"fold_{fold}")
    for f in os.listdir(d):
        if f.startswith("epoch_"):
            os.remove(os.path.join(d, f))


# ── Macro F1 computation ────────────────────────────────────────────────────

def compute_macro_f1(probs, labels, threshold=0.5):
    """Compute Macro F1 across all 24 factors."""
    preds = (probs >= threshold).astype(int)
    per_factor = []
    for i in range(NUM_FACTORS):
        f1 = f1_score(labels[:, i], preds[:, i], zero_division=0)
        per_factor.append(f1)
    return np.mean(per_factor), per_factor


# ── Training ─────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total, steps = 0, 0
    use_amp = device.type == "cuda"
    pbar = tqdm(loader, desc="  train", leave=False, unit="batch", dynamic_ncols=True)
    for batch in pbar:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        ttype = batch.get("token_type_ids")
        if ttype is not None: ttype = ttype.to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(ids, mask, ttype)
        loss = criterion(logits.float(), labels) / GRAD_ACCUM

        loss.backward()

        if (steps+1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total += loss.item() * GRAD_ACCUM
        steps += 1
        pbar.set_postfix(loss=f"{loss.item() * GRAD_ACCUM:.4f}")
    pbar.close()
    return total / max(steps,1)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    all_probs, all_labels, total_loss = [], [], 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        ttype = batch.get("token_type_ids")
        if ttype is not None: ttype = ttype.to(device)

        logits = model(ids, mask, ttype)
        total_loss += criterion(logits, labels).item()
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    mf1, per_f = compute_macro_f1(probs, labels)
    return {"loss": total_loss/len(all_probs), "macro_f1": mf1,
            "per_factor": per_f, "probs": probs, "labels": labels}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--oversample", type=str, default=None,
                        choices=["gpt", "claude", "llama", "all_sources", "verified", "all",
                                 "factor_gpt", "factor_claude", "factor_llama",
                                 "factor_all", "factor_verified"],
                        help="Oversample source. factor_* keys use Sub2 factor-targeted posts.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory (default: outputs/)")
    parser.add_argument("--loss", type=str, default="weighted",
                        choices=["weighted", "vanilla", "focal"],
                        help="Loss: weighted=per-factor pos_weight (default), "
                             "vanilla=plain BCE, focal=multi-label focal loss "
                             "(untried lever for rare-factor Macro F1)")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Focal loss focusing parameter (only used with --loss focal)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override config.SEED -- GroupKFold splits are unaffected "
                             "(deterministic on anon_user_id groups), so this only "
                             "changes model init/dropout/batch order. Use different "
                             "seeds to bag multiple runs of the same config.")
    args = parser.parse_args()

    if args.output_dir:
        globals()["OUTPUT_DIR"] = os.path.abspath(args.output_dir)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    seed = args.seed if args.seed is not None else SEED
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Seed: {seed}")
    torch.manual_seed(seed); np.random.seed(seed)
    init_log()

    tokenizer = get_tokenizer()
    df = load_train(oversample_source=args.oversample)
    print(f"Dataset: {len(df)} | Oversample: {args.oversample or 'none'} | Loss: {args.loss} | Output: {OUTPUT_DIR}")

    # Split synthetic rows — they go to training only, never validation
    orig_df = df[df["anon_user_id"] != "SYN"].reset_index(drop=True)
    syn_df  = df[df["anon_user_id"] == "SYN"].reset_index(drop=True)

    gkf = GroupKFold(n_splits=args.folds)
    groups = orig_df["anon_user_id"].values

    # OOF arrays sized to original data only (no synthetic rows in validation)
    oof_probs  = np.zeros((len(orig_df), NUM_FACTORS))
    oof_labels = np.zeros((len(orig_df), NUM_FACTORS))

    all_metrics = []
    for fold, (tr_i, va_i) in enumerate(gkf.split(orig_df, groups=groups)):
        if fold_completed(fold):
            print(f"\nFold {fold} done, skipping.")
            with open(os.path.join(OUTPUT_DIR,"completed_folds.json")) as f:
                all_metrics.append(json.load(f)[str(fold)])
            continue

        tr_df = pd.concat([orig_df.iloc[tr_i], syn_df], ignore_index=True)
        va_df = orig_df.iloc[va_i]
        print(f"\n{'='*60}\n  FOLD {fold+1}/{args.folds}  "
              f"Train={len(tr_df)} Val={len(va_df)}\n{'='*60}")

        tr_loader = DataLoader(FactorDataset(tr_df, tokenizer), batch_size=BATCH_SIZE,
                               shuffle=True, num_workers=2, pin_memory=True)
        va_loader = DataLoader(FactorDataset(va_df, tokenizer), batch_size=BATCH_SIZE*2,
                               shuffle=False, num_workers=2, pin_memory=True)

        model = FactorClassifier().to(device).float()
        model.backbone.gradient_checkpointing_enable()

        if args.loss in ("weighted", "focal"):
            # Per-factor neg/pos ratio clipped at 50 to handle extreme imbalance --
            # used as BCE's pos_weight directly, or converted to focal loss's
            # per-class alpha (pw/(pw+1), so rare factors up-weight their positives).
            pos_counts = np.zeros(NUM_FACTORS)
            for _, row in tr_df.iterrows():
                for f in row["factors_list"]:
                    fl = f.lower().strip()
                    if fl in FACTOR2ID:
                        pos_counts[FACTOR2ID[fl]] += 1
            neg_counts = len(tr_df) - pos_counts
            pw = np.clip(neg_counts / (pos_counts + 1e-8), 1.0, 50.0)
            print(f"  pos_weight range: [{pw.min():.1f}, {pw.max():.1f}] "
                  f"(median {np.median(pw):.1f})")
            if args.loss == "weighted":
                pos_weight = torch.tensor(pw, dtype=torch.float32).to(device)
                criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                alpha = torch.tensor(pw / (pw + 1.0), dtype=torch.float32).to(device)
                print(f"  focal loss: gamma={args.focal_gamma}, "
                      f"alpha range=[{alpha.min():.2f}, {alpha.max():.2f}]")
                criterion = FocalLossMultiLabel(alpha=alpha, gamma=args.focal_gamma)
        else:
            print("  loss: vanilla BCE (no pos_weight)")
            criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        total_steps = len(tr_loader) * args.epochs // GRAD_ACCUM
        scheduler = get_linear_schedule_with_warmup(optimizer,
                    int(total_steps*WARMUP_RATIO), total_steps)

        start_epoch, best, patience_counter = 0, 0, 0
        ckpt = load_checkpoint(fold)
        if ckpt:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"]+1; best = ckpt["best"]
            print(f"  Resumed epoch {start_epoch}, best_mf1={best:.4f}")

        for epoch in range(start_epoch, args.epochs):
            t0 = time.time()
            t_loss = train_epoch(model, tr_loader, optimizer, scheduler, criterion, device)
            val = eval_epoch(model, va_loader, criterion, device)
            elapsed = time.time() - t0
            is_best = val["macro_f1"] > best
            if is_best:
                best = val["macro_f1"]
                patience_counter = 0
            else:
                patience_counter += 1
            lr = scheduler.get_last_lr()[0]

            print(f"  Epoch {epoch+1}/{args.epochs} ({elapsed:.0f}s) lr={lr:.2e}")
            print(f"    Train loss={t_loss:.4f}  Val loss={val['loss']:.4f}  "
                  f"MacroF1={val['macro_f1']:.4f}{'  **BEST**' if is_best else ''}")

            sorted_f = sorted(zip(FACTOR_LIST, val["per_factor"]), key=lambda x: x[1])
            bottom = ", ".join(f"{n[:15]}={v:.2f}" for n,v in sorted_f[:3])
            print(f"    Bottom 3: {bottom}")

            log_row({"fold":fold,"epoch":epoch+1,"train_loss":f"{t_loss:.4f}",
                     "val_loss":f"{val['loss']:.4f}","val_macro_f1":f"{val['macro_f1']:.4f}",
                     "best_mf1":f"{best:.4f}","lr":f"{lr:.2e}","elapsed":f"{elapsed:.0f}"})
            save_checkpoint(fold, epoch, model, optimizer, scheduler, None, best, is_best)

            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        # Final OOF predictions with best model
        best_path = os.path.join(OUTPUT_DIR, f"fold_{fold}", "best.pt")
        model.load_state_dict(torch.load(best_path, weights_only=False)["model"])
        final = eval_epoch(model, va_loader, criterion, device)
        oof_probs[va_i] = final["probs"]
        oof_labels[va_i] = final["labels"]

        metrics = {"macro_f1": final["macro_f1"]}
        mark_fold_done(fold, metrics)
        all_metrics.append(metrics)

    # Save OOF for threshold tuning
    np.save(os.path.join(OUTPUT_DIR, "oof_probs.npy"), oof_probs)
    np.save(os.path.join(OUTPUT_DIR, "oof_labels.npy"), oof_labels)
    print(f"\nOOF probabilities saved for threshold tuning.")

    # Summary
    print(f"\n{'='*60}\n  CV RESULTS\n{'='*60}")
    vals = [m["macro_f1"] for m in all_metrics]
    print(f"  Macro F1 (default t=0.5): {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    json.dump({"mean_macro_f1": float(np.mean(vals)), "folds": all_metrics},
              open(os.path.join(OUTPUT_DIR,"cv_results.json"),"w"), indent=2)


if __name__ == "__main__":
    main()

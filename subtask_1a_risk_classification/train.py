"""Subtask 1a: Train risk classifier with GroupKFold, checkpointing, live logging."""

import os, csv, json, time, argparse, shutil
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, classification_report

from config import *
from dataset import load_train, get_tokenizer, RiskDataset
from model import RiskClassifier

# ── Logging ──────────────────────────────────────────────────────────────────

LOG_FIELDS = ["fold", "epoch", "train_loss", "val_loss",
              "val_wf1", "val_mf1", "val_acc",
              "best_wf1", "lr", "elapsed"]

def init_log():
    path = os.path.join(OUTPUT_DIR, "training_log.csv")
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=LOG_FIELDS).writeheader()

def log_row(row):
    with open(os.path.join(OUTPUT_DIR, "training_log.csv"), "a", newline="") as f:
        csv.DictWriter(f, fieldnames=LOG_FIELDS).writerow(row)


# ── Checkpoint management (keep at most MAX_CHECKPOINTS, always keep best) ──

def save_checkpoint(fold, epoch, model, optimizer, scheduler, scaler, best_wf1, is_best):
    """Save checkpoint; manage max count."""
    ckpt_dir = os.path.join(OUTPUT_DIR, f"fold_{fold}")
    os.makedirs(ckpt_dir, exist_ok=True)

    state = {
        "fold": fold, "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler else None,
        "best_wf1": best_wf1,
    }

    # Save epoch checkpoint
    path = os.path.join(ckpt_dir, f"epoch_{epoch}.pt")
    torch.save(state, path)

    # Save best separately
    if is_best:
        best_path = os.path.join(ckpt_dir, "best.pt")
        torch.save(state, best_path)

    # Prune old checkpoints (keep best + latest MAX_CHECKPOINTS-1)
    epoch_files = sorted(
        [f for f in os.listdir(ckpt_dir) if f.startswith("epoch_")],
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )
    while len(epoch_files) > MAX_CHECKPOINTS - 1:  # -1 for best.pt
        old = epoch_files.pop(0)
        os.remove(os.path.join(ckpt_dir, old))


def load_checkpoint(fold):
    """Load latest epoch checkpoint for resume."""
    ckpt_dir = os.path.join(OUTPUT_DIR, f"fold_{fold}")
    if not os.path.isdir(ckpt_dir):
        return None
    epoch_files = sorted(
        [f for f in os.listdir(ckpt_dir) if f.startswith("epoch_")],
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )
    if not epoch_files:
        return None
    path = os.path.join(ckpt_dir, epoch_files[-1])
    return torch.load(path, weights_only=False)


def fold_completed(fold):
    meta = os.path.join(OUTPUT_DIR, "completed_folds.json")
    if os.path.exists(meta):
        with open(meta) as f:
            return fold in json.load(f)
    return False

def mark_fold_done(fold, metrics):
    meta = os.path.join(OUTPUT_DIR, "completed_folds.json")
    data = {}
    if os.path.exists(meta):
        with open(meta) as f:
            data = json.load(f)
    data[fold] = metrics
    with open(meta, "w") as f:
        json.dump(data, f, indent=2)


# ── Training ─────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, scaler, criterion, device):
    model.train()
    total_loss, steps = 0, 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        ttype = batch.get("token_type_ids")
        if ttype is not None:
            ttype = ttype.to(device)

        with torch.amp.autocast("cuda", enabled=scaler is not None):
            logits = model(ids, mask, ttype)
            loss = criterion(logits, labels) / GRAD_ACCUM

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (steps + 1) % GRAD_ACCUM == 0:
            if scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * GRAD_ACCUM
        steps += 1

    return total_loss / max(steps, 1)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        ttype = batch.get("token_type_ids")
        if ttype is not None:
            ttype = ttype.to(device)

        logits = model(ids, mask, ttype)
        total_loss += criterion(logits, labels).item()
        all_preds.extend(logits.argmax(-1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    n = max(len(all_preds), 1)
    preds_str = [ID2RISK[p] for p in all_preds]
    labels_str = [ID2RISK[l] for l in all_labels]
    wf1 = f1_score(labels_str, preds_str, labels=RISK_LEVELS, average="weighted", zero_division=0)
    mf1 = f1_score(labels_str, preds_str, labels=RISK_LEVELS, average="macro", zero_division=0)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / n
    report = classification_report(labels_str, preds_str, labels=RISK_LEVELS, zero_division=0)
    return {"loss": total_loss / n, "wf1": wf1, "mf1": mf1, "acc": acc, "report": report}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--oversample", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    init_log()

    tokenizer = get_tokenizer()
    df = load_train(use_oversample=args.oversample)
    print(f"Dataset: {len(df)} posts | Oversample: {args.oversample}")
    print(f"Risk distribution: {df['risk_norm'].value_counts().to_dict()}")

    gkf = GroupKFold(n_splits=args.folds)
    groups = df["anon_user_id"].values
    class_w = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(device)

    all_metrics = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups=groups)):
        if fold_completed(fold):
            print(f"\nFold {fold} already completed, skipping.")
            with open(os.path.join(OUTPUT_DIR, "completed_folds.json")) as f:
                all_metrics.append(json.load(f)[str(fold)])
            continue

        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        print(f"\n{'='*60}\n  FOLD {fold+1}/{args.folds}  "
              f"Train={len(train_df)} Val={len(val_df)}\n{'='*60}")

        train_loader = DataLoader(RiskDataset(train_df, tokenizer), batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(RiskDataset(val_df, tokenizer), batch_size=BATCH_SIZE*2,
                                shuffle=False, num_workers=2, pin_memory=True)

        model = RiskClassifier().to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_w, label_smoothing=0.05)
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        total_steps = len(train_loader) * args.epochs // GRAD_ACCUM
        scheduler = get_linear_schedule_with_warmup(optimizer,
                    int(total_steps * WARMUP_RATIO), total_steps)
        scaler = torch.amp.GradScaler("cuda") if FP16 and device.type == "cuda" else None

        start_epoch = 0
        best_wf1 = 0

        # Resume
        ckpt = load_checkpoint(fold)
        if ckpt:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            if scaler and ckpt["scaler"]:
                scaler.load_state_dict(ckpt["scaler"])
            start_epoch = ckpt["epoch"] + 1
            best_wf1 = ckpt["best_wf1"]
            print(f"  Resumed from epoch {start_epoch}, best_wf1={best_wf1:.4f}")

        for epoch in range(start_epoch, args.epochs):
            t0 = time.time()
            train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                     scaler, criterion, device)
            val = eval_epoch(model, val_loader, criterion, device)
            elapsed = time.time() - t0

            is_best = val["wf1"] > best_wf1
            if is_best:
                best_wf1 = val["wf1"]

            lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch+1}/{args.epochs} ({elapsed:.0f}s) lr={lr:.2e}")
            print(f"    Train loss={train_loss:.4f}  Val loss={val['loss']:.4f}")
            print(f"    WF1={val['wf1']:.4f}  MF1={val['mf1']:.4f}  Acc={val['acc']:.4f}"
                  f"{'  ** BEST **' if is_best else ''}")

            log_row({"fold": fold, "epoch": epoch+1,
                     "train_loss": f"{train_loss:.4f}", "val_loss": f"{val['loss']:.4f}",
                     "val_wf1": f"{val['wf1']:.4f}", "val_mf1": f"{val['mf1']:.4f}",
                     "val_acc": f"{val['acc']:.4f}", "best_wf1": f"{best_wf1:.4f}",
                     "lr": f"{lr:.2e}", "elapsed": f"{elapsed:.0f}"})

            save_checkpoint(fold, epoch, model, optimizer, scheduler,
                            scaler, best_wf1, is_best)

        # Final eval with best model
        best_path = os.path.join(OUTPUT_DIR, f"fold_{fold}", "best.pt")
        model.load_state_dict(torch.load(best_path, weights_only=False)["model"])
        final = eval_epoch(model, val_loader, criterion, device)
        print(f"\n  Fold {fold+1} final: WF1={final['wf1']:.4f}")
        print(final["report"])

        metrics = {"wf1": final["wf1"], "mf1": final["mf1"], "acc": final["acc"]}
        mark_fold_done(fold, metrics)
        all_metrics.append(metrics)

    # Summary
    print(f"\n{'='*60}\n  CV RESULTS ({args.folds} folds)\n{'='*60}")
    for k in ["wf1", "mf1", "acc"]:
        vals = [m[k] for m in all_metrics]
        print(f"  {k}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    with open(os.path.join(OUTPUT_DIR, "cv_results.json"), "w") as f:
        json.dump({"mean": {k: float(np.mean([m[k] for m in all_metrics]))
                            for k in ["wf1","mf1","acc"]},
                   "folds": all_metrics}, f, indent=2)


if __name__ == "__main__":
    main()

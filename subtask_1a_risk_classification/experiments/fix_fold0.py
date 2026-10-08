"""
Re-run fold 0 only for oversample experiments where fold 0 had synthetic
data in its validation set (GroupKFold artefact).

What this script does per experiment:
  - Keeps training_log.csv untouched (original contaminated fold 0 rows preserved)
  - Writes corrected fold 0 training rows to training_log_fold0_corrected.csv
  - Updates summary.json in-place:
      · fold_wf1_original   — original 5 fold scores (before any fix)
      · mean_wf1_original   — original mean (before fix)
      · std_wf1_original    — original std  (before fix)
      · fold_wf1            — corrected 5 fold scores (fold 0 replaced)
      · mean_wf1            — corrected mean
      · std_wf1             — corrected std
      · note                — explains what changed
  - Saves corrected model to best_fold0_corrected.pt (does NOT overwrite best_fold0.pt)

Usage:
    python fix_fold0.py                                    # fix all 6 oversample exps
    python fix_fold0.py exp05_deberta_large_oversample_gpt # one only
"""

import os, sys, csv, json, time
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from trainer import (
    Classifier, FocalLoss, TextDataset,
    load_data, RISK2ID, ID2RISK, RISK_LEVELS, SEED,
    BASE_DIR, DATA_DIR, RESULTS_DIR,
)
from experiment_config import EXPERIMENTS

OVERSAMPLE_EXPS = [
    "exp05_deberta_large_oversample_gpt",
    "exp05b_deberta_large_oversample_claude",
    "exp05c_deberta_large_oversample_llama",
    "exp05d_deberta_large_oversample_all_3_sources",
    "exp06_deberta_large_oversample_verified",
    "exp11_deberta_large_focal_oversample",
]


def run_fold0(name, cfg):
    exp_dir = os.path.join(RESULTS_DIR, name)
    summary_path = os.path.join(exp_dir, "summary.json")
    log_path_corrected = os.path.join(exp_dir, "training_log_fold0_corrected.csv")

    if not os.path.exists(summary_path):
        print(f"  SKIP {name} — no summary.json yet (not finished?)")
        return

    with open(summary_path) as f:
        summary = json.load(f)

    # If already fixed, grab the already-stored originals so we don't overwrite them
    old_fold_wf1 = summary.get("fold_wf1_original", summary["fold_wf1"])
    old_fold0    = old_fold_wf1[0]
    old_mean     = summary.get("mean_wf1_original", summary["mean_wf1"])
    old_std      = summary.get("std_wf1_original",  summary["std_wf1"])

    print(f"\n{'='*64}")
    print(f"  {name}")
    print(f"  Original fold 0 WF1 : {old_fold0:.4f}  (val included synthetic)")
    print(f"  Original mean WF1   : {old_mean:.4f}  std={old_std:.4f}")
    print(f"{'='*64}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    df = load_data(cfg["data"])

    orig_df = df[df["anon_user_id"] != "SYN"].reset_index(drop=True)
    syn_df  = df[df["anon_user_id"] == "SYN"].reset_index(drop=True)
    print(f"  {len(orig_df)} original + {len(syn_df)} synthetic (train-only)")

    gkf = GroupKFold(n_splits=5)
    groups = orig_df["anon_user_id"].values
    fold0_tr_i, fold0_va_i = next(iter(gkf.split(orig_df, groups=groups)))

    tr_df = pd.concat([orig_df.iloc[fold0_tr_i], syn_df], ignore_index=True)
    va_df = orig_df.iloc[fold0_va_i]
    print(f"  Fold 0  Train={len(tr_df)}  Val={len(va_df)} (original only)")

    tr_ds = TextDataset(tr_df["post"].tolist(), [RISK2ID[r] for r in tr_df["risk_norm"]],
                        tokenizer, cfg["max_length"])
    va_ds = TextDataset(va_df["post"].tolist(), [RISK2ID[r] for r in va_df["risk_norm"]],
                        tokenizer, cfg["max_length"])
    tr_ld = DataLoader(tr_ds, batch_size=cfg["batch_size"], shuffle=True,
                       num_workers=2, pin_memory=True)
    va_ld = DataLoader(va_ds, batch_size=cfg["batch_size"] * 2, shuffle=False,
                       num_workers=2, pin_memory=True)

    model = Classifier(cfg["model_name"]).to(device).float()
    model.backbone.gradient_checkpointing_enable()

    if cfg["loss"] == "focal":
        criterion = FocalLoss(alpha=cfg.get("focal_alpha"),
                              gamma=cfg.get("focal_gamma", 2.0)).to(device)
    elif cfg["loss"] == "weighted_ce":
        w = torch.tensor(cfg["class_weights"], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=w,
                                        label_smoothing=cfg.get("label_smoothing", 0))
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0))

    optimizer = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    total_steps  = len(tr_ld) * cfg["epochs"] // cfg["grad_accum"]
    warmup_steps = int(total_steps * cfg.get("warmup_ratio", 0.1))

    sched_type = cfg.get("scheduler", "cosine_warmup")
    if sched_type == "linear_warmup":
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    elif sched_type == "cosine_restarts":
        scheduler = CosineAnnealingWarmRestarts(optimizer,
                                                T_0=cfg.get("cosine_t0", 3), T_mult=2)
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    use_amp = device.type == "cuda"
    best_wf1, patience_counter = 0.0, 0

    log_rows = []

    for epoch in range(cfg["epochs"]):
        t0 = time.time()
        model.train()
        train_loss, steps = 0.0, 0
        current_lr = optimizer.param_groups[0]["lr"]

        pbar = tqdm(tr_ld, desc=f"  E{epoch+1:02d} train", leave=False,
                    unit="batch", dynamic_ncols=True)
        for batch in pbar:
            ids    = batch["input_ids"].to(device)
            mask   = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(ids, mask)
            loss = criterion(logits.float(), labels) / cfg["grad_accum"]
            loss.backward()

            if (steps + 1) % cfg["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_loss += loss.item() * cfg["grad_accum"]
            steps += 1
            pbar.set_postfix(loss=f"{loss.item() * cfg['grad_accum']:.4f}")
        pbar.close()
        train_loss /= max(steps, 1)

        model.eval()
        all_preds, all_labels, val_loss = [], [], 0.0
        with torch.no_grad():
            for batch in va_ld:
                ids    = batch["input_ids"].to(device)
                mask   = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                logits = model(ids, mask)
                val_loss += criterion(logits.float(), labels).item()
                all_preds.extend(logits.argmax(-1).cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        val_loss /= max(len(va_ld), 1)
        ps  = [ID2RISK[p] for p in all_preds]
        ls  = [ID2RISK[l] for l in all_labels]
        wf1 = f1_score(ls, ps, labels=RISK_LEVELS, average="weighted", zero_division=0)
        mf1 = f1_score(ls, ps, labels=RISK_LEVELS, average="macro",    zero_division=0)
        acc = accuracy_score(ls, ps)
        elapsed = int(time.time() - t0)

        is_best = wf1 > best_wf1
        if is_best:
            best_wf1 = wf1
            patience_counter = 0
            torch.save(model.state_dict(),
                       os.path.join(exp_dir, "best_fold0_corrected.pt"))
        else:
            patience_counter += 1

        log_rows.append({
            "fold": 0, "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_wf1": round(wf1, 4),
            "val_mf1": round(mf1, 4),
            "val_acc": round(acc, 4),
            "best_wf1": round(best_wf1, 4),
            "lr": f"{current_lr:.2e}",
            "elapsed": elapsed,
        })

        print(f"    E{epoch+1:02d} ({elapsed}s) "
              f"loss={train_loss:.4f}/{val_loss:.4f} "
              f"WF1={wf1:.4f}{'  **BEST**' if is_best else ''}")

        if patience_counter >= cfg["patience"]:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    print(f"\n  Fold 0 before fix : {old_fold0:.4f}")
    print(f"  Fold 0 after  fix : {best_wf1:.4f}")

    del model, optimizer, scheduler, criterion
    torch.cuda.empty_cache()

    # Write corrected fold 0 training log to a SEPARATE file
    fieldnames = ["fold","epoch","train_loss","val_loss","val_wf1",
                  "val_mf1","val_acc","best_wf1","lr","elapsed"]
    with open(log_path_corrected, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"  Corrected training log → training_log_fold0_corrected.csv")

    # Patch summary.json — preserve originals, add corrected values side by side
    new_fold_wf1 = [best_wf1] + old_fold_wf1[1:]
    new_mean     = float(np.mean(new_fold_wf1))
    new_std      = float(np.std(new_fold_wf1))

    summary["experiment"]        = name          # fix stale name if needed
    summary["fold_wf1_original"] = old_fold_wf1  # [old_f0, f1, f2, f3, f4]
    summary["mean_wf1_original"] = old_mean
    summary["std_wf1_original"]  = old_std
    summary["fold_wf1"]          = new_fold_wf1  # [new_f0, f1, f2, f3, f4]
    summary["mean_wf1"]          = new_mean
    summary["std_wf1"]           = new_std
    summary["note"] = (
        f"fold0 re-run with fixed split (synthetic train-only, never in val). "
        f"Fold 0: {old_fold0:.4f} → {best_wf1:.4f}. "
        f"Mean WF1: {old_mean:.4f} → {new_mean:.4f}."
    )

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  summary.json updated:")
    print(f"    mean_wf1  {old_mean:.4f} → {new_mean:.4f}")
    print(f"    fold_wf1_original = {[round(x,4) for x in old_fold_wf1]}")
    print(f"    fold_wf1          = {[round(x,4) for x in new_fold_wf1]}")
    return summary


def print_before_after(target):
    """Print a side-by-side comparison of original vs corrected fold 0 and mean WF1."""
    print(f"\n{'='*80}")
    print(f"  FOLD 0 FIX — BEFORE vs AFTER")
    print(f"{'='*80}")
    print(f"  {'Experiment':<42} {'F0 before':>9} {'F0 after':>9} {'Mean before':>11} {'Mean after':>10}")
    print(f"  {'-'*80}")
    for name in target:
        summary_path = os.path.join(RESULTS_DIR, name, "summary.json")
        if not os.path.exists(summary_path):
            continue
        with open(summary_path) as f:
            s = json.load(f)
        if "fold_wf1_original" not in s:
            continue
        f0_before   = s["fold_wf1_original"][0]
        f0_after    = s["fold_wf1"][0]
        mean_before = s["mean_wf1_original"]
        mean_after  = s["mean_wf1"]
        delta_f0    = f0_after - f0_before
        delta_mean  = mean_after - mean_before
        print(f"  {name:<42} {f0_before:>9.4f} {f0_after:>9.4f} "
              f"  {mean_before:>9.4f}   {mean_after:>9.4f}  "
              f"(Δmean={delta_mean:+.4f})")
    print(f"{'='*80}\n")


def main():
    target = sys.argv[1:] if len(sys.argv) > 1 else OVERSAMPLE_EXPS
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    for name in target:
        if name not in EXPERIMENTS:
            print(f"Unknown experiment: {name}")
            continue
        if name not in OVERSAMPLE_EXPS:
            print(f"WARNING: {name} is not an oversample experiment — skipping")
            continue
        run_fold0(name, EXPERIMENTS[name])

    print_before_after(target)

    from trainer import aggregate_results
    aggregate_results()


if __name__ == "__main__":
    main()

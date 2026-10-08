"""Train the GCN-gated Sub2 head (model_gcn.py) with the same GroupKFold/
oversample/logging discipline as train.py and train_chain.py.

Only used if it beats the existing independent-head split-ensemble's OOF
Macro F1 (0.5806 for factor_verified, 0.5785 for factor_all, 0.5982 for the
2-way ensemble) -- same validate-before-trusting discipline as everything
else this project. Not directly comparable to model_chain.py's result except
by which true-OOF Macro F1 is higher -- both are candidate Sub2 heads.

Usage:
    python train_gcn.py --oversample factor_verified --tau 0.3 --output-dir outputs_gcn_factor_verified
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
from model_gcn import GCNFactorClassifier, build_adjacency, build_factor_name_embeddings

LOG_FIELDS = ["fold", "epoch", "train_loss", "val_loss", "val_macro_f1", "best_mf1", "lr", "elapsed"]


def init_log(output_dir):
    path = os.path.join(output_dir, "training_log.csv")
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=LOG_FIELDS).writeheader()


def log_row(output_dir, row):
    with open(os.path.join(output_dir, "training_log.csv"), "a", newline="") as f:
        csv.DictWriter(f, fieldnames=LOG_FIELDS).writerow(row)


def compute_macro_f1(probs, labels, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    per_factor = [f1_score(labels[:, i], preds[:, i], zero_division=0) for i in range(NUM_FACTORS)]
    return float(np.mean(per_factor)), per_factor


def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total, steps = 0, 0
    pbar = tqdm(loader, desc="  train", leave=False, unit="batch", dynamic_ncols=True)
    for batch in pbar:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        ttype = batch.get("token_type_ids")
        if ttype is not None: ttype = ttype.to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(ids, mask, ttype)
        loss = criterion(logits.float(), labels) / GRAD_ACCUM
        loss.backward()

        if (steps + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total += loss.item() * GRAD_ACCUM
        steps += 1
        pbar.set_postfix(loss=f"{loss.item() * GRAD_ACCUM:.4f}")
    pbar.close()
    return total / max(steps, 1)


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
    return {"loss": total_loss / len(all_probs), "macro_f1": mf1, "per_factor": per_f,
            "probs": probs, "labels": labels}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--oversample", type=str, default=None,
                        choices=["factor_gpt", "factor_claude", "factor_llama",
                                 "factor_all", "factor_verified"])
    parser.add_argument("--tau", type=float, default=0.3,
                        help="Co-occurrence binarization threshold (ML-GCN default is 0.4, "
                             "lowered here given our much smaller/noisier co-occurrence estimates)")
    parser.add_argument("--gcn-p", type=float, default=0.2, help="ML-GCN re-weighting mass (paper default 0.2)")
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Output: {output_dir}")
    torch.manual_seed(SEED); np.random.seed(SEED)
    init_log(output_dir)

    tokenizer = get_tokenizer()
    df = load_train(oversample_source=args.oversample)
    print(f"Dataset: {len(df)} | Oversample: {args.oversample or 'none'}")

    orig_df = df[df["anon_user_id"] != "SYN"].reset_index(drop=True)
    syn_df = df[df["anon_user_id"] == "SYN"].reset_index(drop=True)

    # Adjacency computed from REAL rows only (not synthetic -- avoids the graph
    # itself being skewed by oversampling), same principle as the chain order.
    orig_labels_full = np.zeros((len(orig_df), NUM_FACTORS))
    for i, row in orig_df.iterrows():
        for f in row["factors_list"]:
            fl = f.lower().strip()
            if fl in FACTOR2ID:
                orig_labels_full[i, FACTOR2ID[fl]] = 1
    adjacency = build_adjacency(orig_labels_full, tau=args.tau, p=args.gcn_p)
    np.save(os.path.join(output_dir, "adjacency.npy"), adjacency)

    print("  Building factor-name embeddings from backbone's own embedding table...")
    factor_name_embeds, shared_backbone_template = build_factor_name_embeddings(MODEL_NAME)
    torch.save(factor_name_embeds, os.path.join(output_dir, "factor_name_embeds.pt"))
    del shared_backbone_template  # only needed transiently to read the embedding table

    gkf = GroupKFold(n_splits=args.folds)
    groups = orig_df["anon_user_id"].values

    oof_probs = np.zeros((len(orig_df), NUM_FACTORS))
    oof_labels = np.zeros((len(orig_df), NUM_FACTORS))

    for fold, (tr_i, va_i) in enumerate(gkf.split(orig_df, groups=groups)):
        tr_df = pd.concat([orig_df.iloc[tr_i], syn_df], ignore_index=True) if len(syn_df) else orig_df.iloc[tr_i]
        va_df = orig_df.iloc[va_i]
        print(f"\n{'='*60}\n  FOLD {fold+1}/{args.folds}  Train={len(tr_df)} Val={len(va_df)}\n{'='*60}")

        tr_loader = DataLoader(FactorDataset(tr_df, tokenizer), batch_size=BATCH_SIZE,
                               shuffle=True, num_workers=2, pin_memory=True)
        va_loader = DataLoader(FactorDataset(va_df, tokenizer), batch_size=BATCH_SIZE * 2,
                               shuffle=False, num_workers=2, pin_memory=True)

        model = GCNFactorClassifier(adjacency=adjacency, factor_name_embeds=factor_name_embeds).to(device).float()
        model.backbone.gradient_checkpointing_enable()
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        total_steps = len(tr_loader) * args.epochs // GRAD_ACCUM
        scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * WARMUP_RATIO), total_steps)

        best, patience_counter, best_state = 0, 0, None
        for epoch in range(args.epochs):
            t0 = time.time()
            t_loss = train_epoch(model, tr_loader, optimizer, scheduler, criterion, device)
            val = eval_epoch(model, va_loader, criterion, device)
            elapsed = time.time() - t0
            is_best = val["macro_f1"] > best
            if is_best:
                best, patience_counter, best_state = val["macro_f1"], 0, {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
            lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch+1}/{args.epochs} ({elapsed:.0f}s) lr={lr:.2e}  "
                  f"Train loss={t_loss:.4f}  Val loss={val['loss']:.4f}  MacroF1={val['macro_f1']:.4f}"
                  f"{'  **BEST**' if is_best else ''}")
            log_row(output_dir, {"fold": fold, "epoch": epoch + 1, "train_loss": f"{t_loss:.4f}",
                     "val_loss": f"{val['loss']:.4f}", "val_macro_f1": f"{val['macro_f1']:.4f}",
                     "best_mf1": f"{best:.4f}", "lr": f"{lr:.2e}", "elapsed": f"{elapsed:.0f}"})
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        model.load_state_dict(best_state)
        fold_dir = os.path.join(output_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        torch.save({"model": model.state_dict()}, os.path.join(fold_dir, "best.pt"))

        final = eval_epoch(model, va_loader, criterion, device)
        oof_probs[va_i] = final["probs"]
        oof_labels[va_i] = final["labels"]
        del model
        torch.cuda.empty_cache()

    np.save(os.path.join(output_dir, "oof_probs.npy"), oof_probs)
    np.save(os.path.join(output_dir, "oof_labels.npy"), oof_labels)
    oof_mf1, _ = compute_macro_f1(oof_probs, oof_labels)
    print(f"\n{'='*60}\n  True 5-fold OOF Macro F1 (t=0.5): {oof_mf1:.4f}\n{'='*60}")
    print(f"  Compare to independent-head baseline for the same oversample source "
          f"(factor_verified: 0.5806, factor_all: 0.5785) and the 2-way ensemble (0.5982).")

    json.dump({"oof_macro_f1": float(oof_mf1), "oversample": args.oversample,
                "tau": args.tau, "gcn_p": args.gcn_p},
               open(os.path.join(output_dir, "oof_summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()

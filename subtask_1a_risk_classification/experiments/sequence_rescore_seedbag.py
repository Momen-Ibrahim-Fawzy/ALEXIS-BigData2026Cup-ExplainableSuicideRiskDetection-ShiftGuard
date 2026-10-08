"""Parametrized copy of sequence_rescore.py -- identical LP-FT-free training
logic (this rescorer was already a from-scratch small Transformer, no
freeze/unfreeze stages needed), but:
  - takes --seq-dir (source of {train_oof_logits,leaderboard_logits}.npy)
    and --output-dir (where to save results) as CLI args instead of hardcoded
    constants, so it works for the production exp09 run (read-only source,
    regenerating its rescored logits since the original run never saved them)
    or any seed-bagged run.
  - SAVES the rescored OOF + leaderboard logits as .npy instead of writing
    the submission CSV directly -- the final CSV is only built by
    merge_seedbag_1a.py, from the ENSEMBLE of all sources, never from one
    source alone.

Usage:
    python sequence_rescore_seedbag.py --seq-dir ../results/exp09_deberta_large_lr_high/sequence_data --output-dir ../results_seedbag_1a/production_rescored
    python sequence_rescore_seedbag.py --seq-dir ../results/exp09_deberta_large_lr_high_seed101/sequence_data --output-dir ../results_seedbag_1a/seed101_rescored
"""

import argparse, os, sys, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(__file__))
from trainer import RISK2ID, ID2RISK, RISK_LEVELS

D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
DROPOUT = 0.1
N_FOLDS = 5
EPOCHS = 60
PATIENCE = 8
LR = 3e-4
CLASS_WEIGHTS = [0.669, 0.788, 1.045, 3.586]
SEED = 42


class SeqRescorer(nn.Module):
    def __init__(self, n_classes=4, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                 dropout=DROPOUT, max_len=64):
        super().__init__()
        self.in_proj = nn.Linear(n_classes, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model * 4,
                                            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.out_proj = nn.Linear(d_model, n_classes)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)  # start as pure identity (correction=0)

    def forward(self, logits, post_ids, pad_mask):
        h = self.in_proj(logits) + self.pos_emb(post_ids.clamp(max=self.pos_emb.num_embeddings - 1))
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        correction = self.out_proj(h)
        return logits + correction


class UserSeqDataset(Dataset):
    def __init__(self, users):
        self.users = users

    def __len__(self):
        return len(self.users)

    def __getitem__(self, i):
        return self.users[i]


def collate(batch, has_labels=True):
    max_len = max(len(u["post_ids"]) for u in batch)
    B = len(batch)
    logits = torch.zeros(B, max_len, 4)
    post_ids = torch.zeros(B, max_len, dtype=torch.long)
    pad_mask = torch.ones(B, max_len, dtype=torch.bool)
    labels = torch.full((B, max_len), -100, dtype=torch.long) if has_labels else None
    row_idx = [None] * B
    for bi, u in enumerate(batch):
        n = len(u["post_ids"])
        logits[bi, :n] = torch.tensor(u["logits"], dtype=torch.float32)
        post_ids[bi, :n] = torch.tensor(u["post_ids"], dtype=torch.long)
        pad_mask[bi, :n] = False
        if has_labels:
            labels[bi, :n] = torch.tensor(u["labels"], dtype=torch.long)
        row_idx[bi] = u["row_indices"]
    return logits, post_ids, pad_mask, labels, row_idx


def build_users(meta_df, logits, label_col=None):
    users = []
    for uid, g in meta_df.groupby("anon_user_id"):
        g = g.sort_values("post_id")
        idx = g.index.to_numpy()
        u = {"logits": logits[idx], "post_ids": g["post_id"].to_numpy(), "row_indices": idx}
        if label_col:
            u["labels"] = np.array([RISK2ID[r] for r in g[label_col]])
        users.append(u)
    return users


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", required=True, help="Dir with train_oof_logits.npy etc.")
    ap.add_argument("--output-dir", required=True, help="Where to save rescored logits")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    train_logits = np.load(os.path.join(args.seq_dir, "train_oof_logits.npy"))
    train_meta = pd.read_csv(os.path.join(args.seq_dir, "train_meta.csv"))
    lb_logits = np.load(os.path.join(args.seq_dir, "leaderboard_logits.npy"))
    lb_meta = pd.read_csv(os.path.join(args.seq_dir, "leaderboard_meta.csv"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_preds = train_logits.argmax(axis=1)
    true_labels = np.array([RISK2ID[r] for r in train_meta["risk_norm"]])
    base_wf1 = f1_score(true_labels, base_preds, average="weighted", zero_division=0)
    print(f"Baseline (classifier alone, true OOF)  WF1={base_wf1:.4f}")

    gkf = GroupKFold(n_splits=N_FOLDS)
    groups = train_meta["anon_user_id"].values
    class_w = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(device)

    all_rescored = np.zeros_like(train_logits)
    fold_models = []

    for fold, (tr_i, va_i) in enumerate(gkf.split(train_meta, groups=groups)):
        torch.manual_seed(SEED + fold)
        tr_meta, va_meta = train_meta.iloc[tr_i], train_meta.iloc[va_i]

        tr_users = build_users(tr_meta, train_logits, "risk_norm")
        va_users = build_users(va_meta, train_logits, "risk_norm")

        tr_ld = DataLoader(UserSeqDataset(tr_users), batch_size=16, shuffle=True,
                            collate_fn=lambda b: collate(b, True))
        va_ld = DataLoader(UserSeqDataset(va_users), batch_size=32, shuffle=False,
                            collate_fn=lambda b: collate(b, True))

        model = SeqRescorer().to(device)
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(weight=class_w, ignore_index=-100)

        best_wf1, best_state, patience_ctr = 0, None, 0
        for epoch in range(EPOCHS):
            model.train()
            for logits_b, post_ids_b, pad_mask_b, labels_b, _ in tr_ld:
                logits_b, post_ids_b = logits_b.to(device), post_ids_b.to(device)
                pad_mask_b, labels_b = pad_mask_b.to(device), labels_b.to(device)
                out = model(logits_b, post_ids_b, pad_mask_b)
                loss = criterion(out.reshape(-1, 4), labels_b.reshape(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            preds_all, labels_all = [], []
            with torch.no_grad():
                for logits_b, post_ids_b, pad_mask_b, labels_b, _ in va_ld:
                    logits_b, post_ids_b = logits_b.to(device), post_ids_b.to(device)
                    pad_mask_b = pad_mask_b.to(device)
                    out = model(logits_b, post_ids_b, pad_mask_b).cpu().numpy()
                    labels_np = labels_b.numpy()
                    mask_np = ~pad_mask_b.cpu().numpy()
                    preds_all.append(out[mask_np].argmax(axis=1))
                    labels_all.append(labels_np[mask_np])
            preds_all = np.concatenate(preds_all)
            labels_all = np.concatenate(labels_all)
            wf1 = f1_score(labels_all, preds_all, average="weighted", zero_division=0)

            if wf1 > best_wf1:
                best_wf1, patience_ctr = wf1, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE:
                    break

        print(f"  Fold {fold}: best val WF1={best_wf1:.4f} (epoch stopped {epoch+1})")
        model.load_state_dict(best_state)
        fold_models.append(model)

        model.eval()
        with torch.no_grad():
            for logits_b, post_ids_b, pad_mask_b, labels_b, row_idx_b in va_ld:
                logits_b, post_ids_b = logits_b.to(device), post_ids_b.to(device)
                pad_mask_b = pad_mask_b.to(device)
                out = model(logits_b, post_ids_b, pad_mask_b).cpu().numpy()
                mask_np = ~pad_mask_b.cpu().numpy()
                for bi, ridx in enumerate(row_idx_b):
                    n = len(ridx)
                    all_rescored[ridx] = out[bi, :n]

    rescored_preds = all_rescored.argmax(axis=1)
    rescored_wf1 = f1_score(true_labels, rescored_preds, average="weighted", zero_division=0)
    rescored_mf1 = f1_score(true_labels, rescored_preds, average="macro", zero_division=0)
    print(f"\nBaseline WF1:  {base_wf1:.4f}")
    print(f"Rescored WF1:  {rescored_wf1:.4f}  (true held-out OOF across all 5 folds)")
    print(f"Rescored MF1:  {rescored_mf1:.4f}")
    print(f"Improvement:   {rescored_wf1 - base_wf1:+.4f} WF1")

    # ── Apply the 5 fold-models (ensemble) to the leaderboard sequences ──
    lb_users = build_users(lb_meta.reset_index(drop=True), lb_logits, label_col=None)
    lb_ld = DataLoader(UserSeqDataset(lb_users), batch_size=32, shuffle=False,
                        collate_fn=lambda b: collate(b, False))

    lb_rescored = np.zeros_like(lb_logits)
    lb_counts = np.zeros(len(lb_logits))
    for model in fold_models:
        model.eval()
        with torch.no_grad():
            for logits_b, post_ids_b, pad_mask_b, _, row_idx_b in lb_ld:
                logits_b, post_ids_b = logits_b.to(device), post_ids_b.to(device)
                pad_mask_b = pad_mask_b.to(device)
                out = model(logits_b, post_ids_b, pad_mask_b).cpu().numpy()
                for bi, ridx in enumerate(row_idx_b):
                    n = len(ridx)
                    lb_rescored[ridx] += out[bi, :n]
                    lb_counts[ridx] += 1
    lb_rescored /= lb_counts[:, None]

    # Save rescored LOGITS (not yet softmaxed -- merge_seedbag_1a.py handles that,
    # consistently across all sources it ensembles).
    np.save(os.path.join(args.output_dir, "train_rescored_logits.npy"), all_rescored)
    np.save(os.path.join(args.output_dir, "leaderboard_rescored_logits.npy"), lb_rescored)
    train_meta.to_csv(os.path.join(args.output_dir, "train_meta.csv"), index=False)
    lb_meta.to_csv(os.path.join(args.output_dir, "leaderboard_meta.csv"), index=False)

    result = {
        "baseline_wf1": float(base_wf1),
        "rescored_wf1": float(rescored_wf1),
        "rescored_mf1": float(rescored_mf1),
        "improvement": float(rescored_wf1 - base_wf1),
    }
    with open(os.path.join(args.output_dir, "sequence_rescore_result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved rescored logits + result -> {args.output_dir}")


if __name__ == "__main__":
    main()

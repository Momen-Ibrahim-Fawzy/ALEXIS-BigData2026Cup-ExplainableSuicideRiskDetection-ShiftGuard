"""
User-sequence re-scoring layer for Sub2 (backlog item #1, part 2).

Same idea as Sub1a's version: every post is scored by the split-ensemble
independently, ignoring that train/leaderboard posts come from only 153/36
users (avg ~10.7/10.5 posts/user) with a post_id giving chronological order.
This trains a small Transformer over each user's *sequence* of combined
split-ensemble logits (24-dim, one per factor) to produce a residual
correction per post — multi-label version of the Sub1a rescorer (BCE loss
instead of CE, per-factor threshold re-tuning instead of argmax).

Usage:
    python extract_seq_data.py   # run once first
    python sequence_rescore.py
"""

import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from config import *

SEQ_DIR = os.path.join(os.path.dirname(__file__), "sequence_data")
SUBMISSIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "submissions")

D_MODEL = 96
N_HEADS = 4
N_LAYERS = 2
DROPOUT = 0.1
N_FOLDS = 5
EPOCHS = 60
PATIENCE = 8
LR = 3e-4
SEED = 42


class SeqRescorer(nn.Module):
    def __init__(self, n_factors=NUM_FACTORS, d_model=D_MODEL, n_heads=N_HEADS,
                 n_layers=N_LAYERS, dropout=DROPOUT, max_len=64):
        super().__init__()
        self.in_proj = nn.Linear(n_factors, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model * 4,
                                            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.out_proj = nn.Linear(d_model, n_factors)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)  # start as identity (correction=0)

    def forward(self, logits, post_ids, pad_mask):
        h = self.in_proj(logits) + self.pos_emb(post_ids.clamp(max=self.pos_emb.num_embeddings - 1))
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        return logits + self.out_proj(h)


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
    logits = torch.zeros(B, max_len, NUM_FACTORS)
    post_ids = torch.zeros(B, max_len, dtype=torch.long)
    pad_mask = torch.ones(B, max_len, dtype=torch.bool)
    labels = torch.zeros(B, max_len, NUM_FACTORS) if has_labels else None
    valid = torch.zeros(B, max_len, dtype=torch.bool)
    row_idx = [None] * B
    for bi, u in enumerate(batch):
        n = len(u["post_ids"])
        logits[bi, :n] = torch.tensor(u["logits"], dtype=torch.float32)
        post_ids[bi, :n] = torch.tensor(u["post_ids"], dtype=torch.long)
        pad_mask[bi, :n] = False
        valid[bi, :n] = True
        if has_labels:
            labels[bi, :n] = torch.tensor(u["labels"], dtype=torch.float32)
        row_idx[bi] = u["row_indices"]
    return logits, post_ids, pad_mask, labels, valid, row_idx


def build_users(meta_df, logits, labels=None):
    """meta_df must keep its ORIGINAL index (do not reset_index) — row_indices
    are used both to slice `logits`/`labels` and to write results back into a
    full-size output array, so they must refer to the same original positions."""
    users = []
    for uid, g in meta_df.groupby("anon_user_id"):
        g = g.sort_values("post_id")
        idx = g.index.to_numpy()
        u = {"logits": logits[idx], "post_ids": g["post_id"].to_numpy(), "row_indices": idx}
        if labels is not None:
            u["labels"] = labels[idx]
        users.append(u)
    return users


def tune_thresholds(probs, labels):
    candidates = np.arange(0.05, 0.95, 0.025)
    thresholds = np.full(NUM_FACTORS, 0.5)
    f1s = []
    for i in range(NUM_FACTORS):
        best_f1, best_t = 0.0, 0.5
        for t in candidates:
            f1 = f1_score(labels[:, i], (probs[:, i] >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[i] = best_t
        f1s.append(best_f1)
    return thresholds, np.mean(f1s)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():
    train_logits = np.load(os.path.join(SEQ_DIR, "train_oof_logits.npy"))
    train_labels = np.load(os.path.join(SEQ_DIR, "train_oof_labels.npy"))
    train_meta = pd.read_csv(os.path.join(SEQ_DIR, "train_meta.csv"))
    lb_logits = np.load(os.path.join(SEQ_DIR, "leaderboard_logits.npy"))
    lb_meta = pd.read_csv(os.path.join(SEQ_DIR, "leaderboard_meta.csv"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_probs = sigmoid(train_logits)
    _, base_macro = tune_thresholds(base_probs, train_labels)
    print(f"Baseline (split-ensemble alone, true OOF) tuned Macro F1={base_macro:.4f}")

    gkf = GroupKFold(n_splits=N_FOLDS)
    groups = train_meta["anon_user_id"].values

    all_rescored = np.zeros_like(train_logits)
    fold_models = []

    for fold, (tr_i, va_i) in enumerate(gkf.split(train_meta, groups=groups)):
        torch.manual_seed(SEED + fold)
        tr_meta, va_meta = train_meta.iloc[tr_i], train_meta.iloc[va_i]

        tr_users = build_users(tr_meta, train_logits, train_labels)
        va_users = build_users(va_meta, train_logits, train_labels)

        tr_ld = DataLoader(UserSeqDataset(tr_users), batch_size=16, shuffle=True,
                            collate_fn=lambda b: collate(b, True))
        va_ld = DataLoader(UserSeqDataset(va_users), batch_size=32, shuffle=False,
                            collate_fn=lambda b: collate(b, True))

        model = SeqRescorer().to(device)
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss(reduction="none")

        best_macro, best_state, patience_ctr = 0, None, 0
        for epoch in range(EPOCHS):
            model.train()
            for logits_b, post_ids_b, pad_mask_b, labels_b, valid_b, _ in tr_ld:
                logits_b, post_ids_b = logits_b.to(device), post_ids_b.to(device)
                pad_mask_b, labels_b, valid_b = pad_mask_b.to(device), labels_b.to(device), valid_b.to(device)
                out = model(logits_b, post_ids_b, pad_mask_b)
                loss_raw = criterion(out, labels_b)
                loss = (loss_raw * valid_b.unsqueeze(-1)).sum() / valid_b.sum().clamp(min=1) / NUM_FACTORS
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            probs_all, labels_all = [], []
            with torch.no_grad():
                for logits_b, post_ids_b, pad_mask_b, labels_b, valid_b, _ in va_ld:
                    logits_b, post_ids_b = logits_b.to(device), post_ids_b.to(device)
                    pad_mask_b = pad_mask_b.to(device)
                    out = torch.sigmoid(model(logits_b, post_ids_b, pad_mask_b)).cpu().numpy()
                    valid_np = valid_b.numpy()
                    probs_all.append(out[valid_np])
                    labels_all.append(labels_b.numpy()[valid_np])
            probs_all = np.concatenate(probs_all)
            labels_all = np.concatenate(labels_all)
            _, macro = tune_thresholds(probs_all, labels_all)

            if macro > best_macro:
                best_macro, patience_ctr = macro, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE:
                    break

        print(f"  Fold {fold}: best val tuned Macro F1={best_macro:.4f} (epoch stopped {epoch+1})")
        model.load_state_dict(best_state)
        fold_models.append(model)

        model.eval()
        with torch.no_grad():
            for logits_b, post_ids_b, pad_mask_b, labels_b, valid_b, row_idx_b in va_ld:
                logits_b, post_ids_b = logits_b.to(device), post_ids_b.to(device)
                pad_mask_b = pad_mask_b.to(device)
                out = model(logits_b, post_ids_b, pad_mask_b).cpu().numpy()
                for bi, ridx in enumerate(row_idx_b):
                    n = len(ridx)
                    all_rescored[ridx] = out[bi, :n]

    rescored_probs = sigmoid(all_rescored)
    _, rescored_macro = tune_thresholds(rescored_probs, train_labels)
    print(f"\nBaseline tuned Macro F1:  {base_macro:.4f}")
    print(f"Rescored tuned Macro F1:  {rescored_macro:.4f}  (true held-out OOF across all 5 folds)")
    print(f"Improvement:              {rescored_macro - base_macro:+.4f}")

    result = {
        "baseline_macro_f1": float(base_macro),
        "rescored_macro_f1": float(rescored_macro),
        "improvement": float(rescored_macro - base_macro),
    }
    with open(os.path.join(SEQ_DIR, "sequence_rescore_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    if rescored_macro <= base_macro:
        print("\n*** Sequence rescoring did NOT beat baseline on true OOF. "
              "Do not use it for a submission. ***")
        return

    # Final thresholds tuned on the full rescored OOF set (for leaderboard application)
    final_thresholds, _ = tune_thresholds(rescored_probs, train_labels)

    # ── Apply the 5 fold-models (ensemble) to the leaderboard sequences ──
    lb_users = build_users(lb_meta, lb_logits, labels=None)
    lb_ld = DataLoader(UserSeqDataset(lb_users), batch_size=32, shuffle=False,
                        collate_fn=lambda b: collate(b, False))

    lb_rescored = np.zeros_like(lb_logits)
    lb_counts = np.zeros(len(lb_logits))
    for model in fold_models:
        model.eval()
        with torch.no_grad():
            for logits_b, post_ids_b, pad_mask_b, _, valid_b, row_idx_b in lb_ld:
                logits_b, post_ids_b = logits_b.to(device), post_ids_b.to(device)
                pad_mask_b = pad_mask_b.to(device)
                out = model(logits_b, post_ids_b, pad_mask_b).cpu().numpy()
                for bi, ridx in enumerate(row_idx_b):
                    n = len(ridx)
                    lb_rescored[ridx] += out[bi, :n]
                    lb_counts[ridx] += 1
    lb_rescored /= lb_counts[:, None]
    lb_probs = sigmoid(lb_rescored)
    preds = (lb_probs >= final_thresholds).astype(int)

    rows = []
    for i in range(len(lb_meta)):
        facs = [FACTOR_LIST[j] for j in range(NUM_FACTORS) if preds[i, j] == 1]
        rows.append({"row_id": lb_meta.iloc[i]["row_id"],
                     "factors": str(facs) if facs else "[]"})
    out_df = pd.DataFrame(rows)
    n = out_df["factors"].apply(lambda x: len(eval(x)) if x != "[]" else 0)
    print(f"\nMean factors/post: {n.mean():.2f}")

    out_path = os.path.join(os.path.dirname(__file__), "factor_predictions_seqrescored.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
